/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iomanip>
#include <memory>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <cuda.h>
#include <curand_kernel.h>
#include <nvrtc.h>
#include <thrust/device_vector.h>

#include <hololink/common/cuda_helper.hpp>
#include <hololink/core/logging_internal.hpp>

#include "expr_eval.cuh"

namespace hololink::expr_eval {

static const size_t thread_block_size = 1024;

// A map of all supported device native functions
static const std::unordered_map<std::string, std::string> native_function_map {
  { "acos", "acosf"},
  { "acosh", "acoshf"},
  { "asin", "asinf"},
  { "asinh", "asinhf"},
  { "atan2", "atan2f"},
  { "atan", "atanf"},
  { "atanh", "atanhf"},
  { "cbrt", "cbrtf"},
  { "ceil", "ceilf"},
  { "copysign", "copysignf"},
  { "cos", "cosf"},
  { "cosh", "coshf"},
  { "cospi", "cospif"},
  { "cyl_bessel_i0", "cyl_bessel_i0f"},
  { "cyl_bessel_i1", "cyl_bessel_i1f"},
  { "erfc", "erfcf"},
  { "erfcinv", "erfcinvf"},
  { "erfcx", "erfcxf"},
  { "erf", "erff"},
  { "erfinv", "erfinvf"},
  { "exp10", "exp10f"},
  { "exp2", "exp2f"},
  { "exp", "expf"},
  { "expm1", "expm1f"},
  { "abs", "fabsf"},
  { "dim", "fdimf"},
  { "divide", "fdividef"},
  { "floor", "floorf"},
  { "fma", "fmaf"},
  { "fmax", "fmaxf"},
  { "fmin", "fminf"},
  { "fmod", "fmodf"},
  { "hypot", "hypotf"},
  { "j0", "j0f"},
  { "j1", "j1f"},
  { "lgamma", "lgammaf"},
  { "log10", "log10f"},
  { "log1p", "log1pf"},
  { "log2", "log2f"},
  { "logb", "logbf"},
  { "log", "logf"},
  { "max", "max"},
  { "min", "min"},
  { "modf", "modff"},
  { "nearbyint", "nearbyintf"},
  { "nextafter", "nextafterf"},
  { "norm3d", "norm3df"},
  { "norm4d", "norm4df"},
  { "normcdf", "normcdff"},
  { "normcdfinv", "normcdfinvf"},
  { "norm", "normf"},
  { "pow", "powf"},
  { "rcbrt", "rcbrtf"},
  { "remainder", "remainderf"},
  { "remquo", "remquof"},
  { "rhypot", "rhypotf"},
  { "rint", "rintf"},
  { "rnorm3d", "rnorm3df"},
  { "rnorm4d", "rnorm4df"},
  { "rnorm", "rnormf"},
  { "round", "roundf"},
  { "rsqrt", "rsqrtf"},
  { "sin", "sinf"},
  { "sinh", "sinhf"},
  { "sinpi", "sinpif"},
  { "sqrt", "sqrtf"},
  { "tan", "tanf"},
  { "tanh", "tanhf"},
  { "tgamma", "tgammaf"},
  { "trunc", "truncf"},
  { "y0", "y0f"},
  { "y1", "y1f"}
};

// A map of all non-native supported functions
static const std::unordered_map<std::string, std::string> custom_function_map {
  { "uniform_rand", "_uniform_rand" }
};

// Initialize curand
__global__ void rand_init_kernel(curandState* states, size_t size, unsigned long long seed)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= size)
      return;
    // Each thread gets same seed, a different sequence number, no offset
    curand_init(seed, id, 0, &states[id]);
}

static void log_and_throw(const std::string& msg) {
  HSB_LOG_ERROR(msg);
  throw std::runtime_error(msg);
}

// Parallel Thread Execution - a Cuda compilation result.
using CudaPtx = std::string;

// A wrapper class for nvrtcProgram
class CudaProgram {
public:
  CudaProgram(const std::string& cuda_code) {
      // Generate a unique program name
      std::ostringstream oss;
      oss << "expr_eval_" << std::setw(sizeof(this) * 2) << std::setfill('0') << std::hex << reinterpret_cast<uintptr_t>(this) << ".cu" << std::dec;
      name_ = oss.str();
      HSB_LOG_DEBUG("Creating Cuda program: {}", name_);
      NVRTC_CHECK(nvrtcCreateProgram(&program_, cuda_code.c_str(), name_.c_str(), 0, nullptr, nullptr));
  }

  // Non-copyable
  CudaProgram(CudaProgram&) = delete;
  CudaProgram& operator=(CudaProgram&) = delete;
  
  ~CudaProgram() try {
    HSB_LOG_DEBUG("Destroying Cuda program: {}", name_);
    NVRTC_CHECK(nvrtcDestroyProgram(&program_));
  } catch (const std::exception&) {
    return;
  }

  // Compiling the Cuda source code provided in the constructor
  void compile(const std::vector<const char*>& options = std::vector<const char*>()) try {
    HSB_LOG_DEBUG("Compiling Cuda program: {}", name_);
    NVRTC_CHECK(nvrtcCompileProgram(program_, options.size(), options.size() ? &options.front() : nullptr));
  } catch (const std::exception&) {
    HSB_LOG_ERROR("{}", get_log());
    throw;
  }

  // Get the program log
  std::string get_log() const {
    size_t log_size;
    NVRTC_CHECK(nvrtcGetProgramLogSize(program_, &log_size));
    std::string log(log_size, '\0');
    NVRTC_CHECK(nvrtcGetProgramLog(program_, log.data()));
    return log;
  }

  // Get the PTX (The compilation artifact)
  CudaPtx get_ptx() const {
    size_t ptx_size;
    NVRTC_CHECK(nvrtcGetPTXSize(program_, &ptx_size));
    CudaPtx ptx(ptx_size, '\0');
    NVRTC_CHECK(nvrtcGetPTX(program_, ptx.data()));
    return ptx;
  }

private:
  std::string name_;
  nvrtcProgram program_;
};

// A wrapper class for CUmodule
// Movable only
class CudaModule {
public:
  CudaModule() = default;
  CudaModule(const CudaPtx& ptx) {
    CudaCheck(cuModuleLoadDataEx(&module_, ptx.data(), 0, nullptr, nullptr));
  }

  CudaModule(CudaModule&) = delete;
  CudaModule(CudaModule&& other) :
    CudaModule() {
    swap(*this, other);
  }
  CudaModule& operator=(CudaModule&) = delete;
  CudaModule& operator=(CudaModule&& other) {
    swap(*this, other);
    return *this;
  }
  
  ~CudaModule() try {
    if (module_)
      CudaCheck(cuModuleUnload(module_));
  } catch (const std::exception&) {
    return;
  }

  // Get a Cuda function from a Cuda module
  CUfunction get_function(const char* function_name) {
    CUfunction cuda_function;
    CudaCheck(cuModuleGetFunction(&cuda_function, module_, function_name));
    return cuda_function;
  }

  void swap(CudaModule& lhs, CudaModule& rhs) noexcept {
    using std::swap;
    swap(lhs.module_, rhs.module_);
  }

private:
  CUmodule module_{};
};

// The string expression is first parsed and converted into a string of Tokens
enum class Token {
  NIL,         // An empty token
  IDENTIFIER,  // Variables and function names
  NUMBER,
  LPARAM,
  RPARAM,
  PLUS,
  MINUS,
  MUL,
  DIV,
  COMMA
};

void add_predefined_constants(ConstantsSymbolTable& constants_symbol_table) {
  constants_symbol_table.emplace("PI", M_PI);
}

// Impl holds the CudaModule and the Cuda function that evaluates the expression
struct Expression::Impl {
  Impl(CudaModule cuda_module, CUfunction cuda_function, const std::string& cuda_toolkit_include_path) :
    cuda_module_(std::move(cuda_module)),
    cuda_function_(std::move(cuda_function)),
    cuda_toolkit_include_path_(cuda_toolkit_include_path),
    seed_([]{
      // Generate a seed for random numbers
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_int_distribution<SeedType> distrib(std::numeric_limits<SeedType>::min(), std::numeric_limits<SeedType>::max());
      return distrib(gen);
    }())
  {
  }
  void evaluate(float* device_output, size_t count, size_t stride, void* device_data);

    CudaModule cuda_module_;
  CUfunction cuda_function_;
  using SeedType  = unsigned long long;
  SeedType seed_;
  std::string cuda_toolkit_include_path_;  // The path to curand is required if curand functions are used
  thrust::device_vector<curandState> curand_states_;
};

Expression::Expression(std::shared_ptr<Impl> pimpl) :
  pimpl_(std::move(pimpl))
{
}

void Expression::Impl::evaluate(float* device_output, size_t count, size_t stride, void* device_data) {

  if (stride == 0)
    log_and_throw("The stride cannot be 0");

  // Curand states is costy and therefore is initialized separately and only once.
  curandState* curand_states = nullptr;
  if (!cuda_toolkit_include_path_.empty()) {
    // Initialize curand_states.
    if (curand_states_.size() != count) {
      curand_states_.resize(count);
      rand_init_kernel<<<static_cast<size_t>(std::ceil(1.0 * count / thread_block_size)),
                        thread_block_size>>>(curand_states_.data().get(), count, seed_);
    }
    curand_states = curand_states_.data().get();

  }

  // Launch the kernel
  void* args[] = {&device_output, &count, &stride, &device_data, &curand_states };
  cuLaunchKernel(cuda_function_,
    static_cast<size_t>(std::ceil(1.0 * count / thread_block_size)), 1, 1,
    thread_block_size, 1, 1,
    0, nullptr, args, nullptr);
}

void Expression::evaluate(float* device_output, size_t count, size_t stride, void* device_data) const {
  if (!pimpl_)
    log_and_throw("Invalid expression");
  pimpl_->evaluate(device_output, count, stride, device_data);
}

Expression::operator bool() const {
  return static_cast<bool>(pimpl_);
}

Expression Parser::compile(const std::string& expression_str,
                           const VariablesSymbolTable& variables_symbol_table,
                           const ConstantsSymbolTable& constants_symbol_table) try {
  current_token_index_ = 0;
  tokens_ = tokenize(expression_str); // Validate expression_str and create a string of tokens

  std::stringstream ss;
  // Check if curand is available
  if (!cuda_toolkit_include_path_.empty())
    ss << 
R"(
#include <curand_kernel.h>
)";
  else 
    ss << 
R"(
// Dummy curand structures to satisfy the compilation
struct curandState {};
__device__ inline float curand_uniform(curandState*) {
  return 0.0f;
}
)";
  ss <<
R"(
// The struct holds data that can be used by the Custom Functions
struct CustomFunctionData {
  curandState* curand_state_ptr_;
};

__device__ float _uniform_rand(const CustomFunctionData* custom_function_data) {
  auto curand_state_ptr = custom_function_data->curand_state_ptr_;
  return curand_uniform(curand_state_ptr);
}
)";

  // For each of the variables in the variables_symbol_table,
  // define a variable getter function
  for (const auto& [variable_name, variable_code] : variables_symbol_table)
    ss << "__device__ float _get_" << variable_name << "(const void* data) {\n" <<
      // Brings in the user's Cuda code
      variable_code << "\n\n}";

  // Defining the main Cuda kernel function
  static const char* cuda_kernel_name = "evaluate_expression_kernel";
  ss <<
R"(
extern "C" __global__ void )" << cuda_kernel_name << R"((float* output, int count, int stride, const void* data, curandState* curand_states) {
  size_t output_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (output_index >= count) return; // Check to see of out-of-bounds

  // Initialize the CustomFunctionData
  curandState* curand_state_ptr = nullptr;
  if (curand_states)
    curand_state_ptr = &curand_states[output_index];
  CustomFunctionData custom_function_data {
    curand_state_ptr
  };

  output_index *= stride;
)";

  // Define constants
  for (const auto& [constant_name, constant_value] : constants_symbol_table)
    ss <<
      "  float " << constant_name << " = " << constant_value << ";\n";

  // Define variables
  for (const auto& [variable_name, _] : variables_symbol_table)
    ss <<
      "  float " << variable_name << " = _get_" << variable_name << "(data);\n";

  // Evaluate the expression and set the result
  ss << "  output[output_index] = ";
  expression(ss);
  ss << ";\n";
  
  // Close the Cuda Kernel function
  ss << "}\n";

  // Check that the full expression was parsed
  if (current_token_index_ != tokens_.size()) log_and_throw("expression ended unexpectedly");

  // Create a Cuda Program
  auto cuda_code = ss.str();
  HSB_LOG_DEBUG("Expression Evaluator Cuda code:\n{}", cuda_code);
  CudaProgram cuda_program(cuda_code);

  // Compile
  std::vector<const char*> options;
  std::string opt_cuda_toolkit_include_path;
  if (!cuda_toolkit_include_path_.empty()) {
    opt_cuda_toolkit_include_path = "--include-path=" + cuda_toolkit_include_path_;
    options.push_back(&opt_cuda_toolkit_include_path.front());
  }
  cuda_program.compile(options);

  // Create a Cuda module and get the Cuda Function
  CudaModule cuda_module(cuda_program.get_ptx());
  CUfunction cuda_function(cuda_module.get_function(cuda_kernel_name));
  
  // Create an Expression object and return it
  return expr_eval::Expression(std::make_shared<expr_eval::Expression::Impl>(std::move(cuda_module), std::move(cuda_function), cuda_toolkit_include_path_));
} catch (const std::exception& err) { return expr_eval::Expression(); }

void Parser::set_cuda_toolkit_include_path(const std::string& cuda_toolkit_include_path) try {
  // Validate curand_include_ path
  if (!cuda_toolkit_include_path.empty()) {
    CudaProgram cuda_program("#include <curand_kernel.h>");
    // Compile
    std::string opt_cuda_toolkit_include_path = "--include-path=" + cuda_toolkit_include_path;
    std::vector<const char*> options { &opt_cuda_toolkit_include_path.front() };
    cuda_program.compile(options);
  }
  cuda_toolkit_include_path_ = cuda_toolkit_include_path;
} catch (const std::exception&) {
  std::stringstream ss;
  ss << "Invalid cuda toolkit include path: " << cuda_toolkit_include_path;
  log_and_throw(ss.str());
}

class Parser::Tokenizer {
 public:
  Parser::Tokens tokenize(const std::string& expression_string) {
    static std::unordered_map<char, Token> kChar2TokenMap{{'+', Token::PLUS},
                                                          {'-', Token::MINUS},
                                                          {'*', Token::MUL},
                                                          {'/', Token::DIV},
                                                          {'(', Token::LPARAM},
                                                          {')', Token::RPARAM},
                                                          {',', Token::COMMA}};

    for (auto c : expression_string) {
      if (c == ' ' || c == '\t' || c == '\n') {
        push_current_token();
        continue;
      }
      if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_') {
        if (current_token_ == Token::NIL) current_token_ = Token::IDENTIFIER;
        else if (current_token_ != Token::IDENTIFIER)
          log_and_throw("Invalid identifier");
        current_token_string_.push_back(c);
        continue;
      }
      if (c >= '0' && c <= '9') {
        if (current_token_ == Token::NIL) {
          current_token_ = Token::NUMBER;
          decimal_ = false;
        } else if (current_token_ != Token::IDENTIFIER && current_token_ != Token::NUMBER)
          log_and_throw("Invalid number");
        current_token_string_.push_back(c);
        continue;
      }
      if (c == '.') {
        if (current_token_ == Token::NIL) current_token_ = Token::NUMBER;
        else if (current_token_ != Token::NUMBER || decimal_)
          log_and_throw("Invalid token");
        decimal_ = true;
        current_token_string_.push_back(c);
        continue;
      }
      auto iter = kChar2TokenMap.find(c);
      if (iter != kChar2TokenMap.end()) {
        push_current_token();
        current_token_string_.push_back(c);
        current_token_ = iter->second;
        push_current_token();
        continue;
      }
      log_and_throw("Unexpected character");
    }
    push_current_token();
    return tokens_;
  }

 private:
  void push_current_token() {
    if ((current_token_ != Token::NIL) && current_token_string_.size()) {
      tokens_.emplace_back(current_token_, current_token_string_);
      current_token_ = Token::NIL;
      current_token_string_.clear();
    };
  }
  Tokens tokens_;
  Token current_token_ = Token::NIL;
  std::string current_token_string_;
  bool decimal_;
};

Parser::Tokens Parser::tokenize(const std::string& expression_string) {
  Tokenizer tokenizer;
  return tokenizer.tokenize(expression_string);
}

Token Parser::current_token() const {
  if (current_token_index_ >= tokens_.size())
    return Token::NIL;
  return tokens_[current_token_index_].first;
}

void Parser::next_token() {
  ++current_token_index_;
}

bool Parser::accept(Token token) {
  if (current_token() == token) {
    next_token();
    return true;
  }
  return false;
}

bool Parser::expect(Token token) {
  if (accept(token)) return true;
  log_and_throw("Unexpected token");
  return false;
}

void Parser::factor(std::ostream& os) {
  if (accept(Token::IDENTIFIER)) {
    auto identifier = tokens_[current_token_index_ - 1].second;
    if (accept(Token::LPARAM)) {
      if (custom_function_map.count(identifier))
        os << custom_function_map.at(identifier) << '(';
      else if (native_function_map.count(identifier))
        os << native_function_map.at(identifier) << '(';
      else {
        std::stringstream ss;
        ss << "Function '" << identifier << "' is not supported";
        log_and_throw(ss.str());
      }

      if (current_token() != Token::RPARAM) { // If function with arguments
        expression(os);

        while (accept(Token::COMMA)) {
          os << ',';
          expression(os);
        }

        if (custom_function_map.count(identifier))  // custom function data is the next argument
          os << ", ";
      }

      if (custom_function_map.count(identifier))  // custom function data
        os << "&custom_function_data";

      expect(Token::RPARAM);
      os << ')';
    }
    else
      os << identifier;
  } else if (accept(Token::NUMBER)) {
    // current_token_index_ already incremented
    float value = std::stof(tokens_[current_token_index_ - 1].second);
    if (std::fmod(value, 1.0f) == 0)
      os << std::fixed;
    else
      os << std::defaultfloat;
    os << value << 'f';
  } else if (accept(Token::LPARAM)) {
      os << '(';
    expression(os);
    expect(Token::RPARAM);
    os << ')';
  } else
    log_and_throw("factor: syntax error");
}

void Parser::term(std::ostream& os) {
  factor(os);
  while (current_token() == Token::MUL || current_token() == Token::DIV) {
    auto token = current_token();
    os << (token == Token::MUL ? '*' : '/');
    next_token();
    factor(os);
  }
}

void Parser::expression(std::ostream& os) {
  if (current_token() == Token::PLUS) next_token();
  else if (current_token() == Token::MINUS) {
    os << '-';
    next_token();  // Unary operator
  }
  term(os);
  while (current_token() == Token::PLUS || current_token() == Token::MINUS) {
    auto token = current_token();
    os << (token == Token::PLUS ? '+' : '-');
    next_token();
    term(os);
  }
}

}  // namespace hololink::expr_eval
