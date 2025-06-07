/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

#ifndef SRC_HOLOLINK_EXPR_EVAL_EXPR_EVAL_CUH
#define SRC_HOLOLINK_EXPR_EVAL_EXPR_EVAL_CUH

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <hololink/common/cuda_error.hpp>
#include <hololink/core/logging_internal.hpp>

namespace hololink::expr_eval {

enum class Token;

/***
 * A Map from variable name to the Cuda source code that returns it
 * Key   - Variable name
 * Value - Cuda source code that returns the variable value
 * 
 * Variable values should be accessible in Cuda.
 * Users need to provide the Cuda code that returns the variable value.
 * Users can assume the existence of "void* data" variable that is passed by the "evaluate" function.
 * This variable should point to a device memory.
 * Users can use any standard Cuda variables or functions like blockIdx and threadIdx. 
 * 
 * Examples:
 * 1. Cuda source code example that returns a fixed value - 0:
 *    return 0;
 * 
 * 2. Cuda source code that uses the "data" variable that holds the variable's value.
 *    struct CustomData {
 *        double value;
 *    };
 *    return reinterpret_cast<const CustomData*>(data)->value;
 */
using VariablesSymbolTable = std::unordered_map<std::string, std::string>;

/**
 * A map from constant name to its value.
 * Key   - Constant name
 * Value - Constant value
 * 
 * Similar to the VariablesSymbolTable but the values are defined during the compilation
 * of the expression instead of during the evaluation of the expression.
 */
using ConstantsSymbolTable = std::unordered_map<std::string, float>;

/**
 * Add predefined constants to constants_symbol_table
 * 
 * PI = 3.14159265358979323846
 */
void add_predefined_constants(ConstantsSymbolTable& constants_symbol_table);

// The Expression object is the result of the compile function.
// It's a movable only object
class Expression {
 public:
  struct Impl;
  Expression() = default;;
  Expression(std::shared_ptr<Impl> pimpl);
  Expression(Expression&&) = default;
  Expression& operator=(Expression&&) = default;

  /**
   * Evaluate the expression 'count' times.
   * 
   * output      - A pointer to device memory to write the evaluation results. 
   *               The memory is required to be big enough to hold 'count' results.
   * count       - The number of evaluations to perform. Also the size of device_output
   * device_data - An optional pointer to user defined data. This pointer can be used by
   *               the user defined variables. See VariablesSymbolTable for more details.
   */
  void evaluate(float* device_output, size_t count, size_t stride = 1, void* device_data = nullptr) const;

  // Returns true if expression is valid
  explicit operator bool() const;

 private:
  std::shared_ptr<Impl> pimpl_;
};

// The Parser object can parse an expression string and build an Expression object out of it.
class Parser {
 public:
  /**
   * Compiles the expression_string and returns an Expression object
   * 
   * expression_string      - The expression in a string format
   * variables_symbol_table - The variable symbol table. See VariablesSymbolTable for more info
   * constants_symbol_table - The constant symbol table. See ConstantsSymbolTable for more info
   */
  expr_eval::Expression compile(const std::string& expression_string,
                                const VariablesSymbolTable& variables_symbol_table,
                                const ConstantsSymbolTable& constants_symbol_table);
  
  /**
   * To use the curand (random) functions, the path to cuda toolkit include folder should be provided
   */
  void set_cuda_toolkit_include_path(const std::string& cuda_toolkit_include_path);

 private:
  Token current_token() const;  // Returns the current token
  void next_token();            // Sets the current token to the next one

  bool accept(Token token);  // Calls NextToken if current token is token
  bool expect(Token token);  // Same as Accept but throws if current token is not token

  /*
  expression = ["+"|"-"] term {("+"|"-") term}
  */
  void expression(std::ostream& os);

  /*
  term = factor {("*"|"/") factor}
  */
  void term(std::ostream& os);

  /*
    factor =
       identifier
       | number
       | "(" expression ")"
  */
  void factor(std::ostream& os);

  // Each token is a pair of the token enum (Token) and the string that defines it
  using TokenPair = std::pair<Token, std::string>;
  using Tokens = std::vector<TokenPair>;
  // Creates Tokens from an expression_string
  class Tokenizer;
  static Tokens tokenize(const std::string& expression_string);

  size_t current_token_index_;
  Tokens tokens_;
  std::string cuda_toolkit_include_path_;
};

}  // namespace hololink::expr_eval

#endif /* SRC_HOLOLINK_EXPR_EVAL_EXPR_EVAL_CUH */
