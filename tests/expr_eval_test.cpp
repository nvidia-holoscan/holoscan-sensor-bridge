/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// The expression evaluator (hololink/expr_eval/expr_eval.cuh) writes the constants and the
// numbers of an expression into the CUDA source it compiles. These tests check that a value
// keeps its float precision in that source.

#include <hololink/expr_eval/expr_eval.cuh>

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <cmath>
#include <string>

namespace {

using namespace hololink::expr_eval;

// Compiles expression_string with the predefined constants, evaluates it once, and returns
// the result.
float evaluate(const std::string& expression_string)
{
    // NVRTC compiles for the device of the current context; make the primary context current.
    EXPECT_EQ(cudaFree(nullptr), cudaSuccess);

    ConstantsSymbolTable constants_symbol_table;
    add_predefined_constants(constants_symbol_table);
    Parser parser;
    Expression expression = parser.compile(expression_string, VariablesSymbolTable {}, constants_symbol_table);
    EXPECT_TRUE(expression);
    if (!expression) {
        return NAN;
    }

    float* device_output = nullptr;
    EXPECT_EQ(cudaMalloc(&device_output, sizeof(float)), cudaSuccess);
    expression.evaluate(device_output, 1);
    float result = NAN;
    EXPECT_EQ(cudaMemcpy(&result, device_output, sizeof(result), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(cudaFree(device_output), cudaSuccess);
    return result;
}

TEST(ExprEvalTest, PiHasFloatPrecision)
{
    EXPECT_EQ(evaluate("PI"), static_cast<float>(M_PI));
}

TEST(ExprEvalTest, NumbersHaveFloatPrecision)
{
    EXPECT_EQ(evaluate("0.1234567"), 0.1234567f);
    EXPECT_EQ(evaluate("1.0000001"), 1.0000001f);
    EXPECT_EQ(evaluate("16777215"), 16777215.0f);
}

} // namespace
