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
#ifndef SRC_HOLOLINK_OPERATORS_SIG_GEN_SIG_GEN_OP_CUH
#define SRC_HOLOLINK_OPERATORS_SIG_GEN_SIG_GEN_OP_CUH

#include "hololink/expr_eval/expr_eval.cuh"

#include "hololink/operators/sig_gen/sig_gen_op_rational.hpp"

namespace hololink::operators {

// Creates a signal expression object from a string, ready to be evaluated in the GPU.
expr_eval::Expression create_signal_expression(expr_eval::Parser& parser, const std::string& signal_expression_string);

// Evaluates an expression and stores the results on a GPU buffer (dvc_buffer).
// dvc_buffer is a pointer to Cuda memory
// Returns the next samples index to be evaluated
size_t evaluate_expression(const expr_eval::Expression& expression, float* dvc_buffer, size_t size,
                        size_t samples_index, const Rational& sampling_interval, size_t stride = 1);

}  // namespace hololink::operators

#endif /* SRC_HOLOLINK_OPERATORS_SIG_GEN_SIG_GEN_OP_CUH */
