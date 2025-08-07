/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef PYTHON_HOLOLINK_OPERATORS_OPERATOR_UTIL
#define PYTHON_HOLOLINK_OPERATORS_OPERATOR_UTIL

#include <pybind11/pybind11.h>

#include <memory>

#include <hololink/core/logging_internal.hpp>
#include <holoscan/core/condition.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/resource.hpp>

namespace py = pybind11;

namespace hololink {

/**
 * Currently there is a limitation in Python operators that wrap an underlying C++ operator that
 * conditions like CountCondition or PeriodicCondition cannot be passed directly. This is being
 * resolved for Holoscan SDK's built-in operators in a future release (v2.1).
 *
 * @param op
 * @param args
 */
void add_positional_condition_and_resource_args(holoscan::Operator* op, const py::args& args)
{
    for (auto it = args.begin(); it != args.end(); ++it) {
        if (py::isinstance<holoscan::Condition>(*it)) {
            op->add_arg(it->cast<std::shared_ptr<holoscan::Condition>>());
        } else if (py::isinstance<holoscan::Resource>(*it)) {
            op->add_arg(it->cast<std::shared_ptr<holoscan::Resource>>());
        } else {
            HSB_LOG_WARN("Unhandled positional argument detected (only Condition and Resource "
                         "objects can be parsed positionally)");
        }
    }
}

} // namespace hololink

#endif /* PYTHON_HOLOLINK_OPERATORS_OPERATOR_UTIL */
