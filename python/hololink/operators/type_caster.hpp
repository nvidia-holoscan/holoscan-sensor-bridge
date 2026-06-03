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

#ifndef PYTHON_HOLOLINK_OPERATORS_TYPE_CASTER
#define PYTHON_HOLOLINK_OPERATORS_TYPE_CASTER

#include <cuda.h>
#include <pybind11/pybind11.h>

// Pybind can't handle incomplete types, e.g., `CUstream` is defined as `typedef struct CUstream_st *CUstream` in CUDA headers.
struct CUstream_st { };
struct CUevent_st { };

namespace pybind11 {
namespace detail {

    /**
     * Type caster for CUDA types.
     */
    template <typename CUDA_TYPE>
    class cuda_type_caster : public type_caster<void_type> {
    public:
        using type_caster<void_type>::cast;

        /**
         * C++ -> Python: convert a CUstream into a Python object.
         * The second and third arguments are used to indicate the return
         * value policy and parent object (for ``return_value_policy::reference_internal``) and are generally
         * ignored by implicit casters.
         */
        static handle cast(const CUDA_TYPE* ptr, [[maybe_unused]] return_value_policy policy, [[maybe_unused]] handle parent)
        {
            if (ptr) {
                return capsule(ptr).release();
            }
            return none().release();
        }

        /**
         * Python->C++: convert a PyObject into a
         * CUstream or return false upon failure. The
         * second argument indicates whether implicit conversions should be applied.
         */
        bool load(handle h, [[maybe_unused]] bool convert)
        {
            if (!h) {
                return false;
            }
            if (h.is_none()) {
                value = nullptr;
                return true;
            }

            // Check if handle is a capsule
            if (!isinstance<capsule>(h)) {
                return false;
            }
            value = reinterpret_borrow<capsule>(h);
            return true;
        }

        template <typename T>
        using cast_op_type = CUDA_TYPE*&;
        explicit operator CUDA_TYPE*&() { return value; }
        static constexpr auto name = const_name("CUDA_TYPE");

    private:
        CUDA_TYPE* value = nullptr;
    };

    template <>
    class type_caster<CUstream_st>
        : public cuda_type_caster<CUstream_st> {
    };

    template <>
    class type_caster<CUevent_st>
        : public cuda_type_caster<CUevent_st> {
    };

} // namespace detail
} // namespace pybind11

#endif /* PYTHON_HOLOLINK_OPERATORS_TYPE_CASTER */
