/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef SRC_HOLOLINK_NATIVE_NULLABLE_POINTER
#define SRC_HOLOLINK_NATIVE_NULLABLE_POINTER

#include <cstddef>

/**
 * Helper class for using handles with std::unique_ptr which requires that a custom
 * handle type satisfies NullablePointer https://en.cppreference.com/w/cpp/named_req/NullablePointer.
 *
 * @tparam T type to hold
 * @tparam Null sentinel value that represents "no handle". Defaults to 0, which
 *   is correct for pointer-like handles (e.g. CUdeviceptr). File descriptors must
 *   use -1 instead: 0 is a *valid* fd (it is stdin), so a Nullable<int, 0> would
 *   treat a socket that legitimately lands on fd 0 (once stdin has been closed)
 *   as empty, and callers checking `!handle` would wrongly report failure.
 */
template <typename T, T Null = T {}>
class Nullable {
public:
    Nullable(T value = Null)
        : value_(value)
    {
    }
    Nullable(std::nullptr_t)
        : value_(Null)
    {
    }
    operator T() const { return value_; };
    explicit operator bool() const { return value_ != Null; }

    friend bool operator==(Nullable l, Nullable r) { return l.value_ == r.value_; }
    friend bool operator!=(Nullable l, Nullable r) { return !(l == r); }

    /**
     * Deleter, call the function when the object is deleted.
     *
     * @tparam F function to call
     */
    template <typename RESULT, RESULT func(T)>
    struct Deleter {
        typedef Nullable<T, Null> pointer;
        void operator()(T value) const { func(value); }
    };

private:
    T value_;
};

#endif /* SRC_HOLOLINK_NATIVE_NULLABLE_POINTER */
