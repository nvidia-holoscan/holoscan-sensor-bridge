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
 */
template <typename T>
class Nullable {
public:
    Nullable(T value = 0)
        : value_(value)
    {
    }
    Nullable(std::nullptr_t)
        : value_(0)
    {
    }
    operator T() const { return value_; };
    explicit operator bool() { return value_ != 0; }

    friend bool operator==(Nullable l, Nullable r) { return l.value_ == r.value_; }
    friend bool operator!=(Nullable l, Nullable r) { return !(l == r); }

    /**
     * Deleter, call the function when the object is deleted.
     *
     * @tparam F function to call
     */
    template <typename RESULT, RESULT func(T)>
    struct Deleter {
        typedef Nullable<T> pointer;
        void operator()(T value) const { func(value); }
    };

private:
    T value_;
};

#endif /* SRC_HOLOLINK_NATIVE_NULLABLE_POINTER */
