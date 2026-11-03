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

#ifndef SRC_HOLOLINK_CORE_ON_SCOPE_EXIT
#define SRC_HOLOLINK_CORE_ON_SCOPE_EXIT

#include <utility>

namespace hololink::core {

/**
 * Runs a cleanup action on scope exit, including during stack unwinding, unless
 * dismiss() has been called. Construct with a callable taking no arguments; the
 * action fires unconditionally unless dismiss() is called first, e.g. when a
 * later step takes over responsibility for the cleanup (see
 * FusaCoeCaptureOp::compute()'s release_fusa_buffer for that pattern).
 */
template <typename F>
class OnScopeExit {
public:
    explicit OnScopeExit(F fn)
        : fn_(std::move(fn))
    {
    }
    ~OnScopeExit()
    {
        if (armed_) {
            fn_();
        }
    }
    OnScopeExit(const OnScopeExit&) = delete;
    OnScopeExit& operator=(const OnScopeExit&) = delete;

    void dismiss() { armed_ = false; }

private:
    F fn_;
    bool armed_ = true;
};

} // namespace hololink::core

#endif /* SRC_HOLOLINK_CORE_ON_SCOPE_EXIT */
