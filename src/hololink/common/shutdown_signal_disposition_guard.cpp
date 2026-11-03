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

#include "shutdown_signal_disposition_guard.hpp"

#include <sys/syscall.h>
#include <unistd.h>

namespace hololink::common {

namespace {

    std::mutex& disposition_mutex()
    {
        static std::mutex mutex;
        return mutex;
    }

} // namespace

static_assert(sizeof(void*) == 8, "ShutdownSignalDispositionGuard assumes the 64-bit kernel sigaction layout");

ShutdownSignalDispositionGuard::ShutdownSignalDispositionGuard()
    : lock_(disposition_mutex())
{
    for (size_t i = 0; i < kSignals.size(); ++i) {
        saved_[i] = syscall(SYS_rt_sigaction, kSignals[i], nullptr, &dispositions_[i],
                        sizeof(dispositions_[i].mask))
            == 0;
    }
}

ShutdownSignalDispositionGuard::~ShutdownSignalDispositionGuard()
{
    for (size_t i = 0; i < kSignals.size(); ++i) {
        if (saved_[i]) {
            syscall(SYS_rt_sigaction, kSignals[i], &dispositions_[i], nullptr, sizeof(dispositions_[i].mask));
        }
    }
}

} // namespace hololink::common
