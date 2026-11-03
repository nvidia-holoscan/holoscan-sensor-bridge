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

#ifndef SRC_HOLOLINK_COMMON_SHUTDOWN_SIGNAL_DISPOSITION_GUARD
#define SRC_HOLOLINK_COMMON_SHUTDOWN_SIGNAL_DISPOSITION_GUARD

#include <signal.h>

#include <array>
#include <mutex>

namespace hololink::common {

/**
 * RAII type class restoring SIGINT's and SIGTERM's dispositions exactly as they were on
 * construction; scope it around nvrtcCompileProgram().
 *
 * nvrtcCompileProgram() installs its own SIGINT/SIGTERM handlers while it runs and then resets
 * them with signal(SIG_DFL), which is not the disposition it found: signal() adds SA_RESTART.
 * Holoscan's managed shutdown signals (holoscan::ManagedShutdownSignals) require both
 * dispositions to be unchanged when a signal arrives and when the run ends, and otherwise end
 * the run with RUNTIME_MANAGED_SIGNAL_RESTORE_FAILED -- so every run of an app whose operators
 * compiled kernels on start would fail. The dispositions are saved and restored through the
 * rt_sigaction system call rather than sigaction(): glibc's sigaction() adds SA_RESTORER on
 * x86_64, so it cannot restore a disposition that was never set. A signal that arrives while
 * the compiler's handlers are installed still counts as a changed disposition.
 *
 * Guards are serialized process-wide: each one holds a lock from saving the dispositions until
 * it restores them, so operators compiling kernels at the same time cannot save each other's
 * temporary compiler handlers. Don't nest guards on one thread.
 */
class ShutdownSignalDispositionGuard {
public:
    ShutdownSignalDispositionGuard();
    ~ShutdownSignalDispositionGuard();
    ShutdownSignalDispositionGuard(const ShutdownSignalDispositionGuard&) = delete;
    ShutdownSignalDispositionGuard& operator=(const ShutdownSignalDispositionGuard&) = delete;

private:
    /// The kernel's sigaction for rt_sigaction on 64-bit Linux (x86_64, aarch64).
    struct KernelSigaction {
        void (*handler)(int);
        unsigned long flags;
        void (*restorer)();
        unsigned long mask;
    };
    static constexpr std::array<int, 2> kSignals { SIGINT, SIGTERM };
    // Declared first: held from before the dispositions are saved until after they're restored.
    std::unique_lock<std::mutex> lock_;
    std::array<KernelSigaction, 2> dispositions_ {};
    std::array<bool, 2> saved_ {};
};

} // namespace hololink::common

#endif /* SRC_HOLOLINK_COMMON_SHUTDOWN_SIGNAL_DISPOSITION_GUARD */
