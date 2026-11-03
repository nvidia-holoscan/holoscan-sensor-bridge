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

// nvrtcCompileProgram() resets SIGINT and SIGTERM to a disposition other than the one it found,
// which makes a Holoscan run with managed shutdown signals fail. Every kernel compile in hololink
// must therefore leave both dispositions exactly as they were, including when several operators
// compile at the same time. NVRTC touches the dispositions only in a process's first compile, so
// each test must run in a process of its own (tests/CMakeLists.txt registers one per test).

#include <gtest/gtest.h>

#include <signal.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <array>
#include <cstring>
#include <thread>
#include <vector>

#include <cuda.h>

#include <hololink/common/cuda_helper.hpp>
#if defined(HOLOLINK_TEST_MODULE_OPERATORS)
#include <hololink/module/operators/cuda_function_launcher.hpp>
#endif

namespace hololink::tests {

namespace {

    constexpr const char* kSource = R"(
extern "C" __global__ void noop()
{
}
)";

    // The kernel's view of a disposition (what rt_sigaction reads and writes on 64-bit Linux).
    struct KernelSigaction {
        void (*handler)(int);
        unsigned long flags;
        void (*restorer)();
        unsigned long mask;
    };

    std::array<KernelSigaction, 2> read_dispositions()
    {
        std::array<KernelSigaction, 2> dispositions {};
        const std::array<int, 2> signals { SIGINT, SIGTERM };
        for (size_t i = 0; i < signals.size(); ++i) {
            EXPECT_EQ(syscall(SYS_rt_sigaction, signals[i], nullptr, &dispositions[i],
                          sizeof(dispositions[i].mask)),
                0);
        }
        return dispositions;
    }

    void expect_same(const std::array<KernelSigaction, 2>& before,
        const std::array<KernelSigaction, 2>& after)
    {
        for (size_t i = 0; i < before.size(); ++i) {
            SCOPED_TRACE(i == 0 ? "SIGINT" : "SIGTERM");
            EXPECT_EQ(before[i].handler, after[i].handler);
            EXPECT_EQ(before[i].flags, after[i].flags);
            EXPECT_EQ(before[i].restorer, after[i].restorer);
            EXPECT_EQ(before[i].mask, after[i].mask);
        }
    }

    class NvrtcSignalDispositions : public ::testing::Test {
    protected:
        void SetUp() override
        {
            ASSERT_EQ(cuInit(0), CUDA_SUCCESS);
            CUdevice device;
            ASSERT_EQ(cuDeviceGet(&device, 0), CUDA_SUCCESS);
            ASSERT_EQ(cuDevicePrimaryCtxRetain(&context_, device), CUDA_SUCCESS);
            ASSERT_EQ(cuCtxSetCurrent(context_), CUDA_SUCCESS);
            // The default disposition without SA_RESTART, as a process starts with and as
            // managed shutdown signals expect it. nvrtcCompileProgram() resets it with
            // signal(SIG_DFL), which adds SA_RESTART.
            struct sigaction action;
            std::memset(&action, 0, sizeof(action));
            action.sa_handler = SIG_DFL;
            sigemptyset(&action.sa_mask);
            for (int signal_number : { SIGINT, SIGTERM }) {
                ASSERT_EQ(sigaction(signal_number, &action, &previous_[signal_number == SIGINT ? 0 : 1]), 0);
            }
        }

        void TearDown() override
        {
            sigaction(SIGINT, &previous_[0], nullptr);
            sigaction(SIGTERM, &previous_[1], nullptr);
            if (context_) {
                CUdevice device;
                if (cuCtxGetDevice(&device) == CUDA_SUCCESS) {
                    cuDevicePrimaryCtxRelease(device);
                }
            }
        }

        CUcontext context_ = nullptr;
        std::array<struct sigaction, 2> previous_ {};
    };

    template <typename Launcher>
    void compile_on(CUcontext context)
    {
        ASSERT_EQ(cuCtxSetCurrent(context), CUDA_SUCCESS);
        Launcher launcher(kSource, { "noop" });
    }

} // namespace

TEST_F(NvrtcSignalDispositions, NativeCompileKeepsThem)
{
    const auto before = read_dispositions();
    compile_on<hololink::common::CudaFunctionLauncher>(context_);
    expect_same(before, read_dispositions());
}

#if defined(HOLOLINK_TEST_MODULE_OPERATORS)
TEST_F(NvrtcSignalDispositions, ModuleCompileKeepsThem)
{
    const auto before = read_dispositions();
    compile_on<hololink::module::operators::CudaFunctionLauncher>(context_);
    expect_same(before, read_dispositions());
}
#endif

// Without serialization, a guard could save another compile's temporary handler and restore it.
TEST_F(NvrtcSignalDispositions, ConcurrentCompilesKeepThem)
{
    const auto before = read_dispositions();
    std::vector<std::thread> threads;
    for (int i = 0; i < 8; ++i) {
        threads.emplace_back([this, i] {
#if defined(HOLOLINK_TEST_MODULE_OPERATORS)
            if (i % 2) {
                compile_on<hololink::module::operators::CudaFunctionLauncher>(context_);
                return;
            }
#endif
            compile_on<hololink::common::CudaFunctionLauncher>(context_);
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    expect_same(before, read_dispositions());
}

} // namespace hololink::tests
