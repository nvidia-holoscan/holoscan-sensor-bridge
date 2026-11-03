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

// hololink_module's LinuxReceiverOp, the receiver of the module_linux_* players, against the
// software HSB emulator (serve_linux_file), as linux_receiver_op_test runs the native
// LinuxReceiverOp. No hololink_module module claims the emulator's FPGA UUID, so the test presents
// the emulated board to the HSB-Lite module: the Hololink, LinuxDataChannel and LinuxReceiver
// services LinuxReceiverOp resolves through it wrap the same core classes that serve the emulator
// natively.
//
// Requirements:
//   - serve_linux_file (built with the examples when zlib is found), at the SERVE_LINUX_FILE env
//     var, which ctest sets, at /tmp/build/examples/serve_linux_file, or in examples/ under the
//     working directory
//   - the HSB-Lite module, in HOLOLINK_MODULE_TEST_MODULE_DIR (or the Adapter's default directory)
//   - GPU access and the loopback interface

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <mutex>
#include <signal.h>
#include <spawn.h>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <vector>

extern char** environ;

#include <cuda.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "hololink/module/adapter.hpp"
#include "hololink/module/enumeration.hpp"
#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/hololink.hpp"
#include "hololink/module/operators/linux_receiver_op.hpp"
#include "hololink/module/status.h"

#include "fake_receiver.hpp" // PairedSinkOp

namespace hololink::tests {

namespace {

    using namespace hololink::tests::fake_receiver;
    namespace mo = hololink::module::operators;

    constexpr const char* kEmulatorIp = "127.0.0.1";
    // The HSB-Lite FPGA UUID and the compat_id of its canonical module (module/hsb_lite/).
    constexpr const char* kHsbLiteUuid = "889b7ce3-65a5-4247-8b05-4ff1904c3359";
    constexpr int64_t kHsbLiteCompatId = 0x2603;
    constexpr std::size_t kFrameSize = 256;
    constexpr int kFramesToVerify = 5;

    std::string find_serve_linux_file()
    {
        const char* env = std::getenv("SERVE_LINUX_FILE");
        if (env && std::filesystem::exists(env)) {
            return env;
        }
        for (const char* candidate : {
                 "/tmp/build/examples/serve_linux_file",
                 "examples/serve_linux_file",
                 "../examples/serve_linux_file",
             }) {
            if (std::filesystem::exists(candidate)) {
                return candidate;
            }
        }
        return {};
    }

    class ModuleLinuxReceiverOpTest : public ::testing::Test {
    protected:
        void SetUp() override
        {
            serve_bin_ = find_serve_linux_file();
            if (serve_bin_.empty()) {
                GTEST_SKIP() << "serve_linux_file binary not found; "
                             << "build the examples (serve_linux_file is built with them when zlib "
                             << "is found) or set SERVE_LINUX_FILE";
            }
            // Every byte kFrameByte, as PairedSinkOp checks.
            frame_file_ = std::filesystem::temp_directory_path() / "hsb_module_test_frame.bin";
            FILE* f = fopen(frame_file_.c_str(), "wb");
            ASSERT_NE(f, nullptr);
            const std::vector<uint8_t> data(kFrameSize, kFrameByte);
            fwrite(data.data(), 1, data.size(), f);
            fclose(f);
        }

        void TearDown() override
        {
            if (emulator_pid_ > 0) {
                kill(emulator_pid_, SIGTERM);
                waitpid(emulator_pid_, nullptr, 0);
                emulator_pid_ = -1;
            }
            if (!frame_file_.empty()) {
                std::filesystem::remove(frame_file_);
            }
        }

        bool start_emulator()
        {
            std::string ip_arg = kEmulatorIp;
            std::string file_arg = frame_file_.string();
            std::string rate_arg = "--frame-rate=30";
            char* const argv[] = {
                const_cast<char*>(serve_bin_.c_str()),
                const_cast<char*>(ip_arg.c_str()),
                const_cast<char*>(file_arg.c_str()),
                const_cast<char*>(rate_arg.c_str()),
                nullptr,
            };
            if (posix_spawn(&emulator_pid_, serve_bin_.c_str(), nullptr, nullptr, argv, environ) != 0) {
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            return true;
        }

        // Receives at least kFramesToVerify frames, each paired with its metadata. With
        // \p frame_limit, the receiver stops after that many frames, and the run completes by itself.
        void receive_frames(int64_t frame_limit = 0)
        {
            ASSERT_TRUE(start_emulator()) << "Failed to start serve_linux_file";

            ASSERT_EQ(cuInit(0), CUDA_SUCCESS);
            CUdevice cu_dev;
            ASSERT_EQ(cuDeviceGet(&cu_dev, 0), CUDA_SUCCESS);
            CUcontext cu_ctx;
            ASSERT_EQ(cuDevicePrimaryCtxRetain(&cu_ctx, cu_dev), CUDA_SUCCESS);
            struct CtxGuard {
                CUdevice dev;
                ~CtxGuard() { cuDevicePrimaryCtxRelease(dev); }
            } ctx_guard { cu_dev };
            cuCtxSetCurrent(cu_ctx);

            auto& adapter = hololink::module::Adapter::get_adapter();
            if (const char* module_dir = std::getenv("HOLOLINK_MODULE_TEST_MODULE_DIR")) {
                adapter.set_module_directory(std::filesystem::path(module_dir));
            }
            hololink::module::EnumerationMetadata metadata;
            try {
                metadata = adapter.wait_for_channel(kEmulatorIp, std::chrono::seconds(10));
            } catch (const std::exception& e) {
                // The emulator is running: not finding it is a failure, not a missing prerequisite.
                FAIL() << "The module Adapter could not find the emulator at " << kEmulatorIp << ": "
                       << e.what();
            }
            // Present the emulated board as an HSB-Lite and let that module enrich its metadata, as
            // Adapter::enumerate() does for a board it has a module for.
            metadata["fpga_uuid"] = std::string(kHsbLiteUuid);
            metadata["compat_id"] = kHsbLiteCompatId;
            auto module = adapter.get_module(metadata);
            ASSERT_EQ(hololink::module::EnumerationInterfaceV1::get_service(module)->update_metadata(
                          metadata, nullptr, 0),
                HOLOLINK_MODULE_OK);

            auto hololink = hololink::module::HololinkInterfaceV1::get_service(metadata);
            ASSERT_EQ(hololink->start(), HOLOLINK_MODULE_OK);
            struct DevGuard {
                std::shared_ptr<hololink::module::HololinkInterfaceV1> dev;
                ~DevGuard() { (void)dev->stop(); }
            } dev_guard { hololink };
            ASSERT_EQ(hololink->reset(), HOLOLINK_MODULE_OK);

            auto paired_log = std::make_shared<PairedLog>();
            auto sink_count = std::make_shared<SinkCount>();
            holoscan::Graph graph { "module_linux_receiver_op_test" };
            auto rx = graph.op<mo::LinuxReceiverOp>(
                "rx", metadata, [] {}, [] {});
            rx.set_params(mo::LinuxReceiverParams {
                .frame_context = cu_ctx,
                .frame_size = kFrameSize,
                .frame_limit = frame_limit,
            });
            auto sink = graph.op<PairedSinkOp>("sink", paired_log, sink_count);
            const holoscan::ConnectionOptions edge { .queue_depth = 4U };
            graph.add_flow(rx->tensor, sink->tensor, edge);
            graph.add_flow(rx->metadata, sink->metadata, edge);
            graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
            holoscan::CompileOptions options;
            options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement {
                .operator_path = "rx",
                .output_port = "tensor",
                .device = holoscan::DeviceId { 0 },
            });
            auto plan = holoscan::compile(graph, std::move(options));
            if (!plan.ok()) {
                holoscan::log_diagnostics(plan);
            }
            ASSERT_TRUE(plan.ok()) << "graph compilation failed";

            auto session = holoscan::run_async(plan);
            if (frame_limit > 0) {
                EXPECT_TRUE(session.wait_for(std::chrono::seconds(15))) << "the run did not complete";
            }
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
            while (sink_count->received.load() < kFramesToVerify && session.is_running()
                && std::chrono::steady_clock::now() < deadline) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            session.request_stop();
            session.wait();

            EXPECT_NE(session.termination(), holoscan::RunTermination::kFailed);
            if (frame_limit > 0) {
                EXPECT_EQ(session.termination(), holoscan::RunTermination::kCompleted);
                EXPECT_EQ(sink_count->received.load(), frame_limit);
            }
            EXPECT_GE(sink_count->received.load(), kFramesToVerify)
                << "only received " << sink_count->received.load() << " frames";
            for (const auto& diagnostic : session.diagnostics()) {
                if (diagnostic.severity == holoscan::DiagnosticSeverity::kError) {
                    ADD_FAILURE() << holoscan::format_diagnostic(diagnostic, plan.graph());
                }
            }

            // Each tensor is its frame's, whole and paired with that frame's metadata: the emulator
            // writes the whole file, the frame counter only advances, and the receiver stamps the
            // frame's arrival.
            std::lock_guard lock(paired_log->mutex);
            EXPECT_EQ(paired_log->mismatches, 0) << "tensors arrived with another frame's metadata";
            int64_t previous_frame_number = -1;
            for (const PairedFrame& frame : paired_log->frames) {
                SCOPED_TRACE("frame_number " + std::to_string(frame.metadata.frame_number));
                EXPECT_EQ(frame.frame_id, static_cast<uint64_t>(frame.metadata.frame_number));
                EXPECT_EQ(frame.nbytes, int64_t(kFrameSize));
                EXPECT_TRUE(frame.intact) << "the tensor doesn't hold the served frame";
                EXPECT_EQ(frame.metadata.bytes_written, int64_t(kFrameSize));
                EXPECT_GT(frame.metadata.frame_number, previous_frame_number);
                EXPECT_NE(frame.metadata.received_s, 0);
                previous_frame_number = frame.metadata.frame_number;
            }
        }

        std::string serve_bin_;
        std::filesystem::path frame_file_;
        pid_t emulator_pid_ { -1 };
    };

} // namespace

TEST_F(ModuleLinuxReceiverOpTest, ReceivesFramesFromEmulator)
{
    receive_frames();
}

// As HSDK4's CountCondition on the receiver: frame_limit frames, then the run completes. A spare
// activation waits in get_next_frame() for the next frame, so every compute() call delivers one.
TEST_F(ModuleLinuxReceiverOpTest, CompletesAfterItsFrameLimit)
{
    receive_frames(kFramesToVerify);
}

} // namespace hololink::tests
