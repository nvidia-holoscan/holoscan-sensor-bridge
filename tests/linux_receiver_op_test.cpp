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

// Integration test for LinuxReceiverOp against the software HSB emulator.
//
// Spawns serve_linux_file as a subprocess, enumerates the emulated device,
// runs a minimal HSDK5 graph (LinuxReceiverOp → FrameCountSinkOp), and
// verifies that at least N frames are received. The emulator sends the whole
// file as each frame, so a file larger than the receiver's pages makes the
// device write past them. LinuxCoeReceiverOp's stall timeout is tested the same
// way against serve_coe_file, over the loopback interface.
//
// Requirements:
//   - the serve_linux_file binary (built with the examples when zlib is found): at the
//     SERVE_LINUX_FILE env var, which ctest sets, or /tmp/build/examples/serve_linux_file
//   - for the LinuxCoeReceiverOp tests, the serve_coe_file binary (built with the examples):
//     at the SERVE_COE_FILE env var, which ctest sets, or /tmp/build/examples/serve_coe_file
//   - GPU access (--gpus all)
//   - Network loopback (127.0.0.1)
//   - CAP_NET_RAW (root in the container), for the emulators' and the CoE receiver's raw sockets

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <mutex>
#include <sched.h>
#include <signal.h>
#include <spawn.h>
#include <stdexcept>

extern char** environ;
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include <capture_metadata_traits.hpp>
#include <hololink/core/data_channel.hpp>
#include <hololink/core/enumerator.hpp>
#include <hololink/core/hololink.hpp>
#include <hololink/core/networking.hpp>
#include <hololink/operators/capture_metadata_utils.hpp>
#include <hololink/operators/linux_coe_receiver/linux_coe_receiver_op.hpp>
#include <hololink/operators/linux_receiver/linux_receiver_op.hpp>

namespace hololink::tests {

// ---------------------------------------------------------------------------
// Shared state between the test and the sink operator.
// ---------------------------------------------------------------------------

struct FrameResults {
    std::atomic<int> count { 0 };
    std::atomic<bool> done { false };
    std::mutex mutex;
    std::condition_variable cv;
    int expect_frames { 5 };
    std::vector<hololink::CaptureMetadataT> frames; // under mutex; only with metadata connected
    std::vector<int64_t> tensor_bytes; // under mutex; each received tensor's size

    // Counts one received frame of \p nbytes, with its metadata when \p metadata is set.
    void record(int64_t nbytes, const hololink::CaptureMetadataT* metadata = nullptr)
    {
        {
            std::lock_guard<std::mutex> lock(mutex);
            tensor_bytes.push_back(nbytes);
            if (metadata) {
                frames.push_back(*metadata);
            }
        }
        count++;
        notify();
    }

    void notify()
    {
        if (count.load() >= expect_frames) {
            done = true;
            cv.notify_all();
        }
    }

    bool wait(std::chrono::seconds timeout)
    {
        std::unique_lock<std::mutex> lock(mutex);
        return cv.wait_for(lock, timeout, [this] { return done.load(); });
    }
};

// ---------------------------------------------------------------------------
// Sink operator.
// ---------------------------------------------------------------------------

struct FrameCountSinkParams {
    std::shared_ptr<FrameResults> results;
};

class FrameCountSinkOp : public holoscan::Operator<FrameCountSinkParams> {
public:
    holoscan::Input<holoscan::Tensor> tensor;
    holoscan::Input<hololink::CaptureMetadataT> metadata;

    void setup(holoscan::OperatorSpec& spec) override
    {
        spec.input(tensor, "tensor")
            .queue_depth(4U)
            .expects_tensor(holoscan::TensorInputSpec {
                .representation = holoscan::TensorRepresentation {
                    .memory_kind = holoscan::MemoryKind::kCudaDevice,
                },
            });
        spec.input(metadata, "metadata").queue_depth(4U);
    }

    [[nodiscard]] holoscan::Contract contract() const override
    {
        holoscan::Contract c;
        // Upstream stamps both ports with an identical capture_time, so OnAligned pairs them on
        // an exact match (see kMetadataPairingWindow).
        c.trigger(holoscan::OnAligned {
                      .ports = holoscan::ports(tensor, metadata),
                      .window = hololink::operators::kMetadataPairingWindow,
                  })
            .input(tensor, holoscan::Exact {})
            .input(metadata, holoscan::Exact {});
        return c;
    }

    [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
        holoscan::ExecutionContext&) override
    {
        // The pairing every downstream operator relies on: the tensor's frame_id is its
        // metadata's frame_number.
        auto maybe_paired = hololink::operators::receive_paired_tensor_metadata(tensor, metadata);
        if (!maybe_paired)
            return holoscan::make_unexpected(std::move(maybe_paired).error());
        if (!*maybe_paired)
            return holoscan::make_unexpected(
                holoscan::Error { holoscan::ErrorCode::kInvalidArgument, "frame identity mismatch" });
        auto& paired = **maybe_paired;

        // Verify the tensor is present and non-empty.
        if (paired.tensor.data.nbytes() == 0)
            return holoscan::make_unexpected(holoscan::Error {
                holoscan::ErrorCode::kFailure, "received empty tensor" });

        params().results->record(paired.tensor.data.nbytes(), &paired.metadata.data);
        return {};
    }
};

// Counts tensors alone, for a graph that leaves the receiver's metadata output unconnected.
class TensorCountSinkOp : public holoscan::Operator<FrameCountSinkParams> {
public:
    holoscan::Input<holoscan::Tensor> tensor;

    void setup(holoscan::OperatorSpec& spec) override
    {
        spec.input(tensor, "tensor")
            .queue_depth(4U)
            .expects_tensor(holoscan::TensorInputSpec {
                .representation = holoscan::TensorRepresentation {
                    .memory_kind = holoscan::MemoryKind::kCudaDevice,
                },
            });
    }

    [[nodiscard]] holoscan::Contract contract() const override
    {
        holoscan::Contract c;
        c.trigger(holoscan::OnEach { tensor });
        return c;
    }

    [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
        holoscan::ExecutionContext&) override
    {
        auto tensor_sample = tensor.receive();
        if (!tensor_sample)
            return holoscan::make_unexpected(std::move(tensor_sample).error());
        if (tensor_sample->data.nbytes() == 0)
            return holoscan::make_unexpected(holoscan::Error {
                holoscan::ErrorCode::kFailure, "received empty tensor" });
        params().results->record(tensor_sample->data.nbytes());
        return {};
    }
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::string find_serve_linux_file()
{
    const char* env = std::getenv("SERVE_LINUX_FILE");
    if (env && std::filesystem::exists(env))
        return env;
    for (const char* candidate : {
             "/tmp/build/examples/serve_linux_file",
             "examples/serve_linux_file",
             "../examples/serve_linux_file",
         }) {
        if (std::filesystem::exists(candidate))
            return candidate;
    }
    return {};
}

// The CoE emulator, at the SERVE_COE_FILE env var (set by ctest) or in the build's examples.
static std::string find_serve_coe_file()
{
    const char* env = std::getenv("SERVE_COE_FILE");
    if (env && std::filesystem::exists(env))
        return env;
    for (const char* candidate : {
             "/tmp/build/examples/serve_coe_file",
             "examples/serve_coe_file",
             "../examples/serve_coe_file",
         }) {
        if (std::filesystem::exists(candidate))
            return candidate;
    }
    return {};
}

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------

class LinuxReceiverOpTest : public ::testing::Test {
protected:
    static constexpr const char* kEmulatorIp = "127.0.0.1";
    static constexpr std::size_t kFrameSize = 256;
    static constexpr uint32_t kPages = 2;
    static constexpr int kFramesToVerify = 5;

    std::string serve_bin;
    std::filesystem::path frame_file;
    pid_t emulator_pid { -1 };

    // A receiver page: the frame, rounded up to the HSB's 128-byte pages, then its metadata (see
    // LinuxReceiverOp::start_receiver()).
    static std::size_t page_bytes()
    {
        return hololink::core::round_up(kFrameSize, hololink::core::PAGE_SIZE) + hololink::METADATA_SIZE;
    }

    // Twice the receiver's whole buffer of kPages pages, with the host page its allocation adds
    // for alignment: every frame of this size overruns its page, and its buffer.
    static std::size_t oversized_frame_bytes()
    {
        const auto host_page = static_cast<std::size_t>(getpagesize());
        return 2 * (hololink::core::round_up(page_bytes() * kPages, host_page) + host_page);
    }

    // The file the emulator serves: \p bytes, all 0x42, sent whole as every frame.
    void write_frame_file(std::size_t bytes)
    {
        FILE* f = fopen(frame_file.c_str(), "wb");
        ASSERT_NE(f, nullptr);
        std::vector<uint8_t> data(bytes, 0x42);
        fwrite(data.data(), 1, data.size(), f);
        fclose(f);
    }

    void SetUp() override
    {
        serve_bin = find_serve_linux_file();
        if (serve_bin.empty()) {
            GTEST_SKIP() << "serve_linux_file binary not found; "
                         << "build the examples (serve_linux_file is built with them when zlib is "
                         << "found) or set SERVE_LINUX_FILE";
        }

        frame_file = std::filesystem::temp_directory_path() / "hsb_test_frame.bin";
        write_frame_file(kFrameSize);
    }

    void TearDown() override
    {
        if (emulator_pid > 0) {
            kill(emulator_pid, SIGTERM);
            waitpid(emulator_pid, nullptr, 0);
            emulator_pid = -1;
        }
        if (!frame_file.empty())
            std::filesystem::remove(frame_file);
    }

    bool start_emulator()
    {
        // Use posix_spawn to start the emulator as a tracked child process.
        // popen() + "&" would background the process under the shell, making
        // pclose() wait for the shell instead of the emulator — leaving it
        // orphaned after the test and able to interfere with the next run.
        std::string rate_arg = "--frame-rate=30";
        std::string ip_arg = kEmulatorIp;
        std::string file_arg = frame_file.string();
        char* const argv[] = {
            const_cast<char*>(serve_bin.c_str()),
            const_cast<char*>(ip_arg.c_str()),
            const_cast<char*>(file_arg.c_str()),
            const_cast<char*>(rate_arg.c_str()),
            nullptr
        };
        if (posix_spawn(&emulator_pid, serve_bin.c_str(), nullptr, nullptr, argv, environ) != 0)
            return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        return true;
    }

    // Receives kFramesToVerify frames. With \p frame_limit, the receiver stops after that many
    // frames, and the run completes by itself. The emulator sends \p served_bytes as each frame;
    // \p trim is LinuxReceiverParams::trim.
    void receive_frames(bool connect_metadata, int64_t frame_limit = 0,
        std::size_t served_bytes = kFrameSize, bool trim = false)
    {
        if (served_bytes != kFrameSize) {
            write_frame_file(served_bytes);
        }
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

        Metadata channel_meta;
        try {
            channel_meta = Enumerator::find_channel(
                kEmulatorIp, std::make_shared<Timeout>(10.f));
        } catch (const std::exception& e) {
            // The emulator is running: not finding it is a failure, not a missing prerequisite.
            FAIL() << "Enumerator could not find emulator at "
                   << kEmulatorIp << ": " << e.what();
        }

        auto data_channel = std::make_shared<DataChannel>(channel_meta);
        auto hololink_dev = data_channel->hololink();
        hololink_dev->start();
        struct DevGuard {
            std::shared_ptr<hololink::Hololink> dev;
            ~DevGuard() { dev->stop(); }
        } dev_guard { hololink_dev };
        hololink_dev->reset();

        auto results = std::make_shared<FrameResults>();
        results->expect_frames = kFramesToVerify;

        holoscan::Graph graph { "linux_receiver_op_test" };

        auto rx = graph.op<hololink::operators::LinuxReceiverOp>("rx", data_channel);
        // Exercise graph-held ownership after the caller releases its reference.
        data_channel.reset();
        rx.set_params(hololink::operators::LinuxReceiverParams {
            .frame_context = cu_ctx,
            .frame_size = kFrameSize,
            .trim = trim,
            .pages = kPages,
            .queue_size = 1,
            .frame_limit = frame_limit,
        });

        if (connect_metadata) {
            auto sink = graph.op<FrameCountSinkOp>("sink");
            sink.set_params(FrameCountSinkParams { .results = results });
            graph.add_flow(rx->tensor, sink->tensor,
                holoscan::ConnectionOptions { .queue_depth = 4U });
            graph.add_flow(rx->metadata, sink->metadata,
                holoscan::ConnectionOptions { .queue_depth = 4U });
        } else {
            // Apps may leave the metadata output unconnected; the receiver must keep running.
            auto sink = graph.op<TensorCountSinkOp>("sink");
            sink.set_params(FrameCountSinkParams { .results = results });
            graph.add_flow(rx->tensor, sink->tensor,
                holoscan::ConnectionOptions { .queue_depth = 4U });
        }
        graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

        holoscan::CompileOptions options;
        options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement {
            .operator_path = "rx",
            .output_port = "tensor",
            .device = holoscan::DeviceId { 0 },
        });
        auto plan = holoscan::compile(graph, std::move(options));
        if (!plan.ok()) {
            for (const auto& d : plan.diagnostics()) {
                std::cerr << "Diagnostic: token=" << d.token
                          << " required=" << d.required
                          << " provided=" << d.provided << "\n";
            }
        }
        ASSERT_TRUE(plan.ok()) << "Graph compilation failed";

        auto session = holoscan::run_async(plan);

        if (frame_limit > 0) {
            EXPECT_TRUE(session.wait_for(std::chrono::seconds(15))) << "the run did not complete";
        }
        const bool got_frames = results->wait(std::chrono::seconds(15));

        session.request_stop();
        session.wait();
        if (frame_limit > 0) {
            EXPECT_EQ(session.termination(), holoscan::RunTermination::kCompleted);
            EXPECT_EQ(results->count.load(), frame_limit);
        }

        EXPECT_TRUE(got_frames)
            << "Timed out: only received " << results->count.load()
            << "/" << kFramesToVerify << " frames";
        EXPECT_NE(session.termination(), holoscan::RunTermination::kFailed);
        for (const auto& diagnostic : session.diagnostics()) {
            ADD_FAILURE() << holoscan::format_diagnostic(diagnostic, plan.graph());
        }

        // Each frame's metadata describes it: the emulator writes the whole file, the frame
        // counter only advances, and the receiver stamps its arrival.
        std::lock_guard<std::mutex> lock(results->mutex);
        EXPECT_EQ(results->frames.empty(), !connect_metadata);
        int64_t previous_frame_number = -1;
        for (const hololink::CaptureMetadataT& frame : results->frames) {
            SCOPED_TRACE("frame_number " + std::to_string(frame.frame_number));
            if (served_bytes == kFrameSize) {
                EXPECT_EQ(frame.bytes_written, int64_t(kFrameSize));
            } else {
                // The device's own report passes through; only the copies stop at the page.
                EXPECT_GT(frame.bytes_written, int64_t(page_bytes())) << "no frame overran its page";
            }
            EXPECT_GT(frame.frame_number, previous_frame_number);
            EXPECT_NE(frame.received_s, 0);
            previous_frame_number = frame.frame_number;
        }
        // Every tensor is the frame_size frame, never more, whatever the device reported writing.
        EXPECT_FALSE(results->tensor_bytes.empty());
        for (const int64_t nbytes : results->tensor_bytes) {
            EXPECT_EQ(nbytes, int64_t(kFrameSize));
        }
    }
};

// The LinuxCoeReceiverOp tests need serve_coe_file, not serve_linux_file (which is built only when
// zlib is found): they skip on their own when it is missing.
class LinuxCoeReceiverOpTest : public LinuxReceiverOpTest {
protected:
    void SetUp() override
    {
        frame_file = std::filesystem::temp_directory_path() / "hsb_test_frame.bin";
        write_frame_file(kFrameSize);
    }

    // What run_coe_receiver() saw.
    struct CoeRun {
        holoscan::RunTermination termination { holoscan::RunTermination::kRunning };
        int frames { 0 };
        std::vector<std::string> diagnostics;
    };

    // Runs a LinuxCoeReceiverOp with \p stall_timeout_ms on the loopback interface. serve_coe_file
    // sends it \p frames frames, at kCoeFrameRate a second once the receiver has configured it, and
    // then exits. The test stops the run after \p run_for if it is still going. The raw CoE
    // sockets need CAP_NET_RAW, as the emulators do.
    void run_coe_receiver(uint32_t stall_timeout_ms, unsigned frames,
        std::chrono::seconds run_for, CoeRun& run)
    {
        const std::string serve_coe_bin = find_serve_coe_file();
        if (serve_coe_bin.empty()) {
            GTEST_SKIP() << "serve_coe_file binary not found; build the examples or set SERVE_COE_FILE";
        }
        const std::string rate_arg = "--frame-rate=" + std::to_string(kCoeFrameRate);
        const std::string limit_arg = "--frame-limit=" + std::to_string(frames);
        const std::string ip_arg = kEmulatorIp;
        const std::string file_arg = frame_file.string();
        char* const argv[] = {
            const_cast<char*>(serve_coe_bin.c_str()),
            const_cast<char*>(ip_arg.c_str()),
            const_cast<char*>(file_arg.c_str()),
            const_cast<char*>(rate_arg.c_str()),
            const_cast<char*>(limit_arg.c_str()),
            nullptr
        };
        ASSERT_EQ(posix_spawn(&emulator_pid, serve_coe_bin.c_str(), nullptr, nullptr, argv, environ), 0)
            << "Failed to start serve_coe_file";
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

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

        Metadata channel_meta;
        ASSERT_NO_THROW(channel_meta = Enumerator::find_channel(kEmulatorIp, std::make_shared<Timeout>(10.f)));
        auto data_channel = std::make_shared<DataChannel>(channel_meta);
        // configure_coe() writes the data plane's registers through the board's control plane.
        auto hololink_dev = data_channel->hololink();
        hololink_dev->start();
        struct DevGuard {
            std::shared_ptr<hololink::Hololink> dev;
            ~DevGuard() { dev->stop(); }
        } dev_guard { hololink_dev };

        auto results = std::make_shared<FrameResults>();
        results->expect_frames = static_cast<int>(frames);

        holoscan::Graph graph { "linux_coe_receiver_op_test" };
        auto rx = graph.op<hololink::operators::LinuxCoeReceiverOp>("rx", data_channel);
        rx.set_params(hololink::operators::LinuxCoeReceiverParams {
            .frame_context = cu_ctx,
            .frame_size = kFrameSize,
            .coe_interface = "lo",
            .pixel_width = kFrameSize,
            .stall_timeout_ms = stall_timeout_ms,
        });
        auto sink = graph.op<TensorCountSinkOp>("sink");
        sink.set_params(FrameCountSinkParams { .results = results });
        graph.add_flow(rx->tensor, sink->tensor, holoscan::ConnectionOptions { .queue_depth = 4U });
        graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

        holoscan::CompileOptions options;
        options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement {
            .operator_path = "rx",
            .output_port = "tensor",
            .device = holoscan::DeviceId { 0 },
        });
        auto plan = holoscan::compile(graph, std::move(options));
        ASSERT_TRUE(plan.ok()) << "Graph compilation failed";

        auto session = holoscan::run_async(plan);
        if (!session.wait_for(run_for)) {
            session.request_stop();
            session.wait();
        }
        run.termination = session.termination();
        run.frames = results->count.load();
        for (const auto& diagnostic : session.diagnostics()) {
            run.diagnostics.push_back(holoscan::format_diagnostic(diagnostic, plan.graph()));
        }
    }

    static constexpr unsigned kCoeFrameRate = 50;
};

// ---------------------------------------------------------------------------
// Test
// ---------------------------------------------------------------------------

TEST_F(LinuxReceiverOpTest, ReceivesFramesFromEmulator)
{
    receive_frames(true);
}

TEST_F(LinuxReceiverOpTest, RunsWithItsMetadataOutputUnconnected)
{
    receive_frames(false);
}

// As HSDK4's CountCondition on the receiver: frame_limit frames, then the run completes.
TEST_F(LinuxReceiverOpTest, CompletesAfterItsFrameLimit)
{
    receive_frames(true, kFramesToVerify);
}

// A device that writes past the receiver's pages (as serve_linux_single_imx274_frames does into a
// sub-frame player's pages) costs the frame's tail, not the process or the run: LinuxReceiver logs
// each packet that spans a page boundary (that message's format once threw on the receive thread
// and aborted the process), copies a ready frame only up to its page (an unchecked bytes_written
// once copied past the buffer and failed the run with CUDA_ERROR_INVALID_VALUE), and
// BaseReceiverOp ignores the oversized bytes_written for trimming.
TEST_F(LinuxReceiverOpTest, SurvivesFramesLargerThanItsPages)
{
    receive_frames(true, 0, oversized_frame_bytes(), /*trim=*/true);
}

// A receiver CPU affinity outside [0, CPU_SETSIZE) fails the start, with a diagnostic. The range
// check used to throw on the receiver thread, which ended the process.
TEST_F(LinuxReceiverOpTest, RejectsAnOutOfRangeAffinity)
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

    Metadata channel_meta;
    ASSERT_NO_THROW(channel_meta = Enumerator::find_channel(kEmulatorIp, std::make_shared<Timeout>(10.f)));

    holoscan::Graph graph { "linux_receiver_op_affinity_test" };
    auto rx = graph.op<hololink::operators::LinuxReceiverOp>("rx", std::make_shared<DataChannel>(channel_meta));
    rx.set_params(hololink::operators::LinuxReceiverParams {
        .frame_context = cu_ctx,
        .frame_size = kFrameSize,
        .receiver_affinity = { CPU_SETSIZE },
        .pages = kPages,
        .queue_size = 1,
    });
    auto sink = graph.op<TensorCountSinkOp>("sink");
    sink.set_params(FrameCountSinkParams { .results = std::make_shared<FrameResults>() });
    graph.add_flow(rx->tensor, sink->tensor, holoscan::ConnectionOptions { .queue_depth = 4U });
    graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

    holoscan::CompileOptions options;
    options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement {
        .operator_path = "rx",
        .output_port = "tensor",
        .device = holoscan::DeviceId { 0 },
    });
    auto plan = holoscan::compile(graph, std::move(options));
    ASSERT_TRUE(plan.ok()) << "Graph compilation failed";

    auto session = holoscan::run_async(plan);
    ASSERT_TRUE(session.wait_for(std::chrono::seconds(15))) << "the run did not end";
    EXPECT_EQ(session.termination(), holoscan::RunTermination::kFailed);
    bool reported = false;
    for (const auto& diagnostic : session.diagnostics()) {
        reported = reported
            || holoscan::format_diagnostic(diagnostic, plan.graph()).find("CPU affinity") != std::string::npos;
    }
    EXPECT_TRUE(reported) << "no diagnostic names the out-of-range affinity";
}

// The receiver thread starts before device_start, and a failed start() gets no stop(): the
// receiver must stop that thread itself, or destroying the operator terminates the process.
TEST_F(LinuxReceiverOpTest, DeviceStartFailureFailsTheRun)
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

    Metadata channel_meta;
    try {
        channel_meta = Enumerator::find_channel(
            kEmulatorIp, std::make_shared<Timeout>(10.f));
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Enumerator could not find emulator at "
                     << kEmulatorIp << ": " << e.what();
    }

    auto data_channel = std::make_shared<DataChannel>(channel_meta);
    auto hololink_dev = data_channel->hololink();
    hololink_dev->start();
    struct DevGuard {
        std::shared_ptr<hololink::Hololink> dev;
        ~DevGuard() { dev->stop(); }
    } dev_guard { hololink_dev };
    hololink_dev->reset();

    {
        holoscan::Graph graph { "linux_receiver_op_device_start_failure_test" };
        auto rx = graph.op<hololink::operators::LinuxReceiverOp>(
            "rx", data_channel, [] { throw std::runtime_error("device start failed"); }, [] {});
        rx.set_params(hololink::operators::LinuxReceiverParams {
            .frame_context = cu_ctx,
            .frame_size = kFrameSize,
            .trim = false,
            .pages = 2,
            .queue_size = 1,
        });
        auto sink = graph.op<FrameCountSinkOp>("sink");
        sink.set_params(FrameCountSinkParams { .results = std::make_shared<FrameResults>() });
        graph.add_flow(rx->tensor, sink->tensor, holoscan::ConnectionOptions { .queue_depth = 4U });
        graph.add_flow(rx->metadata, sink->metadata, holoscan::ConnectionOptions { .queue_depth = 4U });
        graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

        holoscan::CompileOptions options;
        options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement {
            .operator_path = "rx",
            .output_port = "tensor",
            .device = holoscan::DeviceId { 0 },
        });
        auto plan = holoscan::compile(graph, std::move(options));
        ASSERT_TRUE(plan.ok()) << "Graph compilation failed";

        auto session = holoscan::run_async(plan);
        ASSERT_TRUE(session.wait_for(std::chrono::seconds(15)));
        EXPECT_EQ(session.termination(), holoscan::RunTermination::kFailed);
    }
    // Reaching this point means destroying the operator didn't terminate the process.
    SUCCEED();
}

// A LinuxCoeReceiverOp with a stall timeout runs while frames come and fails the run, with a
// diagnostic naming the stall, once the timeout passes without one. The frames come for twice the
// timeout, so a timeout that the frames didn't reset would fail the run while they still came.
TEST_F(LinuxCoeReceiverOpTest, FailsTheRunWhenTheCaptureStalls)
{
    constexpr unsigned kFrames = 30; // 600 ms at kCoeFrameRate
    constexpr uint32_t kStallTimeoutMs = 300;
    CoeRun run;
    run_coe_receiver(kStallTimeoutMs, kFrames, std::chrono::seconds(15), run);
    if (IsSkipped() || HasFatalFailure()) {
        return;
    }
    EXPECT_EQ(run.termination, holoscan::RunTermination::kFailed) << "the stalled run did not fail";
    // A few frames may be lost on the way; a timeout that the frames didn't reset fails the run
    // after kStallTimeoutMs of them, half as many.
    EXPECT_GE(run.frames, static_cast<int>(kFrames) - 5);
    bool reported = false;
    for (const auto& diagnostic : run.diagnostics) {
        reported = reported || diagnostic.find("stalled for") != std::string::npos;
    }
    EXPECT_TRUE(reported) << "no diagnostic names the stall";
}

// Without a stall timeout (the default), the run outlives its last frame until it is stopped.
TEST_F(LinuxCoeReceiverOpTest, RunsOnWithoutAStallTimeout)
{
    constexpr unsigned kFrames = 10;
    CoeRun run;
    run_coe_receiver(0, kFrames, std::chrono::seconds(2), run);
    if (IsSkipped() || HasFatalFailure()) {
        return;
    }
    EXPECT_EQ(run.termination, holoscan::RunTermination::kStopped);
    EXPECT_GE(run.frames, static_cast<int>(kFrames) - 5);
    for (const auto& diagnostic : run.diagnostics) {
        ADD_FAILURE() << diagnostic;
    }
}

} // namespace hololink::tests
