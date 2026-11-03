/**
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
 *
 * Forward-loopback counterpart of examples/gpu_roce_loopback.py for
 * CpuRoceTransceiverOp. It configures the HSB ram_player, receives each
 * payload through CPU-polled RoCE, and immediately sends it back over the
 * same UC QP. The example supports host or CUDA DMA-BUF payload memory and
 * optional HSB frame metadata; board-specific ILA tooling lives outside the
 * installed example.
 */

#include <getopt.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include <cuda.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/managed_shutdown_signals.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include <hololink/common/cuda_helper.hpp>
#include <hololink/common/tools.hpp>
#include <hololink/core/data_channel.hpp>
#include <hololink/core/enumerator.hpp>
#include <hololink/core/hololink.hpp>
#include <hololink/operators/cpu_roce_transceiver/cpu_roce_transceiver_op.hpp>

#define HOLOLINK_PLAYER_SUPPORT_NO_DISPLAY
#include "player_support.hpp"

namespace hx = hololink::examples;

namespace {

class DataGen {
public:
    DataGen(std::shared_ptr<hololink::Hololink> hololink, uint32_t frame_size,
        bool frame_metadata)
        : hololink_(std::move(hololink))
        , frame_size_(frame_size)
        , frame_metadata_(frame_metadata)
    {
    }

    void start()
    {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        try {
            hololink_->write_uint32(PLAYER_ADDR + PLAYER_ENABLE_OFFSET, 0);
            const uint32_t metadata
                = hololink_->read_uint32(METADATA_PACKET_OFFSET);
            hololink_->write_uint32(METADATA_PACKET_OFFSET,
                frame_metadata_ ? metadata & ~(1u << 16)
                                : metadata | (1u << 16));
            hololink_->write_uint32(TX_PKT_PROC_ADDR, 0x00000005);
            hololink_->write_uint32(
                PLAYER_ADDR + PLAYER_WINDOW_SIZE_OFFSET,
                std::max(frame_size_, std::uint32_t { 64 }));
            hololink_->write_uint32(
                PLAYER_ADDR + PLAYER_WINDOW_NUMBER_OFFSET, 4096);
            hololink_->write_uint32(
                PLAYER_ADDR + PLAYER_TIMER_OFFSET, 322 * DATA_GEN_SPACING_US);
            hololink_->write_uint32(
                PLAYER_ADDR + PLAYER_ENABLE_OFFSET, 0x00000003);
        } catch (...) {
            // BaseReceiverOp cannot call device_stop after device_start throws.
            // Best-effort disable here so a partially started player is not
            // left running.
            try {
                stop();
            } catch (...) {
            }
            throw;
        }
    }

    void stop()
    {
        hololink_->write_uint32(PLAYER_ADDR + PLAYER_ENABLE_OFFSET, 0);
    }

private:
    static constexpr uint32_t DATA_GEN_SPACING_US = 10;
    static constexpr uint32_t PLAYER_ADDR = 0x5000'0000;
    static constexpr uint32_t PLAYER_ENABLE_OFFSET = 0x0004;
    static constexpr uint32_t PLAYER_TIMER_OFFSET = 0x0008;
    static constexpr uint32_t PLAYER_WINDOW_SIZE_OFFSET = 0x000C;
    static constexpr uint32_t PLAYER_WINDOW_NUMBER_OFFSET = 0x0010;
    static constexpr uint32_t METADATA_PACKET_OFFSET = 0x102C;
    static constexpr uint32_t TX_PKT_PROC_ADDR = 0x0120'0000;

    std::shared_ptr<hololink::Hololink> hololink_;
    uint32_t frame_size_;
    bool frame_metadata_;
};

// Takes the transceiver's tensor output, on which it never emits: Forward mode sends each
// payload straight back to the FPGA. Like every receiver's, that output still needs a consumer.
class DiscardTensorOp : public holoscan::Operator<> {
public:
    holoscan::Input<holoscan::Tensor> input { "input" };

    void setup(holoscan::OperatorSpec& spec) override
    {
        spec.input(input, "input")
            .expects_tensor(holoscan::TensorInputSpec {
                .representation = holoscan::TensorRepresentation {
                    .memory_kind = holoscan::MemoryKind::kCudaDevice,
                },
            });
    }

    [[nodiscard]] holoscan::Contract contract() const override
    {
        holoscan::Contract c;
        c.trigger(holoscan::OnEach { input });
        return c;
    }

    [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override
    {
        auto sample = input.receive();
        if (!sample) {
            return holoscan::make_unexpected(std::move(sample).error());
        }
        return {};
    }
};

struct LoopbackOptions {
    std::string hololink_ip = hololink::env_hololink_ip(0, "192.168.0.2");
    std::string ibv_name; // empty: the transceiver uses the device on the board's network interface
    uint32_t ibv_port = 1;
    uint32_t tx_qp = 2;
    uint32_t frame_size = 32;
    uint32_t mtu = 256;
    uint32_t slots = 1024;
    int poll_cpu = -1;
    bool gpu_memory = false;
    bool frame_metadata = false;
};

int run_loopback(const LoopbackOptions& options, holoscan::ManagedShutdownSignals& signals)
{
    holoscan::set_log_level(holoscan::LogLevel::INFO);
    CudaCheck(cuInit(0));
    CUdevice device;
    CudaCheck(cuDeviceGet(&device, 0));
    CUcontext context;
    CudaCheck(cuDevicePrimaryCtxRetain(&context, device));

    auto uuid_strategy
        = std::make_shared<hololink::BasicEnumerationStrategy>(
            hololink::Metadata {}, 1, 1, 2);
    hololink::Enumerator::set_uuid_strategy(
        hololink::HOLOLINK_LITE_UUID, uuid_strategy);
    auto metadata = hololink::Enumerator::find_channel(options.hololink_ip);
    hololink::DataChannel::use_sensor(metadata, 0);
    hololink::DataChannel::use_mtu(metadata, options.mtu);
    auto channel = std::make_shared<hololink::DataChannel>(metadata);
    auto hololink = channel->hololink();
    auto data_gen = std::make_shared<DataGen>(
        hololink, options.frame_size, options.frame_metadata);

    holoscan::Graph graph { "cpu_roce_loopback" };
    auto receiver = graph.op<hololink::operators::CpuRoceTransceiverOp>(
        "cpu_roce_fwd", channel, [data_gen] { data_gen->start(); },
        [data_gen] { data_gen->stop(); });
    receiver.set_params(hololink::operators::CpuRoceTransceiverParams {
        .frame_context = context,
        .frame_size = options.frame_size,
        .ibv_name = options.ibv_name,
        .ibv_port = options.ibv_port,
        .tx_ibv_qp = options.tx_qp,
        .slots = options.slots,
        .forward = 1,
        .gpu_memory = options.gpu_memory,
        .frame_metadata = options.frame_metadata,
        .poll_cpu = options.poll_cpu,
    });
    auto sink = graph.op<DiscardTensorOp>("sink");
    graph.add_flow(receiver->tensor, sink->input);
    graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

    holoscan::CompileOptions compile_options;
    compile_options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement {
        .operator_path = "cpu_roce_fwd",
        .output_port = "tensor",
        .device = holoscan::DeviceId { 0 },
    });
    const auto plan = holoscan::compile(graph, std::move(compile_options));
    if (!plan.ok()) {
        hx::print_compile_failure(plan);
        CudaCheck(cuDevicePrimaryCtxRelease(device));
        return EXIT_FAILURE;
    }

    hololink->start();
    hololink->reset();
    std::cout << "Running CPU RoCE FPGA Forward loopback"
              << (options.gpu_memory
                         ? " with GPU DMA-BUF payload memory"
                         : "")
              << (options.frame_metadata ? " with HSB frame metadata" : "")
              << "; press Ctrl+C to stop." << std::endl;
    // The run lasts until Ctrl+C or SIGTERM stops it, or the transceiver's receive path fails.
    auto session = holoscan::run_async(plan, signals);
    session.wait();
    const bool failed = hx::report_session(session, plan);
    hololink->stop();
    CudaCheck(cuDevicePrimaryCtxRelease(device));
    return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}

uint32_t parse_uint32_option(
    const std::string& name, const std::string& argument)
{
    size_t parsed = 0;
    const unsigned long long value = std::stoull(argument, &parsed);
    if (parsed != argument.size()
        || value > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument(
            "--" + name + " requires an unsigned 32-bit integer");
    }
    return static_cast<uint32_t>(value);
}

int parse_int_option(const std::string& name, const std::string& argument)
{
    size_t parsed = 0;
    const long value = std::stol(argument, &parsed);
    if (parsed != argument.size()
        || value < std::numeric_limits<int>::min()
        || value > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("--" + name + " requires an integer");
    }
    return static_cast<int>(value);
}

void usage(const char* name)
{
    std::cout
        << "Usage: " << name << " [options]\n"
        << "  --hololink IP       HSB address (default 192.168.0.2)\n"
        << "  --ibv-name NAME     verbs device (default the one on the board's network interface)\n"
        << "  --ibv-port N        verbs port, with --ibv-name (default 1)\n"
        << "  --tx-ibv-qp N       FPGA receive QP (default 2)\n"
        << "  --frame-size N      payload bytes (default 32)\n"
        << "  --mtu N             HSB MTU (default 256)\n"
        << "  --slots N           RX payload-ring slots (default 1024)\n"
        << "  --poll-cpu N        pin Forward CQ poller (default -1 unset)\n"
        << "  --gpu-memory        use CUDA device memory via DMA-BUF (default off)\n"
        << "  --frame-metadata    reserve the RX trailer and enable it on HSB\n";
}

} // namespace

int main(int argc, char** argv)
{
    LoopbackOptions options;

    const option long_options[] = {
        { "help", no_argument, nullptr, 'h' },
        { "hololink", required_argument, nullptr, 0 },
        { "ibv-name", required_argument, nullptr, 0 },
        { "ibv-port", required_argument, nullptr, 0 },
        { "tx-ibv-qp", required_argument, nullptr, 0 },
        { "frame-size", required_argument, nullptr, 0 },
        { "mtu", required_argument, nullptr, 0 },
        { "slots", required_argument, nullptr, 0 },
        { "poll-cpu", required_argument, nullptr, 0 },
        { "gpu-memory", no_argument, nullptr, 0 },
        { "frame-metadata", no_argument, nullptr, 0 },
        { nullptr, 0, nullptr, 0 },
    };

    try {
        while (true) {
            int index = 0;
            const int value
                = getopt_long(argc, argv, "h", long_options, &index);
            if (value == -1) {
                break;
            }
            if (value == 'h') {
                usage(argv[0]);
                return EXIT_SUCCESS;
            }
            if (value != 0) {
                usage(argv[0]);
                return EXIT_FAILURE;
            }

            const std::string name(long_options[index].name);
            if (name == "gpu-memory") {
                options.gpu_memory = true;
                continue;
            }
            if (name == "frame-metadata") {
                options.frame_metadata = true;
                continue;
            }

            const std::string argument(optarg);
            if (name == "hololink") {
                options.hololink_ip = argument;
            } else if (name == "ibv-name") {
                options.ibv_name = argument;
            } else if (name == "ibv-port") {
                options.ibv_port = parse_uint32_option(name, argument);
            } else if (name == "tx-ibv-qp") {
                options.tx_qp = parse_uint32_option(name, argument);
            } else if (name == "frame-size") {
                options.frame_size = parse_uint32_option(name, argument);
            } else if (name == "mtu") {
                options.mtu = parse_uint32_option(name, argument);
            } else if (name == "slots") {
                options.slots = parse_uint32_option(name, argument);
            } else if (name == "poll-cpu") {
                options.poll_cpu = parse_int_option(name, argument);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "cpu_roce_loopback: invalid option: " << e.what()
                  << std::endl;
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    // With managed signals, Ctrl+C stops the session cooperatively, so the data player and the
    // board are stopped on the way out.
    return hx::run_with_managed_signals(
        [&](holoscan::ManagedShutdownSignals& signals) { return run_loopback(options, signals); });
}
