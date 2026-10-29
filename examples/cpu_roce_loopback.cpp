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
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include <hololink/common/cuda_helper.hpp>
#include <hololink/common/tools.hpp>
#include <hololink/core/data_channel.hpp>
#include <hololink/core/enumerator.hpp>
#include <hololink/core/hololink.hpp>
#include <hololink/operators/cpu_roce_transceiver/cpu_roce_transceiver_op.hpp>

#include <holoscan/holoscan.hpp>

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

class SinkOp : public holoscan::Operator {
public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(SinkOp);

    SinkOp() = default;

    void setup(holoscan::OperatorSpec& spec) override
    {
        spec.input<holoscan::gxf::Entity>("input");
    }

    void compute(holoscan::InputContext& input, holoscan::OutputContext&,
        holoscan::ExecutionContext&) override
    {
        (void)input.receive<holoscan::gxf::Entity>("input");
    }
};

class KeepAlive : public holoscan::Operator {
public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(KeepAlive);

    KeepAlive() = default;

    void setup(holoscan::OperatorSpec&) override { }

    void compute(holoscan::InputContext&, holoscan::OutputContext&,
        holoscan::ExecutionContext&) override
    {
    }
};

class LoopbackApp : public holoscan::Application {
public:
    LoopbackApp(CUcontext cuda_context, hololink::DataChannel& channel,
        std::shared_ptr<DataGen> data_gen, std::string ibv_name,
        uint32_t ibv_port, uint32_t tx_qp, uint32_t frame_size,
        uint32_t slots, int poll_cpu, bool gpu_memory, bool frame_metadata)
        : cuda_context_(cuda_context)
        , channel_(channel)
        , data_gen_(std::move(data_gen))
        , ibv_name_(std::move(ibv_name))
        , ibv_port_(ibv_port)
        , tx_qp_(tx_qp)
        , frame_size_(frame_size)
        , slots_(slots)
        , poll_cpu_(poll_cpu)
        , gpu_memory_(gpu_memory)
        , frame_metadata_(frame_metadata)
    {
    }

    void compose() override
    {
        auto receiver = make_operator<hololink::operators::CpuRoceTransceiverOp>(
            "cpu_roce_fwd",
            holoscan::Arg("frame_context", cuda_context_),
            holoscan::Arg("frame_size", static_cast<size_t>(frame_size_)),
            holoscan::Arg("hololink_channel", &channel_),
            holoscan::Arg("ibv_name", ibv_name_),
            holoscan::Arg("ibv_port", ibv_port_),
            holoscan::Arg("tx_ibv_qp", tx_qp_),
            holoscan::Arg("slots", slots_),
            holoscan::Arg("forward", 1u),
            holoscan::Arg("gpu_memory", gpu_memory_),
            holoscan::Arg("frame_metadata", frame_metadata_),
            holoscan::Arg("poll_cpu", poll_cpu_),
            holoscan::Arg("device_start",
                std::function<void()>(
                    [data_gen = data_gen_] { data_gen->start(); })),
            holoscan::Arg("device_stop",
                std::function<void()>(
                    [data_gen = data_gen_] { data_gen->stop(); })));
        auto sink = make_operator<SinkOp>("sink");
        add_flow(receiver, sink, { { "output", "input" } });

        // CpuRoceTransceiverOp has no scheduled frame output. Tick a no-op at
        // a low rate to keep the graph alive without spinning a scheduler CPU.
        add_operator(make_operator<KeepAlive>("keepalive",
            make_condition<holoscan::PeriodicCondition>(
                "keepalive_tick", std::chrono::milliseconds(100))));
    }

private:
    CUcontext cuda_context_;
    hololink::DataChannel& channel_;
    std::shared_ptr<DataGen> data_gen_;
    std::string ibv_name_;
    uint32_t ibv_port_;
    uint32_t tx_qp_;
    uint32_t frame_size_;
    uint32_t slots_;
    int poll_cpu_;
    bool gpu_memory_;
    bool frame_metadata_;
};

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
        << "  --ibv-name NAME     verbs device (default first device)\n"
        << "  --ibv-port N        verbs port (default 1)\n"
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
    std::string hololink_ip
        = hololink::env_hololink_ip(0, "192.168.0.2");
    std::string ibv_name;
    uint32_t ibv_port = 1;
    uint32_t tx_qp = 2;
    uint32_t frame_size = 32;
    uint32_t mtu = 256;
    uint32_t slots = 1024;
    int poll_cpu = -1;
    bool gpu_memory = false;
    bool frame_metadata = false;

    const option options[] = {
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
                = getopt_long(argc, argv, "h", options, &index);
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
            const std::string name(options[index].name);
            if (name == "gpu-memory") {
                gpu_memory = true;
                continue;
            }
            if (name == "frame-metadata") {
                frame_metadata = true;
                continue;
            }
            const std::string argument(optarg);
            if (name == "hololink") {
                hololink_ip = argument;
            } else if (name == "ibv-name") {
                ibv_name = argument;
            } else if (name == "ibv-port") {
                ibv_port = parse_uint32_option(name, argument);
            } else if (name == "tx-ibv-qp") {
                tx_qp = parse_uint32_option(name, argument);
            } else if (name == "frame-size") {
                frame_size = parse_uint32_option(name, argument);
            } else if (name == "mtu") {
                mtu = parse_uint32_option(name, argument);
            } else if (name == "slots") {
                slots = parse_uint32_option(name, argument);
            } else if (name == "poll-cpu") {
                poll_cpu = parse_int_option(name, argument);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "cpu_roce_loopback: invalid option: " << e.what()
                  << std::endl;
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    try {
        holoscan::set_log_level(holoscan::LogLevel::INFO);

        CudaCheck(cuInit(0));
        CUdevice device;
        CudaCheck(cuDeviceGet(&device, 0));
        CUcontext context;
        CudaCheck(cuDevicePrimaryCtxRetain(&context, device));

        if (ibv_name.empty()) {
            const auto devices = hololink::infiniband_devices();
            if (devices.empty()) {
                throw std::runtime_error("No InfiniBand devices found");
            }
            ibv_name = devices.front();
        }

        auto uuid_strategy
            = std::make_shared<hololink::BasicEnumerationStrategy>(
                hololink::Metadata {}, 1, 1, 2);
        hololink::Enumerator::set_uuid_strategy(
            hololink::HOLOLINK_LITE_UUID, uuid_strategy);

        auto metadata = hololink::Enumerator::find_channel(hololink_ip);
        hololink::DataChannel::use_sensor(metadata, 0);
        hololink::DataChannel::use_mtu(metadata, mtu);
        hololink::DataChannel channel(metadata);
        auto hololink = channel.hololink();
        auto data_gen = std::make_shared<DataGen>(
            hololink, frame_size, frame_metadata);
        auto app = holoscan::make_application<LoopbackApp>(
            context, channel, data_gen, ibv_name, ibv_port, tx_qp,
            frame_size, slots, poll_cpu, gpu_memory, frame_metadata);

        hololink->start();
        hololink->reset();
        std::cout << "Running CPU RoCE FPGA Forward loopback"
                  << (gpu_memory
                             ? " with GPU DMA-BUF payload memory"
                             : "")
                  << (frame_metadata ? " with HSB frame metadata" : "")
                  << "; press Ctrl+C to stop." << std::endl;
        app->run();
        hololink->stop();
        CudaCheck(cuDevicePrimaryCtxRelease(device));
    } catch (const std::exception& e) {
        std::cerr << "cpu_roce_loopback: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
