/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Signal generator over an HSB100G (AD9986 MxFE):
//
//   TX (--tx-hololink): SignalGeneratorOp -> IQEncoderOp -> RoceTransmitterOp
//   RX (--rx-hololink): RoceReceiverOp -> IQDecoderOp -> SignalViewerOp
//                                                     -> UdpTransmitterOp (--udp-ip, e.g. for the
//                                                        GNU Radio client in
//                                                        signal_generator_client.grc)
//
// Either leg or both run in one graph. The operators' GUIs share one ImGui window (--no-gui to
// disable); closing it stops the app.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include <sched.h>
#include <unistd.h>

#include <cuda.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/lifecycle.hpp>
#include <holoscan/core/managed_shutdown_signals.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include <hololink/common/cuda_helper.hpp>
#include <hololink/common/gui_renderer.hpp>
#include <hololink/common/holoargs.hpp>
#include <hololink/core/data_channel.hpp>
#include <hololink/core/enumerator.hpp>
#include <hololink/core/hololink.hpp>
#include <hololink/core/jesd.hpp>
#include <hololink/core/logging_internal.hpp>
#include <hololink/core/networking.hpp>
#include <hololink/operators/discard_metadata/discard_metadata.hpp>
#include <hololink/operators/iq_dec/iq_dec_op.hpp>
#include <hololink/operators/iq_enc/iq_enc_op.hpp>
#include <hololink/operators/roce_receiver/roce_receiver_op.hpp>
#include <hololink/operators/roce_transmitter/roce_transmitter_op.hpp>
#include <hololink/operators/sig_gen/sig_gen_op.hpp>
#include <hololink/operators/sig_viewer/sig_viewer_op.hpp>
#include <hololink/operators/udp_transmitter/udp_transmitter_op.hpp>

namespace {

using Rational = hololink::operators::Rational;

// The receiver's ceiling for one frame (see BaseReceiverOp::kMaxFrameBytes).
constexpr uint64_t kMaxFrameBytes = 64ULL * 1024 * 1024;
// Each IQ sample travels as two int16 components.
constexpr uint64_t kEncodedBytesPerSample = 4;

// Configures the AD9986 MxFE's JESD links when the receiver starts.
class MxfeDevice {
public:
    MxfeDevice(std::shared_ptr<hololink::Hololink> hololink, uint32_t host_pause_mapping)
        : hololink_(std::move(hololink))
        , host_pause_mapping_(host_pause_mapping)
    {
        if (host_pause_mapping_ != 1 && host_pause_mapping_ != 2) {
            throw std::runtime_error("Host pause mapping is not valid. Mask must be 0x01 or 0x02");
        }
    }

    void start()
    {
        HSB_LOG_INFO("Configuring the AD9986 MxFE");
        // Kept only once the link is up: a failed start() gets no stop(), and destroying the
        // configuration stops its SPI daemon thread.
        auto config = std::make_shared<hololink::AD9986Config>(*hololink_);
        config->host_pause_mapping(host_pause_mapping_);
        config->apply();
        if (!config->verify_link_status()) {
            throw std::runtime_error("JESD link status check failed");
        }
        config_ = std::move(config);
    }

    // Stops the configuration's SPI daemon thread while the board is still up.
    void stop()
    {
        HSB_LOG_INFO("MxFE stop");
        config_.reset();
    }

private:
    std::shared_ptr<hololink::Hololink> hololink_;
    uint32_t host_pause_mapping_;
    std::shared_ptr<hololink::AD9986Config> config_;
};

struct FrameLimitStatus {
    std::atomic<bool> done { false };
};

// Counts received frames (on the receiver's metadata, which it also drains) toward --frame-limit.
class FrameLimitOp : public holoscan::Operator<> {
public:
    FrameLimitOp(uint64_t max_frames, std::shared_ptr<FrameLimitStatus> status)
        : max_frames_(max_frames)
        , status_(std::move(status))
    {
    }

    holoscan::Input<hololink::CaptureMetadataT> input_metadata { "metadata" };

    void setup(holoscan::OperatorSpec& spec) override { spec.input(input_metadata, "metadata").queue_depth(4U); }

    [[nodiscard]] holoscan::Contract contract() const override
    {
        holoscan::Contract contract;
        contract.trigger(holoscan::OnEach { input_metadata });
        return contract;
    }

    holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override
    {
        auto sample = input_metadata.receive();
        if (!sample) {
            return holoscan::make_unexpected(std::move(sample).error());
        }
        if (++frames_ >= max_frames_) {
            status_->done = true;
        }
        return {};
    }

private:
    uint64_t max_frames_;
    std::shared_ptr<FrameLimitStatus> status_;
    uint64_t frames_ = 0;
};

// The data plane the TX board is enumerated on picks which host-pause input the RX board's
// MxFE configuration uses.
uint32_t host_pause_mapping(const std::string& tx_hololink_ip)
{
    auto tx_channel_metadata = hololink::Enumerator::find_channel(tx_hololink_ip);
    const auto tx_data_plane = tx_channel_metadata.get<int64_t>("data_plane");
    if (!tx_data_plane || (*tx_data_plane != 0 && *tx_data_plane != 1)) {
        HSB_LOG_ERROR("HSB100G host pause mapping must have tx-hololink mapped to data plane 0 or 1. "
                      "tx-hololink ({}) is mapped to data plane {}",
            tx_hololink_ip, tx_data_plane ? std::to_string(*tx_data_plane) : std::string("(none)"));
        throw std::runtime_error("TX data plane must be 0 or 1");
    }
    HSB_LOG_INFO("TX data plane={}", *tx_data_plane);
    return static_cast<uint32_t>(1 << *tx_data_plane);
}

// The IB device and port for the board at board_ip: ibv_name and ibv_port if ibv_name is given,
// else the device on the network interface the board's DataChannel uses. The first devices the
// system lists may not reach the board: on a DGX Spark each port shows up as two devices, and
// only one of them has the board's RoCE v2 GID.
std::pair<std::string, uint32_t> ibv_device(const std::string& board_ip, const std::string& ibv_name,
    uint32_t ibv_port, const hololink::Metadata* channel_metadata = nullptr)
{
    if (!ibv_name.empty()) {
        return { ibv_name, ibv_port };
    }
    const hololink::Metadata metadata
        = channel_metadata ? *channel_metadata : hololink::Enumerator::find_channel(board_ip);
    const auto device = hololink::core::infiniband_device_for_peer(
        board_ip, metadata.get<std::string>("interface").value_or(std::string()));
    HSB_LOG_INFO("Using IB device {} port {} for the board at {}", device.first, device.second, board_ip);
    return device;
}

int run(int argc, char** argv, holoscan::ManagedShutdownSignals& signals)
{
    using namespace hololink::args;
    OptionsDescription options_description("Signal generator options");
    static const unsigned default_samples_count = 4096 * 1000 * 3;
    static const Rational default_sampling_interval(1, 128);

    // clang-format off
    options_description.add_options()
        ("real-time", bool_switch()->default_value(false), "Set the process to real-time priority")
        ("no-gui", bool_switch()->default_value(false), "Disable the graphic user interface")
        ("frame-limit", value<int64_t>()->default_value(0), "Stop after receiving this many frames (0: no limit)")
        ("expression-i", value<std::string>()->default_value("cos(2*PI*x)"), "Signal expression for the in-phase component. Use 'x' as the expression's variable")
        ("expression-q", value<std::string>()->default_value("sin(2*PI*x)"), "Signal expression for the quadrature component. Use 'x' as the expression's variable")
        ("samples-count", value<unsigned>()->default_value(default_samples_count), "Number of samples to be generated")
        ("sampling-interval", value<Rational>()->default_value(default_sampling_interval), "The interval between sequential samples, must be in the following format: \"num/den\"")

        ("tx-hololink", value<std::string>()->default_value(""), "IP address of Hololink board that the data is transmitted to")
        ("tx-ibv-name", value<std::string>()->default_value(""), "IBV device name used for transmission (default: the device on the TX board's network interface)")
        ("tx-ibv-port", value<unsigned>()->default_value(1), "Port number of IBV device used for transmission (with --tx-ibv-name)")
        ("tx-ibv-qp", value<unsigned>()->default_value(2), "QP number for the IBV stream that the data is transmitted to")
        ("tx-queue-size", value<uint64_t>()->default_value(3), "The number of buffers that can wait to be transmitted")

        ("rx-hololink", value<std::string>()->default_value(""), "IP address of Hololink board that the data is received from")
        ("rx-ibv-name", value<std::string>()->default_value(""), "Local IB device name used for reception (default: the device on the RX board's network interface)")
        ("rx-ibv-port", value<unsigned>()->default_value(1), "Port of the local IB device used for reception (with --rx-ibv-name)")

        ("udp-ip", value<std::string>()->default_value(""), "IP to transmit the data to")
        ("udp-port", value<uint16_t>()->default_value(5000), "Port to transmit the data to")
        ;
    // clang-format on
    auto variables_map = Parser().parse_command_line(argc, argv, options_description);

    const auto tx_hololink = variables_map["tx-hololink"].as<std::string>();
    const auto rx_hololink = variables_map["rx-hololink"].as<std::string>();
    const auto samples_count = variables_map["samples-count"].as<unsigned>();
    const auto frame_limit = variables_map["frame-limit"].as<int64_t>();
    const auto tx_queue_size = variables_map["tx-queue-size"].as<uint64_t>();
    const uint64_t buffer_size = uint64_t(samples_count) * kEncodedBytesPerSample;

    if (tx_hololink.empty() && rx_hololink.empty()) {
        HSB_LOG_ERROR("Either tx-hololink or rx-hololink must be provided");
        return 1;
    }
    if (!rx_hololink.empty() && tx_hololink.empty()) {
        // The RX board's MxFE configuration depends on the TX board's data plane.
        HSB_LOG_ERROR("rx-hololink also needs tx-hololink");
        return 1;
    }
    if ((samples_count == 0) || (samples_count % 4)) {
        HSB_LOG_ERROR("samples-count must be a positive multiple of 4");
        return 1;
    }
    if (frame_limit < 0) {
        HSB_LOG_ERROR("frame-limit must not be negative");
        return 1;
    }
    if (!rx_hololink.empty() && (buffer_size > kMaxFrameBytes)) {
        HSB_LOG_ERROR("samples-count {} makes {}-byte frames, more than the receiver's {}", samples_count,
            buffer_size, kMaxFrameBytes);
        return 1;
    }

    // Set the process to real-time priority
    if (variables_map["real-time"].as<bool>()) {
        ::sched_param sched {};
        sched.sched_priority = 99; // Maximum priority
        if (::sched_setscheduler(::getpid(), SCHED_FIFO, &sched) != 0) {
            HSB_LOG_ERROR("Failed to set scheduler for PID={}. Need root privileges for real-time scheduling.", ::getpid());
        } else {
            HSB_LOG_INFO("Set PID={} priority to real-time", ::getpid());
        }
    }

    HSB_LOG_INFO("Initializing");
    CudaCheck(cuInit(0));
    const int cu_device_ordinal = 0;
    CUdevice cu_device;
    CudaCheck(cuDeviceGet(&cu_device, cu_device_ordinal));
    CUcontext cu_context;
    CudaCheck(cuDevicePrimaryCtxRetain(&cu_context, cu_device));
    CudaCheck(cuCtxSetCurrent(cu_context));

    // HolovizOp has no Dear ImGui support, so the operators' GUIs go to a separate renderer,
    // with its own window and thread.
    std::shared_ptr<hololink::ImGuiRenderer> renderer;
    if (!variables_map["no-gui"].as<bool>()) {
        renderer = hololink::ImGuiRenderer::shared();
        // Top to bottom, the pipeline's order: the operators add their windows as they start, in
        // an order the runtime picks.
        renderer->set_window_order({ "Signal Generator", "IQ Encoder", "IQ Decoder", "Signal Viewer" });
    }

    holoscan::Graph graph { "signal_generator" };
    holoscan::CompileOptions options;
    const auto bind_output_device = [&options, cu_device_ordinal](const char* operator_path, const char* output_port) {
        options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement {
            .operator_path = operator_path,
            .output_port = output_port,
            .device = holoscan::DeviceId { cu_device_ordinal },
        });
    };

    std::optional<holoscan::OperatorHandle<hololink::operators::RoceTransmitterOp>> transmitter;
    if (!tx_hololink.empty()) {
        auto signal_generator = graph.op<hololink::operators::SignalGeneratorOp>("signal_generator",
            samples_count,
            variables_map["sampling-interval"].as<Rational>(),
            variables_map["expression-i"].as<std::string>(),
            variables_map["expression-q"].as<std::string>(),
            std::string {},
            renderer);
        auto iq_encoder = graph.op<hololink::operators::IQEncoderOp>("iq_encoder", 1.0f, renderer);
        const auto [tx_ibv_name, tx_ibv_port] = ibv_device(tx_hololink,
            variables_map["tx-ibv-name"].as<std::string>(), variables_map["tx-ibv-port"].as<unsigned>());
        transmitter = graph.op<hololink::operators::RoceTransmitterOp>("roce_transmitter");
        transmitter->set_params(hololink::operators::RoceTransmitterParams {
            .ibv_name = tx_ibv_name,
            .ibv_port = tx_ibv_port,
            .hololink_ip = tx_hololink,
            .ibv_qp = variables_map["tx-ibv-qp"].as<unsigned>(),
            .buffer_size = buffer_size,
            .queue_size = tx_queue_size,
        });

        graph.add_flow(signal_generator->output, iq_encoder->input);
        // The transmitter holds up to tx-queue-size encoded buffers while they're sent; size the
        // queue (and so the encoder's buffer pool) beyond that.
        graph.add_flow(iq_encoder->output, (*transmitter)->input,
            holoscan::ConnectionOptions { .queue_depth = static_cast<uint32_t>(tx_queue_size + 2) });
        bind_output_device("signal_generator", "output");
        bind_output_device("iq_encoder", "output");
    }

    auto frame_limit_status = std::make_shared<FrameLimitStatus>();
    std::shared_ptr<hololink::Hololink> rx_hololink_device;
    if (!rx_hololink.empty()) {
        auto channel_metadata = hololink::Enumerator::find_channel(rx_hololink);
        hololink::DataChannel::use_sensor(channel_metadata, 0); // Support Tx/Rx on the same interface
        HSB_LOG_INFO("channel_metadata={}", channel_metadata);
        const auto [rx_ibv_name, rx_ibv_port] = ibv_device(rx_hololink,
            variables_map["rx-ibv-name"].as<std::string>(), variables_map["rx-ibv-port"].as<unsigned>(), &channel_metadata);
        auto rx_channel = std::make_shared<hololink::DataChannel>(channel_metadata);
        rx_hololink_device = rx_channel->hololink();
        auto mxfe = std::make_shared<MxfeDevice>(rx_hololink_device, host_pause_mapping(tx_hololink));

        auto receiver = graph.op<hololink::operators::RoceReceiverOp>(
            "roce_receiver", rx_channel, [mxfe] { mxfe->start(); }, [mxfe] { mxfe->stop(); });
        receiver.set_params(hololink::operators::RoceReceiverParams {
            .frame_context = cu_context,
            .frame_size = buffer_size,
            .ibv_name = rx_ibv_name,
            .ibv_port = rx_ibv_port,
        });
        auto iq_decoder = graph.op<hololink::operators::IQDecoderOp>("iq_decoder", uint64_t { samples_count }, 1.0f, renderer);
        auto signal_viewer = graph.op<hololink::operators::SignalViewerOp>("signal_viewer", renderer);

        // The receiver fails the run if an emit is backpressured, so its consumers drop the
        // oldest frames instead (a live stream can't be paused).
        const holoscan::ConnectionOptions receiver_edge {
            .queue_policy = holoscan::QueuePolicy::kDropOldest,
            .queue_depth = 4U,
        };
        graph.add_flow(receiver->tensor, iq_decoder->input, receiver_edge);
        graph.add_flow(iq_decoder->output, signal_viewer->input,
            holoscan::ConnectionOptions { .queue_policy = holoscan::QueuePolicy::kDropOldest, .queue_depth = 1U });
        if (frame_limit) {
            auto limit = graph.op<FrameLimitOp>("frame_limit", static_cast<uint64_t>(frame_limit), frame_limit_status);
            graph.add_flow(receiver->metadata, limit->input_metadata, receiver_edge);
        } else {
            auto discard = graph.op<hololink::operators::DiscardMetadataOp>("discard_metadata");
            graph.add_flow(receiver->metadata, discard->input_metadata, receiver_edge);
        }

        const auto udp_ip = variables_map["udp-ip"].as<std::string>();
        if (!udp_ip.empty()) {
            auto udp_transmitter = graph.op<hololink::operators::UdpTransmitterOp>("udp_transmitter");
            udp_transmitter.set_params(hololink::operators::UdpTransmitterParams {
                .ip = udp_ip,
                .port = variables_map["udp-port"].as<uint16_t>(),
                // The max_buffer_size value should match the gnu radio configuration file
                .max_buffer_size = static_cast<uint16_t>(8192 * sizeof(float)),
            });
            graph.add_flow(iq_decoder->output, udp_transmitter->input, holoscan::ConnectionOptions { .queue_depth = 4U });
        }
        bind_output_device("roce_receiver", "tensor");
        bind_output_device("iq_decoder", "output");

        // Configure the MxFE (in the receiver's start) before the transmitter connects.
        if (transmitter) {
            graph.lifecycle().before(receiver, *transmitter, holoscan::LifecycleStage::kStart);
        }
    }
    graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

    auto plan = holoscan::compile(graph, std::move(options));
    if (!plan.ok()) {
        holoscan::log_diagnostics(plan);
        HSB_LOG_ERROR("Graph compilation failed");
        CudaCheck(cuDevicePrimaryCtxRelease(cu_device));
        return 1;
    }

    if (rx_hololink_device) {
        rx_hololink_device->start();
        rx_hololink_device->reset();
    }

    auto session = holoscan::run_async(plan, signals);
    while (session.is_running()) {
        if (frame_limit && frame_limit_status->done) {
            HSB_LOG_INFO("Received {} frames", frame_limit);
            break;
        }
        if (renderer && renderer->window_closed()) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    session.request_stop();
    session.wait();
    const bool failed = (session.termination() == holoscan::RunTermination::kFailed);

    if (rx_hololink_device) {
        try {
            rx_hololink_device->reset();
        } catch (const std::exception& e) {
            HSB_LOG_ERROR("Error during hololink reset: {}", e.what());
        }
        rx_hololink_device->stop();
    }
    CudaCheck(cuDevicePrimaryCtxRelease(cu_device));
    return failed ? 1 : 0;
}

} // namespace

int main(int argc, char** argv)
{
    // First, before anything that may start threads: managed signals must cover them all.
    auto created_signals = holoscan::ManagedShutdownSignals::create();
    if (!created_signals) {
        std::cerr << holoscan::runtime_error_message(created_signals.error()) << '\n';
        return 1;
    }
    auto signals = std::move(created_signals).value();

    int result = 1;
    try {
        result = run(argc, argv, signals);
    } catch (const std::exception& e) {
        HSB_LOG_ERROR("{}", e.what());
    }

    auto closed = signals.close();
    if (!closed) {
        std::cerr << holoscan::runtime_error_message(closed.error()) << '\n';
        return 1;
    }
    return result;
}
