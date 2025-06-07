/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <functional>
#include <future>
#include <sstream>
#include <stdexcept>
#include <thread>

#include <sched.h>

#include <hololink/common/holoargs.hpp>
#include <hololink/core/data_channel.hpp>
#include <hololink/core/hololink.hpp>
#include <hololink/core/jesd.hpp>
#include <hololink/core/tools.hpp>
#include <hololink/operators/iq_dec/iq_dec_op.hpp>
#include <hololink/operators/iq_enc/iq_enc_op.hpp>
#include <hololink/operators/roce_receiver/roce_receiver_op.hpp>
#include <hololink/operators/roce_transmitter/roce_transmitter_op.hpp>
#include <hololink/operators/sig_gen/sig_gen_op.hpp>
#include <hololink/operators/sig_viewer/sig_viewer_op.hpp>
#include <hololink/operators/udp_transmitter/udp_transmitter_op.hpp>

using Rational = hololink::operators::Rational;

uint32_t get_infiniband_interface_index(const std::string& interface_name)
{
    auto infiniband_devices = hololink::core::infiniband_devices();

    for (size_t i = 0; i < infiniband_devices.size(); ++i)
        if (infiniband_devices[i] == interface_name)
            return i;
    return -1;
}

class HololinkDevice {
public:
    HololinkDevice(hololink::Hololink& hololink, size_t frame_size, uint32_t host_pause_mapping)
        : hololink_(hololink)
        , frame_size_(frame_size)
        , host_pause_mapping_(host_pause_mapping)
    {
        HSB_LOG_INFO("HololinkDevice frame_size={}", frame_size_);
    }

    void start()
    {
        HSB_LOG_INFO("HololinkDevice start");

        // Configure the AD9986 MxFE.
        mxfe_config_ = std::make_shared<hololink::AD9986Config>(hololink_);
        mxfe_config_->host_pause_mapping(host_pause_mapping_);
        mxfe_config_->apply();
    }

    void stop()
    {
        HSB_LOG_INFO("HololinkDevice stop");
    }

private:
    hololink::Hololink& hololink_;
    size_t frame_size_;
    uint32_t host_pause_mapping_;

    std::shared_ptr<hololink::AD9986Config> mxfe_config_;
};

/**
 * Helper class to handle initialization and resource management for RoCE receiver setup.
 * This class encapsulates the complex initialization logic while providing proper error handling
 * and resource cleanup. It follows the same pattern as other examples in the codebase while
 * maintaining clean separation of concerns.
 */
class RoceReceiverInitializer {
public:
    RoceReceiverInitializer(const std::string& hololink_ip, size_t frame_size, uint32_t host_pause_mapping)
        : hololink_ip_(hololink_ip)
        , frame_size_(frame_size)
        , host_pause_mapping_(host_pause_mapping)
    {
        initialize();
    }

    ~RoceReceiverInitializer()
    {
        cleanup();
    }

    // Initialize CUDA context and Hololink resources.
    void initialize()
    {
        try {
            // # Get a handle to the GPU
            CudaCheck(cuInit(0));
            int cu_device_ordinal = 0;

            CudaCheck(cuDeviceGet(&cu_device_, cu_device_ordinal));
            CudaCheck(cuDevicePrimaryCtxRetain(&cu_context_, cu_device_));

            // Get a handle to the data source
            auto channel_metadata = hololink::Enumerator::find_channel(hololink_ip_);
            hololink::DataChannel::use_sensor(channel_metadata, 0); // Support Tx/Rx on the same interface
            HSB_LOG_INFO("channel_metadata={}", channel_metadata);
            hololink_channel_.reset(new hololink::DataChannel(channel_metadata));
            hololink_ = hololink_channel_->hololink();
            hololink_device_.reset(new HololinkDevice(*hololink_, frame_size_, host_pause_mapping_));

            hololink_->start();
            hololink_->reset();
            initialized_ = true;
        } catch (const std::exception& e) {
            cleanup();
            throw std::runtime_error("Failed to initialize RoCE receiver: " + std::string(e.what()));
        }
    }

    // Clean up resources in reverse order of initialization.
    void cleanup()
    {
        if (hololink_)
            try {
                hololink_->reset();
            } catch (const std::exception& e) {
                HSB_LOG_ERROR("Error during hololink reset: {}", e.what());
            }
        if (hololink_device_)
            try {
                hololink_device_->stop();
            } catch (const std::exception& e) {
                HSB_LOG_ERROR("Error during device stop: {}", e.what());
            }
        hololink_.reset();
        hololink_channel_.reset();
        hololink_device_.reset();
        cu_context_ = nullptr;
        initialized_ = false;
    }

    std::shared_ptr<hololink::operators::RoceReceiverOp> create_receiver_op(holoscan::Application& app, const std::string& name, const std::string& ibv_name, uint32_t ibv_port)
    {
        if (!initialized_)
            throw std::runtime_error("RoceReceiverInitializer not initialized");

        return app.make_operator<hololink::operators::RoceReceiverOp>(
            name,
            holoscan::Arg("frame_size", frame_size_),
            holoscan::Arg("frame_context", cu_context_),
            holoscan::Arg("ibv_name", ibv_name),
            holoscan::Arg("ibv_port", ibv_port),
            holoscan::Arg("hololink_channel", hololink_channel_.get()),
            holoscan::Arg("device_start", std::function<void()>([this] { hololink_device_->start(); })),
            holoscan::Arg("device_stop", std::function<void()>([this] { hololink_device_->stop(); })));
    }

private:
    std::string hololink_ip_;
    size_t frame_size_;
    uint32_t host_pause_mapping_;
    CUdevice cu_device_;
    CUcontext cu_context_;
    std::unique_ptr<hololink::DataChannel> hololink_channel_;
    std::shared_ptr<hololink::Hololink> hololink_;
    std::unique_ptr<HololinkDevice> hololink_device_;
    bool initialized_ = false;
};

/**
 * The SignalGeneratorTxApp is a sample app that does the following:
 * 1. Generates a signal with IQ components.
 * 2. The signal is encoded into an IQ buffer
 * 3. The IQ buffer is transmitted by a RoCE transmitter.
 */
class SignalGeneratorTxApp : public holoscan::Application {
public:
    SignalGeneratorTxApp(hololink::args::VariablesMap& variables_map, hololink::ImGuiRenderer* renderer)
        : variables_map_(variables_map)
        , renderer_(renderer)
    {
    }

    void compose() override
    {
        using namespace hololink;
        HSB_LOG_INFO("compose");
        const auto samples_count = variables_map_["samples-count"].as<unsigned>();
        const auto sampling_interval = variables_map_["sampling-interval"].as<Rational>();
        auto buffer_size = static_cast<uint64_t>(samples_count * 4);

        signal_generator_ = make_operator<operators::SignalGeneratorOp>(
            "Signal Generator",
            holoscan::Arg("renderer", renderer_),
            holoscan::Arg("samples_count", samples_count),
            holoscan::Arg("sampling_interval", sampling_interval),
            holoscan::Arg("in_phase", variables_map_["expression-i"].as<std::string>()),
            holoscan::Arg("quadrature", variables_map_["expression-q"].as<std::string>()));
        iq_encoder_ = make_operator<operators::IQEncoderOp>(
            "IQ Encoder",
            holoscan::Arg("renderer", renderer_));
        roce_transmitter_ = make_operator<operators::RoceTransmitterOp>(
            "Roce Transmitter",
            holoscan::Arg("ibv_name", variables_map_["tx-ibv-name"].as<std::string>()),
            holoscan::Arg("ibv_port", variables_map_["tx-ibv-port"].as<unsigned>()),
            holoscan::Arg("hololink_ip", variables_map_["tx-hololink"].as<std::string>()),
            holoscan::Arg("ibv_qp", variables_map_["tx-ibv-qp"].as<unsigned>()),
            holoscan::Arg("buffer_size", buffer_size),
            holoscan::Arg("queue_size", variables_map_["tx-queue-size"].as<uint64_t>()));

        add_flow(signal_generator_, iq_encoder_, { { "output", "input" } });
        add_flow(iq_encoder_, roce_transmitter_, { { "output", "input" } });
    }

private:
    hololink::args::VariablesMap& variables_map_;
    hololink::ImGuiRenderer* renderer_;
    std::shared_ptr<holoscan::Operator> signal_generator_;
    std::shared_ptr<holoscan::Operator> iq_encoder_;
    std::shared_ptr<holoscan::Operator> roce_transmitter_;
};

/**
 * The SignalGeneratorRxApp is a sample app that does the following:
 * 1. Receives an IQ buffer from a RoCE receiver.
 * 2. The IQ buffer is decoded back into a signal (with IQ components).
 * 3. The signal is viewed by the SignalViewerOp.
 * 4. The signal is transmitted to a UDP port and can be viewed by the GNU Radio app.
 */
class SignalGeneratorRxApp : public holoscan::Application {
public:
    SignalGeneratorRxApp(hololink::args::VariablesMap& variables_map, hololink::ImGuiRenderer* renderer)
        : variables_map_(variables_map)
        , renderer_(renderer)
    {
    }

    void compose() override
    {
        using namespace hololink;
        HSB_LOG_INFO("compose");
        const bool udp_enabled = !variables_map_["udp-ip"].as<std::string>().empty();
        const auto samples_count = variables_map_["samples-count"].as<unsigned>();
        auto buffer_size = static_cast<uint64_t>(samples_count * 4);

        uint32_t host_pause_mapping = 1 << get_infiniband_interface_index(variables_map_["tx-ibv-name"].as<std::string>());

        // Initialize receiver resources
        roce_receiver_initializer_ = std::make_unique<RoceReceiverInitializer>(
            variables_map_["rx-hololink"].as<std::string>(),
            buffer_size,
            host_pause_mapping);

        roce_receiver_ = roce_receiver_initializer_->create_receiver_op(
            *this,
            "receiver",
            variables_map_["rx-ibv-name"].as<std::string>(),
            variables_map_["rx-ibv-port"].as<unsigned>());
        iq_decoder_ = make_operator<operators::IQDecoderOp>(
            "IQ Decoder",
            holoscan::Arg("renderer", renderer_));
        signal_viewer_ = make_operator<operators::SignalViewerOp>(
            "Signal Viewer",
            holoscan::Arg("renderer", renderer_));

        add_flow(roce_receiver_, iq_decoder_, { { "output", "input" } });
        add_flow(iq_decoder_, signal_viewer_, { { "output", "input" } });

        if (udp_enabled) {
            udp_transmitter_ = make_operator<operators::UdpTransmitterOp>(
                "UDP Transmitter",
                holoscan::Arg("ip", variables_map_["udp-ip"].as<std::string>()),
                holoscan::Arg("port", variables_map_["udp-port"].as<uint16_t>()),
                // The max_buffer_size value should match the gnu radio configuration file
                holoscan::Arg("max_buffer_size", uint16_t(8192 * sizeof(float))));
            add_flow(iq_decoder_, udp_transmitter_, { { "output", "input" } });
        }
    }

private:
    hololink::args::VariablesMap& variables_map_;
    hololink::ImGuiRenderer* renderer_;
    std::unique_ptr<RoceReceiverInitializer> roce_receiver_initializer_;
    std::shared_ptr<holoscan::Operator> roce_receiver_;
    std::shared_ptr<holoscan::Operator> iq_decoder_;
    std::shared_ptr<holoscan::Operator> signal_viewer_;
    std::shared_ptr<holoscan::Operator> udp_transmitter_;
};

int main([[maybe_unused]] int argc, [[maybe_unused]] char** argv)
{
    using namespace hololink::args;
    OptionsDescription options_description("Allowed options");
    static const int default_samples_count = 4096 * 1000 * 3;
    using Rational = hololink::operators::Rational;
    static const Rational default_sampling_interval(1, 128);

    std::string rx_default_infiniband_interface("mlx5_0");
    std::string tx_default_infiniband_interface("mlx5_1");
    auto infiniband_devices = hololink::core::infiniband_devices();
    if (infiniband_devices.size() >= 2) {
        rx_default_infiniband_interface = infiniband_devices[0];
        tx_default_infiniband_interface = infiniband_devices[1];
    }

    // clang-format off
    options_description.add_options()
        ("real-time", bool_switch()->default_value(false), "Set the process to real-time priority")
        ("no-gui", bool_switch()->default_value(false), "Disable the graphic user interface")
        ("expression-i", value<std::string>()->default_value("cos(2*PI*x)"), "Signal expression for the in-phase component. Use 'x' as the expression's variable")
        ("expression-q", value<std::string>()->default_value("sin(2*PI*x)"), "Signal expression for the quadrature component. Use 'x' as the expression's variable")
        ("samples-count", value<unsigned>()->default_value(default_samples_count), "Number of samples to be generated")
        ("sampling-interval", value<Rational>()->default_value(default_sampling_interval), "The interval between sequential samples, must be in the following format: \"num/den\"")

        ("tx-hololink", value<std::string>()->default_value(""), "IP address of Hololink board that the data is transitted to")
        ("tx-ibv-name", value<std::string>()->default_value(tx_default_infiniband_interface), "IBV device name used for transmission")
        ("tx-ibv-port", value<unsigned>()->default_value(1), "Port number of IBV device used for transmission")
        ("tx-ibv-qp", value<unsigned>()->default_value(2), "QP number for the IBV stream that the data is transmitted to")
        ("tx-queue-size", value<uint64_t>()->default_value(3), "The number of buffers that can wait to be transmitted")

        ("rx-hololink", value<std::string>()->default_value(""), "IP address of Hololink board that the data is received from")
        ("rx-ibv-name", value<std::string>()->default_value(rx_default_infiniband_interface), "Local IB device name used for reception")
        ("rx-ibv-port", value<unsigned>()->default_value(1), "Port of the local IB device used for reception")

        ("udp-ip", value<std::string>()->default_value(""), "IP to transmit the data to")
        ("udp-port", value<uint16_t>()->default_value(5000), "Port to transmit the data to")
        ;
    // clang-format on
    auto variables_map = hololink::args::Parser().parse_command_line(argc, argv, options_description);

    // Set the process to real-time priority
    if (variables_map["real-time"].as<bool>()) {
        ::sched_param sched;
        sched.sched_priority = 99; // Maximum priority
        if (::sched_setscheduler(::getpid(), SCHED_FIFO, &sched) != 0) {
            HSB_LOG_ERROR("Failed to set scheduler for PID={}", ::getpid());
        }
        HSB_LOG_INFO("Set PID={} priority to real-time", ::getpid());
    }

    HSB_LOG_INFO("Initializing");
    // Since Dear ImGui is not supported by the Holoviz operator,
    // A separate renderer is used to render the GUI.
    // The renderer uses the Holoviz module to draw to the screen.
    std::unique_ptr<hololink::ImGuiRenderer> renderer;
    if (!variables_map["no-gui"].as<bool>())
        renderer.reset(new hololink::ImGuiRenderer());

    std::future<void> tx_future;
    std::future<void> rx_future;

    if (!variables_map["tx-hololink"].as<std::string>().empty()) {
        auto tx_app = holoscan::make_application<SignalGeneratorTxApp>(variables_map, renderer.get());
        tx_future = std::async(std::launch::async, std::bind(&holoscan::Application::run, tx_app));
    }

    if (!variables_map["rx-hololink"].as<std::string>().empty()) {
        auto rx_app = holoscan::make_application<SignalGeneratorRxApp>(variables_map, renderer.get());
        rx_future = std::async(std::launch::async, std::bind(&holoscan::Application::run, rx_app));
    }

    if (!tx_future.valid() && !rx_future.valid()) {
        HSB_LOG_ERROR("Either tx-hololink or rx-hololink must be provided");
        return 1;
    }

    // Wait for both futures to complete
    if (tx_future.valid())
        tx_future.get();
    if (rx_future.valid())
        rx_future.get();

    return 0;
}
