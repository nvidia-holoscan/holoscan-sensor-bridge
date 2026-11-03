/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * IMX274 camera player built on the hololink_module API, ported to Holoscan SDK 5.0's
 * Graph / Operator<Params> shape.
 *
 * Runnable companion to the "IMX274 Module Tutorial" user guide. It plays one IMX274 camera
 * behind a Holoscan Sensor Bridge board through the hardware (RoCE) receiver and displays the
 * video with Holoviz. Configuration is intentionally hard-coded so the tutorial can focus on
 * the API rather than on argument parsing -- see examples/module_imx274_player.cpp for the
 * CLI-driven sibling this mirrors (same operators/chain, no --frame-limit or CLI parsing here).
 */

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <utility>

#include <cuda.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/managed_shutdown_signals.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/time/realtime_clock.hpp>
#include <holoscan/vision/bayer_demosaic_op.hpp>
#include <holoscan/visualization/holoviz_op.hpp>

#include "hololink/module/cuda_unique.hpp"

#include "hololink/module/adapter.hpp"
#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/hololink.hpp"
#include "hololink/module/operators/csi_to_bayer_op.hpp"
#include "hololink/module/operators/image_processor_op.hpp"
#include "hololink/module/operators/roce_receiver_op.hpp"

#include "hololink/module/sensors/imx274/imx274_cam.hpp"
#include "hololink/module/sensors/imx274/imx274_mode.hpp"

#include "player_support.hpp"

namespace hx = hololink::examples;

namespace {

// The board announces itself over bootp; this is the peer IP we wait for.
constexpr const char* const HOLOLINK_IP = "192.168.0.2";
// Seconds to wait for that announcement before giving up.
constexpr std::chrono::seconds DISCOVERY_TIMEOUT(30);
constexpr auto CAMERA_MODE
    = hololink::module::sensors::imx274::Imx274_Mode::IMX274_MODE_1920X1080_60FPS;

} // namespace

int main()
{
    // With managed signals, Ctrl+C stops the session cooperatively, so the camera and the board
    // are stopped on the way out.
    return hx::run_with_managed_signals([](holoscan::ManagedShutdownSignals& signals) {
        HOLOLINK_MODULE_CUDA_CHECK(cuInit(0));
        const int cu_device_ordinal = 0;
        CUdevice cu_device;
        HOLOLINK_MODULE_CUDA_CHECK(cuDeviceGet(&cu_device, cu_device_ordinal));
        CUcontext cu_context;
        HOLOLINK_MODULE_CUDA_CHECK(cuDevicePrimaryCtxRetain(&cu_context, cu_device));
        cuCtxSetCurrent(cu_context);

        // Find the board via bootp enumeration and get its metadata.
        auto& adapter = hololink::module::Adapter::get_adapter();
        hololink::module::EnumerationMetadata metadata
            = adapter.wait_for_channel(HOLOLINK_IP, DISCOVERY_TIMEOUT);

        // The sensor driver is linked directly into the application; it is not a
        // service. It resolves the services it needs from the enumeration metadata.
        auto camera = std::make_shared<hololink::module::sensors::imx274::Imx274Cam>(metadata);

        // The control plane is a versioned service fetched by metadata.
        auto hololink = hololink::module::HololinkInterfaceV1::get_service(metadata);
        // start() opens the control-plane socket; without it, no device I/O works.
        if (hololink->start() != HOLOLINK_MODULE_OK) {
            throw std::runtime_error("HololinkInterface::start failed");
        }
        // The framework leaves device state alone unless the application asks;
        // reset() drives the board into a known state.
        if (hololink->reset() != HOLOLINK_MODULE_OK) {
            throw std::runtime_error("HololinkInterface::reset failed");
        }
        // configure() writes device registers, so it must run after reset().
        camera->configure(CAMERA_MODE);
        camera->set_digital_gain_reg(4);
        // Re-assert the mode so configure_converter() below reads freshly-updated
        // width_/height_/pixel_format_ state (mirrors the original HSDK4 compose()'s own
        // set_mode() call, made right before configure_converter() there too).
        camera->set_mode(CAMERA_MODE);

        // Build the graph.
        holoscan::Graph graph { "imx274_module_tutorial" };

        hololink::module::csi::CsiGeometry geometry;
        // Configuring the converter tells us how big each received frame is.
        camera->configure_converter(
            hololink::module::operators::CsiToBayerOp::make_geometry_adapter(geometry));
        const size_t frame_size = geometry.start_byte
            + geometry.bytes_per_line * geometry.pixel_height + geometry.trailing_bytes;

        const auto bayer_format = camera->bayer_format();
        const auto pixel_format = camera->pixel_format();

        holoscan::visualization::LayerStack layers;
        layers.add_image_layer("video", hx::rgba16_layer(camera->width(), camera->height()));

        // A 1920x1080 window titled "Holoviz" that renders each frame as it arrives.
        holoscan::CompileOptions compile_options;
        auto viz = hx::add_holoviz(graph, "holoviz", std::move(layers), hx::DisplayConfig {}, compile_options,
            cu_device_ordinal);

        auto receiver = graph.op<hololink::module::operators::RoceReceiverOp>(
            "receiver",
            metadata,
            [camera] { camera->start(); },
            [camera] { camera->stop(); });
        receiver.set_params(hololink::module::operators::RoceReceiverParams {
            .frame_context = cu_context,
            .frame_size = frame_size,
        });

        auto csi = graph.op<hololink::module::operators::CsiToBayerOp>("csi_to_bayer");
        csi.set_params(hololink::module::operators::CsiToBayerParams {
            .cuda_device_ordinal = cu_device_ordinal,
            .geometry = geometry,
            .bayer_format = bayer_format,
        });

        auto ip = graph.op<hololink::module::operators::ImageProcessorOp>("image_processor");
        ip.set_params(hololink::module::operators::ImageProcessorParams {
            .pixel_format = static_cast<int>(pixel_format),
            .bayer_format = static_cast<int>(bayer_format),
            // Optical black value for IMX274 is 50.
            .optical_black = 50,
            .cuda_device_ordinal = cu_device_ordinal,
            .pixel_width = geometry.pixel_width,
            .pixel_height = geometry.pixel_height,
        });

        const auto bayer_grid_pos = static_cast<holoscan::vision::BayerGridPosition>(bayer_format);
        auto demosaic = graph.op<holoscan::vision::BayerDemosaicOp>("bayer_demosaic");
        demosaic.set_params(holoscan::vision::BayerDemosaicOpParams {
            .bayer_grid_pos = bayer_grid_pos,
            .generate_alpha = true,
            .alpha_value = 65535U,
        });

        graph.add_flow(receiver->tensor, csi->input_tensor, hx::pipeline_edge());
        graph.add_flow(receiver->metadata, csi->input_metadata, hx::pipeline_edge());
        graph.add_flow(csi->output_tensor, ip->input_tensor, hx::pipeline_edge());
        graph.add_flow(csi->output_metadata, ip->input_metadata, hx::pipeline_edge());
        graph.add_flow(ip->output_tensor, demosaic->input, hx::pipeline_edge());
        graph.add_flow(demosaic->output, viz->image_layer_input(0), hx::display_edge());

        graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

        for (const auto& operator_path : { std::string_view("receiver"), std::string_view("csi_to_bayer"), std::string_view("image_processor") }) {
            compile_options.deployment.bind_tensor_output_device(
                holoscan::TensorOutputDevicePlacement {
                    .operator_path = operator_path,
                    .output_port = "tensor",
                    .device = holoscan::DeviceId { cu_device_ordinal },
                });
        }
        compile_options.deployment.bind_tensor_output_device(
            holoscan::TensorOutputDevicePlacement {
                .operator_path = "bayer_demosaic",
                .output_port = "output",
                .device = holoscan::DeviceId { cu_device_ordinal },
            });

        const auto plan = holoscan::compile(graph, std::move(compile_options));
        if (!plan.ok()) {
            hx::print_compile_failure(plan);
            hololink->stop();
            HOLOLINK_MODULE_CUDA_CHECK(cuDevicePrimaryCtxRelease(cu_device));
            return EXIT_FAILURE;
        }

        // Run until the window closes or Ctrl+C, then report how the run ended.
        auto session = holoscan::run_async(plan, signals);
        session.wait();
        const bool failed = hx::report_session(session, plan);

        // Stop streaming data and close up sockets.
        hololink->stop();
        HOLOLINK_MODULE_CUDA_CHECK(cuDevicePrimaryCtxRelease(cu_device));
        return failed ? EXIT_FAILURE : EXIT_SUCCESS;
    });
}
