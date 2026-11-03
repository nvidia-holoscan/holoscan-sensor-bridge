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

// VB1940 sub-frame live viewer over a single RoCE network, drawing each band as it lands:
//   RoceReceiverOp -> CsiToBayerOp (sub-frame) -> ImageProcessorOp -> BayerDemosaicOp ->
//     SubFrameVisualizerOp
//
// The sensor frame travels and is processed as horizontal bands of --sub-frame-rows rows.
// Each band is demosaiced into RGBA16 and drawn by SubFrameVisualizerOp at its row offset as
// soon as it is processed. In the (fullscreen) window, a frame is presented once its last band
// is drawn; with --exclusive-display the visualizer scans out directly to the display in shared
// continuous refresh mode, so each band reaches the display as soon as it is drawn.
// SubFrameVisualizerOp also schedules the sensor's next capture against the display's refresh
// (first pixel out) through the board's PTP pulse output.
//
// BayerDemosaicOp carries only the image; SubFrameVisualizerOp pairs it with the band's metadata
// (its row offset), which comes straight from ImageProcessorOp.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#include <cuda.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/managed_shutdown_signals.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/logger/logger.hpp>
#include <holoscan/time/realtime_clock.hpp>
#include <holoscan/vision/bayer_demosaic_op.hpp>

#include <capture_metadata_traits.hpp>
#include <hololink/common/cuda_helper.hpp>
#include <hololink/common/holoargs.hpp>
#include <hololink/common/tools.hpp>
#include <hololink/core/data_channel.hpp>
#include <hololink/core/enumerator.hpp>
#include <hololink/core/hololink.hpp>
#include <hololink/operators/csi_to_bayer/csi_to_bayer.hpp>
#include <hololink/operators/image_processor/image_processor.hpp>
#include <hololink/operators/roce_receiver/roce_receiver_op.hpp>
#include <hololink/operators/sub_frame_visualizer/sub_frame_visualizer.hpp>
#include <hololink/sensors/camera/vb1940/native_vb1940_sensor.hpp>

// SubFrameVisualizerOp draws the bands, not HolovizOp.
#define HOLOLINK_PLAYER_SUPPORT_NO_DISPLAY
#include "player_support.hpp"

namespace hx = hololink::examples;

static int run_player(int argc, char** argv, holoscan::ManagedShutdownSignals& signals)
{
    using namespace hololink::args;
    OptionsDescription options_description("Allowed options");

    // clang-format off
    options_description.add_options()
        ("camera-mode", value<int>()->default_value(static_cast<int>(hololink::sensors::vb1940_mode::VB1940_MODE_2560X1984_60FPS)), "VB1940 mode (default: 3)")
        ("fullscreen", bool_switch()->default_value(true), "Run in fullscreen mode")
        ("exclusive-display", bool_switch()->default_value(false), "Run in exclusive display mode")
        ("frame-limit", value<int64_t>()->default_value(0), "Exit after receiving this many frames")
        ("sub-frame-rows", value<uint32_t>()->default_value(0), "Number of rows per sub-frame (0 = full camera height). Camera height must be evenly divisible by the sub frame rows.")
        ("configuration", value<std::string>()->default_value(""), "Configuration file")
        ("hololink", value<std::string>()->default_value(hololink::env_hololink_ip(0, "192.168.0.2")), "IP address of Hololink board")
        ("ibv-name", value<std::string>()->default_value(""), "IBV device to use (default: the device on the board's network interface)")
        ("ibv-port", value<uint32_t>()->default_value(1), "Port number of IBV device, with --ibv-name (default: 1)")
        ("use-sensor", value<uint32_t>()->default_value(0), "Use the specific sensor (0 or 1, default=0)");
    // clang-format on

    auto variables_map = Parser().parse_command_line(argc, argv, options_description);
    int use_sensor = variables_map["use-sensor"].as<uint32_t>();
    if (use_sensor >= 2) {
        HOLOSCAN_LOG_ERROR("Invalid value use_sensor={}; must be 0 or 1.", use_sensor);
        return -1;
    }
    if (const auto& configuration = variables_map["configuration"].as<std::string>();
        !configuration.empty() && !hx::check_configuration(configuration)) {
        return -1;
    }
    const std::string ibv_name = variables_map["ibv-name"].as<std::string>();
    const int64_t frame_limit = variables_map["frame-limit"].as<int64_t>();

    try {
        // Get a handle to the GPU
        CudaCheck(cuInit(0));
        int cu_device_ordinal = 0;
        CUdevice cu_device;
        CudaCheck(cuDeviceGet(&cu_device, cu_device_ordinal));
        CUcontext cu_context;
        CudaCheck(cuDevicePrimaryCtxRetain(&cu_context, cu_device));

        // Get a handle to the Hololink device
        auto channel_metadata = hololink::Enumerator::find_channel(variables_map["hololink"].as<std::string>());
        hololink::DataChannel::use_sensor(channel_metadata, use_sensor);
        auto hololink_channel = std::make_shared<hololink::DataChannel>(channel_metadata);

        // Create a PTP synchronizer for single pulse, the delay is then set per frame in the SubFrameVisualizerOp
        auto ptp_synchronizer = hololink_channel->hololink()->ptp_pps_output(1);
        // Get a handle to the camera
        auto camera = std::make_shared<hololink::sensors::NativeVb1940Sensor>(*hololink_channel, ptp_synchronizer);

        // Convert camera_mode to proper enum
        auto camera_mode = static_cast<hololink::sensors::vb1940_mode::Mode>(variables_map["camera-mode"].as<int>());
        camera->set_mode(camera_mode);

        const auto width = static_cast<uint32_t>(camera->get_width());
        const auto height = static_cast<uint32_t>(camera->get_height());
        const auto bayer_format = camera->get_bayer_format();
        const auto pixel_format = camera->get_pixel_format();
        if (width == 0 || height == 0) {
            throw std::runtime_error(fmt::format("Unsupported --camera-mode {}", static_cast<int>(camera_mode)));
        }

        uint32_t sub_frame_rows = variables_map["sub-frame-rows"].as<uint32_t>();
        if (sub_frame_rows == 0) {
            sub_frame_rows = height;
        } else if (sub_frame_rows > height) {
            throw std::runtime_error(fmt::format("Sub frame rows of {} is greater than camera height of {}", sub_frame_rows, height));
        } else if (height % sub_frame_rows != 0) {
            throw std::runtime_error(fmt::format("Camera height of {} is not evenly divisible by sub frame rows of {}", height, sub_frame_rows));
        } else if (sub_frame_rows % 2 != 0) {
            // Each band is demosaiced on its own: it needs an even height, and starting on an
            // even row keeps the color filter phase.
            throw std::runtime_error(fmt::format("Sub frame rows of {} is odd; demosaicing needs an even number", sub_frame_rows));
        }
        const uint32_t slices_per_frame = height / sub_frame_rows;

        // Receiver pages, at most this many: a 248-row VB1940 band page is about 0.8 MB, a
        // full-frame page about 6.3 MB, so the ring takes at most about 52 MB or 76 MB.
        constexpr uint32_t kMaxPages = 66;
        // The slices run through the capture lane in step, so these queues seldom hold more than
        // one; hx::pipeline_edge() keeps them shallow, since every queue entry costs a 64 MiB output
        // pool slot in the receiver, CsiToBayerOp and ImageProcessorOp.
        const holoscan::ConnectionOptions sub_frame_edge = hx::pipeline_edge();
        const uint32_t sub_frame_queue_depth = sub_frame_edge.queue_depth.value();

        holoscan::Graph graph { "sub_frame_vb1940_player" };

        hololink::operators::CsiToBayerParams csi_params {
            .cuda_device_ordinal = cu_device_ordinal,
            .sub_frame_rows = sub_frame_rows,
            .bayer_format = bayer_format,
        };
        camera->configure_converter(csi_params.geometry);
        auto csi = graph.op<hololink::operators::CsiToBayerOp>("csi_to_bayer");
        csi.set_params(csi_params);

        // The receiver delivers one slice at a time (what CsiToBayerOp computes for its sub-frame
        // size); a frame takes one more chunk than it has slices when the CSI data carries a
        // header. The board writes the pages round-robin and reuses a page `pages` chunks after
        // it wrote it, whether or not the host is done with it: RoceReceiverOp copies a page out
        // on its CUDA stream some time after taking it, and nothing holds the board off meanwhile.
        // So the ring holds six frames of chunks (up to kMaxPages), and queue_size, which caps how
        // far behind the newest chunk a page can be when the receiver hands it out, leaves a
        // frame's worth of chunks before the board writes that page again. With queue_size equal
        // to pages, the oldest page handed out could already be being rewritten.
        const auto& geometry = csi_params.geometry;
        const size_t frame_size = static_cast<size_t>(geometry.bytes_per_line) * sub_frame_rows;
        const size_t csi_length = geometry.start_byte
            + static_cast<size_t>(geometry.bytes_per_line) * height + geometry.trailing_bytes;
        const auto chunks_per_frame = static_cast<uint32_t>((csi_length + frame_size - 1) / frame_size);
        const uint32_t receiver_pages = std::min(chunks_per_frame * 6U, kMaxPages);
        const uint32_t receiver_queue_size
            = receiver_pages > chunks_per_frame ? receiver_pages - chunks_per_frame : 1U;

        auto receiver = graph.op<hololink::operators::RoceReceiverOp>(
            "receiver", hololink_channel, [camera] { camera->start(); }, [camera] { camera->stop(); });
        receiver.set_params(hololink::operators::RoceReceiverParams {
            .frame_context = cu_context,
            .frame_size = frame_size,
            // Trim the frame to the size of the received data (when doing sub-frame processing, not
            // each received packet might have the same size when then camera CSI data has a header)
            .trim = true,
            .ibv_name = ibv_name,
            .ibv_port = variables_map["ibv-port"].as<uint32_t>(),
            .pages = receiver_pages,
            // One frame of chunks short of pages: the margin before the board reuses a page.
            .queue_size = receiver_queue_size,
            // Report "Ingress frame timeout" when no data comes, as HSDK4's receiver did here
            // (use_frame_ready_condition=false). frame_limit stays 0: frames are counted at the end
            // of the chain.
            .watch_ingress = true,
        });

        auto ip = graph.op<hololink::operators::ImageProcessorOp>("image_processor");
        ip.set_params(hololink::operators::ImageProcessorParams {
            .pixel_format = int(pixel_format),
            .bayer_format = int(bayer_format),
            // Optical black value for vb1940 is 8 for RAW10
            .optical_black = 8,
            .cuda_device_ordinal = cu_device_ordinal,
            .pixel_width = geometry.pixel_width,
            .pixel_height = geometry.pixel_height,
        });

        auto demosaic = graph.op<holoscan::vision::BayerDemosaicOp>("bayer_demosaic");
        demosaic.set_params(holoscan::vision::BayerDemosaicOpParams {
            .bayer_grid_pos = static_cast<holoscan::vision::BayerGridPosition>(bayer_format),
            .generate_alpha = true,
            .alpha_value = 65535U,
        });

        // The bands are drawn at their row offset into a camera-sized image, so the display
        // takes the camera's dimensions. As in HSDK4, the exclusive display presents in shared
        // continuous refresh mode (the visualizer falls back to the automatic mode, with a
        // warning, where the display has no shared presentable images).
        const bool exclusive_display = variables_map["exclusive-display"].as<bool>();
        auto sub_frame_visualizer = graph.op<hololink::operators::SubFrameVisualizerOp>(
            "sub_frame_visualizer",
            /*full_frame_height=*/height,
            /*fullscreen=*/variables_map["fullscreen"].as<bool>(),
            /*use_exclusive_display=*/exclusive_display,
            /*display_name=*/std::string {},
            /*display_width=*/width,
            /*display_height=*/height,
            /*display_framerate=*/59'950U,
            /*window_title=*/std::string { "Sub-Frame Visualizer" },
            /*ptp_synchronizer=*/std::static_pointer_cast<hololink::Hololink::PtpSynchronizer>(ptp_synchronizer),
            /*present_mode=*/exclusive_display ? holoviz::PresentMode::kSharedContinuousRefresh : holoviz::PresentMode::kAuto);

        graph.add_flow(receiver->tensor, csi->input_tensor, sub_frame_edge);
        graph.add_flow(receiver->metadata, csi->input_metadata, sub_frame_edge);
        graph.add_flow(csi->output_tensor, ip->input_tensor, sub_frame_edge);
        graph.add_flow(csi->output_metadata, ip->input_metadata, sub_frame_edge);
        graph.add_flow(ip->output_tensor, demosaic->input, sub_frame_edge);
        // The visualizer blocks on the swapchain at the display rate; when the camera outruns it,
        // shed whole bands rather than stall. At most four bands are buffered: each RGBA16 band
        // slot of the demosaic's pool is sized from ImageProcessorOp's output bound (4 x 64 MiB).
        // The metadata reaches the visualizer directly while its band waits in two queues (to and
        // from the demosaic), so its queue holds both of those; the visualizer pairs a band with
        // its metadata by exact capture time and drops a band whose partner was still shed.
        const uint32_t band_queue_depth = std::min(slices_per_frame, 4U);
        graph.add_flow(demosaic->output, sub_frame_visualizer->input_tensor,
            holoscan::ConnectionOptions { .queue_policy = holoscan::QueuePolicy::kDropOldest,
                .queue_depth = band_queue_depth });
        graph.add_flow(ip->output_metadata, sub_frame_visualizer->input_metadata,
            holoscan::ConnectionOptions { .queue_policy = holoscan::QueuePolicy::kDropOldest,
                .queue_depth = sub_frame_queue_depth + band_queue_depth });
        if (frame_limit > 0) {
            // A frame counts once a band of the next one leaves ImageProcessorOp, so a lost or
            // shed band never hides a frame.
            hx::add_frame_limit(graph, "frame_limit", ip->output_metadata,
                std::make_shared<hx::FrameLimitGroup>(frame_limit, 1U), /*count_frame_number_changes=*/true);
        }

        // The receiver runs on a lane of its own: its compute() blocks in get_next_frame() when an
        // activation finds no chunk queued (up to a frame period between the last chunk of one
        // frame and the first of the next), and on a shared lane that wait stalls CsiToBayerOp and
        // ImageProcessorOp, which then cannot keep up with the chunks (9 per frame at 248 rows)
        // and the receiver's queue to CsiToBayerOp overflows. A lane of its own also gives the
        // receiver a CUDA stream of its own (one per partition by default), so its copy out of the
        // board's page does not queue behind the band kernels. The band processing runs at slice
        // rate on another lane, and the visualizer's swapchain waits stay off both.
        graph.partition("receiver").add(receiver);
        graph.partition("capture").add(csi, ip);
        graph.auto_partition_remaining();

        graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

        holoscan::CompileOptions compile_options;
        for (const auto& [operator_path, output_port] :
            { std::pair<std::string_view, std::string_view> { "receiver", "tensor" },
                { "csi_to_bayer", "tensor" }, { "image_processor", "tensor" }, { "bayer_demosaic", "output" } }) {
            compile_options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement {
                .operator_path = operator_path,
                .output_port = output_port,
                .device = holoscan::DeviceId { cu_device_ordinal },
            });
        }

        const auto plan = holoscan::compile(graph, std::move(compile_options));
        if (!plan.ok()) {
            hx::print_compile_failure(plan);
            CudaCheck(cuDevicePrimaryCtxRelease(cu_device));
            return -1;
        }

        // Run it.
        std::shared_ptr<hololink::Hololink> hololink = hololink_channel->hololink();
        hololink->start();
        hololink->reset();
        // Wait for PTP synchronization
        hololink->ptp_synchronize();
        hololink->write_uint32(0x8, 0x0); // Keep the sensor RESET at low
        camera->setup_clock();
        // Release the sensor RESET to high
        hololink->write_uint32(0x8, use_sensor == 0 ? 0x1 : 0x2);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        camera->configure(camera_mode);

        auto session = holoscan::run_async(plan, signals);
        session.wait();
        const bool failed = hx::report_session(session, plan);
        hololink->stop();

        CudaCheck(cuDevicePrimaryCtxRelease(cu_device));
        if (failed) {
            return -1;
        }

    } catch (std::exception& e) {
        std::cout << e.what() << std::endl;
        return -1;
    }

    return 0;
}

int main(int argc, char** argv)
{
    // With managed signals, Ctrl+C stops the session cooperatively, so the camera and the board
    // are stopped on the way out.
    return hx::run_with_managed_signals(
        [&](holoscan::ManagedShutdownSignals& signals) { return run_player(argc, argv, signals); });
}
