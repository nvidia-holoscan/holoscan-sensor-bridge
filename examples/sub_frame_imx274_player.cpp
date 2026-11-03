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

// Single IMX274 sub-frame live viewer over a single RoCE network:
//   RoceReceiverOp -> CsiToBayerOp (sub-frame) -> ImageProcessorOp -> SubFrameCombinerOp ->
//     BayerDemosaicOp -> HolovizOp
//
// The sensor frame travels and is processed as horizontal bands of --sub-frame-rows rows: the
// receiver delivers one band at a time, and CsiToBayerOp and ImageProcessorOp each run on a band
// as soon as it lands. SubFrameCombinerOp reassembles the bands into a full frame, which is
// demosaiced once into RGBA16 and displayed. Combining before demosaicing (the HSDK4 version of
// this player demosaiced each band) avoids seams at the band boundaries. To display each band as
// it arrives instead, see linux_sub_frame_imx274_hsdk5_player --sub-frame-display. This is the
// RoCE counterpart of linux_sub_frame_imx274_player.

#include <getopt.h>

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
#include <vector>

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
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/time/realtime_clock.hpp>
#include <holoscan/vision/bayer_demosaic_op.hpp>
#include <holoscan/visualization/holoviz_op.hpp>
#include <holoscan/visualization/sink_desc.hpp>

#include <capture_metadata_traits.hpp>
#include <hololink/common/cuda_helper.hpp>
#include <hololink/common/tools.hpp>
#include <hololink/core/data_channel.hpp>
#include <hololink/core/enumerator.hpp>
#include <hololink/core/hololink.hpp>
#include <hololink/operators/csi_to_bayer/csi_to_bayer.hpp>
#include <hololink/operators/image_processor/image_processor.hpp>
#include <hololink/operators/roce_receiver/roce_receiver_op.hpp>
#include <hololink/operators/sub_frame_combiner/sub_frame_combiner.hpp>
#include <hololink/sensors/camera/imx274/native_imx274_sensor.hpp>

#include "player_support.hpp"

namespace hx = hololink::examples;

static int run_player(int argc, char** argv, holoscan::ManagedShutdownSignals& signals)
{
    int32_t camera_mode = 0;
    bool headless = false;
    bool fullscreen = false;
    int64_t frame_limit = 0;
    // the camera height must be evenly divisible by the sub frame rows, by choosing a default of 216,
    // this is the case for the modes supported by the IMX274 sensor.
    uint32_t sub_frame_rows = 216;
    std::string hololink_ip = hololink::env_hololink_ip(0, "192.168.0.2");
    holoscan::LogLevel log_level = holoscan::LogLevel::INFO;
    int32_t expander_configuration = 0;
    int32_t pattern = 0;
    bool pattern_set = false;
    uint32_t ibv_port = 1;
    std::string ibv_name; // empty: the receiver uses the device on the board's network interface

    // Parse args
    const struct option long_options[] = { { "help", no_argument, nullptr, 'h' },
        { "camera-mode", required_argument, nullptr, 0 }, { "headless", no_argument, nullptr, 0 },
        { "fullscreen", no_argument, nullptr, 0 }, { "frame-limit", required_argument, nullptr, 0 },
        { "sub-frame-rows", required_argument, nullptr, 0 },
        { "configuration", required_argument, nullptr, 0 },
        { "hololink", required_argument, nullptr, 0 },
        { "ibv-name", required_argument, nullptr, 0 },
        { "ibv-port", required_argument, nullptr, 0 },
        { "expander-configuration", required_argument, nullptr, 0 },
        { "pattern", required_argument, nullptr, 0 },
        { "log-level", required_argument, nullptr, 0 }, { 0, 0, nullptr, 0 } };
    while (true) {
        int option_index = 0;
        const int c = getopt_long(argc, argv, "h", long_options, &option_index);

        if (c == -1) {
            break;
        }

        const std::string argument(optarg ? optarg : "");
        if (c == 0) {
            const struct option* cur_option = &long_options[option_index];
            if (cur_option->name == std::string("camera-mode")) {
                camera_mode = std::stoi(argument);
            } else if (cur_option->name == std::string("headless")) {
                headless = true;
            } else if (cur_option->name == std::string("fullscreen")) {
                fullscreen = true;
            } else if (cur_option->name == std::string("frame-limit")) {
                frame_limit = std::stoll(argument);
            } else if (cur_option->name == std::string("sub-frame-rows")) {
                sub_frame_rows = std::stoul(argument);
            } else if (cur_option->name == std::string("configuration")) {
                if (!hx::check_configuration(argument)) {
                    return -1;
                }
            } else if (cur_option->name == std::string("hololink")) {
                hololink_ip = argument;
            } else if (cur_option->name == std::string("log-level")) {
                const auto level = hx::parse_log_level(argument);
                if (!level) {
                    throw std::runtime_error(fmt::format("Unhandled log level \"{}\"", argument));
                }
                log_level = *level;
            } else if (cur_option->name == std::string("ibv-name")) {
                ibv_name = argument;
            } else if (cur_option->name == std::string("ibv-port")) {
                ibv_port = std::stoul(argument);
            } else if (cur_option->name == std::string("expander-configuration")) {
                expander_configuration = std::stoul(argument);
            } else if (cur_option->name == std::string("pattern")) {
                pattern = std::stoi(argument);
                pattern_set = true;
            } else {
                throw std::runtime_error(fmt::format("Unhandled option \"{}\"", cur_option->name));
            }
        } else {
            switch (c) {
            case 'h':
                std::cout << "Usage: " << argv[0] << " [options]\n"
                          << "Options:\n"
                          << "  -h, --help                    display this information\n"
                          << "  --hololink <ip>               IP address of Hololink board (default "
                          << hololink_ip << ")\n"
                          << "  --ibv-name <name>             IBV device to use (default: the device on the\n"
                          << "                                board's network interface)\n"
                          << "  --ibv-port <n>                Port number of the IBV device, with --ibv-name\n"
                          << "                                (default " << ibv_port << ")\n"
                          << "  --camera-mode <0|1|2>         IMX274 mode (default 0 = 3840x2160@60)\n"
                          << "  --sub-frame-rows <n>          Rows per sub-frame (default 216; must divide\n"
                          << "                                the camera height evenly; 0 = whole frames)\n"
                          << "  --frame-limit <n>             Stop after N frames (0 = run until the window\n"
                          << "                                closes)\n"
                          << "  --fullscreen                  Run in fullscreen mode\n"
                          << "  --headless                    Render without a window\n"
                          << "  --expander-configuration <n>  I2C expander configuration (default 0)\n"
                          << "  --pattern <n>                 Show sensor test pattern N\n"
                          << "  --log-level <level>           trace|debug|info|warn|error|critical|off\n"
                          << "  --configuration <file>        Configuration file (ignored)\n";
                return EXIT_SUCCESS;

            default:
                throw std::runtime_error("Unhandled option ");
            }
        }
    }

    try {
        holoscan::set_log_level(log_level);
        // As in HSDK4, --log-level also sets the camera driver's log level; the native driver
        // logs through hololink's own logger.
        hx::set_hsb_log_level(log_level);

        std::cout << "Initializing." << std::endl;

        const int cuda_device_ordinal = 0;
        CudaCheck(cuInit(0));
        CUdevice cu_device;
        CudaCheck(cuDeviceGet(&cu_device, cuda_device_ordinal));
        CUcontext cu_context;
        CudaCheck(cuDevicePrimaryCtxRetain(&cu_context, cu_device));
        cuCtxSetCurrent(cu_context);

        // Get a handle to the data source
        hololink::Metadata channel_metadata = hololink::Enumerator::find_channel(hololink_ip);
        auto hololink_channel = std::make_shared<hololink::DataChannel>(channel_metadata);

        auto camera = std::make_shared<hololink::sensors::NativeImx274Sensor>(
            *hololink_channel, expander_configuration);
        const auto imx274_mode = static_cast<hololink::sensors::imx274_mode::Mode>(camera_mode);
        camera->set_mode(imx274_mode);

        const auto width = static_cast<uint32_t>(camera->get_width());
        const auto height = static_cast<uint32_t>(camera->get_height());
        const auto bayer_format = camera->get_bayer_format();
        const auto pixel_format = camera->get_pixel_format();
        if (width == 0 || height == 0) {
            throw std::runtime_error(fmt::format("Unsupported --camera-mode {}", camera_mode));
        }

        if (sub_frame_rows == 0) {
            sub_frame_rows = height;
        } else if (sub_frame_rows > height) {
            throw std::runtime_error(fmt::format(
                "Sub frame rows of {} is greater than camera height of {}", sub_frame_rows, height));
        } else if (height % sub_frame_rows != 0) {
            throw std::runtime_error(fmt::format(
                "Camera height of {} is not evenly divisible by sub frame rows of {}", height, sub_frame_rows));
        } else if (sub_frame_rows % 2 != 0) {
            // ImageProcessorOp takes the color filter phase from each band's own first row, so
            // every band must start on an even image row.
            throw std::runtime_error(fmt::format("Sub frame rows of {} is odd; it must be even", sub_frame_rows));
        }

        // Receiver pages, at most this many: a 4K band page is about 1.25 MB, so 66 of them
        // take about 83 MB of GPU memory.
        constexpr uint32_t kMaxPages = 66;
        // The slices run through the capture lane in step, so the queues up to the combiner
        // seldom hold more than one; hx::pipeline_edge() keeps them shallow, since every queue
        // entry costs a 64 MiB output pool slot in the receiver, CsiToBayerOp and ImageProcessorOp.
        const holoscan::ConnectionOptions sub_frame_edge = hx::pipeline_edge();

        holoscan::Graph graph { "sub_frame_imx274_player" };

        hololink::operators::CsiToBayerParams csi_params {
            .cuda_device_ordinal = cuda_device_ordinal,
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
        // to pages, the oldest page handed out could already be being rewritten: a band of a later
        // frame then shows up in the assembled frame (a tear that drifts down the image).
        const auto& geometry = csi_params.geometry;
        const size_t frame_size = static_cast<size_t>(geometry.bytes_per_line) * sub_frame_rows;
        const size_t csi_length = geometry.start_byte
            + static_cast<size_t>(geometry.bytes_per_line) * height + geometry.trailing_bytes;
        const auto chunks_per_frame = static_cast<uint32_t>((csi_length + frame_size - 1) / frame_size);
        const uint32_t receiver_pages = std::min(chunks_per_frame * 6U, kMaxPages);
        const uint32_t receiver_queue_size
            = receiver_pages > chunks_per_frame ? receiver_pages - chunks_per_frame : 1U;

        auto receiver = graph.op<hololink::operators::RoceReceiverOp>(
            "receiver", hololink_channel, [camera] { camera->start(); },
            [camera] {
                camera->stop();
                // Let the egress buffer drain, as HSDK4's IMX274 driver did.
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            });
        receiver.set_params(hololink::operators::RoceReceiverParams {
            .frame_context = cu_context,
            .frame_size = frame_size,
            // Trim each slice to the bytes received: the last slice of a frame is shorter when
            // the CSI data carries a header.
            .trim = true,
            .ibv_name = ibv_name,
            .ibv_port = ibv_port,
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
            .optical_black = 50, // Optical black value for imx274 is 50
            .cuda_device_ordinal = cuda_device_ordinal,
            .pixel_width = geometry.pixel_width,
            .pixel_height = geometry.pixel_height,
        });

        auto combiner = graph.op<hololink::operators::SubFrameCombinerOp>("sub_frame_combiner");
        combiner.set_params(hololink::operators::SubFrameCombinerParams {
            .cuda_device_ordinal = cuda_device_ordinal,
            .height = height,
        });

        auto demosaic = graph.op<holoscan::vision::BayerDemosaicOp>("bayer_demosaic");
        demosaic.set_params(holoscan::vision::BayerDemosaicOpParams {
            .bayer_grid_pos = static_cast<holoscan::vision::BayerGridPosition>(bayer_format),
            .generate_alpha = true,
            .alpha_value = 65535U,
        });

        holoscan::CompileOptions compile_options;
        holoscan::visualization::LayerStack layers;
        layers.add_image_layer("video", hx::rgba16_layer(width, height));
        // Render one frame per assembled frame.
        auto viz = hx::add_holoviz(graph, "holoviz", std::move(layers),
            hx::DisplayConfig { .fullscreen = fullscreen, .headless = headless }, compile_options,
            cuda_device_ordinal);

        graph.add_flow(receiver->tensor, csi->input_tensor, sub_frame_edge);
        graph.add_flow(receiver->metadata, csi->input_metadata, sub_frame_edge);
        graph.add_flow(csi->output_tensor, ip->input_tensor, sub_frame_edge);
        graph.add_flow(csi->output_metadata, ip->input_metadata, sub_frame_edge);
        graph.add_flow(ip->output_tensor, combiner->input_tensor, sub_frame_edge);
        graph.add_flow(ip->output_metadata, combiner->input_metadata, sub_frame_edge);
        // The display runs on a lane of its own (below), at the display's rate. The combined frame
        // goes there through a one-frame queue that drops the oldest: the newest frame is shown,
        // and the capture lane never waits for the display.
        graph.add_flow(combiner->output_tensor, demosaic->input,
            holoscan::ConnectionOptions { .queue_policy = holoscan::QueuePolicy::kDropOldest, .queue_depth = 1U });
        hx::add_frame_limit(graph, "frame_limit", combiner->output_metadata,
            std::make_shared<hx::FrameLimitGroup>(frame_limit, 1U));
        graph.add_flow(demosaic->output, viz->image_layer_input(0), hx::display_edge());

        // HolovizOp blocks on the swapchain at the display refresh rate, and band processing can
        // hold up whatever shares its lane: either stall on the receiver's lane is long enough to
        // miss slices. The receiver runs on a lane of its own, which also gives it a CUDA stream
        // of its own (one per partition by default): its copy out of the board's page then no
        // longer queues behind the band kernels, which matters here because the board reuses the
        // page on its own schedule. Everything up to the combiner runs at slice rate on another
        // lane; the rest runs at frame rate.
        graph.partition("receiver").add(receiver);
        graph.partition("capture").add(csi, ip, combiner);
        graph.auto_partition_remaining();

        graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

        for (const auto& [operator_path, output_port] :
            { std::pair<std::string_view, std::string_view> { "receiver", "tensor" }, { "csi_to_bayer", "tensor" },
                { "image_processor", "tensor" }, { "sub_frame_combiner", "tensor" },
                { "bayer_demosaic", "output" } }) {
            compile_options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement {
                .operator_path = operator_path,
                .output_port = output_port,
                .device = holoscan::DeviceId { cuda_device_ordinal },
            });
        }

        const auto plan = holoscan::compile(graph, std::move(compile_options));
        if (!plan.ok()) {
            hx::print_compile_failure(plan);
            CudaCheck(cuDevicePrimaryCtxRelease(cu_device));
            return -1;
        }

        auto hololink_dev = hololink_channel->hololink();
        hololink_dev->start();
        hololink_dev->reset();
        camera->setup_clock();
        camera->configure(imx274_mode);
        camera->set_digital_gain_reg(0x4);
        if (pattern_set) {
            // Any --pattern, 0 included, selects that test pattern.
            camera->enable_test_pattern(static_cast<uint8_t>(pattern));
        }

        std::cout << "Calling run" << std::endl;
        auto session = holoscan::run_async(plan, signals);
        session.wait();
        const bool failed = hx::report_session(session, plan);

        hololink_dev->stop();
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
