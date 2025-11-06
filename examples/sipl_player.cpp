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

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/bayer_demosaic/bayer_demosaic.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

#include <hololink/common/holoargs.hpp>
#include <hololink/core/logging.hpp>
#include <hololink/operators/image_processor/image_processor.hpp>
#include <hololink/operators/packed_format_converter/packed_format_converter.hpp>
#include <hololink/operators/sipl_capture/sipl_capture.hpp>

class SIPLMetadataPrinterOp : public holoscan::Operator {
public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(SIPLMetadataPrinterOp)

    SIPLMetadataPrinterOp(const std::vector<hololink::operators::SIPLCaptureOp::CameraInfo>& camera_info)
        : camera_info_(camera_info)
    {
    }

    void setup(holoscan::OperatorSpec& spec) override
    {
        spec.input<holoscan::gxf::Entity>("input");
    }

    void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output, holoscan::ExecutionContext& context) override
    {
        auto entity = op_input.receive<holoscan::gxf::Entity>("input");
        if (entity) {
            for (const auto& camera : camera_info_) {
                // Print out the HSB metadata.
                HOLOSCAN_LOG_INFO("[{}] frame={}, ts={}.{:09}, meta_ts={}.{:09}, crc=0x{:08x}, bytes={}",
                    camera.output_name,
                    metadata()->get<int64_t>(camera.output_name + "_frame_number"),
                    metadata()->get<int64_t>(camera.output_name + "_timestamp_s"),
                    metadata()->get<int64_t>(camera.output_name + "_timestamp_ns"),
                    metadata()->get<int64_t>(camera.output_name + "_metadata_s"),
                    metadata()->get<int64_t>(camera.output_name + "_metadata_ns"),
                    metadata()->get<int64_t>(camera.output_name + "_crc"),
                    metadata()->get<int64_t>(camera.output_name + "_bytes_written"));

                // The SIPL image metadata is written to "{camera_name}_metadata"
                // Print out the start/end timestamps as an example.
                const auto& image_metadata = metadata()->get<nvsipl::INvSIPLClient::ImageMetaData>(camera.output_name + "_metadata");
                HOLOSCAN_LOG_INFO("    SIPL Metadata: start time={}, end time={}",
                    image_metadata.frameCaptureStartTSC,
                    image_metadata.frameCaptureTSC);

                // The ISP statistics, when using ISP, are written to "{camera_name}_isp_stats"
                // Print out the average values for the first LAC region as an example.
                if (metadata()->has_key(camera.output_name + "_isp_stats")) {
                    const auto& isp_stats = metadata()->get<nvsipl::IspStatsInfo>(camera.output_name + "_isp_stats");
                    const auto& lac_data = isp_stats.ispStatsData.lacStatsData[0].data[0];
                    HOLOSCAN_LOG_INFO("    LAC Averages (window 0): ({:.4f}, {:.4f}, {:.4f}, {:.4f})",
                        lac_data.average[0][0],
                        lac_data.average[0][1],
                        lac_data.average[0][2],
                        lac_data.average[0][3]);
                }
            }
        }
    }

private:
    const std::vector<hololink::operators::SIPLCaptureOp::CameraInfo>& camera_info_;
};

class Application : public holoscan::Application {
public:
    Application(
        const std::string& camera_config,
        const std::string& json_config,
        bool raw_output,
        bool headless,
        bool fullscreen,
        int frame_limit,
        bool print_metadata)
        : camera_config_(camera_config)
        , json_config_(json_config)
        , raw_output_(raw_output)
        , headless_(headless)
        , fullscreen_(fullscreen)
        , frame_limit_(frame_limit)
        , print_metadata_(print_metadata)
    {
        enable_metadata(print_metadata);

        if (print_metadata && raw_output) {
            // Since the application flow will split into two for the sake of image
            // conversion/ISP for multiple sensors, Holoviz will end up receiving duplicate
            // copies of the metadata from the two streams. We set the metadata policy to
            // reject these duplicates instead of generating an error.
            metadata_policy(holoscan::MetadataPolicy::kReject);
        }
    }

    void compose() override
    {
        using namespace holoscan;

        std::shared_ptr<Condition> condition;
        if (frame_limit_) {
            condition = make_condition<CountCondition>("count", frame_limit_);
        } else {
            condition = make_condition<BooleanCondition>("ok", true);
        }

        // SIPL capture operator captures either the RAW or ISP-Processed image via NvSIPL.
        auto sipl_capture = make_operator<hololink::operators::SIPLCaptureOp>(
            "sipl_capture",
            camera_config_,
            json_config_,
            raw_output_,
            condition);
        const auto& camera_info = sipl_capture->get_camera_info();

        if (print_metadata_) {
            // Print out the metadata provided by NvSIPL.
            auto sipl_metadata_printer = make_operator<SIPLMetadataPrinterOp>(
                "sipl_metadata_printer",
                camera_info);
            add_flow(sipl_capture, sipl_metadata_printer, { { "output", "input" } });
        }

        // Holoviz is used to render the image(s).
        const float view_width = 1.0f / camera_info.size();
        std::vector<ops::HolovizOp::InputSpec> specs;
        for (uint32_t i = 0; i < camera_info.size(); ++i) {
            ops::HolovizOp::InputSpec::View view;
            view.offset_x_ = view_width * i;
            view.offset_y_ = 0;
            view.width_ = view_width;
            view.height_ = 1;

            ops::HolovizOp::InputSpec spec(camera_info[i].output_name, ops::HolovizOp::InputType::COLOR);
            spec.views_.push_back(view);
            specs.push_back(spec);
        }

        auto visualizer = make_operator<holoscan::ops::HolovizOp>(
            "holoviz",
            Arg("tensors", specs),
            Arg("fullscreen", fullscreen_),
            Arg("headless", headless_));

        if (!raw_output_) {
            // When capturing ISP-processed images from SIPL, output directly
            // from SIPL to the vizualizer.
            add_flow(sipl_capture, visualizer, { { "output", "receivers" } });
        } else {
            // If capturing RAW, we need to do the following between the SIPL capture and Holoviz:
            //   1) Convert the packed RAW to 16-bit Bayer (PackedFormatConverterOp)
            //   2) Perform basic ISP operations (ImageProcessorOp)
            //   3) Demosaic the Bayer to RGB.
            // These operators only process one buffer each, so each camera needs separate instances.
            for (uint32_t camera_index = 0; camera_index < camera_info.size(); ++camera_index) {
                const auto& info = camera_info[camera_index];

                // Convert packed RAW to 16-bit Bayer.
                const int32_t storage_type_device_memory = 1;
                const size_t num_blocks = 2;
                size_t size = info.width * info.height * 2; // 16-bit Bayer
                auto converter_pool = make_resource<holoscan::BlockMemoryPool>(
                    fmt::format("converter_pool_{}", info.output_name),
                    storage_type_device_memory, size, num_blocks);

                auto packed_format_converter = make_operator<hololink::operators::PackedFormatConverterOp>(
                    fmt::format("packed_format_converter_{}", info.output_name),
                    Arg("allocator", converter_pool),
                    Arg("in_tensor_name", info.output_name),
                    Arg("out_tensor_name", info.output_name));
                packed_format_converter->configure(
                    info.offset, info.bytes_per_line, info.width, info.height, info.pixel_format, 0);

                // Perform basic ISP operations.
                auto image_processor = make_operator<hololink::operators::ImageProcessorOp>(
                    fmt::format("image_processor_{}", info.output_name),
                    Arg("optical_black", 0),
                    Arg("bayer_format", static_cast<int>(info.bayer_format)),
                    Arg("pixel_format", static_cast<int>(info.pixel_format)));

                // Bayer demosaic to RGBA buffer.
                size = info.width * info.height * 2 * 4; // 16-bit RGBA
                auto demosaic_pool = make_resource<holoscan::BlockMemoryPool>(
                    fmt::format("demosaic_pool_{}", info.output_name),
                    storage_type_device_memory, size, num_blocks);

                auto bayer_demosaic = make_operator<holoscan::ops::BayerDemosaicOp>(
                    fmt::format("bayer_demosaic_{}", info.output_name),
                    Arg("pool", demosaic_pool),
                    Arg("generate_alpha", true),
                    Arg("alpha_value", 65535),
                    Arg("bayer_grid_pos", static_cast<int>(info.bayer_format)),
                    Arg("in_tensor_name", info.output_name),
                    Arg("out_tensor_name", info.output_name));

                // Define the application flow.
                add_flow(sipl_capture, packed_format_converter, { { "output", "input" } });
                add_flow(packed_format_converter, image_processor, { { "output", "input" } });
                add_flow(image_processor, bayer_demosaic, { { "output", "receiver" } });
                add_flow(bayer_demosaic, visualizer, { { "transmitter", "receivers" } });
            }
        }
    }

private:
    std::string camera_config_;
    std::string json_config_;
    bool raw_output_;
    bool headless_;
    bool fullscreen_;
    int frame_limit_;
    bool print_metadata_;
};

int main(int argc, char** argv)
{
    using namespace hololink::args;
    OptionsDescription options_description("Allowed options");

    hololink::logging::hsb_log_level = hololink::logging::HSB_LOG_LEVEL_DEBUG;

    // clang-format off
    options_description.add_options()
        ("camera-config", value<std::string>()->default_value(""), "Camera configuration to use")
        ("json-config", value<std::string>()->default_value(""), "JSON configuration file to use")
        ("list-configs", bool_switch()->default_value(false), "List available camera configurations then exit")
        ("raw", bool_switch()->default_value(false), "Use RAW capture path (uses CUDA-based ISP)")
        ("headless", bool_switch()->default_value(false), "Run in headless mode")
        ("fullscreen", bool_switch()->default_value(false), "Run in fullscreen mode")
        ("frame-limit", value<int>()->default_value(0), "Exit after receiving this many frames")
        ("metadata", bool_switch()->default_value(false), "Print frame metadata")
        ;
    // clang-format on

    auto variables_map = Parser().parse_command_line(argc, argv, options_description);

    try {
        if (variables_map["list-configs"].as<bool>()) {
            hololink::operators::SIPLCaptureOp::list_available_configs(
                variables_map["json-config"].as<std::string>());
        } else {
            auto app = std::make_unique<Application>(
                variables_map["camera-config"].as<std::string>(),
                variables_map["json-config"].as<std::string>(),
                variables_map["raw"].as<bool>(),
                variables_map["headless"].as<bool>(),
                variables_map["fullscreen"].as<bool>(),
                variables_map["frame-limit"].as<int>(),
                variables_map["metadata"].as<bool>());
            app->run();
        }
        return 0;
    } catch (const std::exception& e) {
        HOLOSCAN_LOG_ERROR("Application failed: {}", e.what());
        return -1;
    }
}
