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
#include <holoscan/operators/holoviz/holoviz.hpp>

#include <hololink/common/holoargs.hpp>
#include <hololink/core/logging.hpp>
#include <hololink/operators/sipl_capture/sipl_capture.hpp>

class Application : public holoscan::Application {
public:
    Application(
        const std::string& camera_config,
        const std::string& json_config,
        bool headless,
        bool fullscreen,
        int frame_limit)
        : camera_config_(camera_config)
        , json_config_(json_config)
        , headless_(headless)
        , fullscreen_(fullscreen)
        , frame_limit_(frame_limit)
    {
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

        auto sipl_capture = make_operator<hololink::operators::SIPLCaptureOp>(
            "sipl_capture",
            camera_config_,
            json_config_,
            condition);

        const auto& output_names = sipl_capture->get_output_names();
        const float view_width = 1.0f / output_names.size();
        std::vector<ops::HolovizOp::InputSpec> specs;
        for (uint32_t i = 0; i < output_names.size(); ++i) {
            ops::HolovizOp::InputSpec::View view;
            view.offset_x_ = view_width * i;
            view.offset_y_ = 0;
            view.width_ = view_width;
            view.height_ = 1;

            ops::HolovizOp::InputSpec spec(output_names[i], ops::HolovizOp::InputType::COLOR);
            spec.views_.push_back(view);
            specs.push_back(spec);
        }
        auto visualizer = make_operator<holoscan::ops::HolovizOp>(
            "holoviz",
            Arg("tensors", specs),
            Arg("fullscreen", fullscreen_),
            Arg("headless", headless_));

        add_flow(sipl_capture, visualizer, { { "output", "receivers" } });
    }

private:
    std::string camera_config_;
    std::string json_config_;
    bool headless_;
    bool fullscreen_;
    int frame_limit_;
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
        ("headless", bool_switch()->default_value(false), "Run in headless mode")
        ("fullscreen", bool_switch()->default_value(false), "Run in fullscreen mode")
        ("frame-limit", value<int>()->default_value(0), "Exit after receiving this many frames")
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
                variables_map["headless"].as<bool>(),
                variables_map["fullscreen"].as<bool>(),
                variables_map["frame-limit"].as<int>());
            app->run();
        }
        return 0;
    } catch (const std::exception& e) {
        HOLOSCAN_LOG_ERROR("Application failed: {}", e.what());
        return -1;
    }
}
