# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# See README.md for detailed information.

import argparse
import logging

import holoscan

import hololink as hololink_module


class HoloscanApplication(holoscan.core.Application):
    def __init__(
        self,
        camera_config,
        json_config,
        headless,
        fullscreen,
        frame_limit,
    ):
        logging.info("__init__")
        super().__init__()
        self._camera_config = camera_config
        self._json_config = json_config
        self._headless = headless
        self._fullscreen = fullscreen
        self._frame_limit = frame_limit

    def compose(self):
        if self._frame_limit:
            self._count = holoscan.conditions.CountCondition(
                self,
                name="count",
                count=self._frame_limit,
            )
            condition = self._count
        else:
            self._ok = holoscan.conditions.BooleanCondition(
                self, name="ok", enable_tick=True
            )
            condition = self._ok

        sipl_capture = hololink_module.operators.SIPLCaptureOp(
            self,
            condition,
            name="sipl_capture",
            camera_config=self._camera_config,
            json_config=self._json_config,
        )

        output_names = sipl_capture.get_output_names()
        view_width = 1.0 / len(output_names)
        specs = []
        output_index = 0
        for name in output_names:
            view = holoscan.operators.HolovizOp.InputSpec.View()
            view.offset_x = view_width * output_index
            view.offset_y = 0
            view.width = view_width
            view.height = 1
            output_index = output_index + 1

            spec = holoscan.operators.HolovizOp.InputSpec(
                name, holoscan.operators.HolovizOp.InputType.COLOR
            )
            spec.views = [view]
            specs = specs + [spec]

        visualizer = holoscan.operators.HolovizOp(
            self,
            name="holoviz",
            fullscreen=self._fullscreen,
            headless=self._headless,
            tensors=specs,
        )

        self.add_flow(sipl_capture, visualizer, {("output", "receivers")})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--list-configs", action="store_true", help="List available configs then exit"
    )
    parser.add_argument("--camera-config", default="", help="Camera config to use")
    parser.add_argument(
        "--json-config", default="", help="JSON configuration file to use"
    )
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument(
        "--fullscreen", action="store_true", help="Run in fullscreen mode"
    )
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=None,
        help="Exit after receiving this many frames",
    )
    parser.add_argument(
        "--log-level",
        type=int,
        default=20,
        help="Logging level to display",
    )
    args = parser.parse_args()
    hololink_module.logging_level(args.log_level)

    if args.list_configs:
        hololink_module.operators.SIPLCaptureOp.list_available_configs(args.json_config)
    else:
        application = HoloscanApplication(
            args.camera_config,
            args.json_config,
            args.headless,
            args.fullscreen,
            args.frame_limit,
        )
        application.run()


if __name__ == "__main__":
    main()
