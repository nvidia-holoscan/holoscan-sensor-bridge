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

import logging

import holoscan

import hololink as hololink_module


class HsbControllerOp(holoscan.core.Operator):
    def __init__(
        self,
        *args,
        sensor_factory=None,
        network_receiver=None,
        fallback_image=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._controller = hololink_module.hsb_controller.HsbController(
            sensor_factory, network_receiver
        )
        self._fallback_image = fallback_image
        self._fallback_csi_image = self._fallback_image.csi_image()
        self._out_tensor_name = ""
        #
        self._frame_ready_condition = holoscan.conditions.AsynchronousCondition(
            self.fragment, name="frame_ready_condition"
        )
        self._frame_ready_condition.event_state = (
            holoscan.conditions.AsynchronousEventState.WAIT
        )
        self.add_arg(self._frame_ready_condition)

    def setup(self, spec):
        logging.info("setup")
        spec.output("output")

    def start(self):
        self._controller.start(self)

    def stop(self):
        self._controller.stop()
        self._frame_ready_condition.event_state = (
            holoscan.conditions.AsynchronousEventState.EVENT_NEVER
        )

    def frame_ready(self):
        # The network receiver calls this guy.
        self._frame_ready_condition.event_state = (
            holoscan.conditions.AsynchronousEventState.EVENT_DONE
        )

    def compute(self, op_input, op_output, context):
        tensor = None
        if self._controller.connected():
            # This may time out if we lost connection to HSB.
            tensor = self._controller.get_next_frame()
        if tensor is None:
            logging.info(f"{self.name} Using fallback image.")
            tensor = self._fallback_csi_image
        op_output.emit({self._out_tensor_name: tensor}, "output")
        self._frame_ready_condition.event_state = (
            holoscan.conditions.AsynchronousEventState.EVENT_WAITING
        )

    def lost(self):
        # send the fallback image
        self.frame_ready()
