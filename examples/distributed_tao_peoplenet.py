# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import ctypes
import logging
import os

import cupy as cp
import holoscan
import numpy as np
from cuda import cuda

import hololink as hololink_module


class FormatInferenceInputOp(holoscan.core.Operator):
    """Operator to format input image for inference"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec):
        spec.input("in")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        # Get input message
        in_message = op_input.receive("in")

        # Transpose
        tensor = cp.asarray(in_message.get("preprocessed")).get()
        # OBS: Numpy conversion and moveaxis is needed to avoid strange
        # strides issue when doing inference
        tensor = np.moveaxis(tensor, 2, 0)[None]
        tensor = cp.asarray(tensor)

        # Output tensor
        op_output.emit({"preprocessed": tensor}, "out")


class PostprocessorOp(holoscan.core.Operator):
    """Operator to post-process inference output:
    * Reparameterize bounding boxes
    * Non-max suppression
    * Make boxes compatible with Holoviz

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec):
        spec.input("in")
        spec.output("out")
        spec.param("iou_threshold", 0.15)
        spec.param("score_threshold", 0.5)
        spec.param("image_width", None)
        spec.param("image_height", None)
        spec.param("box_scale", None)
        spec.param("box_offset", None)
        spec.param("grid_height", None)
        spec.param("grid_width", None)

    def compute(self, op_input, op_output, context):
        # Get input message
        in_message = op_input.receive("in")

        # Convert input to cupy array
        # ONNX models produce a singleton batch dimension that we skip.
        boxes = cp.asarray(in_message.get("boxes"))[0, ...]
        scores = cp.asarray(in_message.get("scores"))[0, ...]

        # PeopleNet has three classes:
        # 0. Person
        # 1. Bag
        # 2. Face
        # Here we only keep the Person and Face classes
        boxes = boxes[[0, 1, 2, 3, 8, 9, 10, 11], ...][None]
        scores = scores[[0, 2], ...][None]

        # Loop over label classes
        out = {"person": None, "faces": None}
        for i, label in enumerate(out):
            # Reparameterize boxes
            out[label], scores_nms = self.reparameterize_boxes(
                boxes[:, 0 + i * 4 : 4 + i * 4, ...],
                scores[:, i, ...][None],
            )

            # Non-max suppression
            out[label], _ = self.nms(out[label], scores_nms)

            # Reshape for HoloViz
            if len(out[label]) == 0:
                out[label] = np.zeros([1, 2, 2]).astype(np.float32)
            else:
                out[label][:, [0, 2]] /= self.image_width
                out[label][:, [1, 3]] /= self.image_height
                out[label] = cp.reshape(out[label][None], (1, -1, 2))
                out[label] = cp.asnumpy(out[label])

        # Output tensor
        op_output.emit({"person": out["person"], "faces": out["faces"]}, "out")

    def nms(self, boxes, scores):
        """Non-max suppression (NMS)

        Parameters
        ----------
        boxes : array (4, n)
        scores : array (n,)

        Returns
        ----------
        boxes : array (m, 4)
        scores : array (m,)

        """
        if len(boxes) == 0:
            return cp.asarray([]), cp.asarray([])

        # Get coordinates
        x0, y0, x1, y1 = boxes[0, :], boxes[1, :], boxes[2, :], boxes[3, :]

        # Area of bounding boxes
        area = (x1 - x0 + 1) * (y1 - y0 + 1)

        # Get indices of sorted scores
        indices = cp.argsort(scores)

        # Output boxes and scores
        boxes_out, scores_out = [], []

        # Iterate over bounding boxes
        while len(indices) > 0:
            # Get index with highest score from remaining indices
            index = indices[-1]

            # Pick bounding box with highest score
            boxes_out.append(boxes[:, index])
            scores_out.append(scores[index])

            # Get coordinates
            x00 = cp.maximum(x0[index], x0[indices[:-1]])
            x11 = cp.minimum(x1[index], x1[indices[:-1]])
            y00 = cp.maximum(y0[index], y0[indices[:-1]])
            y11 = cp.minimum(y1[index], y1[indices[:-1]])

            # Compute IOU
            width = cp.maximum(0, x11 - x00 + 1)
            height = cp.maximum(0, y11 - y00 + 1)
            overlap = width * height
            union = area[index] + area[indices[:-1]] - overlap
            iou = overlap / union

            # Threshold and prune
            left = cp.where(iou < self.iou_threshold)
            indices = indices[left]

        # To array
        boxes = cp.asarray(boxes_out)
        scores = cp.asarray(scores_out)

        return boxes, scores

    def reparameterize_boxes(self, boxes, scores):
        """Reparameterize boxes from corner+width+height to corner+corner.

        Parameters
        ----------
        boxes : array (1, 4, grid_height, grid_width)
        scores : array (1, 1, grid_height, grid_width)

        Returns
        ----------
        boxes : array (4, n)
        scores : array (n,)

        """
        cell_height = self.image_height / self.grid_height
        cell_width = self.image_width / self.grid_width

        # Generate the grid coordinates
        mx, my = cp.meshgrid(cp.arange(self.grid_width), cp.arange(self.grid_height))
        mx = mx.astype(np.float32).reshape((1, 1, self.grid_height, self.grid_width))
        my = my.astype(np.float32).reshape((1, 1, self.grid_height, self.grid_width))

        # Compute the box corners
        xmin = -(boxes[0, 0, ...] + self.box_offset) * self.box_scale + mx * cell_width
        ymin = -(boxes[0, 1, ...] + self.box_offset) * self.box_scale + my * cell_height
        xmax = (boxes[0, 2, ...] + self.box_offset) * self.box_scale + mx * cell_width
        ymax = (boxes[0, 3, ...] + self.box_offset) * self.box_scale + my * cell_height
        boxes = cp.concatenate([xmin, ymin, xmax, ymax], axis=1)

        # Select the scores that are above the threshold
        scores_mask = scores > self.score_threshold
        scores = scores[scores_mask]
        scores_mask = cp.repeat(scores_mask, 4, axis=1)
        boxes = boxes[scores_mask]

        # Reshape after masking
        n = int(boxes.size / 4)
        boxes = boxes.reshape(4, n)

        return boxes, scores


class SrcFragment(holoscan.core.Fragment):
    def __init__(
        self,
        app,
        name,
        headless,
        fullscreen,
        cuda_context,
        cuda_device_ordinal,
        hololink_channel,
        ibv_name,
        ibv_port,
        camera,
        camera_mode,
        frame_limit,
    ):
        logging.info("SrcFragment::__init__")
        super().__init__(app, name)
        self._headless = headless
        self._fullscreen = fullscreen
        self._cuda_context = cuda_context
        self._cuda_device_ordinal = cuda_device_ordinal
        self._hololink_channel = hololink_channel
        self._ibv_name = ibv_name
        self._ibv_port = ibv_port
        self._camera = camera
        self._camera_mode = camera_mode
        self._frame_limit = frame_limit

    def compose(self):
        logging.info("SrcFragment::compose")
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
        self._camera.set_mode(self._camera_mode)

        csi_to_bayer_pool = holoscan.resources.BlockMemoryPool(
            self,
            name="pool",
            # storage_type of 1 is device memory
            storage_type=1,
            block_size=self._camera._width
            * ctypes.sizeof(ctypes.c_uint16)
            * self._camera._height,
            num_blocks=3,
        )
        csi_to_bayer_operator = hololink_module.operators.CsiToBayerOp(
            self,
            name="csi_to_bayer",
            allocator=csi_to_bayer_pool,
            cuda_device_ordinal=self._cuda_device_ordinal,
        )
        self._camera.configure_converter(csi_to_bayer_operator)

        frame_size = csi_to_bayer_operator.get_csi_length()
        frame_context = self._cuda_context
        receiver_operator = hololink_module.operators.RoceReceiverOp(
            self,
            condition,
            name="receiver",
            frame_size=frame_size,
            frame_context=frame_context,
            ibv_name=self._ibv_name,
            ibv_port=self._ibv_port,
            hololink_channel=self._hololink_channel,
            device=self._camera,
        )

        bayer_format = self._camera.bayer_format()
        pixel_format = self._camera.pixel_format()
        image_processor_operator = hololink_module.operators.ImageProcessorOp(
            self,
            name="image_processor",
            # Optical black value for imx274 is 50
            optical_black=50,
            bayer_format=bayer_format.value,
            pixel_format=pixel_format.value,
        )

        rgb_components_per_pixel = 3
        # workaround for Holoscan SDK 2.0 regression, see distributed app UCX
        # transmitterissue described here: https://github.com/nvidia-holoscan/holoscan-sdk/releases/tag/v2.0.0
        if True:
            bayer_pool = holoscan.resources.UnboundedAllocator(self, name="pool")
        else:
            bayer_pool = holoscan.resources.BlockMemoryPool(
                self,
                name="pool",
                # storage_type of 1 is device memory
                storage_type=1,
                block_size=self._camera._width
                * rgb_components_per_pixel
                * ctypes.sizeof(ctypes.c_uint16)
                * self._camera._height,
                num_blocks=3,
            )
        demosaic = holoscan.operators.BayerDemosaicOp(
            self,
            name="demosaic",
            pool=bayer_pool,
            generate_alpha=False,
            bayer_grid_pos=bayer_format.value,
            interpolation_mode=0,
        )

        image_shift = hololink_module.operators.ImageShiftToUint8Operator(
            self, name="image_shift", shift=8
        )

        visualizer = holoscan.operators.HolovizOp(
            self,
            name="holoviz",
            fullscreen=self._fullscreen,
            headless=self._headless,
            framebuffer_srgb=True,
            **self.kwargs("holoviz"),
        )
        #
        self.add_flow(receiver_operator, csi_to_bayer_operator, {("output", "input")})
        self.add_flow(
            csi_to_bayer_operator, image_processor_operator, {("output", "input")}
        )
        self.add_flow(image_processor_operator, demosaic, {("output", "receiver")})
        self.add_flow(demosaic, image_shift, {("transmitter", "input")})
        self.add_flow(image_shift, visualizer, {("output", "receivers")})


class InferenceFragment(holoscan.core.Fragment):
    def __init__(
        self,
        app,
        name,
        engine,
    ):
        logging.info("InferenceFragment::__init__")
        super().__init__(app, name)
        self._engine = engine

    def compose(self):
        logging.info("InferenceFragment::compose")
        #
        pool = holoscan.resources.UnboundedAllocator(self)
        preprocessor_args = self.kwargs("preprocessor")
        preprocessor = holoscan.operators.FormatConverterOp(
            self,
            name="preprocessor",
            pool=pool,
            **preprocessor_args,
        )
        format_input = FormatInferenceInputOp(
            self,
            name="transpose",
            pool=pool,
        )
        inference = holoscan.operators.InferenceOp(
            self,
            name="inference",
            allocator=pool,
            model_path_map={
                "face_detect": self._engine,
            },
            **self.kwargs("inference"),
        )
        postprocessor_args = self.kwargs("postprocessor")
        postprocessor_args["image_width"] = preprocessor_args["resize_width"]
        postprocessor_args["image_height"] = preprocessor_args["resize_height"]
        postprocessor = PostprocessorOp(
            self,
            name="postprocessor",
            allocator=pool,
            **postprocessor_args,
        )

        #
        self.add_flow(preprocessor, format_input)
        self.add_flow(format_input, inference, {("", "receivers")})
        self.add_flow(inference, postprocessor, {("transmitter", "in")})


class HoloscanApplication(holoscan.core.Application):
    def __init__(
        self,
        headless,
        fullscreen,
        cuda_context,
        cuda_device_ordinal,
        hololink_channel,
        ibv_name,
        ibv_port,
        camera,
        camera_mode,
        frame_limit,
        engine,
    ):
        logging.info("__init__")
        super().__init__()
        self._headless = headless
        self._fullscreen = fullscreen
        self._cuda_context = cuda_context
        self._cuda_device_ordinal = cuda_device_ordinal
        self._hololink_channel = hololink_channel
        self._ibv_name = ibv_name
        self._ibv_port = ibv_port
        self._camera = camera
        self._camera_mode = camera_mode
        self._frame_limit = frame_limit
        self._engine = engine

    def compose(self):
        logging.info("HoloscanApplication::compose")
        src_fragment = SrcFragment(
            self,
            name="src_fragment",
            headless=self._headless,
            fullscreen=self._fullscreen,
            cuda_context=self._cuda_context,
            cuda_device_ordinal=self._cuda_device_ordinal,
            hololink_channel=self._hololink_channel,
            ibv_name=self._ibv_name,
            ibv_port=self._ibv_port,
            camera=self._camera,
            camera_mode=self._camera_mode,
            frame_limit=self._frame_limit,
        )
        inference_fragment = InferenceFragment(
            self,
            name="inference_fragment",
            engine=self._engine,
        )
        self.add_flow(
            src_fragment,
            inference_fragment,
            {("image_shift.output", "preprocessor.source_video")},
        )
        self.add_flow(
            inference_fragment,
            src_fragment,
            {("postprocessor.out", "holoviz.receivers")},
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--camera-mode",
        type=int,
        default=hololink_module.sensors.imx274.imx274_mode.Imx274_Mode.IMX274_MODE_3840X2160_60FPS.value,
        help="IMX274 mode",
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
    default_configuration = os.path.join(
        os.path.dirname(__file__), "tao_peoplenet.yaml"
    )
    parser.add_argument(
        "--configuration", default=default_configuration, help="Configuration file"
    )
    parser.add_argument(
        "--hololink",
        default="192.168.0.2",
        help="IP address of Hololink board",
    )
    default_engine = os.path.join(
        os.path.dirname(__file__), "resnet34_peoplenet_int8.onnx"
    )
    parser.add_argument(
        "--engine",
        default=default_engine,
        help="TRT engine model",
    )
    parser.add_argument(
        "--log-level",
        type=int,
        default=20,
        help="Logging level to display",
    )
    infiniband_devices = hololink_module.infiniband_devices()
    parser.add_argument(
        "--ibv-name",
        default=infiniband_devices[0],
        help="IBV device to use",
    )
    parser.add_argument(
        "--ibv-port",
        type=int,
        default=1,
        help="Port number of IBV device",
    )
    parser.add_argument(
        "--expander-configuration",
        type=int,
        default=0,
        choices=(0, 1),
        help="I2C Expander configuration",
    )
    parser.add_argument(
        "--address",
        dest="address",
        help="address ('[<IP or hostname>][:<port>]') of the App Driver. If not specified, "
        "the App Driver uses the default host address ('0.0.0.0') with the default port "
        "number ('8765').",
    )
    parser.add_argument(
        "--driver",
        dest="driver",
        action="store_true",
        default=False,
        help="run the App Driver on the current machine. Can be used together with the "
        "'--worker' option "
        "to run both the App Driver and the App Worker on the same machine.",
    )
    parser.add_argument(
        "-f",
        "--fragments",
        dest="fragments",
        help="comma-separated names of the fragments to be executed by the App Worker. "
        "If not specified, only one fragment (selected by the App Driver) will be executed. "
        "'all' can be used to run all the fragments.",
    )
    parser.add_argument(
        "--worker",
        dest="worker",
        action="store_true",
        default=False,
        help="run the App Worker.",
    )
    parser.add_argument(
        "--worker-address",
        dest="worker_address",
        help="address (`[<IP or hostname>][:<port>]`) of the App Worker. If not specified, the App "
        "Worker uses the default host address ('0.0.0.0') with the default port number "
        "randomly chosen from unused ports (between 10000 and 32767).",
    )
    args = parser.parse_args()
    hololink_module.logging_level(args.log_level)
    logging.info("Initializing.")
    # Get a handle to the GPU
    (cu_result,) = cuda.cuInit(0)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_device_ordinal = 0
    cu_result, cu_device = cuda.cuDeviceGet(cu_device_ordinal)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS
    cu_result, cu_context = cuda.cuDevicePrimaryCtxRetain(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS

    is_running_src_fragment = False
    if args.fragments is None:
        is_running_src_fragment = True
    else:
        running_fragments = args.fragments.split(",")
        for fragment in running_fragments:
            if "src_fragment" in fragment or "all" in fragment:
                is_running_src_fragment = True
                break
    hololink = None
    hololink_channel = None
    camera = None
    camera_mode = None

    if is_running_src_fragment:
        # Get a handle to the data source
        channel_metadata = hololink_module.Enumerator.find_channel(
            channel_ip=args.hololink
        )
        hololink_channel = hololink_module.DataChannel(channel_metadata)
        # Get a handle to the camera
        camera = hololink_module.sensors.imx274.dual_imx274.Imx274Cam(
            hololink_channel, expander_configuration=args.expander_configuration
        )
        camera_mode = hololink_module.sensors.imx274.imx274_mode.Imx274_Mode(
            args.camera_mode
        )
    # Set up the application
    application = HoloscanApplication(
        args.headless,
        args.fullscreen,
        cu_context,
        cu_device_ordinal,
        hololink_channel,
        args.ibv_name,
        args.ibv_port,
        camera,
        camera_mode,
        args.frame_limit,
        args.engine,
    )
    application.config(args.configuration)
    # Run it.
    if is_running_src_fragment:
        hololink = hololink_channel.hololink()
        hololink.start()
        hololink.reset()
        camera.setup_clock()
        camera.configure(camera_mode)
        camera.set_digital_gain_reg(0x4)

    # If the app exits early, try uncommenting below code and using different
    # values for scheduler parameters.
    # scheduler = holoscan.schedulers.MultiThreadScheduler(
    #    application,
    #    worker_thread_number=1,
    #    stop_on_deadlock=True,
    #    stop_on_deadlock_timeout=500,
    #    check_recession_period_ms=5.0,
    #    name="multithread_scheduler",
    # )
    # application.scheduler(scheduler)

    application.run()

    if is_running_src_fragment:
        hololink.stop()

    (cu_result,) = cuda.cuDevicePrimaryCtxRelease(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS


if __name__ == "__main__":
    main()
