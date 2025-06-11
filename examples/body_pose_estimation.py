# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import holoscan as hs
import numpy as np
from cuda import cuda
from holoscan.core import Operator, OperatorSpec
from holoscan.gxf import Entity

import hololink as hololink_module


class FormatInferenceInputOp(Operator):
    """Operator to format input image for inference"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
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

        # Create output message
        out_message = Entity(context)
        out_message.add(hs.as_tensor(tensor), "preprocessed")
        op_output.emit(out_message, "out")


class PostprocessorOp(Operator):
    """Operator to post-process inference output:
    * Non-max suppression
    * Make boxes compatible with Holoviz

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Output tensor names
        self.outputs = [
            "boxes",
            "noses",
            "left_eyes",
            "right_eyes",
            "left_ears",
            "right_ears",
            "left_shoulders",
            "right_shoulders",
            "left_elbows",
            "right_elbows",
            "left_wrists",
            "right_wrists",
            "left_hips",
            "right_hips",
            "left_knees",
            "right_knees",
            "left_ankles",
            "right_ankles",
            "segments",
        ]

        # Indices for each keypoint as defined by YOLOv8 pose model
        self.NOSE = slice(5, 7)
        self.LEFT_EYE = slice(8, 10)
        self.RIGHT_EYE = slice(11, 13)
        self.LEFT_EAR = slice(14, 16)
        self.RIGHT_EAR = slice(17, 19)
        self.LEFT_SHOULDER = slice(20, 22)
        self.RIGHT_SHOULDER = slice(23, 25)
        self.LEFT_ELBOW = slice(26, 28)
        self.RIGHT_ELBOW = slice(29, 31)
        self.LEFT_WRIST = slice(32, 34)
        self.RIGHT_WRIST = slice(35, 37)
        self.LEFT_HIP = slice(38, 40)
        self.RIGHT_HIP = slice(41, 43)
        self.LEFT_KNEE = slice(44, 46)
        self.RIGHT_KNEE = slice(47, 49)
        self.LEFT_ANKLE = slice(50, 52)
        self.RIGHT_ANKLE = slice(53, 55)

    def setup(self, spec: OperatorSpec):
        """
        input: "in"    - Input tensors coming from output of inference model
        output: "out"  - The post-processed output after applying thresholding and non-max suppression.
                         Outputs are the boxes, keypoints, and segments.  See self.outputs for the list of outputs.
        params:
            iou_threshold:    Intersection over Union (IoU) threshold for non-max suppression (default: 0.5)
            score_threshold:  Score threshold for filtering out low scores (default: 0.5)
            image_dim:        Image dimensions for normalizing the boxes (default: None)

        Returns:
            None
        """
        spec.input("in")
        spec.output("out")
        spec.param("iou_threshold", 0.5)
        spec.param("score_threshold", 0.5)
        spec.param("image_dim", None)

    def get_keypoints(self, detection):
        # Keypoints to be returned including our own "neck" keypoint
        keypoints = {
            "nose": detection[self.NOSE],
            "left_eye": detection[self.LEFT_EYE],
            "right_eye": detection[self.RIGHT_EYE],
            "left_ear": detection[self.LEFT_EAR],
            "right_ear": detection[self.RIGHT_EAR],
            "neck": (detection[self.LEFT_SHOULDER] + detection[self.RIGHT_SHOULDER])
            / 2,
            "left_shoulder": detection[self.LEFT_SHOULDER],
            "right_shoulder": detection[self.RIGHT_SHOULDER],
            "left_elbow": detection[self.LEFT_ELBOW],
            "right_elbow": detection[self.RIGHT_ELBOW],
            "left_wrist": detection[self.LEFT_WRIST],
            "right_wrist": detection[self.RIGHT_WRIST],
            "left_hip": detection[self.LEFT_HIP],
            "right_hip": detection[self.RIGHT_HIP],
            "left_knee": detection[self.LEFT_KNEE],
            "right_knee": detection[self.RIGHT_KNEE],
            "left_ankle": detection[self.LEFT_ANKLE],
            "right_ankle": detection[self.RIGHT_ANKLE],
        }

        return keypoints

    def compute(self, op_input, op_output, context):
        # Get input message
        in_message = op_input.receive("in")

        # Convert input to cupy array
        results = cp.asarray(in_message.get("inference_output"))[0]

        # Filter out low scores
        results = results[:, results[4, :] > self.score_threshold]
        scores = results[4, :]

        # If no detections, return zeros for all outputs
        if results.shape[1] == 0:
            out_message = Entity(context)
            zeros = hs.as_tensor(np.zeros([1, 2, 2]).astype(np.float32))

            for output in self.outputs:
                out_message.add(zeros, output)
            op_output.emit(out_message, "out")
            return

        results = results.transpose([1, 0])

        segments = []
        for i, detection in enumerate(results):
            # fmt: off
            kp = self.get_keypoints(detection)
            # Every two points defines a segment
            segments.append([kp["nose"], kp["left_eye"],      # nose <-> left eye
                             kp["nose"], kp["right_eye"],     # nose <-> right eye
                             kp["left_eye"], kp["left_ear"],  # ...
                             kp["right_eye"], kp["right_ear"],
                             kp["left_shoulder"], kp["right_shoulder"],
                             kp["left_shoulder"], kp["left_elbow"],
                             kp["left_elbow"], kp["left_wrist"],
                             kp["right_shoulder"], kp["right_elbow"],
                             kp["right_elbow"], kp["right_wrist"],
                             kp["left_shoulder"], kp["left_hip"],
                             kp["left_hip"], kp["left_knee"],
                             kp["left_knee"], kp["left_ankle"],
                             kp["right_shoulder"], kp["right_hip"],
                             kp["right_hip"], kp["right_knee"],
                             kp["right_knee"], kp["right_ankle"],
                             kp["left_hip"], kp["right_hip"],
                             kp["left_ear"], kp["neck"],
                             kp["right_ear"], kp["neck"],
                             ])
            # fmt: on

        cx, cy, w, h = results[:, 0], results[:, 1], results[:, 2], results[:, 3]
        x1, x2 = cx - w / 2, cx + w / 2
        y1, y2 = cy - h / 2, cy + h / 2

        data = {
            "boxes": cp.asarray(np.stack([x1, y1, x2, y2], axis=-1)).transpose([1, 0]),
            "noses": results[:, self.NOSE],
            "left_eyes": results[:, self.LEFT_EYE],
            "right_eyes": results[:, self.RIGHT_EYE],
            "left_ears": results[:, self.LEFT_EAR],
            "right_ears": results[:, self.RIGHT_EAR],
            "left_shoulders": results[:, self.LEFT_SHOULDER],
            "right_shoulders": results[:, self.RIGHT_SHOULDER],
            "left_elbows": results[:, self.LEFT_ELBOW],
            "right_elbows": results[:, self.RIGHT_ELBOW],
            "left_wrists": results[:, self.LEFT_WRIST],
            "right_wrists": results[:, self.RIGHT_WRIST],
            "left_hips": results[:, self.LEFT_HIP],
            "right_hips": results[:, self.RIGHT_HIP],
            "left_knees": results[:, self.LEFT_KNEE],
            "right_knees": results[:, self.RIGHT_KNEE],
            "left_ankles": results[:, self.LEFT_ANKLE],
            "right_ankles": results[:, self.RIGHT_ANKLE],
            "segments": cp.asarray(segments),
        }
        scores = cp.asarray(scores)

        out = self.nms(data, scores)

        # Rearrange boxes to be compatible with Holoviz
        out["boxes"] = cp.reshape(out["boxes"][None], (1, -1, 2))

        # Create output message
        out_message = Entity(context)
        for output in self.outputs:
            out_message.add(hs.as_tensor(out[output] / self.image_dim), output)
        op_output.emit(out_message, "out")

    def nms(self, inputs, scores):
        """Non-max suppression (NMS)
        Performs non-maximum suppression on input boxes according to their intersection-over-union (IoU).
        Filter out detections where the IoU is >= self.iou_threshold

        Parameters
        ----------
        inputs : dictionary containing boxes, keypoints, and segments
        scores : array (n,)

        Returns
        ----------
        outputs : dictionary containing remaining boxes, keypoints, and segments after non-max supprerssion

        """

        boxes = inputs["boxes"]
        segments = inputs["segments"]

        if len(boxes) == 0:
            return cp.asarray([]), cp.asarray([])

        # Get coordinates
        x0, y0, x1, y1 = boxes[0, :], boxes[1, :], boxes[2, :], boxes[3, :]

        # Area of bounding boxes
        area = (x1 - x0 + 1) * (y1 - y0 + 1)

        # Get indices of sorted scores
        indices = cp.argsort(scores)

        # Output boxes and scores
        boxes_out, segments_out, scores_out = [], [], []

        selected_indices = []

        # Iterate over bounding boxes
        while len(indices) > 0:
            # Get index with highest score from remaining indices
            index = indices[-1]
            selected_indices.append(index)
            # Pick bounding box with highest score
            boxes_out.append(boxes[:, index])
            segments_out.extend(segments[index])
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

        selected_indices = cp.asarray(selected_indices)

        outputs = {
            "boxes": cp.asarray(boxes_out),
            "segments": cp.asarray(segments_out),
            "noses": inputs["noses"][selected_indices],
            "left_eyes": inputs["left_eyes"][selected_indices],
            "right_eyes": inputs["right_eyes"][selected_indices],
            "left_ears": inputs["left_ears"][selected_indices],
            "right_ears": inputs["right_ears"][selected_indices],
            "left_shoulders": inputs["left_shoulders"][selected_indices],
            "right_shoulders": inputs["right_shoulders"][selected_indices],
            "left_elbows": inputs["left_elbows"][selected_indices],
            "right_elbows": inputs["right_elbows"][selected_indices],
            "left_wrists": inputs["left_wrists"][selected_indices],
            "right_wrists": inputs["right_wrists"][selected_indices],
            "left_hips": inputs["left_hips"][selected_indices],
            "right_hips": inputs["right_hips"][selected_indices],
            "left_knees": inputs["left_knees"][selected_indices],
            "right_knees": inputs["right_knees"][selected_indices],
            "left_ankles": inputs["left_ankles"][selected_indices],
            "right_ankles": inputs["right_ankles"][selected_indices],
        }

        return outputs


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
        logging.info("compose")
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
            num_blocks=2,
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

        image_shift = hololink_module.operators.ImageShiftToUint8Operator(
            self, name="image_shift", shift=8
        )

        rgb_components_per_pixel = 3
        bayer_pool = holoscan.resources.BlockMemoryPool(
            self,
            name="pool",
            # storage_type of 1 is device memory
            storage_type=1,
            block_size=self._camera._width
            * rgb_components_per_pixel
            * ctypes.sizeof(ctypes.c_uint16)
            * self._camera._height,
            num_blocks=2,
        )
        demosaic = holoscan.operators.BayerDemosaicOp(
            self,
            name="demosaic",
            pool=bayer_pool,
            generate_alpha=False,
            bayer_grid_pos=bayer_format.value,
            interpolation_mode=0,
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
                "yolo_pose": self._engine,
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
        self.add_flow(receiver_operator, csi_to_bayer_operator, {("output", "input")})
        self.add_flow(
            csi_to_bayer_operator, image_processor_operator, {("output", "input")}
        )
        self.add_flow(image_processor_operator, demosaic, {("output", "receiver")})
        self.add_flow(demosaic, image_shift, {("transmitter", "input")})
        self.add_flow(image_shift, visualizer, {("output", "receivers")})
        self.add_flow(image_shift, preprocessor, {("output", "")})
        self.add_flow(preprocessor, format_input)
        self.add_flow(format_input, inference, {("", "receivers")})
        self.add_flow(inference, postprocessor, {("transmitter", "in")})
        self.add_flow(postprocessor, visualizer, {("out", "receivers")})

        # Not using metadata
        self.enable_metadata(False)


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
        os.path.dirname(__file__), "body_pose_estimation.yaml"
    )
    parser.add_argument(
        "--configuration", default=default_configuration, help="Configuration file"
    )
    parser.add_argument(
        "--hololink",
        default="192.168.0.2",
        help="IP address of Hololink board",
    )
    default_engine = os.path.join(os.path.dirname(__file__), "yolov8n-pose.engine.fp32")
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
    # Get a handle to the data source
    channel_metadata = hololink_module.Enumerator.find_channel(channel_ip=args.hololink)
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
    hololink = hololink_channel.hololink()
    hololink.start()
    hololink.reset()
    camera.setup_clock()
    camera.configure(camera_mode)
    camera.set_digital_gain_reg(0x4)
    application.run()
    hololink.stop()

    (cu_result,) = cuda.cuDevicePrimaryCtxRelease(cu_device)
    assert cu_result == cuda.CUresult.CUDA_SUCCESS


if __name__ == "__main__":
    main()
