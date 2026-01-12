from collections import namedtuple
from enum import Enum
from enum import IntEnum


import hololink as hololink_module


class RealSense_StreamId(IntEnum):
    RGB = 0
    DEPTH = 2

class RealSense_StreamCommand(IntEnum):
    START_STREAM = 1
    STOP_STREAM = 2
    SET_PROFILE = 3

# Define Enum
class RealSense_RGB_Mode(Enum):
    RGB_896x504_30FPS = 0 # 17 in single stream
    RGB_896x504_15FPS = 1 # 18
    RGB_896x504_5FPS  = 2 # 19
    RGB_896x504_60FPS = 3 # 20
    RGB_1280x800_30FPS = 4 # 21
    RGB_1280x800_15FPS = 5 # 22
    RGB_1280x720_30FPS = 6 # 23
    RGB_1280x720_15FPS = 7 # 24
    RGB_1280x720_5FPS = 8 # 25
    RGB_640x360_60FPS = 9 # 26
    RGB_640x360_30FPS = 10 # 27
    RGB_640x360_15FPS = 11 # 28
    RGB_640x360_5FPS = 12 # 29
    RGB_448x252_60FPS = 13 # 30
    RGB_448x252_30FPS = 14 # 31
    RGB_448x252_15FPS = 15 # 32
    RGB_448x252_5FPS = 16 # 33

class RealSense_Depth_Mode(Enum):
    DEPTH_896x504_30FPS = 0
    DEPTH_896x504_15FPS = 1
    DEPTH_896x504_5FPS  = 2
    DEPTH_896x504_60FPS = 3
    DEPTH_1280x720_30FPS = 4
    DEPTH_1280x720_15FPS = 5
    DEPTH_1280x720_5FPS = 6
    DEPTH_640x360_60FPS = 7 
    DEPTH_640x360_30FPS = 8
    DEPTH_640x360_15FPS = 9
    DEPTH_640x360_5FPS = 10
    DEPTH_448x252_60FPS = 11
    DEPTH_448x252_30FPS = 12
    DEPTH_448x252_15FPS = 13
    DEPTH_448x252_5FPS = 14
    DEPTH_1280x800_15FPS = 15
    DEPTH_256x144_90FPS = 16


PROFILE_COUNT = 17  # Total number of profiles for RealSense D555 camera per stream

# Merge safely at class creation time
def create_combined_enum(name, *enums):
    combined = {}
    i = 0
    for enum_cls in enums:
        for member in enum_cls:
            combined[member.name] = i
            i += 1
    return Enum(name, combined)

RealSense_Mode = create_combined_enum("RealSense_Mode", RealSense_Depth_Mode, RealSense_RGB_Mode)

# Define namedtuple
stream_info = namedtuple(
    "stream_profile",
    ["width", "height", "framerate", "pixel_format"]
)

# Mapping
depth_stream_profiles = []

depth_profiles = [
    (896, 504, 30), (896, 504, 15), (896, 504, 5), (896, 504, 60),
    (1280, 720, 30), (1280, 720, 15), (1280, 720, 5),
    (640, 360, 60), (640, 360, 30), (640, 360, 15), (640, 360, 5),
    (448, 252, 60), (448, 252, 30), (448, 252, 15), (448, 252, 5),
    (1280, 800, 15), (256, 144, 90),
]

for i, (w, h, fps) in enumerate(depth_profiles):
    depth_stream_profiles.insert(
        i,
        stream_info(
            w, h, fps,
            hololink_module.operators.ImageDecoderOp.PixelFormat.Z16
        )
    )

# RGB profiles has same depth profiles but with different pixel format

rgb_stream_profiles = []

rgb_profiles = [
    (896, 504, 30), (896, 504, 15), (896, 504, 5), (896, 504, 60),
    (1280, 800, 30), (1280, 800, 15), 
    (1280, 720, 30), (1280, 720, 15), (1280, 720, 5),
    (640, 360, 60), (640, 360, 30), (640, 360, 15), (640, 360, 5),
    (448, 252, 60), (448, 252, 30), (448, 252, 15), (448, 252, 5),
]


for i, (w, h, fps) in enumerate(rgb_profiles):
    rgb_stream_profiles.insert(
        i,
        stream_info(
            w, h, fps,
            hololink_module.operators.ImageDecoderOp.PixelFormat.YUYV,
        )
    )
