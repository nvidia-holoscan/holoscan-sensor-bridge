from enum import Enum

class Endianness(Enum):
    LITTLE = "little"
    BIG = "big"

class DataWidth(Enum):
    BITS_8 = 1
    BITS_16 = 2
    BITS_32 = 4
