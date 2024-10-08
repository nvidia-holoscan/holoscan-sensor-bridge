def device_configuration():
    return [
        bytes([0xFC, 0x00, 0x00, 0x00, 0x00,]),
        bytes([0x00, 0x33, 0x10, 0x4A, 0x30, 0x32, 0x02, 0x00, 0x00, 0x04, 0x00, 0x30, 0x00, 0x00, 0x00, 0x00, 0x00,]),
        bytes([0x10, 0x00, 0x00, 0x19, 0x9A, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0x00, 0x00, 0x00, 0x00,]),
        bytes([0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,]),
        bytes([0x30, 0x03, 0x01, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,]),
        bytes([0x40, 0x03, 0x01, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,]),
        bytes([0x50, 0x03, 0x01, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,]),
        bytes([0x60, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,]),
        bytes([0x70, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,]),
        bytes([0x80, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,]),
        bytes([0x90, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,]),
        bytes([0xA0, 0x82, 0x80, 0x36, 0x00, 0x00, 0x38, 0x42, 0x5B, 0x10, 0x11, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0x1F,]),
        bytes([0xB0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00,]),
        bytes([0xC0, 0x90, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x55, 0x01, 0xFF, 0x00,]),
        bytes([0xD0, 0x1F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,]),
        bytes([0xE0, 0x4A, 0x1E, 0x01, 0x81, 0x22, 0x00, 0x5C, 0x8F, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00,]),
        bytes([0xF0, 0x0B, 0x01, 0x00, 0x00, 0x44, 0x00, 0x00, 0x00, 0x0D, 0x4D, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,]),
        bytes([0xFC, 0x00, 0x01, 0x00, 0x00,]),
        bytes([0x00, 0x69, 0x00, 0x0B, 0x6C, 0xB4, 0x03, 0x00, 0x00, 0x9B, 0x81, 0x08, 0x6C, 0xB4, 0x03, 0x00, 0x00,]),
        bytes([0x10, 0x69, 0x00, 0x0B, 0x6C, 0xB4, 0x03, 0x00, 0x00, 0x69, 0x00, 0x0B, 0x6C, 0xB4, 0x03, 0x00, 0x00,]),
        bytes([0x20, 0x00, 0x00, 0x70, 0x00, 0x00, 0x00, 0x70, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,]),
        bytes([0x30, 0x10, 0x2F, 0x00, 0x02, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,]),
        bytes([0x40, 0x21, 0x06, 0x44, 0x09,]),
        bytes([0x45, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x52, 0xB8, 0x1E, 0x05, 0x62,]),
        bytes([0x55, 0x00, 0x23, 0x0D, 0x44, 0x3E, 0x64, 0x27, 0x06, 0x1F, 0x45, 0x0F, 0x04, 0x00, 0x00, 0x7A, 0x80,]),
        bytes([0x65, 0x01, 0x88, 0x00, 0x00, 0x00, 0x00, 0x25, 0x01, 0x00, 0x01, 0x09, 0x00, 0x00, 0x00, 0x00, 0x00,]),
        bytes([0x75, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,]),
        bytes([0x85, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,]),
        bytes([0x95, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x2C, 0x3B, 0x00, 0x77, 0x70,]),
        bytes([0xA5, 0x80, 0x01, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0,]),
        bytes([0xB5, 0x00, 0xD0, 0x03, 0x00, 0x00, 0x00, 0x00, 0xBA, 0x00, 0x00, 0x00, 0x1A, 0xA6, 0x0F, 0x47, 0x24,]),
        bytes([0xC5, 0x00, 0x24, 0x00, 0x00, 0x11, 0x20, 0x12, 0x0B, 0x10, 0x02, 0x30, 0x00, 0x00, 0x00, 0x00, 0x00,]),
        bytes([0xD5, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,]),
        bytes([0x62, 0x00, 0x7A,]),
        bytes([0x62, 0x00, 0x7A,]),
        bytes([0x62, 0x00, 0x7A,]),
        bytes([0x62, 0x00, 0x7A,]),
        bytes([0x62, 0x80, 0x7A,]),
        bytes([0x62, 0x00, 0x7A,]),
        bytes([0xFC, 0x00, 0x00, 0x00, 0x00,]),
        bytes([0x0A, 0x30,]),
        bytes([0x0A, 0x32,]),
        bytes([0x0A, 0x30,]),
        bytes([0xFC, 0x00, 0x01, 0x00, 0x00,]),
        bytes([0x44, 0x01,]),
        bytes([0xFC, 0x00, 0x00, 0x00, 0x00,]),
    ]
