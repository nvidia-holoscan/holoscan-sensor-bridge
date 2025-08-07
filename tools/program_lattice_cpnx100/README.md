# Program Lattice CPNX100 - C++ Version

This is a C++ rewrite of the Python script `python/tools/program_lattice_cpnx100.py` for
programming Lattice CPNX100 flash memory on Hololink boards.

## Overview

The program provides functionality to:

- Program and verify CLNX and CPNX flash memory on Hololink boards
- Handle Winbond W25Q128JW flash memory devices
- Support both traditional and modern SPI interfaces
- Parse YAML manifest files for firmware configuration
- Download firmware from URLs or load from local files
- Verify firmware integrity using MD5 checksums
- Handle EULA acceptance

## Building

### Prerequisites

- CMake 3.20 or later
- C++17 compatible compiler
- yaml-cpp library
- libcurl library
- OpenSSL library
- Hololink core library

### Build Instructions

```bash
# From the hololink root directory
mkdir -p build
cd build
cmake ..
make program_lattice_cpnx100
```

### Dependencies

The following packages are required:

- **yaml-cpp**: YAML parsing library
- **libcurl**: HTTP client library for downloading firmware
- **OpenSSL**: Cryptographic functions for MD5 verification
- **fmt**: Formatting library (included with Hololink)

## Usage

```bash
./program_lattice_cpnx100 [OPTIONS] <manifest>
```

### Options

- `--hololink=IP`: IP address of Hololink board (default: 192.168.0.2)
- `--force`: Don't rely on enumeration data for device connection
- `--log-level=LEVEL`: Logging level to display (default: 20)
- `--archive=FILE`: Use a local zip archive instead of downloading
- `--skip-program-clnx`: Skip programming CLNX
- `--skip-verify-clnx`: Skip verifying CLNX
- `--skip-program-cpnx`: Skip programming CPNX
- `--skip-verify-cpnx`: Skip verifying CPNX
- `--accept-eula`: Provide non-interactive EULA acceptance
- `--skip-power-cycle`: Don't wait for confirmation of power cycle
- `-h, --help`: Display help information

### Example

```bash
./program_lattice_cpnx100 scripts/manifest.yaml
```

## Architecture

### Classes

- **WinbondW25q128jw**: Base class for Winbond W25Q128JW flash memory operations
- **ClnxFlash**: Specialized class for CLNX flash programming
- **CpnxFlash**: Specialized class for CPNX flash programming
- **Programmer**: Main orchestrator class for programming operations

### Key Features

1. **SPI Interface Support**: Supports both traditional and modern SPI interfaces based
   on FPGA version
1. **Flash Memory Operations**: Implements erase, program, and verify operations for
   flash memory
1. **Manifest Parsing**: Parses YAML manifest files for firmware configuration
1. **Content Fetching**: Downloads firmware from URLs or loads from local files
1. **Integrity Verification**: Verifies firmware integrity using MD5 checksums
1. **Error Handling**: Comprehensive error handling and logging

## Differences from Python Version

The C++ version maintains the same functionality as the Python version but with some
implementation differences:

1. **Memory Management**: Uses RAII and smart pointers for automatic memory management
1. **Error Handling**: Uses C++ exceptions instead of Python exceptions
1. **Logging**: Uses Hololink's logging system instead of Python's logging
1. **HTTP Requests**: Uses libcurl instead of Python's requests library
1. **YAML Parsing**: Uses yaml-cpp instead of Python's PyYAML
1. **Cryptographic Functions**: Uses OpenSSL instead of Python's hashlib

## File Structure

```
tools/program_lattice_cpnx100/
├── CMakeLists.txt          # Build configuration
├── README.md              # This file
├── main.cpp               # Main executable entry point
├── programmer.hpp         # Programmer class header
├── programmer.cpp         # Programmer class implementation
├── winbond_w25q128jw.hpp # Winbond flash base class header
├── winbond_w25q128jw.cpp # Winbond flash base class implementation
├── clnx_flash.hpp         # CLNX flash class header
├── clnx_flash.cpp         # CLNX flash class implementation
├── cpnx_flash.hpp         # CPNX flash class header
└── cpnx_flash.cpp         # CPNX flash class implementation
```

## License

This code is licensed under the Apache License, Version 2.0. See the license headers in
each source file for details.
