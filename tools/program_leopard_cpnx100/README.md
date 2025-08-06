# Program Leopard CPNX100

This tool programs the CPNX flash memory on Leopard sensor bridge devices using Macronix
MX25U25645G flash memory.

## Overview

The `program_leopard_cpnx100` tool is a C++ implementation that replaces the Python
version. It provides the same functionality for programming and verifying firmware on
Leopard sensor bridge devices.

## Features

- Programs CPNX flash memory using Macronix MX25U25645G
- Verifies programmed content
- Supports both traditional and modern SPI controllers
- Downloads firmware from URLs or reads from local files
- Validates MD5 checksums
- Handles EULA acceptance
- Supports power cycle confirmation

## Usage

```bash
program_leopard_cpnx100 [OPTIONS] <manifest>
```

### Options

- `--hololink=IP` - IP address of Hololink board (default: 192.168.0.2)
- `--force` - Don't rely on enumeration data for device connection
- `--log-level=LEVEL` - Logging level to display
- `--skip-program-cpnx` - Skip programming CPNX
- `--skip-verify-cpnx` - Skip verifying CPNX
- `--accept-eula` - Provide non-interactive EULA acceptance
- `--skip-power-cycle` - Don't wait for confirmation of power cycle
- `-h, --help` - Display help information

### Example

```bash
program_leopard_cpnx100 --hololink=192.168.0.2 scripts/manifest_leopard_cpnx100.yaml
```

## Dependencies

- yaml-cpp
- libcurl
- OpenSSL
- hololink::core

## Building

The tool is built as part of the main hololink build system. It will be installed to the
system's binary directory.

## Architecture

The tool consists of several key components:

- `MacronixMx25u25645g` - Base class for Macronix flash memory operations
- `CpnxFlash` - Specialized implementation for CPNX flash
- `Programmer` - Main programming logic and manifest handling
- `main.cpp` - Command-line interface and program entry point

## Differences from Python Version

- Uses C++ standard library instead of Python libraries
- Implements custom MD5 calculation using OpenSSL
- Uses libcurl for HTTP downloads instead of requests
- Implements a simple pager for EULA display
- Uses fmt library for string formatting
