/**
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
 *
 * See README.md for detailed information.
 */

#pragma once

#include "csi_formats.hpp"
#include "hololink.hpp"

namespace hololink {

/**
 * Base class for an object that provides the definition for a packetizer program.
 */
class PacketizerProgram {
public:
    virtual ~PacketizerProgram();

    /**
     * Whether enable() turns the packetizer on. False for NullPacketizerProgram, whose
     * enable() leaves the sensor stream passing through untouched.
     */
    virtual bool packetizes() const;

    /**
     * The number of bytes the packetizer emits for one line of input_size bytes: the
     * line as the packetizer's sorting network reshapes it, before the HSB pads it to a
     * 64-byte boundary (see padded_output_size()).
     * The default implementation is that data is not resized (i.e. it is only swizzled).
     */
    virtual uint32_t get_output_size(uint32_t input_size);

    /**
     * The number of bytes the HSB puts on the wire for one line of input_size bytes:
     * get_output_size() padded to the 64-byte boundary the packetizer pads every line to
     * once it is enabled, whether or not the line went through the sorting network (the
     * embedded-data lines it bypasses are padded too).
     */
    uint32_t padded_output_size(uint32_t input_size);

    /// The boundary the enabled packetizer pads every line to, in bytes.
    static constexpr uint32_t LINE_PAD_BYTES = 64;

    /**
     * Enables the packetizer program. Called by a DataChannel at configure() time.
     * This must be implemented by the derived class.
     */
    virtual void enable(Hololink& hololink, uint32_t sif_address) = 0;

    /**
     * Disables the packetizer program. Called by a DataChannel at unconfigure() time.
     */
    virtual void disable(Hololink& hololink, uint32_t sif_address);

protected:
    PacketizerProgram();
};

/**
  Default packetizer program which disables the packetizer.
*/
class NullPacketizerProgram : public PacketizerProgram {
public:
    bool packetizes() const override;
    void enable(Hololink& hololink, uint32_t sif_address) override;
};

/**
  A program for the HSB's programmable packetizer (fpga/nv_hsb_ip/packetizer) is a
  schedule in the PACKETIZER_MODE control word plus the sorting-network RAM contents
  enable() writes. The schedule is what sizes a line, and the packetizer knows nothing of
  pixel formats, only of beats and rows: the loop has MAX_ADDR+1 rows (control word bits
  [23:20]), row r consumes one input beat when bit r of LD_ARR (bits [15:0]) is set, every
  row emits one output beat, and a line that ends part-way through the loop is completed
  with zero input (packetizer_top.sv PACK_RUN -> PACK_PATTERN). So, with a beat the
  64-bit sensor interface these programs are written for,

      output_bytes = ceil(input_bytes / (popcount(LD_ARR) * 8)) * ((MAX_ADDR + 1) * 8)

  before the 64-byte pad (bit 16). Each program below hard-codes that result for its own
  control word; a program whose schedule is not fixed should decode the word instead.

  Converts from tightly-packed 10-bpp CSI data to 10-bit RAW such that
  each 4-byte word is comprised of 3 10-bit pixels and 2 bits of padding.
  Output format: {2'b0, p3[9:0], p2[9:0], p1[9:0]}
  Schedule 0x11f1fff7: 15 input beats (120 bytes, 96 pixels) per 16 output beats (128
  bytes), so a line's output is ceil(line_bytes / 120) * 128.
*/
class Csi10ToPacked10 : public PacketizerProgram {
public:
    uint32_t get_output_size(uint32_t input_size) override;
    void enable(Hololink& hololink, uint32_t sif_address) override;
};

/**
  Swizzles 12-bpp CSI data to contiguous 12-bit pixels.
  Output format: {p2[11:0], p1[11:0]}
  Schedule 0x11210007: 3 input beats per 3 output beats (24 bytes, 16 pixels), so a line's
  output is ceil(line_bytes / 24) * 24 (see Csi10ToPacked10 for the general form).
*/
class Csi12ToPacked12 : public PacketizerProgram {
public:
    uint32_t get_output_size(uint32_t input_size) override;
    void enable(Hololink& hololink, uint32_t sif_address) override;
};

} // namespace hololink
