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
     * Gets the output size of the packetized data given the input size.
     * The default implementation is that data is not resized (i.e. it is only swizzled).
     */
    virtual uint32_t get_output_size(uint32_t input_size);

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
    void enable(Hololink& hololink, uint32_t sif_address) override;
};

/**
  Converts from tightly-packed 10-bpp CSI data to 10-bit RAW such that
  each 4-byte word is comprised of 3 10-bit pixels and 2 bits of padding.
  Output format: {2'b0, p3[9:0], p2[9:0], p1[9:0]}
*/
class Csi10ToPacked10 : public PacketizerProgram {
public:
    uint32_t get_output_size(uint32_t input_size) override;
    void enable(Hololink& hololink, uint32_t sif_address) override;
};

/**
  Swizzles 12-bpp CSI data to contiguous 12-bit pixels.
  Output format: {p2[11:0], p1[11:0]}
*/
class Csi12ToPacked12 : public PacketizerProgram {
public:
    void enable(Hololink& hololink, uint32_t sif_address) override;
};

} // namespace hololink
