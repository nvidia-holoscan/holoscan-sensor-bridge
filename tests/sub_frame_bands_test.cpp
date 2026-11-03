/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 */

// SubFrameBands: how the HSB's sub-frame chunks are cut into bands of image rows. The geometry
// matches sub_frame_pipeline_test's: 72x48 RAW10 (96 bytes per line), a 944-byte header and
// 16-row bands of 1536 bytes, so a 5552-byte frame arrives as chunks of 1536, 1536, 1536 and 944
// bytes and every band straddles two chunks.

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include <hololink/operators/csi_to_bayer/sub_frame_bands.hpp>

namespace hololink::tests {

namespace {

    using hololink::operators::SubFrameBands;

    constexpr std::size_t kBytesPerLine = 96;

    SubFrameBands::Geometry geometry(uint32_t band_rows, std::size_t trailing_bytes = 0)
    {
        return {
            .csi_length = 944 + kBytesPerLine * 48 + trailing_bytes,
            .band_bytes = kBytesPerLine * band_rows,
            .start_byte = 944,
            .band_rows = band_rows,
            .pixel_height = 48,
        };
    }

    // A whole chunk as the HSB sends it: full size, every byte received.
    SubFrameBands::Chunk chunk(int64_t frame_number, std::size_t size)
    {
        return { .frame_number = frame_number,
            .size = size,
            .bytes_written = static_cast<int64_t>(size),
            .bytes_received = static_cast<int64_t>(size) };
    }

    struct Completed {
        int64_t row_offset;
        bool incomplete;
        uint64_t frame_id;
        bool operator==(const Completed&) const = default;
    };

    void PrintTo(const Completed& c, std::ostream* os)
    {
        *os << "{row " << c.row_offset << (c.incomplete ? ", incomplete" : "") << ", frame "
            << c.frame_id << "}";
    }

    std::vector<Completed> completed_bands(SubFrameBands& bands, const std::vector<SubFrameBands::Chunk>& chunks)
    {
        std::vector<Completed> result;
        for (const auto& c : chunks) {
            const SubFrameBands::Step step = bands.next(c);
            if (step.band == SubFrameBands::Band::kComplete) {
                result.push_back({ step.row_offset, step.incomplete, step.frame_id });
            }
        }
        return result;
    }

} // namespace

TEST(SubFrameBandsTest, CutsBandsStraddlingTwoChunks)
{
    SubFrameBands bands(geometry(16));
    SubFrameBands::Step step = bands.next(chunk(0, 1536));
    EXPECT_EQ(step.band, SubFrameBands::Band::kPending);
    EXPECT_EQ(step.payload_offset, 944U); // the header
    EXPECT_EQ(step.band_offset, 0U);
    EXPECT_EQ(step.copy_size, 592U);

    step = bands.next(chunk(1, 1536));
    EXPECT_EQ(step.band, SubFrameBands::Band::kComplete);
    EXPECT_EQ(step.payload_offset, 0U);
    EXPECT_EQ(step.band_offset, 592U);
    EXPECT_EQ(step.copy_size, 944U);
    EXPECT_EQ(step.tail_size, 592U); // starts the next band
    EXPECT_EQ(step.row_offset, 0);
    EXPECT_FALSE(step.incomplete);

    EXPECT_EQ(completed_bands(bands, { chunk(2, 1536), chunk(3, 944) }),
        (std::vector<Completed> { { 16, false, 0 }, { 32, false, 0 } }));
}

// The header is longer than a chunk (8-row bands of 768 bytes): the first chunk is all header.
TEST(SubFrameBandsTest, SkipsAChunkThatIsAllHeader)
{
    SubFrameBands bands(geometry(8));
    EXPECT_EQ(bands.next(chunk(0, 768)).band, SubFrameBands::Band::kHeaderOnly);
    const SubFrameBands::Step step = bands.next(chunk(1, 768));
    EXPECT_EQ(step.band, SubFrameBands::Band::kPending);
    EXPECT_EQ(step.payload_offset, 176U);
    EXPECT_EQ(step.copy_size, 592U);
}

TEST(SubFrameBandsTest, NumbersBandsByTheirSensorFrame)
{
    SubFrameBands bands(geometry(16));
    EXPECT_EQ(completed_bands(bands, { chunk(20, 1536), chunk(21, 1536), chunk(22, 1536), chunk(23, 944) }),
        (std::vector<Completed> { { 0, false, 5 }, { 16, false, 5 }, { 32, false, 5 } }));
}

// A lost chunk leaves the head of the next band stale.
TEST(SubFrameBandsTest, MarksTheBandAfterALostChunkIncomplete)
{
    SubFrameBands bands(geometry(16));
    EXPECT_EQ(completed_bands(bands, { chunk(0, 1536), chunk(2, 1536), chunk(3, 944) }),
        (std::vector<Completed> { { 16, true, 0 }, { 32, false, 0 } }));
}

// A hole (fewer bytes received than written) spoils the band it completes and the one it starts.
TEST(SubFrameBandsTest, MarksBothBandsOfAHoledChunkIncomplete)
{
    SubFrameBands bands(geometry(16));
    SubFrameBands::Chunk holed = chunk(1, 1536);
    holed.bytes_received -= 1408;
    EXPECT_EQ(completed_bands(bands, { chunk(0, 1536), holed, chunk(2, 1536), chunk(3, 944) }),
        (std::vector<Completed> { { 0, true, 0 }, { 16, true, 0 }, { 32, false, 0 } }));
}

// A short chunk (the HSB wrote less) can't complete its band, and leaves the next one stale.
TEST(SubFrameBandsTest, MarksTheBandAfterAShortChunkIncomplete)
{
    SubFrameBands bands(geometry(16));
    EXPECT_EQ(completed_bands(bands, { chunk(0, 1536), chunk(1, 768), chunk(2, 1536), chunk(3, 944) }),
        (std::vector<Completed> { { 16, true, 0 }, { 32, false, 0 } }));
}

// Trailing bytes that add up to a whole band after the image's last row hold no image rows.
TEST(SubFrameBandsTest, ReportsABandOfTrailingBytes)
{
    SubFrameBands bands(geometry(16, kBytesPerLine * 16));
    for (int64_t n = 0; n < 4; ++n) {
        (void)bands.next(chunk(n, 1536));
    }
    const SubFrameBands::Step step = bands.next(chunk(4, 944));
    EXPECT_EQ(step.band, SubFrameBands::Band::kTrailingOnly);
    EXPECT_EQ(step.row_offset, 48);
}

} // namespace hololink::tests
