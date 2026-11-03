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

#ifndef SRC_HOLOLINK_OPERATORS_CSI_TO_BAYER_SUB_FRAME_BANDS
#define SRC_HOLOLINK_OPERATORS_CSI_TO_BAYER_SUB_FRAME_BANDS

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace hololink::operators {

/**
 * Bookkeeping that cuts a sensor's CSI-2 byte stream, as the HSB delivers it in sub-frame mode,
 * into bands of whole image rows. Pure arithmetic and state, no CUDA, shared by the native and
 * hololink_module CsiToBayerOps.
 *
 * In sub-frame mode the HSB sends each frame as consecutive chunks of band_bytes
 * (bytes_per_line * band_rows) bytes, except the frame's last chunk, which holds what is left of
 * the CSI stream; each chunk is a receiver frame with its own frame_number. The sensor header
 * (start_byte) is not a whole number of bands, so a band straddles two chunks: the tail of one
 * chunk starts the band the next chunk finishes. The caller keeps the band being assembled and,
 * for every chunk, asks next() where its bytes go:
 * - copy copy_size bytes from the chunk at payload_offset into the band at band_offset;
 * - when the band is complete, emit it (band rows from row_offset) unless it is incomplete, and
 *   start the next band with the chunk's last tail_size bytes.
 * A chunk larger than a band (from a receiver whose frame_size isn't band_bytes) is rejected:
 * next() marks it oversized, with nothing to copy, so tail_size never exceeds the band.
 */
class SubFrameBands {
public:
    struct Geometry {
        std::size_t csi_length = 0; ///< bytes of one frame's CSI stream: header, lines, trailer
        std::size_t band_bytes = 0; ///< bytes_per_line * band_rows
        std::uint32_t start_byte = 0; ///< header bytes ahead of the first image line
        std::uint32_t band_rows = 0; ///< image rows per band
        std::uint32_t pixel_height = 0; ///< image rows per frame
    };

    struct Chunk {
        std::int64_t frame_number = 0; ///< the receiver's frame (chunk) number
        std::size_t size = 0; ///< bytes delivered (the receiver trims a chunk to bytes_written)
        std::int64_t bytes_written = 0; ///< bytes the HSB wrote, per the receiver
        std::int64_t bytes_received = 0; ///< bytes the receiver got (0: not counted)
    };

    enum class Band {
        kHeaderOnly, ///< nothing to copy: the chunk is all header (or oversized, and rejected)
        kPending, ///< the band isn't complete yet
        kComplete, ///< the band is complete
        kTrailingOnly, ///< the completed "band" is only the frame's trailing bytes: no image rows
    };

    struct Step {
        Band band = Band::kHeaderOnly;
        std::size_t payload_offset = 0; ///< chunk bytes to skip (header)
        std::size_t band_offset = 0; ///< where in the band the chunk's bytes go
        std::size_t copy_size = 0; ///< chunk bytes that go into this band
        std::size_t tail_size = 0; ///< chunk bytes past the completed band: the next band's head
        /// Part of the completed band was lost: its head with an earlier chunk (dropped, short or
        /// with a hole), so the band memory holds stale bytes ahead of this chunk's, or a packet of
        /// this chunk itself.
        bool incomplete = false;
        std::int64_t row_offset = 0; ///< first image row of the band
        std::uint64_t frame_id = 0; ///< the sensor frame the band belongs to
        /// The chunk is larger than a band, so where its bytes go is unknown: rejected (band is
        /// kHeaderOnly), and the band it would have started lacks its head.
        bool oversized = false;
    };

    explicit SubFrameBands(const Geometry& geometry)
        : geometry_(geometry)
        , chunks_per_frame_(
              static_cast<std::int64_t>((geometry.csi_length + geometry.band_bytes - 1) / geometry.band_bytes))
    {
    }

    [[nodiscard]] Step next(const Chunk& chunk)
    {
        if (chunk.size > geometry_.band_bytes) {
            // The HSB never sends more than a band per chunk when the receiver's frame_size is
            // band_bytes; past one band, the chunk's tail would also overrun the band memory.
            previous_frame_number_ = chunk.frame_number;
            previous_chunk_complete_ = false;
            Step rejected;
            rejected.oversized = true;
            return rejected;
        }
        const auto band_bytes = static_cast<std::int64_t>(geometry_.band_bytes);
        const std::int64_t chunk_offset = (chunk.frame_number % chunks_per_frame_) * band_bytes;

        // The HSB sends a full band_bytes chunk except for a frame's last one, which holds what is
        // left of csi_length; a smaller one is a chunk of which the HSB wrote less. Packets lost on
        // the way leave a hole instead, which a receiver counting the bytes it got (the Linux one)
        // shows as fewer received than written.
        const std::size_t expected_size = std::min(geometry_.band_bytes,
            geometry_.csi_length - static_cast<std::size_t>(chunk_offset));
        const bool has_hole = chunk.bytes_received != 0 && chunk.bytes_received < chunk.bytes_written;
        // Whatever of the band is already assembled was carried over from the previous chunk. It
        // belongs to this band only if that chunk was this one's immediate predecessor and arrived
        // whole.
        const bool continues_previous_chunk = previous_chunk_complete_
            && chunk.frame_number == previous_frame_number_ + 1;
        previous_frame_number_ = chunk.frame_number;
        previous_chunk_complete_ = chunk.size >= expected_size && !has_hole;

        Step step;
        // How many bytes at the start of this chunk are still part of the header (all of it when
        // the header is longer than a chunk).
        const std::int64_t skip = std::max(std::int64_t(0),
            static_cast<std::int64_t>(geometry_.start_byte) - chunk_offset);
        if (static_cast<std::size_t>(skip) >= chunk.size) {
            return step;
        }
        step.payload_offset = static_cast<std::size_t>(skip);
        const std::size_t payload_size = chunk.size - step.payload_offset;
        // Derive the row offset from the byte offset with the header already removed, not from
        // the chunk index: start_byte is not a whole number of bands, so the two disagree by one
        // band whenever the CSI stream carries a header.
        const std::int64_t band_byte_offset
            = chunk_offset - static_cast<std::int64_t>(geometry_.start_byte) + skip;
        step.row_offset = (band_byte_offset / band_bytes) * static_cast<std::int64_t>(geometry_.band_rows);
        step.band_offset = static_cast<std::size_t>(band_byte_offset % band_bytes);
        step.copy_size = std::min(payload_size, geometry_.band_bytes - step.band_offset);
        if (step.band_offset + step.copy_size != geometry_.band_bytes) {
            step.band = Band::kPending;
            return step;
        }
        step.tail_size = payload_size - step.copy_size;
        if (step.row_offset >= static_cast<std::int64_t>(geometry_.pixel_height)) {
            // The frame's trailing bytes (the sensor's status lines after the image, when they
            // add up to a whole band).
            step.band = Band::kTrailingOnly;
            return step;
        }
        step.band = Band::kComplete;
        step.incomplete = (step.band_offset != 0 && !continues_previous_chunk) || has_hole;
        step.frame_id = static_cast<std::uint64_t>(chunk.frame_number / chunks_per_frame_);
        return step;
    }

private:
    Geometry geometry_;
    std::int64_t chunks_per_frame_;
    std::int64_t previous_frame_number_ = -1;
    bool previous_chunk_complete_ = false;
};

} // namespace hololink::operators

#endif /* SRC_HOLOLINK_OPERATORS_CSI_TO_BAYER_SUB_FRAME_BANDS */
