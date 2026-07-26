/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_SEQUENCER_HPP
#define HOLOLINK_MODULE_SEQUENCER_HPP

#include <cstdint>
#include <memory>

#include "module.hpp"
#include "service.hpp"
#include "status.h"

namespace hololink::module {

/* A pre-programmed command sequence (reads, writes, polls) that
 * executes atomically on the FPGA, triggered by a hardware event
 * (frame end, frame start, GPIO) or software. The location() method
 * returns the base address in FPGA memory so callers can read back
 * results after execution.
 *
 * Instances come from factory methods on HololinkInterface and from
 * data-channel interfaces' frame_end_sequencer. */
class SequencerInterfaceV1 : public Service<SequencerInterfaceV1> {
public:
    static constexpr const char* type_id = "sequencer.v1";

    virtual ~SequencerInterfaceV1() = default;

    /* Encode a register write into the sequencer; return the index in
     * the sequencer buffer at which the encoded operation starts. */
    virtual unsigned write_uint32(uint32_t address, uint32_t data) = 0;

    /* Encode a register read; return the index where the read result
     * will be deposited at execution time. initial_value seeds the
     * slot before the read runs. */
    virtual unsigned read_uint32(uint32_t address,
        uint32_t initial_value = 0xFFFFFFFFu)
        = 0;

    /* Encode a poll-until-mask-matches operation; return the index
     * where the latest read value will be deposited. */
    virtual unsigned poll(uint32_t address, uint32_t mask, uint32_t match) = 0;

    /* Arm the sequencer for hardware-triggered execution. */
    virtual hololink_module_status_t enable() = 0;

    /* Base address in FPGA memory where the encoded sequencer
     * lives. Callers read back results from this base + per-op
     * indexes after execution. */
    virtual uint32_t location() = 0;
};

} // namespace hololink::module

#endif // HOLOLINK_MODULE_SEQUENCER_HPP
