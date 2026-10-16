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

/**
 * @brief A list of memory operations that an HSB device runs by itself
 * when a trigger event occurs.
 *
 * A sequencer lets time-critical work (for example reading a sensor's
 * registers over I2C at the end of every frame) be done by the device
 * hardware at the moment of the trigger, rather than by the host after
 * the fact. Operations are recorded with write_uint32(), read_uint32()
 * and poll(), then loaded onto the device and armed with enable(). Each
 * time the sequencer's trigger event occurs the device runs the whole
 * list, in the order recorded, without interruption.
 *
 * Which event triggers a sequencer depends on how it was obtained: each
 * `*_sequencer` accessor of HololinkInterfaceV1 (software, GPIO, sensor
 * frame start or end) and of the data channel interfaces yields a
 * sequencer for its own event.
 *
 * Results of read and poll operations are left in the sequencer's memory
 * on the device. A recorded operation is identified by the index
 * returned when it was added: the index of a 32-bit word in the
 * sequence. The device address of the word at index `i` is
 * `location() + 4 * i`.
 *
 * HSB executes sequences one-at-a-time: the current sequence must
 * finish before another event can be handled-- so be aware that a
 * long-running poll operation may affect other event handling.  Also
 * note that some operations (like I2C transactions) always go through
 * sequencer (to avoid reentrancy), so sequencer occupation may not be
 * entirely obvious.
 */
class SequencerInterfaceV1 : public Service<SequencerInterfaceV1> {
public:
    /// @cond INCLUDE_INTERNAL_DOCUMENTATION
    static constexpr const char* type_id = "sequencer.v1";
    static constexpr const char* release_version = "2.7.0";

    virtual ~SequencerInterfaceV1() = default;
    /** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION

    /**
     * @brief Add a memory write to the sequence.
     *
     * When the sequence runs, `data` is written to the HSB local bus
     * at `address`.
     *
     * @param address Device location to write.
     * @param data Value to write.
     * @return Index of the word holding `data`.
     * @throws std::runtime_error if enable() has been called.
     */
    virtual unsigned write_uint32(uint32_t address, uint32_t data) = 0;

    /**
     * @brief Add a memory read to the sequence.
     *
     * When the sequence runs, the memory at `address` is read and
     * the value stored at the returned index.
     *
     * Operations can only be added before enable() is called.
     *
     * @param address Device memory to read.
     * @param initial_value Value held at the returned index until the
     *      first time the sequence runs, so that a result that hasn't
     *      been produced yet can be recognized.
     * @return Index at which the value read is stored.
     * @throws std::runtime_error if enable() has been called.
     */
    virtual unsigned read_uint32(uint32_t address,
        uint32_t initial_value = 0xFFFFFFFFu)
        = 0;

    /**
     * @brief Add a wait to the sequence: the sequence stops at this
     * point until a memory value has the expected value.
     *
     * When the sequence runs, the memory at `address` is read
     * repeatedly until `(value & mask) == match`, and only then does
     * the sequence continue. The poll times out if the match does not
     * occur within the timeout configured on the device.
     *
     * Operations can only be added before enable() is called.
     *
     * @param address Device memory to read.
     * @param mask Bits of the value to compare.
     * @param match Value the masked bits must equal.
     * @return Index at which the most recent value read from
     *      `address` is stored.
     * @throws std::runtime_error if enable() has been called.
     */
    virtual unsigned poll(uint32_t address, uint32_t mask, uint32_t match) = 0;

    /**
     * @brief Load the sequence onto the device and arm it for its
     * trigger.
     *
     * Call this once, after the last operation has been added. From
     * then on the device runs the sequence each time the trigger event
     * occurs, and location() can be used.
     *
     * @return HOLOLINK_MODULE_OK or an error status.
     * @throws std::runtime_error if the sequencer has already been
     *      enabled.
     */
    virtual hololink_module_status_t enable() = 0;

    /**
     * @brief The device address at which the sequence is stored.
     *
     * The word for an operation's index is at this address plus four
     * times the index. Use it to read back results, or to update data to
     * be written.
     *
     * @return The address.
     * @throws std::runtime_error if enable() has not been called.
     */
    virtual uint32_t location() = 0;
};

} // namespace hololink::module

#endif // HOLOLINK_MODULE_SEQUENCER_HPP
