/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_CORE_SEQUENCER_DEFAULT_HPP
#define HOLOLINK_MODULE_CORE_SEQUENCER_DEFAULT_HPP

#include <memory>
#include <utility>

#include "hololink/module/sequencer.hpp"
#include "hololink/module/status.h"

#include "hololink/core/hololink.hpp"

#include "hololink_default.hpp"

namespace hololink::module::module_core {

/* SequencerInterfaceV1 backed by a hololink::Hololink::Sequencer.
 * The shared_ptr to the legacy sequencer keeps the underlying object
 * alive for the wrapper's lifetime. */
class SequencerV1 : public SequencerInterfaceV1 {
public:
    SequencerV1(std::shared_ptr<HololinkV1> owner,
        std::shared_ptr<hololink::Hololink::Sequencer> backing)
        : owner_(std::move(owner))
        , backing_(std::move(backing))
    {
        owner_->register_associated(this);
    }

    unsigned write_uint32(uint32_t address, uint32_t data) override
    {
        return backing_->write_uint32(address, data);
    }

    unsigned read_uint32(uint32_t address, uint32_t initial_value) override
    {
        return backing_->read_uint32(address, initial_value);
    }

    unsigned poll(uint32_t address, uint32_t mask, uint32_t match) override
    {
        return backing_->poll(address, mask, match);
    }

    hololink_module_status_t enable() override
    {
        backing_->enable();
        return HOLOLINK_MODULE_OK;
    }

    uint32_t location() override
    {
        return backing_->location();
    }

private:
    std::shared_ptr<HololinkV1> owner_;
    std::shared_ptr<hololink::Hololink::Sequencer> backing_;
};

} // namespace hololink::module::module_core

#endif // HOLOLINK_MODULE_CORE_SEQUENCER_DEFAULT_HPP
