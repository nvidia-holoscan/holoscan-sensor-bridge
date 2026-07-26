/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_CORE_HOLOLINK_DEFAULT_HPP
#define HOLOLINK_MODULE_CORE_HOLOLINK_DEFAULT_HPP

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "hololink/module/data_channel.hpp"
#include "hololink/module/enumeration_metadata.hpp"
#include "hololink/module/hololink.hpp"
#include "hololink/module/i2c_lock.hpp"
#include "hololink/module/publisher.hpp"
#include "hololink/module/status.h"

#include "hololink/core/hololink.hpp"
#include "hololink/core/named_lock.hpp"

namespace hololink::module::module_core {

/* Subclass of hololink::Hololink that re-exposes the methods Hololink
 * declared protected (`configure_hsb`, `and_uint32`, `or_uint32`).
 * The legacy class restricted these to friends; HololinkInterfaceV1
 * publishes them as part of its public surface, so the supplement
 * constructs this subclass instead of the base. No state is added —
 * only constructor inheritance and three using-declarations. */
class LegacyHololinkAccess : public hololink::Hololink {
public:
    using hololink::Hololink::Hololink;

    using hololink::Hololink::and_uint32;
    using hololink::Hololink::configure_hsb;
    using hololink::Hololink::or_uint32;
};

/* I2cLockV1 wrapper around hololink::NamedLock.
 *
 * NamedLock is a recursive process-level lock (lockf-backed) that
 * predates std::try_lock semantics — it offers lock() / unlock()
 * only. Wrapping it under V1's BasicLockable + Lockable contract
 * means try_lock falls back to a blocking lock(), then returns true.
 * Callers that genuinely need non-blocking acquisition need a
 * different lock implementation. */
class I2cLockNamedV1 : public I2cLockV1 {
public:
    I2cLockNamedV1(std::shared_ptr<LegacyHololinkAccess> hololink,
        hololink::NamedLock* backing)
        : hololink_(std::move(hololink))
        , backing_(backing)
    {
    }

    void lock() override { backing_->lock(); }
    void unlock() override { backing_->unlock(); }
    bool try_lock() override
    {
        backing_->lock();
        return true;
    }

private:
    /* Keep the parent Hololink alive — the NamedLock reference came
     * out of it. */
    std::shared_ptr<LegacyHololinkAccess> hololink_;
    hololink::NamedLock* backing_;
};

/* Adapts a std::function<void()> to the legacy
 * hololink::Hololink::ResetController interface so the module's
 * HololinkInterfaceV1::on_reset (which takes a plain callback) can
 * register against the backing legacy Hololink. Fires the callback once
 * per board reset. */
class ResetCallbackController : public hololink::Hololink::ResetController {
public:
    explicit ResetCallbackController(std::function<void()> callback)
        : callback_(std::move(callback))
    {
    }
    void reset() override
    {
        if (callback_) {
            callback_();
        }
    }

private:
    std::function<void()> callback_;
};

/* HololinkInterfaceV1 backed by a hololink::Hololink.
 *
 * Default-constructed cheap from the per-module service constructor;
 * the supplement publishes it under instance_id "serial=<n>" with no
 * backing yet. The first call to configure(metadata) materializes
 * backing_ from the metadata; subsequent configure calls observe
 * backing_ already set and return immediately, so a cached
 * HololinkV1 can be reached by any number of
 * ConfigurableService::get_service(metadata) calls without
 * re-materializing. */
class HololinkV1 : public HololinkInterfaceV1,
                   public Service<HololinkV1> {
public:
    /* Impl-specific type_id distinct from HololinkInterfaceV1::type_id.
     * The supplement publishes the same shared_ptr under both keys so
     * callers that need the concrete impl surface (legacy_access,
     * ensure_configured) fetch it through HololinkV1::get_service
     * without a static_pointer_cast. */
    static constexpr const char* type_id = "hololink.module_core.v1";

    /* Hides the inherited HololinkInterfaceV1 chain's get_service
     * overloads via the C++ name-hiding rule (a declaration in the
     * derived class hides inherited declarations of the same name
     * regardless of signature). Without this, HololinkV1::get_service
     * would be ambiguous between Service<HololinkV1>::get_service
     * and the V1 chain's inherited overloads (same signature, different
     * return types). The cast inside Service<HololinkV1>::get_service
     * is safe by construction — the supplement only publishes
     * HololinkV1 instances under HololinkV1::type_id. */
    using Service<HololinkV1>::get_service;
    using Service<HololinkV1>::for_each_type_id;

    /* ServiceAlias drives Service<HololinkV1>::for_each_type_id's
     * chain walk: it emits the parent V1 interface's type_id first
     * ("hololink.v1") and then this impl's own type_id. */
    using ServiceAlias = HololinkInterfaceV1;

    HololinkV1() = default;

    void configure(const EnumerationMetadata& metadata) override;

    const EnumerationMetadata& enumeration_metadata() const override
    {
        return enumeration_metadata_;
    }

    std::shared_ptr<DataChannelInterfaceV1> default_data_channel() const override
    {
        return default_data_channel_;
    }

    hololink_module_status_t start() override
    {
        backing_->start();
        return HOLOLINK_MODULE_OK;
    }
    hololink_module_status_t stop() override
    {
        backing_->stop();
        return HOLOLINK_MODULE_OK;
    }
    hololink_module_status_t reset() override
    {
        backing_->reset();
        return HOLOLINK_MODULE_OK;
    }
    hololink_module_status_t configure_hsb() override
    {
        backing_->configure_hsb();
        return HOLOLINK_MODULE_OK;
    }
    hololink_module_status_t device_lost() override
    {
        // Invalidate every service associated with this board, then this
        // Hololink, so re-resolving after rediscovery yields fresh instances.
        // Take ownership of the list (leaving associated_ empty) so a second
        // device_lost() on this instance can't re-invalidate now-freed
        // pointers — whose addresses may have been reused by live services.
        std::vector<const void*> associated;
        {
            std::lock_guard<std::mutex> lock(associated_mutex_);
            associated.swap(associated_);
        }
        for (const void* service : associated) {
            Publisher::invalidate(service);
        }
        Publisher::invalidate(this);
        return HOLOLINK_MODULE_OK;
    }

    /* Associate a per-board service with this Hololink so device_lost()
     * invalidates it too. A sibling service registers itself here when it
     * resolves this Hololink at configure time. */
    void register_associated(const void* service)
    {
        std::lock_guard<std::mutex> lock(associated_mutex_);
        associated_.push_back(service);
    }
    bool ptp_synchronize() override
    {
        // Delegate to the backing legacy Hololink's default-timeout
        // overload; it returns whether the clock synchronized in time.
        return backing_->ptp_synchronize();
    }

    std::shared_ptr<HololinkInterfaceV1::ResetRegistration> on_reset(
        std::function<void()> callback) override
    {
        // The legacy Hololink::on_reset is append-only (no removal) and
        // HololinkV1 is a process-lifetime per-serial singleton, so
        // forwarding each callback straight through would accumulate (and
        // pin its owner) forever. Instead register ONE aggregating
        // controller with the backing Hololink and keep the callbacks in
        // our own registry, handing back an RAII handle that erases its
        // entry when destroyed.
        // Serialize the one-time aggregator creation so exactly one
        // ResetCallbackController is registered with backing_. on_reset is
        // currently only reached via the GIL-holding Python binding, but this
        // keeps the check-and-create safe for any future C++ (off-GIL) caller.
        {
            std::lock_guard<std::mutex> aggregator_lock(reset_aggregator_mutex_);
            if (!reset_aggregator_) {
                auto registry = reset_registry_;
                reset_aggregator_ = std::make_shared<ResetCallbackController>(
                    [registry]() {
                        std::vector<std::function<void()>> snapshot;
                        {
                            std::lock_guard<std::mutex> lock(registry->mutex);
                            snapshot.reserve(registry->callbacks.size());
                            for (auto& entry : registry->callbacks) {
                                snapshot.push_back(entry.second);
                            }
                        }
                        for (auto& cb : snapshot) {
                            if (cb) {
                                cb();
                            }
                        }
                    });
                backing_->on_reset(reset_aggregator_);
            }
        }
        uint64_t id;
        {
            std::lock_guard<std::mutex> lock(reset_registry_->mutex);
            id = reset_registry_->next_id++;
            reset_registry_->callbacks[id] = std::move(callback);
        }
        return std::make_shared<ResetRegistrationImpl>(reset_registry_, id);
    }

    hololink_module_status_t write_uint32(
        const std::vector<uint32_t>& addresses,
        const std::vector<uint32_t>& values) override
    {
        if (addresses.size() != values.size()) {
            return HOLOLINK_MODULE_INVALID_PARAMETER;
        }
        // Batch every (address, value) pair into a single WR_BLOCK UDP message
        // instead of one control-plane round-trip per register. A control-plane
        // timeout throws from the backing Hololink and propagates — like every
        // other wrapper here, this layer does not swallow backing exceptions.
        hololink::Hololink::WriteData write_data;
        for (size_t i = 0; i < addresses.size(); ++i) {
            write_data.queue_write_uint32(addresses[i], values[i]);
        }
        if (!backing_->write_uint32(write_data)) {
            return HOLOLINK_MODULE_NETWORK_ERROR;
        }
        return HOLOLINK_MODULE_OK;
    }

    hololink_module_status_t read_uint32(
        const std::vector<uint32_t>& addresses,
        std::vector<uint32_t>& out_values) override
    {
        out_values.resize(addresses.size());
        if (addresses.empty()) {
            return HOLOLINK_MODULE_OK;
        }
        // A run of consecutive registers (stride 4) goes out as a single
        // RD_BLOCK UDP message; only genuinely scattered addresses fall back to
        // one request per address. Control-plane timeouts propagate (see
        // write_uint32).
        bool consecutive = true;
        for (size_t i = 1; i < addresses.size(); ++i) {
            if (addresses[i] != addresses[0] + static_cast<uint32_t>(i * 4)) {
                consecutive = false;
                break;
            }
        }
        if (consecutive) {
            auto [ok, values] = backing_->read_uint32(
                addresses[0], static_cast<uint32_t>(addresses.size()),
                std::shared_ptr<hololink::Timeout>(), sequence_number_checking_);
            if (!ok) {
                // The block read reports a timed-out control plane by returning
                // false; the scalar read throws, so raise here too to keep every
                // read path's timeout behavior identical and propagating.
                throw hololink::TimeoutError("read_uint32 block read timed out");
            }
            out_values = std::move(values);
        } else {
            for (size_t i = 0; i < addresses.size(); ++i) {
                out_values[i] = backing_->read_uint32(addresses[i]);
            }
        }
        return HOLOLINK_MODULE_OK;
    }

    hololink_module_status_t and_uint32(uint32_t address, uint32_t mask) override
    {
        if (!backing_->and_uint32(address, mask)) {
            return HOLOLINK_MODULE_NETWORK_ERROR;
        }
        return HOLOLINK_MODULE_OK;
    }

    hololink_module_status_t or_uint32(uint32_t address, uint32_t mask) override
    {
        if (!backing_->or_uint32(address, mask)) {
            return HOLOLINK_MODULE_NETWORK_ERROR;
        }
        return HOLOLINK_MODULE_OK;
    }

    hololink_module_status_t i2c_lock(
        std::unique_ptr<I2cLockV1>& out_lock) override
    {
        out_lock = std::make_unique<I2cLockNamedV1>(backing_, &backing_->i2c_lock());
        return HOLOLINK_MODULE_OK;
    }

    /* The materialized LegacyHololinkAccess, or null if configure()
     * has not yet run. Sibling impls in the same module (I2cV1,
     * data-channel / sequencer wrappers, ...) borrow it via the
     * Publisher's construct_service path; abstract callers go
     * through the V1 surface and never touch this. */
    std::shared_ptr<LegacyHololinkAccess> legacy_access() const { return backing_; }

protected:
    std::string roce_data_channel_instance_id(
        const EnumerationMetadata& md) override
    {
        return "serial=" + serial_number_ + ";data_channel="
            + std::to_string(md.get<int64_t>("data_channel", int64_t { 0 }));
    }

    std::string i2c_instance_id(uint32_t bus, uint32_t address) override
    {
        return "serial=" + serial_number_
            + ";bus=" + std::to_string(bus)
            + ";address=" + std::to_string(address);
    }

    std::string ptp_pps_output_instance_id() override
    {
        return "serial=" + serial_number_;
    }

    std::string null_vsync_instance_id() override
    {
        return "serial=" + serial_number_ + ";kind=null";
    }

private:
    // Reset-callback registry. on_reset() keeps callbacks here (keyed by
    // a monotonic id) and registers a single aggregating controller with
    // the backing Hololink; the RAII handle below erases its entry on
    // destruction so registrations don't outlive their owner.
    struct ResetRegistry {
        std::mutex mutex;
        std::map<uint64_t, std::function<void()>> callbacks;
        uint64_t next_id = 0;
    };

    class ResetRegistrationImpl : public HololinkInterfaceV1::ResetRegistration {
    public:
        ResetRegistrationImpl(std::weak_ptr<ResetRegistry> registry, uint64_t id)
            : registry_(std::move(registry))
            , id_(id)
        {
        }
        ~ResetRegistrationImpl() override
        {
            if (auto registry = registry_.lock()) {
                std::lock_guard<std::mutex> lock(registry->mutex);
                registry->callbacks.erase(id_);
            }
        }

    private:
        std::weak_ptr<ResetRegistry> registry_;
        uint64_t id_;
    };

    // Held by shared_ptr so the aggregating controller (captured into the
    // backing Hololink's append-only list) and the outstanding handles
    // (weak_ptr) can all reach it regardless of HololinkV1's lifetime.
    std::shared_ptr<ResetRegistry> reset_registry_ = std::make_shared<ResetRegistry>();
    std::shared_ptr<hololink::Hololink::ResetController> reset_aggregator_;
    // Serializes the one-time creation of reset_aggregator_ in on_reset().
    std::mutex reset_aggregator_mutex_;

    // Per-board services registered via register_associated(); device_lost()
    // invalidates each along with this Hololink.
    std::mutex associated_mutex_;
    std::vector<const void*> associated_;

    std::shared_ptr<LegacyHololinkAccess> backing_;
    // Sequence-number checking the backing Hololink was constructed with.
    // The scalar and batched-write legacy overloads read the backing's own
    // copy of this internally, but the batched block-read overload takes it
    // explicitly (its default is true), so read_uint32() passes this to keep
    // every control-plane request on the same setting. Kept here as the single
    // source of truth for the value configure() hands the backing constructor.
    bool sequence_number_checking_ = false;
    std::string serial_number_;
    // The metadata configure() was called with. Stashed for sibling
    // per-board services (I2c, Oscillator) to read off
    // enumeration_metadata() rather than carrying their own copy.
    EnumerationMetadata enumeration_metadata_;
    // The default per-board DataChannel, constructed by configure()
    // alongside this Hololink and bound to it via shared_from_this().
    // Shares the same metadata blob so the two cannot drift onto
    // inconsistent configurations. Null when configure-time metadata
    // lacked the 'data_plane' field needed to identify a channel.
    std::shared_ptr<DataChannelInterfaceV1> default_data_channel_;
};

} // namespace hololink::module::module_core

#endif // HOLOLINK_MODULE_CORE_HOLOLINK_DEFAULT_HPP
