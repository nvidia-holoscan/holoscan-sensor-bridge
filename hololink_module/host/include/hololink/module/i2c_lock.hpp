/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_I2C_LOCK_HPP
#define HOLOLINK_MODULE_I2C_LOCK_HPP

namespace hololink::module {

/// @cond INCLUDE_INTERNAL_DOCUMENTATION

/**
 * Mutex support for I2C buses; use HololinkInterfaceV1::i2c_lock
 * to fetch an unlocked instance of this; use that with
 * std::unique_lock, std::lock_guard, or std::scoped_lock to
 * guarantee single-threaded access to the I2C device.
 *
 * Note that this control doesn't protect accesses due to
 * background processes like those triggered by
 * SequencerInterfaceV1; when using sequencers, it's best
 * to /only/ use sequencers-- which are serialized by
 * the HSB IP executor.
 */
class I2cLockV1 {
public:
    static constexpr const char* release_version = "2.7.0";
    virtual ~I2cLockV1() = default;
    virtual void lock() = 0;
    virtual void unlock() = 0;
    virtual bool try_lock() = 0;
};
/** @endcond */ // INCLUDE_INTERNAL_DOCUMENTATION

} // namespace hololink::module

#endif // HOLOLINK_MODULE_I2C_LOCK_HPP
