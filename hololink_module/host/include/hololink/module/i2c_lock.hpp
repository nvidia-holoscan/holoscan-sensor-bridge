/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_I2C_LOCK_HPP
#define HOLOLINK_MODULE_I2C_LOCK_HPP

namespace hololink::module {

/* Mutex-like handle for the per-board I2C bus, satisfying the
 * BasicLockable + Lockable concepts so callers can drive it through
 * std::lock_guard / std::unique_lock / std::scoped_lock. Returned
 * unlocked from HololinkInterfaceV1::i2c_lock; callers acquire and
 * release through the standard primitives.
 *
 * Not a Service — a fresh handle comes back per call. */
class I2cLockV1 {
public:
    virtual ~I2cLockV1() = default;
    virtual void lock() = 0;
    virtual void unlock() = 0;
    virtual bool try_lock() = 0;
};

} // namespace hololink::module

#endif // HOLOLINK_MODULE_I2C_LOCK_HPP
