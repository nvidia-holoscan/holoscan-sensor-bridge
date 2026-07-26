/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_CORE_I2C_LOCK_DEFAULT_HPP
#define HOLOLINK_MODULE_CORE_I2C_LOCK_DEFAULT_HPP

#include <memory>
#include <mutex>

#include "hololink/module/i2c_lock.hpp"

namespace hololink::module::module_core {

/* Default I2cLockV1 backed by a std::mutex. The mutex lives in the
 * caller-supplied shared_ptr<std::mutex>; this lets the publisher
 * keep one mutex per board and hand fresh I2cLockImplV1 instances
 * out to each i2c_lock() caller — every handle locks the same
 * underlying mutex. */
class I2cLockImplV1 : public I2cLockV1 {
public:
    explicit I2cLockImplV1(std::shared_ptr<std::mutex> mutex)
        : mutex_(std::move(mutex))
    {
    }

    void lock() override { mutex_->lock(); }
    void unlock() override { mutex_->unlock(); }
    bool try_lock() override { return mutex_->try_lock(); }

private:
    std::shared_ptr<std::mutex> mutex_;
};

} // namespace hololink::module::module_core

#endif // HOLOLINK_MODULE_CORE_I2C_LOCK_DEFAULT_HPP
