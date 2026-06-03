/**
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
 *
 * See README.md for detailed information.
 */

#ifndef SRC_HOLOLINK_CORE_NAMED_LOCK_HPP
#define SRC_HOLOLINK_CORE_NAMED_LOCK_HPP

#include <mutex>
#include <string>

namespace hololink {

/**
 * Used to guarantee serialized access to I2C or SPI controllers.  The FPGA
 * only has a single I2C controller-- what looks like independent instances
 * are really just pin-muxed outputs from a single I2C controller block within
 * the device-- the same is true for SPI.
 *
 * Recursive: the same thread may lock multiple times; it must unlock the same
 * number of times. Other threads and processes block until the lock is
 * completely released (all recursive levels unlocked).
 */
class NamedLock {
public:
    /** Constructs a lock using the lockf call to access a named
     * semaphore with the given name.
     */
    NamedLock(std::string name);
    ~NamedLock() noexcept(false);

    /**
     * Blocks until no other thread/process owns this lock; then takes it.
     * The same thread may call lock() again (recursive); it must call
     * unlock() the same number of times before others can acquire.
     */
    void lock();

    /**
     * Unlocks one level of this lock. When the calling thread has unlocked
     * as many times as it locked, another thread or process blocked in
     * lock() may proceed.
     */
    void unlock();

protected:
    int fd_;
    std::recursive_mutex process_mutex_;
};

} // namespace hololink

#endif /* SRC_HOLOLINK_CORE_NAMED_LOCK_HPP */
