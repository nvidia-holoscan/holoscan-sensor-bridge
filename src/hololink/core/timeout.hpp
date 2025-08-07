/**
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef SRC_HOLOLINK_TIMEOUT
#define SRC_HOLOLINK_TIMEOUT

#include <chrono>
#include <memory>

namespace hololink {

/**
 * @brief This class handle timeouts and retries for bus transactions.
 */
class Timeout {
public:
    /**
     * @brief Construct a new Timeout object, the start time is set to the current time.
     *
     * The retry time should be less than the timeout, otherwise there won't be any retries at all.
     *
     * @param timeout_s duration in seconds after the timepoint should time out
     * @param retry_s duration in seconds before retry, a value of 0.f means no retries
     */
    explicit Timeout(float timeout_s, float retry_s = 0.f);
    Timeout() = delete;

    /**
     * @brief Get the default Timeout object
     *
     * @param timeout If set, use this Timeout object instead of creating one
     * @returns the given timeout unless it's nullptr in which case it returns a default default
     * timeout instance.
     */
    static std::shared_ptr<Timeout> default_timeout(
        const std::shared_ptr<Timeout>& timeout = std::shared_ptr<Timeout>());

    /**
     * @brief Get the Timeout object for I2C transactions
     *
     * @param timeout If set, use this Timeout object instead of creating one
     * @returns the given timeout unless it's nullptr in which case it returns a default I2C timeout
     * instance.
     */
    static std::shared_ptr<Timeout> i2c_timeout(
        const std::shared_ptr<Timeout>& timeout = std::shared_ptr<Timeout>());

    /**
     * @brief Get the Timeout object for SPI transactions
     *
     * @param timeout If set, use this Timeout object instead of creating one
     * @returns the given timeout unless it's nullptr in which case it returns a default SPI timeout
     * instance.
     */
    static std::shared_ptr<Timeout> spi_timeout(
        const std::shared_ptr<Timeout>& timeout = std::shared_ptr<Timeout>());

    /**
     * @returns the value (in fractional seconds) of a monotonic clock.
     *
     * @note the epoch of this time value is undefined
     */
    static double now_s();

    /**
     * @returns the value (in nano seconds) of a monotonic clock.
     *
     * @note the epoch of this time value is undefined
     */
    static int64_t now_ns();

    /**
     * @return true if the timeout expired
     */
    bool expired() const;

    /**
     * @returns the time in seconds until the timeout triggers. Can be negative if the timeout had
     * already been triggered.
     */
    float trigger_s() const;

    /**
     * @brief Retry.
     *
     * @returns true if we should retry the transaction. This only happens if retry_s() has been
     * called and we haven't reached the timeout given in the constructor.
     */
    bool retry();

private:
    /// The clock we use
    using Clock = std::chrono::steady_clock;

    const Clock::time_point start_;
    const Clock::duration timeout_;
    const Clock::time_point expiry_;
    const Clock::duration retry_;
    Clock::time_point deadline_;
};

} // namespace hololink

#endif /* SRC_HOLOLINK_TIMEOUT */
