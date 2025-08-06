/*
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
 */

#include "timeout.hpp"

namespace hololink {

/// default timeout in seconds
constexpr static float DEFAULT_TIMEOUT_S = 0.5f;

/// default retry time in seconds
constexpr static float DEFAULT_RETRY_S = 0.1f;

Timeout::Timeout(float timeout_s, float retry_s)
    : start_(Clock::now())
    , timeout_(std::chrono::duration_cast<Clock::duration>(std::chrono::duration<float>(timeout_s)))
    , expiry_(start_ + timeout_)
    , retry_(std::chrono::duration_cast<Clock::duration>(std::chrono::duration<float>(retry_s)))
    , deadline_(start_ + ((retry_s == 0.f) ? timeout_ : retry_))
{
}

/*static*/ std::shared_ptr<Timeout> Timeout::default_timeout(
    const std::shared_ptr<Timeout>& timeout)
{
    if (!timeout) {
        return std::make_shared<Timeout>(DEFAULT_TIMEOUT_S, DEFAULT_RETRY_S);
    }
    return timeout;
}

/*static*/ std::shared_ptr<Timeout> Timeout::i2c_timeout(const std::shared_ptr<Timeout>& timeout)
{
    if (!timeout) {
        return std::make_shared<Timeout>(DEFAULT_TIMEOUT_S, DEFAULT_RETRY_S);
    }
    return timeout;
}

/*static*/ std::shared_ptr<Timeout> Timeout::spi_timeout(const std::shared_ptr<Timeout>& timeout)
{
    if (!timeout) {
        return std::make_shared<Timeout>(DEFAULT_TIMEOUT_S, DEFAULT_RETRY_S);
    }
    return timeout;
}

/*static*/ double Timeout::now_s()
{
    return std::chrono::duration_cast<std::chrono::duration<double>>(
        Clock::now().time_since_epoch())
        .count();
}

/*static*/ int64_t Timeout::now_ns()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        Clock::now().time_since_epoch())
        .count();
}

bool Timeout::expired() const { return Clock::now() > deadline_; }

float Timeout::trigger_s() const
{
    return std::chrono::duration_cast<std::chrono::duration<float>>(deadline_ - Clock::now())
        .count();
}

bool Timeout::retry()
{
    if (retry_ == Clock::duration::zero()) {
        return false;
    }
    const Clock::time_point now = Clock::now();
    if (now >= expiry_) {
        return false;
    }
    if (deadline_ <= now) {
        deadline_ += retry_;
    }
    return true;
}

} // namespace hololink
