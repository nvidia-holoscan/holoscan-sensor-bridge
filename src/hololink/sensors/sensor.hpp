/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef SENSORS_SENSOR_HPP
#define SENSORS_SENSOR_HPP

#include <memory>
#include <string>
#include <unordered_set>

namespace hololink::sensors {

/**
 * @brief Abstract base class defining the interface for all sensor implementations.
 *
 * The Sensor class provides a common framework for hardware sensor integration.
 * Derived classes must implement the core lifecycle methods:
 * - start(): Begin sensor data acquisition/streaming
 * - stop(): Cease sensor operations and release active resources
 *
 * Typical usage flow:
 * 1. Create concrete sensor instance
 * 2. Call start() to begin sensor operations
 * 3. Use sensor-specific methods to interact with data
 * 4. Call stop() when done
 */
class Sensor {
public:
    Sensor() = default;
    virtual ~Sensor() = default;

    // Start sensor operation
    virtual void start() = 0;

    // Stop sensor operation
    virtual void stop() = 0;

protected:
    /// The unique identifier for the sensor.
    /// @note This is not the sensor's serial number, but a unique identifier for the sensor which can be used to identify the sensor when
    /// the sensor driver object is instantiated from a factory object (e.g. CameraFactory).
    std::string sensor_id_;
};

} // namespace hololink::sensors

#endif /* SENSORS_SENSOR_HPP */
