/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_OPERATORS_SENSOR_DEVICE_HPP
#define HOLOLINK_MODULE_OPERATORS_SENSOR_DEVICE_HPP

#include "hololink/module/status.h"

namespace hololink::module::operators {

/* A wrapper over one sensor driver (the camera, e.g. Imx274Cam),
 * created — already configured + armed — by SensorFactory::new_sensor for
 * a freshly (re)connected device. The application subclasses it per sensor
 * (e.g. Imx274SensorDevice); a reconnection replaces it with a fresh one. */
class SensorDevice {
public:
    virtual ~SensorDevice() = default;

    /* Disarm the sensor. Called before the network receiver tears down. */
    virtual hololink_module_status_t stop_sensor() = 0;
};

} // namespace hololink::module::operators

#endif // HOLOLINK_MODULE_OPERATORS_SENSOR_DEVICE_HPP
