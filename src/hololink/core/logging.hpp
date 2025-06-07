/**
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef SRC_HOLOLINK_LOGGING
#define SRC_HOLOLINK_LOGGING

namespace hololink::logging {

// Supported logging levels.
typedef enum {
    HSB_LOG_LEVEL_TRACE = 10,
    HSB_LOG_LEVEL_DEBUG = 20,
    HSB_LOG_LEVEL_INFO = 30,
    HSB_LOG_LEVEL_WARN = 40,
    HSB_LOG_LEVEL_ERROR = 50,

    // Special value here allows the logging subsystem to
    // initialize itself while still keeping the test for
    // logging levels in the application code (so that the
    // compiler may be able to skip a lot of fmt::format...
    // variable initialization.  This value must be lower than
    // all the actual levels.
    HSB_LOG_LEVEL_INVALID = 0,
} HsbLogLevel;

// Callback with logging data.
typedef void (*HsbLogger)(char const* file, unsigned line, const char* function, HsbLogLevel level, const char* msg);

// Controls which logging calls actually result in calls to hsb_logger.
extern HsbLogLevel hsb_log_level;

// By default this logger writes to stderr; set a new value here
// to visit the data with your own callback.
extern HsbLogger hsb_logger;

} // namespace hololink::logging

#endif /* SRC_HOLOLINK_LOGGING */
