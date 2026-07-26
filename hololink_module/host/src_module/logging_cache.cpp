/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "hololink/module/logging.hpp"

namespace hololink::module {

LoggingInterfaceV1* hsb_logger_cache = nullptr;

void set_hsb_logger_cache(LoggingInterfaceV1* logger)
{
    hsb_logger_cache = logger;
}

} // namespace hololink::module
