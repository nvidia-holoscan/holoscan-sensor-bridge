// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "../../utils.hpp"
#include <stdexcept>

extern "C" void Error_Handler(const char* str)
{
    throw std::runtime_error(str);
}
