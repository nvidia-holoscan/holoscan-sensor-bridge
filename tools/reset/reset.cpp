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

#include "reset.hpp"

#include <hololink/core/data_channel.hpp>
#include <iostream>

namespace hololink::tools {

void reset(const std::string& hololink_ip)
{
    hololink::Metadata metadata = hololink::Enumerator::find_channel(hololink_ip);
    hololink::DataChannel channel(metadata);
    std::shared_ptr<hololink::Hololink> hololink = channel.hololink();
    hololink->start();
    hololink->reset();
}

} // namespace hololink::tools
