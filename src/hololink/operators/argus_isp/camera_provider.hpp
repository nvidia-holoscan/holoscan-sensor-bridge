/**
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#ifndef SRC_HOLOLINK_OPERATORS_ARGUS_ISP_CAMERA_PROVIDER
#define SRC_HOLOLINK_OPERATORS_ARGUS_ISP_CAMERA_PROVIDER

#include <Argus/Argus.h>
#undef Success

#include <memory>
#include <mutex>

namespace hololink::operators {
class CameraProvider {
public:
    CameraProvider();
    ~CameraProvider();
    std::shared_ptr<Argus::CameraProvider> get_camera_provider();

private:
    std::shared_ptr<Argus::CameraProvider> camera_provider_;
    std::mutex mutex_;
};
} // namespace hololink::operators

#endif // SRC_HOLOLINK_OPERATORS_ARGUS_ISP_CAMERA_PROVIDER
