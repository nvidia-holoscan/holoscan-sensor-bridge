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

#include "camera_provider.hpp"

namespace hololink::operators {

// need to define our own delete functor to pass to std::shared_ptr because
// CameraProvider has a protected/deleted destructor

void CameraProviderDeleter(Argus::CameraProvider* p)
{
    p->destroy();
}

CameraProvider::CameraProvider()
    : camera_provider_(nullptr)
{
}

CameraProvider::~CameraProvider() { }

std::shared_ptr<Argus::CameraProvider> CameraProvider::get_camera_provider()
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!camera_provider_) {
        Argus::Status status = Argus::STATUS_OK;
        // Create the CameraProvider object to get the core interface.
        camera_provider_ = std::shared_ptr<Argus::CameraProvider>(
            Argus::CameraProvider::create(&status), CameraProviderDeleter);
        if (status != Argus::STATUS_OK) {
            return NULL;
        }
    }
    return camera_provider_;
}

} // namespace hololink::operators
