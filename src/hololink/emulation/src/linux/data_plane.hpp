/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef LINUX_DATA_PLANE_HPP
#define LINUX_DATA_PLANE_HPP

#include "../../data_plane.hpp"
#include "net.hpp"
#include <cstdlib>
#include <memory>
#include <mutex>
#include <thread>
#include <time.h>

namespace hololink::emulation {

class BootpThread;

/**
 * @brief Linux-specific DataPlane context extension.
 *
 * `base` MUST be the first member: the DataPlane base class stores its `data_plane_ctxt_`
 * as a `DataPlaneCtxt*` pointing into this `base`. Standard-layout C++ guarantees
 * `&this->base == reinterpret_cast<DataPlaneCtxt*>(this)`, so linux/*.cpp downcasts via
 * `reinterpret_cast<LinuxDataPlaneCtxt*>(data_plane_ctxt_.get())`.
 */
struct LinuxDataPlaneCtxt {
    DataPlaneCtxt base;
    // Serializes transmitter metadata updates with sends. Held for the duration of
    // each DataPlane::send().
    std::mutex* metadata_mutex; // we get this from the HSBEmulator ControlThread mutex
    // Serializes read/write access to base.running. DataPlane::start / stop / is_running
    // (linux variants in linux/data_plane.cpp) acquire this around their base.running
    // access. BootpThread reaches base.running indirectly via DataPlane::is_running().
    std::mutex running_mutex;
    // BootP broadcast loop thread + its socket. Created in the DataPlane constructor,
    // deleted in ~LinuxDataPlaneCtxt (defined in linux/data_plane.cpp where BootpThread
    // is complete). Raw pointer instead of std::unique_ptr<BootpThread> because that
    // would force every TU embedding LinuxDataPlaneCtxt (e.g. COECtxt, RoCEv2Ctxt) to
    // see the complete BootpThread type for the implicit default_delete instantiation.
    BootpThread* bootp_thread { nullptr };
    // Bounce buffer for DLTensor payloads that live in CUDA device memory.
    // DataPlane::send(DLTensor&) cudaMemcpy's the device buffer here before forwarding
    // to the transmitter. Owned per DataPlane (one buffer per active stream) so the
    // cudaMemcpy on the hot path doesn't have to allocate. realloc'd on size growth and
    // free()'d in ~LinuxDataPlaneCtxt — see linux/data_plane.cpp. Shared by COE and
    // RoCEv2 because the bounce is platform-side, not transport-side. STM32 has no GPU
    // memory and therefore no equivalent fields.
    uint8_t* double_buffer { nullptr };
    int64_t double_buffer_size { 0 };

    // Body in linux/data_plane.cpp because the implicit destruction of bootp_thread
    // (unique_ptr<BootpThread>) needs the complete BootpThread type, which is defined
    // there. Frees the realloc-grown double_buffer too.
    ~LinuxDataPlaneCtxt();
};

}

#endif
