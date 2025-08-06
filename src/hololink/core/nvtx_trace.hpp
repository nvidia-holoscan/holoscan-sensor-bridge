/**
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOLINK_NVTX_TRACE_H
#define HOLOLINK_NVTX_TRACE_H 1

#include <nvtx3/nvToolsExt.h>
#include <pthread.h>

namespace hololink::core {

class NvtxTrace {
public:
    explicit NvtxTrace(const char* s) { nvtxRangePushA(s); }

    ~NvtxTrace() { nvtxRangePop(); }

    static inline void setThreadName(char const* threadName)
    {
        pthread_setname_np(pthread_self(), threadName);
    }

    static inline void event_u64(char const* message, uint64_t datum)
    {
        nvtxEventAttributes_t event = { 0 };
        event.version = NVTX_VERSION;
        event.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        event.messageType = NVTX_MESSAGE_TYPE_ASCII;
        event.message.ascii = message;
        event.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
        event.payload.ullValue = datum;
        nvtxMarkEx(&event);
    }

    typedef nvtxRangeId_t RangeId;

    static inline RangeId range_start(char const* msg)
    {
        RangeId r = nvtxRangeStartA(msg);
        return r;
    }

    static inline void range_end(RangeId range_id) { nvtxRangeEnd(range_id); }
};

} // namespace hololink::core

#endif /* HOLOLINK_NVTX_TRACE_H */
