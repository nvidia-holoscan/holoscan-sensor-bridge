/**
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

#ifndef SRC_HOLOLINK_OPERATORS_LINUX_RECEIVER_LINUX_COE_RECEIVER
#define SRC_HOLOLINK_OPERATORS_LINUX_RECEIVER_LINUX_COE_RECEIVER

#include <atomic>
#include <semaphore.h>
#include <stdint.h>

#include <cuda.h>

#include <hololink/core/hololink.hpp>

namespace hololink::operators {

class LinuxCoeReceiverMetadata {
public:
    // Data accumulated just in this frame
    unsigned frame_packets_received = 0;
    unsigned frame_bytes_received = 0;
    unsigned received_frame_number = 0;
    uint64_t frame_start_s = 0;
    uint64_t frame_start_ns = 0;
    uint64_t frame_end_s = 0;
    uint64_t frame_end_ns = 0;
    int64_t received_s = 0;
    int64_t received_ns = 0;
    // Data received directly from HSB.
    Hololink::FrameMetadata frame_metadata;
    uint32_t frame_number = 0; // 32-bit extended version of the 16-bit frame_metadata.frame_number
};

class LinuxCoeReceiverDescriptor;

/**
 * Provides native-implementation support for
 * LinuxCoeReceiverOperator, pulling high-performance
 * operations out of python code and into native C++
 * code.  Given a GPU buffer, instantiate this class
 * and call run() on to go into a perpetual loop that
 * receives P1772B COE packets and writes those into
 * the given GPU buffer.  Another thread can call
 * `get_next_frame` to block until a complete data frame
 * arrives; use `set_frame_ready` to run your callback
 * in that background thread when a complete data frame
 * is received, which can tell your application that
 * `get_next_frame` won't need to block.  Don't call
 * `get_next_frame` from within that callback.
 */
class LinuxCoeReceiver {
public:
    LinuxCoeReceiver(CUdeviceptr cu_buffer,
        size_t cu_buffer_size,
        int socket,
        uint16_t channel);

    ~LinuxCoeReceiver();

    /**
     * Runs perpetually, looking for packets received
     * via socket (given in the constructor).  Call
     * close() to inspire this method to return.
     */
    void run();

    /**
     * Set a flag that will encourage run() to return.
     */
    void close();

    /**
     * Block until the next complete frame arrives.
     * @returns false if timeout_ms elapses before
     * the complete frame is observed.
     * @param metadata is updated with statistics
     * collected with the video frame.
     */
    bool get_next_frame(unsigned timeout_ms, LinuxCoeReceiverMetadata& metadata);

    /**
     * If the application schedules the call to get_next_frame after this
     * callback occurs, then get_next_frame won't block.
     */
    void set_frame_ready(std::function<void(const LinuxCoeReceiver&)> frame_ready);

protected:
    // Blocks execution until signal() is called;
    // @returns false if timeout_ms elapses before
    // signal is observed.
    bool wait(unsigned timeout_ms);

    // Pass a message to wait() telling it to wake up.
    void signal();

protected:
    /**
     * What buffer do we fill in GPU memory with our received
     * data?
     */
    CUdeviceptr cu_buffer_;
    size_t cu_buffer_size_;

    /**
     * Socket fd where our received packets can be found.
     */
    int socket_;

    /**
     * Ignore all received data not sent to this channel ID.
     */
    uint16_t channel_;

    /**
     * Works with ready_mutex_ and ready_condition_
     * to notify callers blocked in get_next_frame
     * to wake up.
     */
    std::atomic<bool> ready_;

    /** Flag tells the background thread to terminate. */
    std::atomic<bool> exit_;

    /**
     * Works with ready_condition_ to protect shared
     * resources between the background receiver thread
     * and the foreground calls to get_next_frame.
     */
    pthread_mutex_t ready_mutex_;

    /**
     * get_next_frame waits on this condition variable
     * until a frame is pointed to by `available_`.
     */
    pthread_cond_t ready_condition_;

    /** Points to the host memory where we cache our received data. */
    uint8_t* local_;

    /**
     * This points to the last received data buffer, or null,
     * if get_next_frame was called and we haven't received
     * our next complete data frame yet.
     */
    std::atomic<LinuxCoeReceiverDescriptor*> available_;

    /**
     * Which buffer is used by the application?  May be null.
     */
    LinuxCoeReceiverDescriptor* busy_;

    /**
     * This stream allows us to copy our receiver cache
     * memory into GPU without waiting for the GPU device
     * to be completely idle.
     */
    CUstream cu_stream_;

    /**
     * Callback the application layer can use to learn that
     * a call to get_next_frame won't block.  This is executed
     * within the receiver thread and is not at all synchronized
     * with the foreground thread.
     */
    std::function<void(const LinuxCoeReceiver&)> frame_ready_;

    /** Sign-extended frame_number value. */
    ExtendedCounter<uint32_t, uint16_t> frame_number_;
};

} // namespace hololink::operators

#endif /* SRC_HOLOLINK_OPERATORS_LINUX_RECEIVER_LINUX_COE_RECEIVER */
