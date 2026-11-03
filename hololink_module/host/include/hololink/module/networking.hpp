/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_NETWORKING_HPP
#define HOLOLINK_MODULE_NETWORKING_HPP

#include <algorithm>
#include <array>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring> // strerror

#include <cerrno> // errno
#include <sys/socket.h>

#include "logging.hpp"

/* Adapter-owned networking types and constants. Public counterpart of the
 * internal hololink_module/host/src/networking.hpp, so applications (e.g. the
 * FUSA example player) name no legacy hololink::core type or value. */

namespace hololink::module {

/**
 * @brief A 48-bit Ethernet MAC (medium access control) address.
 *
 * The six octets are stored in the order they are written in the usual
 * text form, so `aa:bb:cc:dd:ee:ff` is `{0xaa, 0xbb, 0xcc, 0xdd, 0xee,
 * 0xff}`: element 0 is the first octet written (the one carrying the
 * multicast and locally-administered flags). This is also the order the
 * octets appear on the network.
 */
using MacAddress = std::array<uint8_t, 6>;

/// Default Ethernet MTU, in bytes. Adapter-owned mirror of the legacy
/// hololink::core value so callers name no legacy hololink::core constant.
constexpr uint32_t DEFAULT_MTU = 1500;

/* Grow a datagram socket's kernel receive buffer to hold at least
 * `bytes`, which callers size to one video frame.
 *
 * The software receive path drains the socket with one recv() per ~1.4kB
 * packet on a single thread, so the kernel buffer is what absorbs every
 * stall in that thread (a scheduling preemption, the per-frame page reset,
 * the CUDA event wait in LinuxReceiver::signal). At the default ~208kB it
 * absorbs only ~0.3ms of a 4K60 stream — less than one preemption — and
 * overflow drops whole runs of packets. Losing the frame-end packet (the
 * only one that completes a frame) means that frame is never delivered at
 * all, so sustained overflow reads downstream as a stalled pipeline rather
 * than as degraded frames. One frame of buffering (~10MB at 4K) covers a
 * full frame period.
 *
 * Best effort: a kernel that clamps the request (rmem_max) leaves the
 * socket usable, just less tolerant, so this warns rather than throws.
 * Mirrors hololink::operators::LinuxReceiverOp::check_buffer_size. New for the
 * module receivers: HSDK4's never sized SO_RCVBUF, so expect a warning on hosts
 * whose rmem_max is below a frame.
 *
 * Inline so loaded modules (which include this header but don't link the
 * module host library) get the symbol without an extra link dep. */
inline void size_receive_buffer(int socket_fd, size_t bytes)
{
    int receiver_buffer_size = 0;
    socklen_t opt_len = sizeof(receiver_buffer_size);
    if (getsockopt(socket_fd, SOL_SOCKET, SO_RCVBUF, &receiver_buffer_size,
            &opt_len)
        < 0) {
        HSB_LOG_WARN("Failed to read the socket receive buffer size, errno={}: \"{}\"",
            errno, strerror(errno));
        return;
    }
    // Linux reports twice what it accepted (the extra half is bookkeeping
    // overhead), so the usable capacity is half the reported value.
    if (static_cast<size_t>(receiver_buffer_size / 2) >= bytes) {
        return;
    }
    // Round the request up to a 64k boundary.
    constexpr size_t BUFFER_ALIGNMENT = 0x10000;
    const size_t request_size = (bytes + BUFFER_ALIGNMENT - 1) & ~(BUFFER_ALIGNMENT - 1);
    // SO_RCVBUF takes an int.
    const int request = static_cast<int>(std::min<size_t>(request_size, INT_MAX));
    if (setsockopt(socket_fd, SOL_SOCKET, SO_RCVBUF, &request, sizeof(request)) < 0) {
        HSB_LOG_WARN("Failed to set the socket receive buffer size, errno={}: \"{}\"",
            errno, strerror(errno));
        return;
    }
    if (getsockopt(socket_fd, SOL_SOCKET, SO_RCVBUF, &receiver_buffer_size,
            &opt_len)
        < 0) {
        HSB_LOG_WARN("Failed to read the socket receive buffer size after setting it, "
                     "errno={}: \"{}\"",
            errno, strerror(errno));
        return;
    }
    HSB_LOG_DEBUG("receiver buffer size={}", receiver_buffer_size);
    // The kernel reports twice what it accepted (bookkeeping overhead), so
    // compare the usable half against the request.
    if (static_cast<size_t>(receiver_buffer_size / 2) < bytes) {
        HSB_LOG_WARN("Kernel receiver buffer size is too small; frames will be dropped.");
        HSB_LOG_WARN("Resolve this with \"echo {} | sudo tee /proc/sys/net/core/rmem_max\"",
            request_size);
    }
}

} // namespace hololink::module

#endif // HOLOLINK_MODULE_NETWORKING_HPP
