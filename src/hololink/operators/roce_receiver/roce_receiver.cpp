/**
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
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

#include "roce_receiver.hpp"

#include <arpa/inet.h>
#include <chrono>
#include <fcntl.h>
#include <poll.h>
#include <stdio.h>
#include <unistd.h>

#include <hololink/native/deserializer.hpp>
#include <hololink/native/nvtx_trace.hpp>

#define TRACE(fmt, ...) /* ignored */
#define DEBUG(fmt...) fprintf(stderr, "DEBUG -- " fmt)
#define ERROR(fmt...) fprintf(stderr, "ERROR -- " fmt)

#undef PERIODIC_STATUS

#define NUM_OF(x) (sizeof(x) / sizeof(x[0]))

namespace hololink::operators {

RoceReceiver::RoceReceiver(
    const char* ibv_name,
    unsigned ibv_port,
    CUdeviceptr cu_buffer,
    size_t cu_buffer_size,
    const char* peer_ip)
    : ibv_name_(strdup(ibv_name))
    , ibv_port_(ibv_port)
    , cu_buffer_(cu_buffer)
    , cu_buffer_size_(cu_buffer_size)
    , peer_ip_(strdup(peer_ip))
    , ib_qp_(NULL)
    , ib_mr_(NULL)
    , ib_cq_(NULL)
    , ib_pd_(NULL)
    , ib_context_(NULL)
    , ready_(false)
    , ready_mutex_(PTHREAD_MUTEX_INITIALIZER)
    , ready_condition_(PTHREAD_COND_INITIALIZER)
    , done_(false)
    , frame_number_(0)
    , rx_write_requests_fd_(-1)
    , rx_write_requests_(0)
    , frame_end_ {}
    , imm_data_(0)
    , event_time_ {}
    , received_ns_(0)
{
    DEBUG("cu_buffer=0x%llX cu_buffer_size=%u\n",
        (unsigned long long)cu_buffer, (unsigned)cu_buffer_size);

    int r = pthread_mutex_init(&ready_mutex_, NULL);
    if (r != 0) {
        ERROR("pthread_mutex_init failed, r=%d.\n", r);
    }
    pthread_condattr_t pthread_condattr;
    pthread_condattr_init(&pthread_condattr);
    pthread_condattr_setclock(&pthread_condattr, CLOCK_MONOTONIC);
    r = pthread_cond_init(&ready_condition_, &pthread_condattr);
    if (r != 0) {
        ERROR("pthread_cond_init failed, r=%d.\n", r);
    }
    int pipe_fds[2] = { -1, -1 };
    // If these aren't updated, we'll get an error when we try to read, which is good.
    r = pipe(pipe_fds);
    if (r != 0) {
        ERROR("Pipe failed.\n");
    }
    control_r_ = pipe_fds[0];
    control_w_ = pipe_fds[1];

    char rx_write_requests_filename[8192];
    int written = snprintf(rx_write_requests_filename, sizeof(rx_write_requests_filename),
        "/sys/class/infiniband/%s/ports/%d/hw_counters/rx_write_requests",
        ibv_name_, ibv_port_);
    if (written < 0) {
        ERROR("Error writing to rx_write_requests_filename.\n");
    } else if (((size_t)written) >= sizeof(rx_write_requests_filename)) {
        ERROR("Buffer isn't large enough to compute rx_write_requests filename.\n");
    } else {
        rx_write_requests_fd_ = open(rx_write_requests_filename, O_RDONLY);
    }
    if (rx_write_requests_fd_ < 0) {
        ERROR("Unable to fetch rx_write_requests.\n");
    }
}

RoceReceiver::~RoceReceiver()
{
    pthread_cond_destroy(&ready_condition_);
    pthread_mutex_destroy(&ready_mutex_);
    ::close(rx_write_requests_fd_); // we ignore an error here if fd==-1
}

bool RoceReceiver::start()
{
    DEBUG("Starting.\n");

    // ibv calls seem to have trouble with
    // reentrancy.  No problem; since we're only
    // at startup, run just one at a time.  This
    // only affects multithreaded schedulers like
    // EventBasedScheduler.
    std::lock_guard lock(get_lock());

    // Find the IB controller
    int num_devices = 0;
    struct ibv_device** ib_devices = ibv_get_device_list(&num_devices);
    if (!ib_devices) {
        ERROR("ibv_get_device_list failed; errno=%d.\n", (int)errno);
        return false;
    }
    if (num_devices < 0) {
        ERROR("ibv_get_device_list set unexpected value for num_devices=%d.\n", num_devices);
        return false;
    }
    struct ibv_device* ib_device = NULL;
    for (unsigned i = 0; i < (unsigned)num_devices; i++) {
        const char* device_name = ibv_get_device_name(ib_devices[i]);
        DEBUG("ibv_get_device_list[%d]=%s.\n", i, device_name);
        if (strcmp(device_name, ibv_name_) != 0) {
            continue;
        }
        ib_device = ib_devices[i];
        break;
    }
    if (ib_device == NULL) {
        ERROR("ibv_get_device_list didnt find a device named \"%s\".\n", ibv_name_);
        ibv_free_device_list(ib_devices);
        return false;
    }

    // Open the IB device
    ib_context_ = ibv_open_device(ib_device);
    if (!ib_context_) {
        ERROR("ibv_open_device failed, errno=%d.\n", (int)errno);
        return false;
    }
    ibv_free_device_list(ib_devices); // Note that "ib_device" is invalid after this.
    ib_device = NULL;
    ib_devices = NULL;

    // Configure for nonblocking reads, this way we can poll
    // for events
    int flags = fcntl(ib_context_->async_fd, F_GETFL);
    int r = fcntl(ib_context_->async_fd, F_SETFL, flags | O_NONBLOCK);
    if (r < 0) {
        ERROR("Can't configure async_fd=%d with O_NONBLOCK, errno=%d.\n", (int)ib_context_->async_fd, (int)errno);
        return false;
    }

    //
    struct ibv_device_attr ib_device_attr = { 0 }; // C fills the rest with 0s
    if (ibv_query_device(ib_context_, &ib_device_attr)) {
        ERROR("ibv_query_device failed, errno=%d.\n", (int)errno);
        free_ib_resources();
        return false;
    }

    struct ibv_port_attr ib_port_attr = { .flags = 0 }; // C fills the rest with 0s
    if (ibv_query_port(ib_context_, ibv_port_, &ib_port_attr)) {
        ERROR("ibv_query_port failed, errno=%d.\n", (int)errno);
        free_ib_resources();
        return false;
    }

    // Fetch the GID
    bool ok = false;
    struct ibv_gid_entry ib_gid_entry = { 0 }; // C fills the rest with 0s
    int gid_index = 0;
    for (gid_index = 0; 1; gid_index++) {
        uint32_t flags = 0;
        r = ibv_query_gid_ex(ib_context_, ibv_port_, gid_index, &ib_gid_entry, flags);
        if (r && (errno != ENODATA)) {
            break;
        }

        struct ibv_gid_entry* u = &(ib_gid_entry);
        DEBUG("gid_index=%u gid_entry(gid_index=%u port_num=%u gid_type=%u ndev_ifindex=%d subnet_prefix=%d interface_id=0x%X)\n", (unsigned)gid_index, (unsigned)u->gid_index, (unsigned)u->port_num, (unsigned)u->gid_type, (unsigned)u->ndev_ifindex, (unsigned)u->gid.global.subnet_prefix, (unsigned)u->gid.global.interface_id);

        if (ib_gid_entry.gid_type != IBV_GID_TYPE_ROCE_V2) {
            continue;
        }
        if (ib_gid_entry.gid.global.subnet_prefix != 0) {
            continue;
        }
        if ((ib_gid_entry.gid.global.interface_id & 0xFFFFFFFF) != 0xFFFF0000) {
            continue;
        }
        ok = true;
        break;
    }
    if (!ok) {
        ERROR("Cannot find GID for IBV_GID_TYPE_ROCE_V2.\n");
        free_ib_resources();
        return false;
    }

    // Create a protection domain
    ib_pd_ = ibv_alloc_pd(ib_context_);
    if (ib_pd_ == NULL) {
        ERROR("Cannot allocate a protection domain, errno=%d.\n", (int)errno);
        free_ib_resources();
        return false;
    }

    // Create a completion channel.
    ib_completion_channel_ = ibv_create_comp_channel(ib_context_);
    if (ib_completion_channel_ == NULL) {
        ERROR("Cannot create a completion channel.\n");
        free_ib_resources();
        return false;
    }
    // Configure for nonblocking reads, this way we can poll
    // for events
    flags = fcntl(ib_completion_channel_->fd, F_GETFL);
    r = fcntl(ib_completion_channel_->fd, F_SETFL, flags | O_NONBLOCK);
    if (r < 0) {
        ERROR("Can't configure fd=%d with O_NONBLOCK, errno=%d.\n", (int)ib_completion_channel_->fd, (int)errno);
        return false;
    }

    // Create a completion queue
    unsigned completion_queue_size = 10;
    ib_cq_ = ibv_create_cq(ib_context_, completion_queue_size, NULL, ib_completion_channel_, 0);
    if (ib_cq_ == NULL) {
        ERROR("Cannot create a completion queue, errno=%d.\n", (int)errno);
        free_ib_resources();
        return false;
    }

    // Provide access to the frame buffer
    int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
    ib_mr_ = ibv_reg_mr(ib_pd_, (void*)cu_buffer_, cu_buffer_size_, access);
    if (ib_mr_ == NULL) {
        ERROR("Cannot register memory region p=0x%llX size=%u, errno=%d.\n", (unsigned long long)cu_buffer_, (unsigned)cu_buffer_size_, (int)errno);
        free_ib_resources();
        return false;
    }

    rkey_ = ib_mr_->rkey;

    // Set up the queue pair
    struct ibv_qp_init_attr ib_qp_init_attr = {
        .send_cq = ib_cq_,
        .recv_cq = ib_cq_,
        .cap = {
            .max_send_wr = 2048,
            .max_recv_wr = 2048,
            .max_send_sge = 1,
            .max_recv_sge = 1,
        },
        .qp_type = IBV_QPT_UC,
        .sq_sig_all = 0,
    }; // C++ sets the rest of the values to 0
    ib_qp_ = ibv_create_qp(ib_pd_, &ib_qp_init_attr);
    if (ib_qp_ == NULL) {
        ERROR("Cannot create queue pair, errno=%d.\n", (int)errno);
        free_ib_resources();
        return false;
    }

    qp_number_ = ib_qp_->qp_num;

    // qp is currently RST; go to INIT
    struct ibv_qp_attr ib_qp_attr = {
        .qp_state = IBV_QPS_INIT,
        .qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE,
        .pkey_index = 0,
        .port_num = (uint8_t)ibv_port_,
    }; // C sets the rest to 0s
    flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
    if (ibv_modify_qp(ib_qp_, &ib_qp_attr, flags)) {
        ERROR("Cannot modify queue pair to IBV_QPS_INIT, errno=%d.\n", (int)errno);
        free_ib_resources();
        return false;
    }

    // Set up the list of work requests
    for (unsigned i = 0x1000; i < 0x1100; i++) {
        struct ibv_recv_wr ib_wr = {
            .wr_id = i,
            .next = NULL,
        }; // C fills the rest with 0s
        struct ibv_recv_wr* bad_wr = NULL;
        if (ibv_post_recv(ib_qp_, &ib_wr, &bad_wr)) {
            ERROR("Cannot post the receiver work list, errno=%d.\n", (int)errno);
            free_ib_resources();
            return false;
        }
    }

    // qp is currently INIT; go to RTR
    unsigned long client_ip = 0;
    if (inet_pton(AF_INET, peer_ip_, &client_ip) != 1) {
        ERROR("Unable to convert \"%s\" to an IP address.\n", peer_ip_);
        free_ib_resources();
        return false;
    }
    // client_ip is in network-byte-order
    uint64_t client_interface_id = client_ip;
    client_interface_id <<= 32;
    client_interface_id |= 0xFFFF0000;
    union ibv_gid remote_gid = {
        .global = {
            .subnet_prefix = 0,
            .interface_id = client_interface_id,
        },
    };
    ib_qp_attr = {
        .qp_state = IBV_QPS_RTR,
        .path_mtu = IBV_MTU_4096,
        .rq_psn = 0,
        .dest_qp_num = 0,
        .ah_attr = {
            .grh = {
                .dgid = remote_gid,
                .sgid_index = (uint8_t)gid_index,
                .hop_limit = 0xFF,
            },
            .dlid = 0,
            .sl = 0,
            .src_path_bits = 0,
            .is_global = 1,
            .port_num = (uint8_t)ibv_port_,
        },
    }; // C sets the rest to 0s
    flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN;
    // We see occasional errno=110 (ETIMEDOUT); no idea
    // what causes this but it works on retry.
    for (int retry = 5; retry--;) {
        r = ibv_modify_qp(ib_qp_, &ib_qp_attr, flags);
        if (!r) {
            break;
        }
        if (!retry) {
            break;
        }
        ERROR("Cannot modify queue pair to IBV_QPS_RTR, errno=%d: \"%s\"; retrying.\n", r, strerror(r));
        useconds_t ms = 200;
        useconds_t us = ms * 1000;
        usleep(us);
    }
    if (r) {
        ERROR("Cannot modify queue pair to IBV_QPS_RTR, errno=%d.\n", r);
        free_ib_resources();
        return false;
    }
    // All ready to go.
    return true;
}

static inline struct timespec add_ms(struct timespec& ts, unsigned long ms)
{
    unsigned long us = ms * 1000;
    unsigned long ns = us * 1000;
    ns += ts.tv_nsec;
#ifndef NS_PER_S
#define MS_PER_S (1000ULL)
#define US_PER_S (1000ULL * MS_PER_S)
#define NS_PER_S (1000ULL * US_PER_S)
#endif /* NS_PER_S */
    lldiv_t d = lldiv(ns, NS_PER_S);
    struct timespec r;
    r.tv_nsec = d.rem;
    r.tv_sec = ts.tv_sec + d.quot;
    return r;
}

#ifdef PERIODIC_STATUS
// Returns true if a is before b.
static inline bool before(struct timespec& a, struct timespec& b)
{
    if (a.tv_sec < b.tv_sec) {
        return true;
    }
    if (a.tv_sec > b.tv_sec) {
        return false;
    }
    if (a.tv_nsec < b.tv_nsec) {
        return true;
    }
    return false;
}
#endif /* PERIODIC_STATUS */

void RoceReceiver::blocking_monitor()
{
    native::NvtxTrace::setThreadName("RoceReceiver::run");
    DEBUG("Running.\n");

    struct ibv_wc ib_wc = { 0 };

#ifdef PERIODIC_STATUS
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    unsigned report_ms = 1000;
    struct timespec report_time = add_ms(now, report_ms);
    unsigned count = 0;
#endif /* PERIODIC_STATUS */

    int r = ibv_req_notify_cq(ib_cq_, 0);
    if (r != 0) {
        ERROR("ibv_req_notify_cq failed, errno=%d.\n", r);
        return;
    }

    struct pollfd poll_fds[2] = {
        {
            .fd = control_r_,
            .events = POLLIN | POLLHUP | POLLERR,
        },
        {
            .fd = ib_completion_channel_->fd,
            .events = POLLIN | POLLHUP | POLLERR,
        }
    };

    while (!done_) {
        check_async_events();

        int timeout = -1; // stay here forever.
        int r = poll(poll_fds, NUM_OF(poll_fds), timeout);
        if (r == -1) {
            ERROR("poll returned r=%d, errno=%d.\n", r, (int)errno);
            break;
        }

        // Keep this as close to the actual message receipt as possible.
        clock_gettime(CLOCK_MONOTONIC, &event_time_);

        // control_r_
        if (poll_fds[0].revents) {
            // Currently, the only activity that we see on control_r_ is a flag
            // telling us that someone closed the control_w_ side (which we do
            // in LinuxReceiver::close).  That specific event is an indication
            // that this loop is instructed to terminate.
            DEBUG("Closing.\n");
            break;
        }

        // CQ event.  The Hololink device, on the last UDP/ROCE request, sends
        // a write-immediate, which wakes us up here.  (We don't see any of the
        // other network traffic at all--it's all RDMA'd directly to GPU memory.)
        if (poll_fds[1].revents) {
            struct ibv_cq* ev_cq = NULL;
            void* ev_ctx = NULL;
            r = ibv_get_cq_event(ib_completion_channel_, &ev_cq, &ev_ctx);
            if (r != 0) {
                ERROR("ibv_get_cq_event returned r=%d.\n", r);
                break;
            }
            // Ack it and queue up another
            ibv_ack_cq_events(ev_cq, 1);
            r = ibv_req_notify_cq(ev_cq, 0);
            if (r != 0) {
                ERROR("ibv_req_notify_cq returned r=%d.\n", r);
                break;
            }

            // Now deal with active events.
            while (!done_) {
                r = ibv_poll_cq(ib_cq_, 1, &ib_wc);
                if (r < 0) {
                    ERROR("ibv_poll_cq failed, errno=%d.\n", (int)errno);
                    break;
                }
                // Is there a message for us?
                if (r == 0) {
                    break;
                }
                // Note some metadata
                char buffer[1024];
                lseek(rx_write_requests_fd_, 0, SEEK_SET); // may fail if fd==-1, we don't care
                ssize_t buffer_size = read(rx_write_requests_fd_, buffer, sizeof(buffer));
                // if rx_write_requests_fd_ is -1, then buffer_size_ will be less than 0
                if ((buffer_size > 0) && (buffer_size < 1000)) {
                    rx_write_requests_ = strtoull(buffer, NULL, 10);
                    // otherwise we'll continue to use the 0 from the constructor
                }
                frame_number_++;
                imm_data_ = ntohl(ib_wc.imm_data); // ibverbs just gives us the bytes here
                frame_end_ = event_time_;
                // frame_end_ uses the monotonic clock, which doesn't define it's epoch;
                // received_ns_ is the same but matches the epoch used by PTP.
                received_ns_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::system_clock::now().time_since_epoch())
                                   .count();
                // Send it
                signal();
                // Add back the work request
                struct ibv_recv_wr ib_wr = {
                    .wr_id = 1,
                    .next = NULL,
                }; // C fills the rest with 0s
                struct ibv_recv_wr* bad_wr = NULL;
                if (ibv_post_recv(ib_qp_, &ib_wr, &bad_wr)) {
                    ERROR("Cannot post a receiver work list, errno=%d.\n", (int)errno);
                    break;
                }
            }
        }

#ifdef PERIODIC_STATUS
        count++;
        clock_gettime(CLOCK_MONOTONIC, &now);
        if (!before(now, report_time)) {
            ERROR("count=%u.\n", count);
            report_time = add_ms(report_time, report_ms);
        }
#endif /* PERIODIC_STATUS */
    }
    free_ib_resources();
    DEBUG("Closed.\n");
}

void RoceReceiver::signal()
{
    int r = pthread_mutex_lock(&ready_mutex_);
    if (r != 0) {
        ERROR("pthread_mutex_lock returned r=%d.\n", r);
    }
    ready_ = true;
    r = pthread_cond_signal(&ready_condition_);
    if (r != 0) {
        ERROR("pthread_cond_signal returned r=%d.\n", r);
    }
    r = pthread_mutex_unlock(&ready_mutex_);
    if (r != 0) {
        ERROR("pthread_mutex_unlock returned r=%d.\n", r);
    }
}

void RoceReceiver::close()
{
    done_ = true;
    ::close(control_w_);
}

void RoceReceiver::free_ib_resources()
{
    if (ib_qp_ != NULL) {
        if (ibv_destroy_qp(ib_qp_)) {
            ERROR("ibv_destroy_qp failed, errno=%d.\n", (int)errno);
        }
        ib_qp_ = NULL;
    }

    if (ib_mr_ != NULL) {
        if (ibv_dereg_mr(ib_mr_)) {
            ERROR("ibv_dereg_mr failed, errno=%d.\n", (int)errno);
        }
        ib_mr_ = NULL;
    }

    if (ib_cq_ != NULL) {
        if (ibv_destroy_cq(ib_cq_)) {
            ERROR("ibv_destroy_cq failed, errno=%d.\n", (int)errno);
        }
        ib_cq_ = NULL;
    }

    if (ib_completion_channel_ != NULL) {
        if (ibv_destroy_comp_channel(ib_completion_channel_)) {
            ERROR("ibv_destroy_comp_channel failed, errno=%d.\n", (int)errno);
        }
        ib_completion_channel_ = NULL;
    }

    if (ib_pd_ != NULL) {
        if (ibv_dealloc_pd(ib_pd_)) {
            ERROR("ibv_dealloc_pd failed, errno=%d.\n", (int)errno);
        }
        ib_pd_ = NULL;
    }

    if (ib_context_ != NULL) {
        if (ibv_close_device(ib_context_)) {
            ERROR("ibv_close_device failed, errno=%d.\n", (int)errno);
        }
        ib_context_ = NULL;
    }

    TRACE("Done.\n");
}

bool RoceReceiver::get_next_frame(unsigned timeout_ms, RoceReceiverMetadata& metadata)
{
    bool r = wait(timeout_ms);
    metadata.frame_number = frame_number_;
    if (r) {
        metadata.rx_write_requests = rx_write_requests_;
        metadata.frame_end_s = frame_end_.tv_sec;
        metadata.frame_end_ns = frame_end_.tv_nsec;
        metadata.imm_data = imm_data_;
        metadata.received_ns = received_ns_;
    } else {
        metadata.rx_write_requests = 0;
        metadata.frame_end_s = 0;
        metadata.frame_end_ns = 0;
        metadata.imm_data = 0;
        metadata.received_ns = 0;
    }
    return r;
}

bool RoceReceiver::wait(unsigned timeout_ms)
{
    int status = pthread_mutex_lock(&ready_mutex_);
    if (status != 0) {
        ERROR("pthread_mutex_lock returned status=%d.\n", status);
        return false;
    }
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        ERROR("clock_gettime failed, errno=%d.\n", (int)errno);
    }
    struct timespec timeout = add_ms(now, timeout_ms);

    while (!ready_) {
        status = pthread_cond_timedwait(&ready_condition_, &ready_mutex_, &timeout);
        if (status == ETIMEDOUT) {
            break;
        }
        if (status != 0) {
            ERROR("pthread_cond_wait returned status=%d.\n", status);
            break;
        }
    }
    bool r = ready_;
    ready_ = false;
    status = pthread_mutex_unlock(&ready_mutex_);
    if (status != 0) {
        ERROR("pthread_mutex_unlock returned status=%d.\n", status);
    }
    return r;
}

bool RoceReceiver::check_async_events()
{
    unsigned i;
    struct ibv_async_event ib_async_event;
    for (i = 0; i < 100; i++) {
        int r = ibv_get_async_event(ib_context_, &ib_async_event);
        if (r != 0) {
            break;
        }

        switch (ib_async_event.event_type) {
        case IBV_EVENT_COMM_EST:
            // Communication established isn't an error; don't complain about it
            break;
        default:
            ERROR("ib_async_event.event_type=%d.\n", (int)ib_async_event.event_type);
            break;
        }

        ibv_ack_async_event(&ib_async_event);
    }
    return true;
}

std::mutex& RoceReceiver::get_lock()
{
    // We want all RoceReceiver instances in this process
    // to share the same lock.
    static std::mutex lock;
    return lock;
}

} // namespace hololink::operators
