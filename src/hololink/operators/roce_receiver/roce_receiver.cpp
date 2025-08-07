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

#include <hololink/common/cuda_helper.hpp>
#include <hololink/core/logging_internal.hpp>
#include <hololink/core/nvtx_trace.hpp>

#undef PERIODIC_STATUS

#define NUM_OF(x) (sizeof(x) / sizeof(x[0]))

namespace hololink::operators {

RoceReceiver::RoceReceiver(
    const char* ibv_name,
    unsigned ibv_port,
    CUdeviceptr cu_buffer,
    size_t cu_buffer_size,
    size_t cu_frame_size,
    size_t cu_page_size,
    unsigned pages,
    size_t metadata_offset,
    const char* peer_ip)
    : ibv_name_(strdup(ibv_name))
    , ibv_port_(ibv_port)
    , cu_buffer_(cu_buffer)
    , cu_buffer_size_(cu_buffer_size)
    , cu_frame_size_(cu_frame_size)
    , cu_page_size_(cu_page_size)
    , pages_(pages)
    , metadata_offset_(metadata_offset)
    , peer_ip_(strdup(peer_ip))
    , ib_qp_(NULL)
    , ib_mr_(NULL)
    , ib_cq_(NULL)
    , ib_pd_(NULL)
    , ib_context_(NULL)
    , ib_completion_channel_(NULL)
    , qp_number_(0)
    , rkey_(0)
    , ready_(false)
    , ready_mutex_(PTHREAD_MUTEX_INITIALIZER)
    , ready_condition_(PTHREAD_COND_INITIALIZER)
    , done_(false)
    , control_r_(-1)
    , control_w_(-1)
    , received_frame_number_(0)
    , rx_write_requests_fd_(-1)
    , rx_write_requests_(0)
    , imm_data_(0)
    , event_time_ {}
    , received_ {}
    , current_buffer_(0)
    , metadata_stream_(0)
    , dropped_(0)
    , metadata_buffer_(nullptr)
    , received_psn_(0)
    , received_page_(0)
    , frame_ready_([](const RoceReceiver&) {})
    , frame_number_()
{
    HSB_LOG_DEBUG("cu_buffer={:#x} cu_frame_size={:#x} cu_page_size={} pages={}",
        cu_buffer, cu_frame_size, cu_page_size, pages);

    int r = pthread_mutex_init(&ready_mutex_, NULL);
    if (r != 0) {
        throw std::runtime_error("pthread_mutex_init failed.");
    }
    pthread_condattr_t pthread_condattr;
    pthread_condattr_init(&pthread_condattr);
    pthread_condattr_setclock(&pthread_condattr, CLOCK_MONOTONIC);
    r = pthread_cond_init(&ready_condition_, &pthread_condattr);
    if (r != 0) {
        throw std::runtime_error("pthread_cond_init failed.");
    }
    int pipe_fds[2] = { -1, -1 };
    // If these aren't updated, we'll get an error when we try to read, which is good.
    r = pipe(pipe_fds);
    if (r != 0) {
        throw std::runtime_error("pipe call failed.");
    }
    control_r_ = pipe_fds[0];
    control_w_ = pipe_fds[1];

    char rx_write_requests_filename[8192];
    int written = snprintf(rx_write_requests_filename, sizeof(rx_write_requests_filename),
        "/sys/class/infiniband/%s/ports/%d/hw_counters/rx_write_requests",
        ibv_name_, ibv_port_);
    if (written < 0) {
        throw std::runtime_error("Error writing to rx_write_requests_filename.");
    } else if (((size_t)written) >= sizeof(rx_write_requests_filename)) {
        throw std::runtime_error("Buffer isn't large enough to compute rx_write_requests filename.");
    } else {
        rx_write_requests_fd_ = open(rx_write_requests_filename, O_RDONLY);
    }
    if (rx_write_requests_fd_ < 0) {
        // Note that the rest of the code is OK if this occurs.
        HSB_LOG_ERROR("Unable to fetch rx_write_requests; ignoring.");
    }

    // We use this to synchronize metadata readout.
    CUresult cu_result = cuStreamCreate(&metadata_stream_, CU_STREAM_NON_BLOCKING);
    if (cu_result != CUDA_SUCCESS) {
        throw std::runtime_error(fmt::format("cuStreamCreate failed, cu_result={}.", cu_result));
    }

    // Set metadata_buffer_ content to some value that's easily distinguished.
    cu_result = cuMemHostAlloc((void**)&metadata_buffer_, hololink::METADATA_SIZE, 0);
    if (cu_result != CUDA_SUCCESS) {
        throw std::runtime_error(fmt::format("cuMemHostAlloc failed, cu_result={}.", cu_result));
    }
    memset(metadata_buffer_, 0xEE, hololink::METADATA_SIZE);
}

RoceReceiver::~RoceReceiver()
{
    pthread_cond_destroy(&ready_condition_);
    pthread_mutex_destroy(&ready_mutex_);
    ::close(rx_write_requests_fd_); // we ignore an error here if fd==-1
    cuMemFreeHost(metadata_buffer_);

    free(ibv_name_);
    free(peer_ip_);
}

bool RoceReceiver::start()
{
    HSB_LOG_DEBUG("Starting.");

    // ibv calls seem to have trouble with
    // reentrancy.  No problem; since we're only
    // at startup, run just one at a time.  This
    // only affects multithreaded schedulers like
    // EventBasedScheduler.
    std::lock_guard lock(get_lock());

    // ibverbs has some reentrancy problems with programs
    // that use fork, which may happen in applications we're
    // used in.
    static bool ibv_fork_init_done = false;
    if (!ibv_fork_init_done) {
        int r = ibv_fork_init();
        if (r != 0) {
            HSB_LOG_ERROR("ibv_fork_init failed; errno={}.", errno);
            return false;
        }
        ibv_fork_init_done = true;
    }

    // Find the IB controller
    int num_devices = 0;
    struct ibv_device** ib_devices = ibv_get_device_list(&num_devices);
    if (!ib_devices) {
        HSB_LOG_ERROR("ibv_get_device_list failed; errno={}.", errno);
        return false;
    }
    if (num_devices < 0) {
        HSB_LOG_ERROR("ibv_get_device_list set unexpected value for num_devices={}.", num_devices);
        return false;
    }
    struct ibv_device* ib_device = NULL;
    for (unsigned i = 0; i < (unsigned)num_devices; i++) {
        const char* device_name = ibv_get_device_name(ib_devices[i]);
        HSB_LOG_DEBUG("ibv_get_device_list[{}]={}.", i, device_name);
        if (strcmp(device_name, ibv_name_) != 0) {
            continue;
        }
        ib_device = ib_devices[i];
        break;
    }
    if (ib_device == NULL) {
        HSB_LOG_ERROR("ibv_get_device_list didn't find a device named \"{}\".", ibv_name_);
        ibv_free_device_list(ib_devices);
        return false;
    }

    // Open the IB device
    ib_context_ = ibv_open_device(ib_device);
    if (!ib_context_) {
        HSB_LOG_ERROR("ibv_open_device failed, errno={}.", errno);
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
        HSB_LOG_ERROR("Can't configure async_fd={} with O_NONBLOCK, errno={}.", ib_context_->async_fd, errno);
        return false;
    }

    //
    struct ibv_device_attr ib_device_attr { };
    if (ibv_query_device(ib_context_, &ib_device_attr)) {
        HSB_LOG_ERROR("ibv_query_device failed, errno={}.", errno);
        free_ib_resources();
        return false;
    }

    struct ibv_port_attr ib_port_attr = { .flags = 0 }; // C fills the rest with 0s
    if (ibv_query_port(ib_context_, ibv_port_, &ib_port_attr)) {
        HSB_LOG_ERROR("ibv_query_port failed, errno={}.", errno);
        free_ib_resources();
        return false;
    }

    // Fetch the GID
    bool ok = false;
    struct ibv_gid_entry ib_gid_entry { };
    int gid_index = 0;
    for (gid_index = 0; 1; gid_index++) {
        uint32_t flags = 0;
        r = ibv_query_gid_ex(ib_context_, ibv_port_, gid_index, &ib_gid_entry, flags);
        if (r && (errno != ENODATA)) {
            break;
        }

        struct ibv_gid_entry* u = &(ib_gid_entry);
        HSB_LOG_DEBUG("gid_index={} gid_entry(gid_index={} port_num={} gid_type={} ndev_ifindex={} subnet_prefix={} interface_id={:#x})", gid_index, u->gid_index, u->port_num, u->gid_type, u->ndev_ifindex, u->gid.global.subnet_prefix, u->gid.global.interface_id);

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
        HSB_LOG_ERROR("Cannot find GID for IBV_GID_TYPE_ROCE_V2.");
        free_ib_resources();
        return false;
    }

    // Create a protection domain
    ib_pd_ = ibv_alloc_pd(ib_context_);
    if (ib_pd_ == NULL) {
        HSB_LOG_ERROR("Cannot allocate a protection domain, errno={}.", errno);
        free_ib_resources();
        return false;
    }

    // Create a completion channel.
    ib_completion_channel_ = ibv_create_comp_channel(ib_context_);
    if (ib_completion_channel_ == NULL) {
        HSB_LOG_ERROR("Cannot create a completion channel.");
        free_ib_resources();
        return false;
    }
    // Configure for nonblocking reads, this way we can poll
    // for events
    flags = fcntl(ib_completion_channel_->fd, F_GETFL);
    r = fcntl(ib_completion_channel_->fd, F_SETFL, flags | O_NONBLOCK);
    if (r < 0) {
        HSB_LOG_ERROR("Can't configure fd={} with O_NONBLOCK, errno={}.", ib_completion_channel_->fd, errno);
        return false;
    }

    // Create a completion queue
    unsigned completion_queue_size = 100;
    ib_cq_ = ibv_create_cq(ib_context_, completion_queue_size, NULL, ib_completion_channel_, 0);
    if (ib_cq_ == NULL) {
        HSB_LOG_ERROR("Cannot create a completion queue, errno={}.", errno);
        free_ib_resources();
        return false;
    }

    // Provide access to the frame buffer
    int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
    ib_mr_ = ibv_reg_mr_iova(ib_pd_, (void*)cu_buffer_, cu_buffer_size_, 0, access);
    if (ib_mr_ == NULL) {
        HSB_LOG_ERROR("Cannot register memory region p={:#x} size={:#x}, errno={}.", cu_buffer_, cu_frame_size_, errno);
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
        HSB_LOG_ERROR("Cannot create queue pair, errno={}.", errno);
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
        HSB_LOG_ERROR("Cannot modify queue pair to IBV_QPS_INIT, errno={}.", errno);
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
            HSB_LOG_ERROR("Cannot post the receiver work list, errno={}.", errno);
            free_ib_resources();
            return false;
        }
    }

    // qp is currently INIT; go to RTR
    unsigned long client_ip = 0;
    if (inet_pton(AF_INET, peer_ip_, &client_ip) != 1) {
        HSB_LOG_ERROR("Unable to convert \"{}\" to an IP address.", peer_ip_);
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
        HSB_LOG_ERROR("Cannot modify queue pair to IBV_QPS_RTR, errno={}: \"{}\"; retrying.", r, strerror(r));
        useconds_t ms = 200;
        useconds_t us = ms * 1000;
        usleep(us);
    }
    if (r) {
        HSB_LOG_ERROR("Cannot modify queue pair to IBV_QPS_RTR, errno={}.", r);
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
    core::NvtxTrace::setThreadName("RoceReceiver::run");
    HSB_LOG_DEBUG("Running.");

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
        throw std::runtime_error(fmt::format("ibv_req_notify_cq failed, errno={}.", r));
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
            throw std::runtime_error(fmt::format("poll returned r={}, errno={}.", r, errno));
        }

        // Keep this as close to the actual message receipt as possible.
        clock_gettime(CLOCK_REALTIME, const_cast<struct timespec*>(&event_time_));

        // control_r_
        if (poll_fds[0].revents) {
            // Currently, the only activity that we see on control_r_ is a flag
            // telling us that someone closed the control_w_ side (which we do
            // in LinuxReceiver::close).  That specific event is an indication
            // that this loop is instructed to terminate.
            HSB_LOG_DEBUG("Closing.");
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
                throw std::runtime_error(fmt::format("ibv_get_cq_event returned r={}.", r));
            }
            // Ack it and queue up another
            ibv_ack_cq_events(ev_cq, 1);
            r = ibv_req_notify_cq(ev_cq, 0);
            if (r != 0) {
                throw std::runtime_error(fmt::format("ibv_req_notify_cq returned r={}.", r));
            }

            // Now deal with active events.
            while (!done_) {
                r = ibv_poll_cq(ib_cq_, 1, &ib_wc);
                if (r < 0) {
                    throw std::runtime_error(fmt::format("ibv_poll_cq failed, errno={}.", errno));
                }
                // Is there a message for us?
                if (r == 0) {
                    break;
                }
                uint64_t q = (uint64_t)this;
                HSB_LOG_TRACE("this={:#x} r={} qp_number_={:#x} imm_data={:#x}", q, r, qp_number_, ntohl(ib_wc.imm_data));
                // Note some metadata
                char buffer[1024];
                lseek(rx_write_requests_fd_, 0, SEEK_SET); // may fail if fd==-1, we don't care
                // if rx_write_requests_fd_ is -1, then buffer_size_ will be less than 0
                ssize_t buffer_size = read(rx_write_requests_fd_, buffer, sizeof(buffer));
                // Do an atomic update
                r = pthread_mutex_lock(&ready_mutex_);
                if (r != 0) {
                    throw std::runtime_error(fmt::format("pthread_mutex_lock returned r={}.", r));
                }
                // If the application didn't set ready_ to false,
                // then we're overwriting a valid frame.
                if (ready_) {
                    dropped_++;
                }
                if ((buffer_size > 0) && (buffer_size < 1000)) {
                    rx_write_requests_ = strtoull(buffer, NULL, 10);
                    // otherwise we'll continue to use the 0 from the constructor
                }
                imm_data_ = ntohl(ib_wc.imm_data); // ibverbs just gives us the bytes here
                received_psn_ = (imm_data_ >> 8) & 0xFFFFFF;
                unsigned page = imm_data_ & 0xFF;
                if (page >= pages_) {
                    throw std::runtime_error(fmt::format("Invalid page={}; ignoring.", page));
                }
                received_page_ = page;
                // Start copying out the metadata chunk.  get_next_frame will synchronize on metadata_stream_.
                // NOTE that we start this request while we hold ready_mutex_ -- this guarantees that we don't
                // start another copy while the foreground is copying data out of this buffer.
                CUdeviceptr page_start = page * cu_page_size_;
                CUdeviceptr metadata_start = page_start + metadata_offset_;
                if ((metadata_start + hololink::METADATA_SIZE) > cu_buffer_size_) {
                    throw std::runtime_error(fmt::format("metadata_start={:#x}+metadata_size={:#x}(which is {:#x}) exceeds cu_buffer_size={:#x}.",
                        metadata_start, hololink::METADATA_SIZE, metadata_start + hololink::METADATA_SIZE, cu_buffer_size_));
                }
                CUresult cu_result = cuMemcpyDtoHAsync(metadata_buffer_, cu_buffer_ + metadata_start, hololink::METADATA_SIZE, metadata_stream_);
                if (cu_result != CUDA_SUCCESS) {
                    throw std::runtime_error(fmt::format("cmMemcpyDtoHAsync failed, cu_result={}.", cu_result));
                }
                current_buffer_ = cu_buffer_ + cu_page_size_ * page;
                received_frame_number_++;
                core::NvtxTrace::event_u64("received_frame_number", received_frame_number_);
                HSB_LOG_TRACE("received_frame_number={}", received_frame_number_);
                received_.tv_sec = event_time_.tv_sec;
                received_.tv_nsec = event_time_.tv_nsec;
                HSB_LOG_TRACE("received_frame_number={} imm_data={:#x} received.tv_sec={:#x} received.tv_nsec={:#x}",
                    received_frame_number_, imm_data_, received_.tv_sec, received_.tv_nsec);
                // Send it
                ready_ = true;
                core::NvtxTrace::event_u64("signal", 1);
                r = pthread_cond_signal(&ready_condition_);
                if (r != 0) {
                    throw std::runtime_error(fmt::format("pthread_cond_signal returned r={}.", r));
                }
                r = pthread_mutex_unlock(&ready_mutex_);
                if (r != 0) {
                    throw std::runtime_error(fmt::format("pthread_mutex_unlock returned r={}.", r));
                }
                // Provide the local callback, letting the application know
                // that get_next_frame won't block.
                frame_ready_(this[0]);
                // Add back the work request
                struct ibv_recv_wr ib_wr = {
                    .wr_id = 1,
                    .next = NULL,
                }; // C fills the rest with 0s
                struct ibv_recv_wr* bad_wr = NULL;
                if (ibv_post_recv(ib_qp_, &ib_wr, &bad_wr)) {
                    throw std::runtime_error(fmt::format("Cannot post a receiver work list, errno={}.", errno));
                }
            }
        }

#ifdef PERIODIC_STATUS
        count++;
        clock_gettime(CLOCK_MONOTONIC, &now);
        if (!before(now, report_time)) {
            HSB_LOG_ERROR("count={}.", count);
            report_time = add_ms(report_time, report_ms);
        }
#endif /* PERIODIC_STATUS */
    }
    free_ib_resources();
    HSB_LOG_DEBUG("Closed.");
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
            HSB_LOG_ERROR("ibv_destroy_qp failed, errno={}.", errno);
        }
        ib_qp_ = NULL;
    }

    if (ib_mr_ != NULL) {
        if (ibv_dereg_mr(ib_mr_)) {
            HSB_LOG_ERROR("ibv_dereg_mr failed, errno={}.", errno);
        }
        ib_mr_ = NULL;
    }

    if (ib_cq_ != NULL) {
        if (ibv_destroy_cq(ib_cq_)) {
            HSB_LOG_ERROR("ibv_destroy_cq failed, errno={}.", errno);
        }
        ib_cq_ = NULL;
    }

    if (ib_completion_channel_ != NULL) {
        if (ibv_destroy_comp_channel(ib_completion_channel_)) {
            HSB_LOG_ERROR("ibv_destroy_comp_channel failed, errno={}.", errno);
        }
        ib_completion_channel_ = NULL;
    }

    if (ib_pd_ != NULL) {
        if (ibv_dealloc_pd(ib_pd_)) {
            HSB_LOG_ERROR("ibv_dealloc_pd failed, errno={}.", errno);
        }
        ib_pd_ = NULL;
    }

    if (ib_context_ != NULL) {
        if (ibv_close_device(ib_context_)) {
            HSB_LOG_ERROR("ibv_close_device failed, errno={}.", errno);
        }
        ib_context_ = NULL;
    }

    HSB_LOG_TRACE("Done.");
}

bool RoceReceiver::get_next_frame(unsigned timeout_ms, RoceReceiverMetadata& metadata)
{
    int status = pthread_mutex_lock(&ready_mutex_);
    if (status != 0) {
        HSB_LOG_ERROR("pthread_mutex_lock returned status={}.", status);
        return false;
    }
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        HSB_LOG_ERROR("clock_gettime failed, errno={}.", errno);
    }
    struct timespec timeout = add_ms(now, timeout_ms);

    while (!ready_) {
        status = pthread_cond_timedwait(&ready_condition_, &ready_mutex_, &timeout);
        if (status == ETIMEDOUT) {
            break;
        }
        if (status != 0) {
            HSB_LOG_ERROR("pthread_cond_wait returned status={}.", status);
            break;
        }
    }
    bool r = ready_;
    ready_ = false;
    metadata.received_frame_number = received_frame_number_;
    metadata.dropped = dropped_;
    if (r) {
        CUresult cu_result = cuStreamSynchronize(metadata_stream_);
        if (cu_result != CUDA_SUCCESS) {
            throw std::runtime_error(fmt::format("cuStreamSynchronize failed, cu_result={}.", cu_result));
        }
        Hololink::FrameMetadata frame_metadata = Hololink::deserialize_metadata(metadata_buffer_, hololink::METADATA_SIZE);
        if (frame_metadata.psn != received_psn_) {
            // This indicates that the distal end rewrote the receiver buffer.
            HSB_LOG_ERROR("Metadata psn={} but received_psn={}.", frame_metadata.psn, received_psn_);
        }
        metadata.rx_write_requests = rx_write_requests_;
        metadata.imm_data = imm_data_;
        metadata.received_s = received_.tv_sec;
        metadata.received_ns = received_.tv_nsec;
        metadata.frame_memory = current_buffer_;
        metadata.metadata_memory = current_buffer_ + metadata_offset_;
        metadata.frame_metadata = frame_metadata;
        metadata.frame_number = frame_number_.update(frame_metadata.frame_number);
    } else {
        metadata.rx_write_requests = 0;
        metadata.imm_data = 0;
        metadata.received_s = 0;
        metadata.received_ns = 0;
        metadata.frame_memory = 0;
        metadata.metadata_memory = 0;
        metadata.frame_metadata = {}; // All 0s
        metadata.frame_number = 0;
    }
    status = pthread_mutex_unlock(&ready_mutex_);
    if (status != 0) {
        HSB_LOG_ERROR("pthread_mutex_unlock returned status={}.", status);
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

        auto event_type = ib_async_event.event_type;
        switch (event_type) {
        case IBV_EVENT_COMM_EST:
            // Communication established isn't an error; don't complain about it
            break;
        default:
            HSB_LOG_ERROR("ib_async_event.event_type={} ({}).", static_cast<int>(event_type), ibv_event_type_str(event_type));
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

uint64_t RoceReceiver::external_frame_memory()
{
    // If we didn't use ibv_reg_mr_iova above, we'd return
    // cu_buffer_ here; but ibv_reg_mr_iova always adds it's
    // address to the address received from the peripheral.
    return 0;
}

void RoceReceiver::set_frame_ready(std::function<void(const RoceReceiver&)> frame_ready)
{
    frame_ready_ = frame_ready;
}

} // namespace hololink::operators
