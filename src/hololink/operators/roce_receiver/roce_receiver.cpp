/**
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES.
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

class RoceReceiverDescriptor {
public:
    uint32_t page_;

    common::UniqueCUhostptr metadata_buffer_;
    common::UniqueCUevent metadata_event_;

    uint64_t received_frame_number_;
    uint64_t rx_write_requests_; // over all of time
    uint32_t imm_data_;
    struct timespec received_;
    uint32_t dropped_;
    uint32_t received_psn_;
};

RoceReceiver::RoceReceiver(
    const char* ibv_name,
    unsigned ibv_port,
    CUdeviceptr cu_buffer,
    size_t cu_buffer_size,
    size_t cu_frame_size,
    size_t cu_page_size,
    unsigned pages,
    size_t metadata_offset,
    const char* peer_ip,
    unsigned queue_size,
    void* host_buffer)
    : ibv_name_(strdup(ibv_name))
    , ibv_port_(ibv_port)
    , host_buffer_(host_buffer)
    , cu_buffer_(cu_buffer)
    , cu_buffer_size_(cu_buffer_size)
    , cu_frame_size_(cu_frame_size)
    , cu_page_size_(cu_page_size)
    , pages_(pages)
    , queue_size_(queue_size)
    , metadata_offset_(metadata_offset)
    , peer_ip_(strdup(peer_ip))
    , ib_qp_(NULL)
    , ib_mr_(NULL)
    , dmabuf_fd_(-1)
    , ib_cq_(NULL)
    , ib_pd_(NULL)
    , ib_context_(NULL)
    , ib_completion_channel_(NULL)
    , qp_number_(0)
    , rkey_(0)
    , ready_mutex_(PTHREAD_MUTEX_INITIALIZER)
    , ready_condition_(PTHREAD_COND_INITIALIZER)
    , dropped_(0)
    , done_(false)
    , control_r_(-1)
    , control_w_(-1)
    , rx_write_requests_fd_(-1)
    , frame_ready_([](const RoceReceiver&) {})
    , frame_number_()
    , monitor_running_()
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

    metadata_stream_.reset([] {
        CUstream stream;
        CudaCheck(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
        return stream;
    }());
}

RoceReceiver::~RoceReceiver()
{
    pthread_cond_destroy(&ready_condition_);
    pthread_mutex_destroy(&ready_mutex_);
    ::close(rx_write_requests_fd_); // we ignore an error here if fd==-1

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
            HSB_LOG_ERROR("ibv_fork_init failed; r={}.", r);
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

    // Register the frame buffer as an RDMA memory region.
    //
    // Preferred path: GPU VRAM via DMA-BUF (host_buffer_ == nullptr).
    // cuMemGetHandleForAddressRange exports the VRAM allocation as a DMA-BUF
    // fd; ibv_reg_dmabuf_mr maps those pages into the NIC's IOMMU domain so
    // the FPGA can RDMA-write directly into GPU VRAM. Requires GPUDirect RDMA
    // to be fully configured (e.g. nvidia_peermem loaded, IOMMU grants write
    // access to the NIC). If DMAR faults appear in dmesg with fault reason
    // 0x05 ("PTE Write access is not set"), the CUDA GPU driver is mapping
    // VRAM pages read-only for non-peer devices; configure GPUDirect RDMA or
    // force the fallback path by allocating with cuMemHostAlloc instead.
    //
    // Fallback path: pinned host memory via ibv_reg_mr_iova (host_buffer_ != nullptr).
    // The IOMMU always maps pinned host pages with full write access for all
    // PCIe devices, so RDMA writes land correctly without any additional
    // GPUDirect configuration. ReceiverMemoryDescriptor sets host_buffer_ when
    // GPU VRAM allocation or DMA-BUF export is unavailable on this system.
    if (host_buffer_ == nullptr) {
        // Preferred: GPU VRAM via DMA-BUF.
        CUresult cu_result = cuMemGetHandleForAddressRange(
            (void*)&dmabuf_fd_, cu_buffer_, cu_buffer_size_,
            CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
        if (cu_result == CUDA_SUCCESS) {
            int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
            ib_mr_ = ibv_reg_dmabuf_mr(ib_pd_, 0, cu_buffer_size_, 0, dmabuf_fd_, access);
            if (ib_mr_ == NULL) {
                HSB_LOG_ERROR("Cannot register dmabuf memory region p={:#x} size={:#x}, errno={}.", cu_buffer_, cu_buffer_size_, errno);
                free_ib_resources();
                return false;
            }
        } else {
            int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
            ib_mr_ = ibv_reg_mr_iova(ib_pd_, (void*)cu_buffer_, cu_buffer_size_, 0, access);
            if (ib_mr_ == NULL) {
                HSB_LOG_ERROR("Cannot register memory region p={:#x} size={:#x}, errno={}.", cu_buffer_, cu_buffer_size_, errno);
                free_ib_resources();
                return false;
            }
        }
    } else {
        // Fallback: pinned host memory.
        int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
        ib_mr_ = ibv_reg_mr_iova(ib_pd_, host_buffer_, cu_buffer_size_, 0, access);
        if (ib_mr_ == NULL) {
            HSB_LOG_ERROR("Cannot register host memory region host={} size={:#x}, errno={}.", host_buffer_, cu_buffer_size_, errno);
            free_ib_resources();
            return false;
        }
        HSB_LOG_INFO("Registered host memory region: host={} size={:#x} rkey={:#x}", host_buffer_, cu_buffer_size_, ib_mr_->rkey);
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

    std::lock_guard monitor_lock(monitor_running_);

    // Construct a descriptor for each page
    std::vector<RoceReceiverDescriptor> descriptors(queue_size_);
    for (uint32_t index = 0; index < queue_size_; index++) {
        descriptors[index].metadata_buffer_.reset([] {
            void* buffer;
            CudaCheck(cuMemHostAlloc(&buffer, hololink::METADATA_SIZE, 0));
            memset(buffer, 0xEE, hololink::METADATA_SIZE);
            return buffer;
        }());
        descriptors[index].metadata_event_.reset([] {
            CUevent event;
            CudaCheck(cuEventCreate(&event, CU_EVENT_DISABLE_TIMING));
            return event;
        }());

        available_.push(&descriptors[index]);
    }

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

    unsigned frame_count = 0;
    struct timespec event_time;

    while (!done_) {
        check_async_events();

        int timeout = -1; // stay here forever.
        int r = poll(poll_fds, NUM_OF(poll_fds), timeout);
        if (r == -1) {
            throw std::runtime_error(fmt::format("poll returned r={}, errno={}.", r, errno));
        }

        // Keep this as close to the actual message receipt as possible.
        clock_gettime(CLOCK_REALTIME, &event_time);

        // control_r_
        if (poll_fds[0].revents) {
            // Currently, the only activity that we see on control_r_ is a flag
            // telling us that someone closed the control_w_ side (which we do
            // in RoceReceiver::close).  That specific event is an indication
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

                frame_count++;
                core::NvtxTrace::event_u64("frame_count", frame_count);

                // Note some metadata
                char buffer[1024];
                lseek(rx_write_requests_fd_, 0, SEEK_SET); // may fail if fd==-1, we don't care
                // if rx_write_requests_fd_ is -1, then buffer_size_ will be less than 0
                ssize_t buffer_size = read(rx_write_requests_fd_, buffer, sizeof(buffer));

                const uint32_t imm_data = ntohl(ib_wc.imm_data); // ibverbs just gives us the bytes here
                const uint32_t page = imm_data & 0xFFF;
                if (page >= pages_) {
                    throw std::runtime_error(fmt::format("Invalid page={}; ignoring.", page));
                }

                // Do an atomic update
                r = pthread_mutex_lock(&ready_mutex_);
                if (r != 0) {
                    throw std::runtime_error(fmt::format("pthread_mutex_lock returned r={}.", r));
                }

                RoceReceiverDescriptor* descriptor;
                if (!consumer_ready_ && !ready_.empty()) {
                    // if the consumer is not ready yet, avoid queueing up more frames
                    descriptor = ready_.front();
                    ready_.pop();
                } else {
                    if (available_.empty()) {
                        // if there is no available descriptor use the oldest ready one and drop the frame
                        descriptor = ready_.front();
                        ready_.pop();
                        if (consumer_ready_) {
                            HSB_LOG_DEBUG("No available descriptors, dropping oldest ready frame {}.", descriptor->received_frame_number_);
                        }
                        dropped_++;
                    } else {
                        descriptor = available_.front();
                        available_.pop();
                    }
                }

                descriptor->page_ = page;

                // Start copying out the metadata chunk. get_next_frame will synchronize with the event.
                copy_metadata_to_host(descriptor->page_, descriptor->metadata_buffer_.get(), descriptor->metadata_event_.get());

                descriptor->received_frame_number_ = frame_count;
                if ((buffer_size > 0) && (buffer_size < 1000)) {
                    buffer[buffer_size] = '\0';
                    descriptor->rx_write_requests_ = strtoull(buffer, NULL, 10);
                    // otherwise we'll continue to use the 0 from the constructor
                } else {
                    descriptor->rx_write_requests_ = 0;
                }
                descriptor->imm_data_ = imm_data;
                descriptor->received_.tv_sec = event_time.tv_sec;
                descriptor->received_.tv_nsec = event_time.tv_nsec;
                descriptor->dropped_ = dropped_;
                descriptor->received_psn_ = (imm_data >> 12) & 0xFFFFF;

                // Send it
                ready_.push(descriptor);
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

void RoceReceiver::copy_metadata_to_host(uint32_t page, void* metadata_buffer, CUevent metadata_event)
{
    CUdeviceptr page_start = page * cu_page_size_;
    CUdeviceptr metadata_start = page_start + metadata_offset_;
    if ((metadata_start + hololink::METADATA_SIZE) > cu_buffer_size_) {
        throw std::runtime_error(fmt::format("metadata_start={:#x}+metadata_size={:#x}(which is {:#x}) exceeds cu_buffer_size={:#x}.",
            metadata_start, hololink::METADATA_SIZE, metadata_start + hololink::METADATA_SIZE, cu_buffer_size_));
    }
    CudaCheck(cuMemcpyDtoHAsync(metadata_buffer, cu_buffer_ + metadata_start, hololink::METADATA_SIZE, metadata_stream_.get()));
    CudaCheck(cuEventRecord(metadata_event, metadata_stream_.get()));
}

void RoceReceiver::close()
{
    done_ = true;
    ::close(control_w_);
    // Wait here until the background thread closes.
    std::lock_guard monitor_lock(monitor_running_);
}

void RoceReceiver::free_ib_resources()
{
    if (ib_qp_ != NULL) {
        // Move QP to ERROR state -- this flushes all posted
        // receive WRs and NAKs incoming RDMA operations.
        struct ibv_qp_attr attr = { .qp_state = IBV_QPS_ERR };
        if (ibv_modify_qp(ib_qp_, &attr, IBV_QP_STATE)) {
            HSB_LOG_ERROR("ibv_modify_qp to ERR failed, errno={}.", errno);
        }

        // Drain the CQ so all flushed WRs are consumed before
        // we destroy the QP and deregister the MR.
        if (ib_cq_ != NULL) {
            struct ibv_wc wc;
            while (ibv_poll_cq(ib_cq_, 1, &wc) > 0) {
                // discard -- these are all error completions
            }
        }

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

    if (dmabuf_fd_ >= 0) {
        ::close(dmabuf_fd_);
        dmabuf_fd_ = -1;
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
        throw std::runtime_error(fmt::format("pthread_mutex_lock returned status={}.", status));
    }

    // signal the background thread that the consumer is ready to receive frames
    consumer_ready_ = true;

    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        HSB_LOG_ERROR("clock_gettime failed, errno={}.", errno);
    }
    struct timespec timeout = add_ms(now, timeout_ms);

    bool result = true;
    while (ready_.empty()) {
        status = pthread_cond_timedwait(&ready_condition_, &ready_mutex_, &timeout);
        if (status == ETIMEDOUT) {
            result = false;
            break;
        }
        if (status != 0) {
            HSB_LOG_ERROR("pthread_cond_wait returned status={}.", status);
            result = false;
            break;
        }
    }

    if (result) {
        RoceReceiverDescriptor* ready_descriptor = ready_.front();
        ready_.pop();

        Hololink::FrameMetadata frame_metadata = get_frame_metadata(ready_descriptor->metadata_buffer_.get(), ready_descriptor->metadata_event_.get());
        if ((frame_metadata.psn & 0xFFFFF) != ready_descriptor->received_psn_) {
            // This indicates that the distal end rewrote the receiver buffer.
            HSB_LOG_ERROR("Metadata psn={} but received_psn={}.", frame_metadata.psn, ready_descriptor->received_psn_);
        }

        metadata.received_frame_number = ready_descriptor->received_frame_number_;
        metadata.dropped = ready_descriptor->dropped_;
        metadata.rx_write_requests = ready_descriptor->rx_write_requests_;
        metadata.imm_data = ready_descriptor->imm_data_;
        metadata.received_s = ready_descriptor->received_.tv_sec;
        metadata.received_ns = ready_descriptor->received_.tv_nsec;
        metadata.frame_memory = cu_buffer_ + cu_page_size_ * ready_descriptor->page_;
        metadata.metadata_memory = metadata.frame_memory + metadata_offset_;
        metadata.frame_metadata = frame_metadata;
        metadata.frame_number = frame_number_.update(frame_metadata.frame_number);

        available_.push(ready_descriptor);
    }

    status = pthread_mutex_unlock(&ready_mutex_);
    if (status != 0) {
        throw std::runtime_error(fmt::format("pthread_mutex_unlock returned status={}.", status));
    }
    return result;
}

bool RoceReceiver::frames_ready()
{
    int status = pthread_mutex_lock(&ready_mutex_);
    if (status != 0) {
        throw std::runtime_error(fmt::format("pthread_mutex_lock returned status={}.", status));
    }

    bool result = ready_.empty();

    status = pthread_mutex_unlock(&ready_mutex_);
    if (status != 0) {
        throw std::runtime_error(fmt::format("pthread_mutex_unlock returned status={}.", status));
    }
    return !result;
}

const Hololink::FrameMetadata RoceReceiver::get_frame_metadata(void* metadata_buffer, CUevent metadata_event)
{
    CudaCheck(cuEventSynchronize(metadata_event));
    Hololink::FrameMetadata frame_metadata = Hololink::deserialize_metadata(
        static_cast<uint8_t*>(metadata_buffer), hololink::METADATA_SIZE);
    return frame_metadata;
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
    // All registration paths (host pinned memory via ibv_reg_mr_iova, or
    // GPU VRAM via ibv_reg_dmabuf_mr) use IOVA=0 as the base. The FPGA is
    // therefore told the frame buffer starts at address 0, and writes to
    // IOVA 0 + page * page_increment. The MR maps IOVA 0 to the start of
    // the frame buffer.
    return 0;
}

void RoceReceiver::set_frame_ready(std::function<void(const RoceReceiver&)> frame_ready)
{
    frame_ready_ = frame_ready;
}

} // namespace hololink::operators
