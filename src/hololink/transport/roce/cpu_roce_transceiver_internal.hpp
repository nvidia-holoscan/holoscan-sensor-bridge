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
 */

#ifndef SRC_HOLOLINK_TRANSPORT_ROCE_CPU_ROCE_TRANSCEIVER_INTERNAL_HPP
#define SRC_HOLOLINK_TRANSPORT_ROCE_CPU_ROCE_TRANSCEIVER_INTERNAL_HPP

#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>

#include <sched.h>
#include <unistd.h>

#include "cpu_roce_transceiver.hpp"
#include "ibv.hpp"

namespace hololink::transport::roce::detail {

inline void cpu_relax() noexcept
{
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    asm volatile("yield" ::
                     : "memory");
#else
    std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

// Restrict the calling thread to one Linux logical CPU ID. Returns EINVAL
// unless 0 <= cpu < CPU_SETSIZE; 0 is valid. Distinct IDs may still be SMT
// siblings. Otherwise returns 0 or the errno from sched_setaffinity().
inline int try_pin_current_thread(int cpu)
{
    if (cpu < 0 || cpu >= CPU_SETSIZE) {
        return EINVAL;
    }
    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);
    CPU_SET(cpu, &cpu_set);
    if (sched_setaffinity(0, sizeof(cpu_set), &cpu_set) != 0) {
        return errno;
    }
    return 0;
}

/**
 * One aligned host buffer; destructor frees it.
 *
 * Payload alignment is the OS memory page (sysconf(_SC_PAGESIZE), typically
 * 4 KiB). That is the kernel granule for ibv_reg_mr, not the ring stride.
 * An unaligned start would register neighboring OS pages. Flags use 64 bytes
 * (a cache line), not an OS page: the NIC never DMA's them.
 *
 * Pinning is ibv_reg_mr's job, not this type's.
 */
class HostAllocation {
public:
    HostAllocation() = default;
    HostAllocation(std::size_t size, std::size_t alignment);
    ~HostAllocation();

    HostAllocation(const HostAllocation&) = delete;
    HostAllocation& operator=(const HostAllocation&) = delete;
    HostAllocation(HostAllocation&& other) noexcept;
    HostAllocation& operator=(HostAllocation&& other) noexcept;

    void* get() const;

private:
    void reset() noexcept;
    void swap(HostAllocation& other) noexcept;

    void* address_ = nullptr;
};

enum class RxPublishResult {
    Published,
    Collision,
    InvalidSlot,
};

enum class TxClaimResult {
    Claimed,
    NotReady,
};

/** Validate Config without creating rings or network resources. */
// any_tx_depth allows a caller-transmit ring whose depth is not a power of two.
void validate_config(
    const CpuRoceTransceiver::Config& config, bool any_tx_depth = false);

/** Validate peer fields that connect() consumes, without touching the QP. */
void validate_peer(
    const CpuRoceTransceiver::Config& config,
    const CpuRoceTransceiver::Peer& peer);

struct RxCompletion {
    // Posted receive to re-arm. Not the payload slot.
    std::uint32_t credit_id = 0;
    // Present for Send-with-Immediate and Write-with-Immediate; host byte order.
    std::optional<std::uint32_t> immediate_data;
};

/** Build the handled-path callback record. The caller names the credit. */
RxCompletion make_rx_completion(
    const ::ibv_wc& wc, std::uint32_t credit_id);

/** Upper bound for enable_handled_receive()'s credit count. */
constexpr std::uint32_t kMaxHandledReceiveCredits = 4096;

inline std::size_t host_page_size()
{
    const long result = sysconf(_SC_PAGESIZE);
    return result > 0 ? static_cast<std::size_t>(result) : 4096;
}

/**
 * Handled receive for Mode::Rx.
 *
 * enable_handled_receive() before setup() selects this path. The credit
 * count is an argument of that call because it applies only here: Stall,
 * Duplex, Forward, and Unified still post one receive WQE per slot. It is not a
 * field of CpuRoceTransceiver::Config. Credits are zero-SGE, so only
 * RDMA Write-with-Immediate is accepted.
 *
 * The callback runs on the monitor thread after the CQE status, opcode, and
 * credit id are checked, and before that credit is reposted. The completion
 * carries the credit id and the host-order immediate. It does not carry a
 * decoded slot. A throw terminalizes the monitor and suppresses repost.
 * Returning reposts the credit, including when the callback drops an
 * immediate whose slot is outside its ring. This path posts
 * receive_credit_count zero-SGE credits instead of one WQE per slot. The
 * callback must return promptly and must not call CpuRoceTransceiver
 * methods. Its context must remain valid until blocking_monitor() returns.
 * The callback is deliberately not noexcept: receiver failures propagate
 * through the monitor's terminal catch. Public RX flags remain zero and
 * get_rx_collision_count() remains zero on this path; the receiver owns
 * publication and collision policy.
 */
class CpuRoceRxCompletionHandler {
public:
    using Fn = void (*)(void* context, RxCompletion completion);

    explicit CpuRoceRxCompletionHandler(CpuRoceTransceiver& transceiver)
        : transceiver_(transceiver)
    {
    }

    void enable_handled_receive(
        Fn fn, void* context, std::uint32_t receive_credit_count);

    /**
     * Ask setup() to place device-ring metadata trailers in host memory.
     * Call before setup(). Only this handled-receive path opts in. Other RX
     * modes keep trailers in the ring slot.
     */
    void request_host_metadata();

    /**
     * Base of the host trailers, or null when setup declined.
     *
     * One trailer is `metadata_size` bytes at `slot * metadata_size`. The
     * pointer is valid while the handled callback can run: until
     * blocking_monitor() returns, or until close(). Read it from the callback.
     * Do not call this method from the callback.
     */
    const std::uint8_t* host_metadata() const;

private:
    CpuRoceTransceiver& transceiver_;
};

using DuplexWorkerBody = std::function<void()>;
using DuplexWorkerSpawn = std::function<std::thread(DuplexWorkerBody)>;

/** Default spawn: direct std::thread construction. */
std::thread spawn_duplex_worker(DuplexWorkerBody body);

/**
 * Duplex coordinator: verbs-free so it is directly testable.
 *
 * Spawns one thread per body, records only the first failure, and sets
 * exit_flag so the peer worker leaves its poll loop. Both threads are
 * joined before returning, including when the second spawn throws, after
 * which the first failure is rethrown. Returns normally when both bodies
 * return. A throwing join() is not recovered: the thread may remain
 * joinable and destroying it would terminate. Under these invariants
 * (joinable check, coordinator is not a worker, each thread joined once)
 * that path is not expected.
 *
 * The bodies and exit_flag must outlive the call. Because the joins are
 * unconditional, a body may safely capture this caller's locals by reference.
 *
 * Each body that does not throw must observe exit_flag and return promptly
 * after it becomes true; otherwise join hangs. spawn must return a joinable
 * thread that runs the supplied body, or throw without detaching a worker.
 */
void run_duplex_workers(
    std::atomic<bool>& exit_flag,
    const DuplexWorkerBody& rx_body,
    const DuplexWorkerBody& tx_body,
    const DuplexWorkerSpawn& spawn);

inline void run_duplex_workers(
    std::atomic<bool>& exit_flag,
    const DuplexWorkerBody& rx_body,
    const DuplexWorkerBody& tx_body)
{
    run_duplex_workers(exit_flag, rx_body, tx_body, spawn_duplex_worker);
}

class ThreadAffinityRestoreError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

/**
 * Temporarily restrict the calling thread to one valid Linux CPU ID.
 * A missing CPU leaves the affinity unchanged. best_effort leaves the
 * current mask in place after a failed affinity read or pin instead of
 * throwing.
 */
class ThreadAffinityScope {
public:
    explicit ThreadAffinityScope(std::optional<int> cpu, bool best_effort = false);
    ~ThreadAffinityScope() noexcept;

    ThreadAffinityScope(const ThreadAffinityScope&) = delete;
    ThreadAffinityScope& operator=(const ThreadAffinityScope&) = delete;
    ThreadAffinityScope(ThreadAffinityScope&&) = delete;
    ThreadAffinityScope& operator=(ThreadAffinityScope&&) = delete;

    /**
     * Restore the original mask. Called explicitly on normal monitor return
     * so a failure can be reported; the destructor retries without throwing
     * while another exception is propagating.
     */
    void restore();

private:
    cpu_set_t original_ {};
    bool active_ = false;
};

/** Local IPv4 RoCE v2 GID chosen for this transceiver (netdev required). */
struct LocalRoceV2Ipv4Gid {
    ::ibv_gid gid {};
    int index = -1;
    std::uint32_t ndev_ifindex = 0;
    std::string netdev;
};

LocalRoceV2Ipv4Gid query_local_roce_v2_ipv4_gid(
    ibv::Context& context, std::uint32_t port, const std::string& local_ip = {});

/** Whether the selected GID-table entry still identifies the same local path. */
bool selected_gid_matches(
    const LocalRoceV2Ipv4Gid& selected,
    const std::optional<::ibv_gid_entry>& current) noexcept;

enum class AsyncEventDisposition {
    Silent,
    Terminal,
    ValidateSelectedGid,
    UnexpectedInvariant,
};

/**
 * Classify an acknowledged verbs async event without querying hardware.
 *
 * GID validation remains a separate operation because it reads the selected
 * GID-table entry. Unknown future event values are terminal.
 */
AsyncEventDisposition classify_async_event(
    ::ibv_event_type type,
    std::uint32_t event_port,
    std::uint32_t device_port) noexcept;

/**
 * Common payload, flag storage, and geometry for one ring.
 *
 * Payload storage is either owned here or borrowed from HostMemory,
 * DmaBufMemory, or GpuPointerMemory. Flag storage is always owned here.
 * This type contains no RX/TX protocol, sequence mapping, CQ policy, or
 * lifecycle state. slot_count must be nonzero; a power-of-two constraint
 * is TxRing's, not storage's.
 */
class RingStorage {
public:
    // gpu_pointer_iova_zero is read only for GpuPointerMemory. RX passes
    // true so the SGE base is 0, matching the peer-visible IOVA. TX leaves
    // it false so the SGE base is the GPU pointer. Host and DMA-BUF ignore it.
    RingStorage(
        const PayloadMemory& memory, std::size_t stride, std::uint32_t slot_count,
        bool gpu_pointer_iova_zero = false);

    RingStorage(const RingStorage&) = delete;
    RingStorage& operator=(const RingStorage&) = delete;
    RingStorage(RingStorage&&) = delete;
    RingStorage& operator=(RingStorage&&) = delete;

    std::uint8_t* data() const noexcept { return data_; }
    std::uint64_t* flags() const noexcept { return flags_; }
    std::size_t stride() const noexcept { return stride_; }
    std::uint32_t slot_count() const noexcept { return slot_count_; }
    std::uint64_t slot_address(std::uint32_t slot) const;
    std::uint64_t slot_sge_address(std::uint32_t slot) const;
    const std::optional<DmaBufMemory>& dmabuf() const noexcept
    {
        return dmabuf_;
    }
    bool gpu_pointer() const noexcept { return gpu_pointer_; }

private:
    // owned_data_ stays empty for borrowed HostMemory, DmaBufMemory, and
    // GpuPointerMemory. data_ points at the selected payload. flags_ is a typed
    // view of owned_flags_.
    HostAllocation owned_data_;
    HostAllocation owned_flags_;
    std::uint8_t* data_ = nullptr;
    std::uint64_t* flags_ = nullptr;
    std::uint64_t sge_base_ = 0;
    std::size_t stride_ = 0;
    std::uint32_t slot_count_ = 0;
    std::optional<DmaBufMemory> dmabuf_;
    bool gpu_pointer_ = false;
};

/**
 * Flag-only publication protocol for an RX ring.
 *
 * A zero flag is free; a non-zero flag is the slot data address. Explicit
 * slots come from a CQE and are rejected when out of range, never wrapped.
 * One monitor owns publication; an acquire load checks occupancy and a
 * release store publishes the completed slot to consumers. CQ loops own
 * overflow, shutdown, and receive-credit policy.
 */
class RxRing {
public:
    // Selected by the payload type. Host and transceiver-owned memory accept
    // Send, so the WQE carries an SGE. DMA-BUF and a GPU pointer accept only
    // Write-with-Immediate, so the WQE is credit-only. This is not a Config
    // field.
    enum class ReceivePolicy {
        SendAndWriteWithImm,
        WriteWithImmOnly,
    };

    RxRing(const PayloadMemory& memory, std::size_t stride, std::uint32_t slot_count);

    RxRing(const RxRing&) = delete;
    RxRing& operator=(const RxRing&) = delete;
    RxRing(RxRing&&) = delete;
    RxRing& operator=(RxRing&&) = delete;

    std::uint8_t* data() const noexcept { return storage_.data(); }
    std::uint64_t* flags() const noexcept { return storage_.flags(); }
    std::size_t stride() const noexcept { return storage_.stride(); }
    std::uint32_t slot_count() const noexcept { return storage_.slot_count(); }
    std::uint64_t slot_address(std::uint32_t slot) const
    {
        return storage_.slot_address(slot);
    }
    std::uint64_t slot_sge_address(std::uint32_t slot) const
    {
        return storage_.slot_sge_address(slot);
    }
    const std::optional<DmaBufMemory>& dmabuf() const noexcept
    {
        return storage_.dmabuf();
    }
    bool gpu_pointer() const noexcept
    {
        return storage_.gpu_pointer();
    }
    // True when this ring's receive policy accepts an ordinary Send.
    bool accepts_send() const noexcept
    {
        return receive_policy_ == ReceivePolicy::SendAndWriteWithImm;
    }
    // The WQE carries an SGE exactly when Send is accepted.
    bool posts_receive_sge() const noexcept { return accepts_send(); }

    /**
     * Publish an explicit RX slot without blocking the CQ thread.
     *
     * `slot` is the ring index decoded from the CQE (IMM or wr_id), not an
     * OS memory page. InvalidSlot reports an index outside the ring.
     *
     * Collision means the previous publication was not released before this
     * CQE arrived. Only the handoff is skipped: the new payload is already in
     * the slot. Nothing here can tell an unread slot from one the consumer is
     * reading, so this reports the missing release and nothing more. The
     * caller may retry this publication handoff after the consumer releases
     * the flag; the CQE and payload are not processed again.
     *
     * Send and Write-with-Imm both consume a receive WQE. For Send, its SGE
     * selects payload placement; for Write-with-Imm, remote_addr selects
     * placement and the WQE carries the immediate notification. Delaying the
     * consumed WQE's repost reduces available RQ credit and can bound future
     * notified operations once the remaining credit is exhausted, but cannot
     * undo DMA that completed before this Collision was observed.
     */
    RxPublishResult publish_rx(std::uint32_t slot);

private:
    RingStorage storage_;
    ReceivePolicy receive_policy_ = ReceivePolicy::SendAndWriteWithImm;
};

/**
 * Doorbell protocol for a TX ring.
 *
 * The public flag is the producer-to-transport handoff: the exact slot
 * address is an unclaimed publication and zero means no publication. It is
 * cleared with a release-store at claim, not at completion. The cyclic TX
 * loop bounds its outstanding sequence window to the ring depth, which
 * prevents posting the same slot twice before its send CQE. The requested SQ
 * depth is that same ring depth; the provider may grant a larger SQ. The
 * window cannot prevent producer writes. Zero never signals that the payload
 * may be reused: inline
 * shortens the NIC's read of the payload to the synchronous ibv_post_send()
 * call that follows the claim, but the flag exposes only claim, not that
 * call's completion, so the producer cannot observe inline copy from the
 * flag. Non-inline payload must additionally remain stable through the send
 * CQE. All payload reuse pacing is the application protocol's responsibility.
 * Exactly one TX poller owns try_claim_tx(). CQ loops own occupancy,
 * shutdown, and verbs policy. The cyclic flag path requires a power-of-two
 * slot_count because slot_index() maps sequence with a mask. Caller
 * transmit passes cyclic false and accepts any depth from 1 to 4096; it
 * does not call slot_index().
 */
class TxRing {
public:
    TxRing(const PayloadMemory& memory, std::size_t stride,
        std::uint32_t slot_count, bool cyclic = true);

    TxRing(const TxRing&) = delete;
    TxRing& operator=(const TxRing&) = delete;
    TxRing(TxRing&&) = delete;
    TxRing& operator=(TxRing&&) = delete;

    std::uint8_t* data() const noexcept { return storage_.data(); }
    std::uint64_t* flags() const noexcept { return storage_.flags(); }
    std::size_t stride() const noexcept { return storage_.stride(); }
    std::uint32_t slot_count() const noexcept { return storage_.slot_count(); }
    std::uint64_t slot_address(std::uint32_t slot) const
    {
        return storage_.slot_address(slot);
    }
    std::uint64_t slot_sge_address(std::uint32_t slot) const
    {
        return storage_.slot_sge_address(slot);
    }
    const std::optional<DmaBufMemory>& dmabuf() const noexcept
    {
        return storage_.dmabuf();
    }
    bool gpu_pointer() const noexcept
    {
        return storage_.gpu_pointer();
    }

    /** Map a local monotonic sequence; TX slot_count is a power of two. */
    std::uint32_t slot_index(std::uint64_t sequence) const;

    /**
     * Try to consume a published TX flag for one post without blocking.
     *
     * `slot` is a ring index, not a monotonic sequence. Both TX operations map
     * next_post_sequence through slot_index() before calling this;
     * WriteWithImm also names that slot in IMM. Out-of-range slots throw
     * std::out_of_range and do not wrap. A successful claim consumes the
     * publication with a release-store of zero. The posted SGE address is
     * always this ring's slot_sge_address(slot), never a value decoded from the
     * flag. An out-of-contract same-address republication is indistinguishable
     * and may coalesce; replacing the load/store with a claim CAS cannot
     * detect or prevent that ABA case. This primitive does not track WQE
     * lifetime; the cyclic TX loop must enforce its occupancy window before
     * calling it.
     */
    TxClaimResult try_claim_tx(std::uint32_t slot);

private:
    RingStorage storage_;
};

/**
 * Whether the cyclic TX loop may consume another publication.
 *
 * The software occupancy window and requested SQ depth are slot_count. The
 * provider may grant a larger SQ; this predicate does not inspect that
 * grant. Unsigned subtraction preserves the small outstanding distance when
 * the monotonic sequence counters wrap.
 */
inline bool cyclic_tx_window_has_capacity(
    std::uint64_t next_post_sequence,
    std::uint64_t next_completion_sequence,
    std::uint32_t slot_count) noexcept
{
    return next_post_sequence - next_completion_sequence < slot_count;
}

/**
 * Build an SGE against the RX ring's local MR.
 *
 * Host rings use their process virtual address. DMA-BUF rings use literal
 * zero, the registration IOVA, plus the slot offset.
 */
inline ibv::ScatterGatherElement make_local_rx_sge(
    const RxRing& ring,
    std::uint32_t slot,
    std::uint32_t length,
    std::uint32_t local_lkey)
{
    return ibv::ScatterGatherElement(
        ring.slot_sge_address(slot), length, local_lkey);
}

// Host send SGEs use their registered virtual address. DMA-BUF send SGEs use
// the fixed zero registration IOVA. Both use the local send lkey.
inline ibv::ScatterGatherElement make_local_tx_sge(
    const TxRing& ring,
    std::uint32_t slot,
    std::uint32_t length,
    std::uint32_t local_lkey)
{
    return ibv::ScatterGatherElement(
        ring.slot_sge_address(slot), length, local_lkey);
}

/**
 * Payload slot and consumed RQ identity from one recv CQE.
 *
 * wr_id is always the posted receive to re-arm. The payload slot is wr_id for
 * IBV_WC_RECV (Send and Send-with-Imm) and the 12-bit IMM page for
 * IBV_WC_RECV_RDMA_WITH_IMM. Those two identities are never collapsed.
 */
struct RxCqeDecode {
    enum class Status {
        Ok,
        FailedStatus,
        UnsupportedOpcode,
        UnexpectedSendForCreditOnly,
        MissingImm,
        InvalidWrId,
        InvalidImmSlot,
        // Appended so the values above stay the ones decode_rx_cqe emits.
        UnexpectedSend,
    };

    Status status = Status::FailedStatus;
    std::uint32_t payload_slot = 0;
    std::uint64_t recv_wr_id = 0;
};

RxCqeDecode decode_rx_cqe(
    const ::ibv_wc& wc,
    std::uint32_t slot_count,
    bool write_with_imm_only);

// Handled path: status, Write-with-Immediate, and credit id. Does not read the
// immediate slot. A Send is rejected because these credits have no buffer.
RxCqeDecode decode_handled_rx_cqe(
    const ::ibv_wc& wc,
    std::uint32_t credit_count);

struct TxCqeDecode {
    enum class Status {
        Ok,
        FailedStatus,
        UnsupportedOpcode,
        OutOfSequence,
    };

    Status status = Status::FailedStatus;
    std::uint64_t sequence = 0;
};

TxCqeDecode decode_tx_cqe(
    const ::ibv_wc& wc,
    std::uint64_t expected_sequence,
    ibv_wc_opcode expected_opcode = IBV_WC_SEND) noexcept;

// Observes the close()/monitor handshake. Not a public readiness API.
// close() already waits on monitor_active; tests wait for the same predicate
// so they do not destroy verbs before blocking_monitor() has claimed it.
struct CpuRoceMonitorTestAccess {
    static bool wait_until_active(CpuRoceTransceiver& transceiver, int timeout_ms);
};

/**
 * Caller-posted UC SEND for Mode::Tx.
 *
 * configure() before setup(). configured() means the seam is selected, not
 * that the queue pair is connected. post() fails until the queue pair is
 * connected and healthy. poll() maps the whole batch, then terminalizes on a
 * non-flush error, so that error is still returned. Flush completions stay
 * readable after the queue pair is moved to error. After the queue pair is
 * destroyed, poll() returns zero and queue_pair_destroyed() is true so the
 * caller can retire leases. The handle is a borrowed view of the transceiver,
 * not of memory verbs.reset() frees. The caller owns any extra host bytes;
 * this seam only registers them.
 *
 * One thread owns post() and poll(). close() runs only after that thread
 * has stopped. fail() does not wait for the owner. The caller excludes
 * that thread before fail(), drain_async(), or close(). Taking the
 * teardown mutex is not that exclusion. post() and poll() do not take the
 * teardown mutex; each call reads the verbs pointer again.
 * drain_async(), queue_pair_destroyed(), close(), and terminalize keep it.
 * There is no ready flag: an errored queue pair stays allocated so poll()
 * can return its flush completions.
 *
 * Plain Mode::Tx keeps the flag loop. This seam does not start it, and it
 * rejects WriteWithImm. Send completions are signaled and not inline.
 */
class CpuRoceCallerTx {
public:
    struct Config {
        // Caller-owned. Null and zero bytes mean one scatter/gather entry.
        void* extra_host = nullptr;
        std::uint32_t extra_host_bytes = 0;
        // Program the path MTU to fit the netdev MTU minus 44 bytes.
        // A failed netdev query keeps the port MTU.
        bool clamp_path_mtu = false;
    };

    explicit CpuRoceCallerTx(CpuRoceTransceiver& transceiver)
        : transceiver_(transceiver)
    {
    }

    CpuRoceCallerTx(const CpuRoceCallerTx&) = delete;
    CpuRoceCallerTx& operator=(const CpuRoceCallerTx&) = delete;

    // Mode::Tx and TxOperation::Send only. Call before setup().
    // Rejected after close() or a terminal failure.
    void configure(Config config);
    bool configured() const;

    enum class Status {
        Ok,
        Flush,
        Error,
    };

    struct Completion {
        std::uint64_t wr_id = 0;
        Status status = Status::Error;
        // True only for a successful SEND. The opcode is defined only then.
        bool send = false;
    };

    // extra is null, or one range inside the registered extra host bytes.
    // Returns the last-packet 24-bit PSN of the accepted message. The send
    // queue starts at 0. The counter advances only after ibv_post_send
    // accepts the request. A verbs post failure moves the queue pair to
    // error and throws. A validation error does not.
    std::uint32_t post(std::uint32_t slot, std::uint32_t length, std::uint64_t wr_id,
        const std::uint8_t* extra, std::uint32_t extra_length);

    // Zero when the queue pair is gone. Flush completions are still returned
    // while it remains in error. A poll failure terminalizes and throws.
    // Zero is not proof that a posted send has retired.
    int poll(Completion* out, int max_entries);

    // A fatal async event terminalizes and throws. Returns when the verbs
    // objects are already gone. Does not wait for the post/poll owner; the
    // caller excludes that thread first.
    void drain_async();

    // True when configured and no queue pair is present: before setup(), and
    // after the queue pair is destroyed. An errored queue pair that is still
    // allocated returns false, so flush completions can still be polled.
    // This is a retirement proof only after setup() has succeeded.
    bool queue_pair_destroyed() const;

    // Legal only on a configured caller-post transmit path. Move the queue
    // pair to error and keep it so flush completions can be polled. A second
    // call does not destroy it. If the transition to error fails, the queue
    // pair is destroyed immediately. close() still releases a queue pair that
    // remained allocated. Does not wait for the post/poll owner.
    void fail();

private:
    CpuRoceTransceiver& transceiver_;
};

} // namespace hololink::transport::roce::detail

#endif /* SRC_HOLOLINK_TRANSPORT_ROCE_CPU_ROCE_TRANSCEIVER_INTERNAL_HPP */
