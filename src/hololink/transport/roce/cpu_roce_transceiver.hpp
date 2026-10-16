/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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
 */

#ifndef SRC_HOLOLINK_TRANSPORT_ROCE_CPU_ROCE_TRANSCEIVER_HPP
#define SRC_HOLOLINK_TRANSPORT_ROCE_CPU_ROCE_TRANSCEIVER_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>

namespace hololink::transport::roce {

/**
 * Transmit DMA-BUF registration failed. setup() throws this only when the
 * transmit ring's ibv_reg_dmabuf_mr call fails, so the frame layer can retry
 * that pool once as a GPU pointer. Receive DMA-BUF registration and every
 * other verbs failure throw ibv::Error.
 */
struct RegistrationError : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

/**
 * Contents of one slot: payload region and optional HSB trailer.
 *
 * Shared ring-slot layout for this namespace, not specific to one engine.
 * Ring pitch and slot count are not here; each RingConfig owns those.
 *
 * For TX, payload_size is the exact fixed payload length used by ordinary
 * work requests. Send and WriteWithImm use inline data when the payload is
 * eligible and the QP supports its full size; otherwise they use registered
 * TX memory.
 * For RX, payload_size is the declared valid region of a
 * stride-sized receive slot, not a guarantee that each CQE wrote that many
 * fresh bytes. Host receive WQEs are posted for stride. DMA-BUF and
 * GPU-pointer receive WQEs are credit-only for RDMA
 * Write-with-Immediate. CQE byte_len is not
 * published. Actual message length lives in the application's in-band
 * framing. Consumers must validate and obey that framing and must not treat
 * unread slot tail as new data. In Unified mode, TX payload_size is the
 * maximum callback response; each dispatch result supplies the actual
 * transmitted byte count.
 *
 * RX metadata_size == 0 selects the no-metadata path; metadata_offset is then
 * unused and is not range-checked. Otherwise the offset must follow the
 * payload and the metadata must fit inside the slot. HSB senders append the
 * metadata as a trailer, after the payload has been padded up to the 128-byte
 * HSB protocol boundary, so metadata_offset is that padded offset rather than
 * a position inside the payload. Like RoceReceiver, this engine uses the
 * offset as given; computing a default from payload_size is the caller's job.
 * The transport validates that layout and delivers stride-sized slots;
 * locating and deserializing the trailer is the consumer's job. There is no
 * metadata accessor. TX metadata_size must remain zero because no TX trailer
 * contract is defined.
 */
struct FrameLayout {
    std::size_t payload_size = 0;
    std::size_t metadata_offset = 0;
    std::size_t metadata_size = 0;
};

/**
 * Request host memory allocated and owned by CpuRoceTransceiver.
 *
 * The transceiver frees that memory in close(). Ring pointers returned by the
 * accessors are invalid afterwards.
 */
struct TransceiverAllocation { };

/**
 * Caller-owned host memory.
 *
 * The allocation must remain valid until close() returns. size is the complete
 * accessible span beginning at address. Borrowed RX memory must be OS-page
 * aligned: the zero-based RX registration used by this transport requires
 * page-aligned host memory. TX uses a virtual-address IOVA, so borrowed TX
 * memory is not held to that alignment.
 */
struct HostMemory {
    void* address = nullptr;
    std::size_t size = 0;
};

/**
 * Borrowed DMA-BUF and its application-visible address.
 *
 * address identifies the same first byte as offset within fd. If they
 * disagree, the application and the NIC operate on different storage. address
 * is opaque and may be a host pointer or a device address; it is not
 * necessarily CPU-dereferenceable. The fd and allocation must remain valid
 * until close() returns. offset is the byte offset into the fd passed to
 * ibv_reg_dmabuf_mr (rdma-core >= 34; building the transport requires it).
 * libibverbs requires offset and the registration IOVA to have the same OS
 * page offset (kernel page). This engine always registers at IOVA zero, so
 * offset must be OS-page-aligned. size is the accessible span beginning at offset. address
 * is used only for application-visible ring pointers; the transport never
 * dereferences it.
 *
 * DMA-BUF changes only payload storage. Slot flags remain CPU-owned host
 * memory; this is not GPU-resident polling. Flag acquire/release order CPU
 * memory only. The allocation owner must provide exporter-specific
 * synchronization: CUDA TX writes must complete and become GPUDirect-visible
 * before the TX flag is published; a CUDA RX consumer must establish CUDA /
 * GPUDirect ordering after observing the RX flag and must not access a region
 * while the NIC writes it. Configure CUDA allocations for GPUDirect,
 * including CU_POINTER_ATTRIBUTE_SYNC_MEMOPS and
 * cuFlushGPUDirectRDMAWrites() when the device requires it.
 *
 * Duplex RX and TX backing storage must be physically disjoint, including
 * dup()ed fds, aliased mappings, and HostMemory/DmaBufMemory/GpuPointerMemory
 * views of one allocation. Setup's same-integer-fd overlap check is
 * best-effort.
 *
 * Inline transmit is disabled for DMA-BUF rings because the CPU cannot copy
 * from an opaque device address while posting the work request. Passing this
 * type selects the Write-with-Immediate receive policy: receive WQEs carry
 * notification credit without an SGE, and an ordinary inbound Send is
 * rejected. There is no separate receive-policy setting.
 */
struct DmaBufMemory {
    void* address = nullptr;
    int fd = -1;
    std::uint64_t offset = 0;
    std::size_t size = 0;
};

/**
 * Caller-owned GPU pointer.
 *
 * address is a GPU pointer, not a CPU pointer. The allocation must remain
 * valid until close() returns. size is the accessible span beginning at
 * address. The transport never dereferences it and does not export a DMA-BUF:
 * the caller chooses this form when it wants ibv_reg_mr_iova. That
 * registration needs nvidia-peermem. DMA-BUF registration does not. If a
 * DMA-BUF export succeeded, pass DmaBufMemory instead. A failed DMA-BUF
 * registration is not retried here.
 *
 * RX is registered at IOVA zero, so address must be OS-page aligned. TX uses
 * address as both the registration IOVA and the SGE base. Passing this type
 * selects the Write-with-Immediate receive policy: the work request carries
 * notification credit and no SGE, and an ordinary inbound Send is rejected.
 * There is no separate receive-policy setting. Unified mode and inline
 * transmit are also rejected, as they are for DMA-BUF.
 */
struct GpuPointerMemory {
    void* address = nullptr;
    std::size_t size = 0;
};

/** Storage policy for an RX or TX ring; reusable by other types in this namespace. */
using PayloadMemory = std::variant<TransceiverAllocation, HostMemory, DmaBufMemory, GpuPointerMemory>;

/** Independent storage and frame geometry for one RX or TX ring. */
struct RingConfig {
    PayloadMemory payload_memory = TransceiverAllocation {};
    std::size_t stride = 0;
    std::uint32_t slot_count = 0;
    FrameLayout frame_layout;
};

namespace detail {
    class CpuRoceCallerTx;
    class CpuRoceRxCompletionHandler;
    struct CpuRoceMonitorTestAccess;
}

/**
 * CPU-driven, CQ-spinning RoCE transceiver.
 *
 * @code
 * CpuRoceTransceiver::Config config;
 * config.ibv_name = "mlx5_0";
 * config.ibv_port = 1;
 * config.rx.stride = 256;
 * config.rx.slot_count = 8;
 * config.rx.frame_layout.payload_size = 128;
 * config.tx.stride = 64;
 * config.tx.slot_count = 8;
 * config.tx.frame_layout.payload_size = 32;
 * CpuRoceTransceiver transceiver { config };
 * transceiver.setup();
 * @endcode
 *
 * Lifecycle:
 *  1. setup() creates and registers only the payload rings used by the mode,
 *     opens the RDMA device, creates two CQs and a UC QP, moves the QP to INIT, and
 *     posts rx.slot_count receive WQEs for every mode with an RX ring,
 *     including CallerDriven. Recv posting is legal in INIT; the peer cannot
 *     send until RTR. Rx, Duplex, Forward, and Unified leave those
 *     completions until blocking_monitor() polls them. CallerDriven leaves them
 *     until the caller drains them with rx_poll(). Tx mode does not post an
 *     RQ.
 *     external_frame_memory() is 0 (peer-visible RX IOVA; this engine does not
 *     advertise any other base).
 *     get_qp_number() and get_rkey() then report the local QP and RX memory
 *     region.
 *  2. connect() supplies the peer information and moves the QP to RTR/RTS.
 *     start() is the setup()+connect() convenience path using config.peer.
 *  3. blocking_monitor() runs the active CQ path until close() or an abnormal
 *     failure that throws. Duplex uses independent RX and TX polling contexts;
 *     Rx, Tx, Forward, and Unified each use one polling context. CallerDriven
 *     starts no thread: blocking_monitor() returns, and the caller drives
 *     rx_poll()/tx_publish(). Rx publishes a completed slot before re-arming
 *     its consumed recv WQE; Tx atomically consumes each public doorbell at
 *     claim and tracks completion privately.
 *  close() is terminal: setup() after close() fails so borrowed payload
 *  memory is not re-registered after the caller is entitled to free it.
 *
 *  Synchronized pairs are close() against blocking_monitor(), and close()
 *  against close(). Stop CallerDriven rx_poll()/tx_publish(), and stop the
 *  caller-post thread, before calling close().
 *  Scalar getters
 *  (get_qp_number(), get_rkey(), geometry, collision count) may be called
 *  concurrently with close() or monitor terminalization and return a
 *  data-race-free snapshot.
 *  Using a previously returned ring pointer concurrent with close() is
 *  unsupported: the snapshot does not keep that storage alive. Every other
 *  combination — setup() concurrent with close() — is unsupported, the same
 *  as calling start() during the destructor. Consumers cache the ring
 *  pointers once between setup() and close(). External RX/TX flag threads
 *  must poll their own cancellation signal; they cannot see the
 *  transceiver's private exit flag. Stop and join those threads before
 *  close() invalidates the cached pointers.
 *
 * Configuration strings are copied at construction. The object is
 * non-copyable and non-movable because it owns verbs resources and registered
 * ring memory. Except for Duplex's two independent polling contexts, the
 * active CQ loop runs on a single polling context; close() interrupts those
 * loops through one shared exit flag. This engine uses UC: a lost UC
 * message is not recovered, so the policy named
 * Stall is unbounded with respect to the peer. Ring primitives are
 * nonblocking; the Rx/Duplex monitor waits on an occupied RX flag and TX
 * waits only by retrying its nonblocking claim. Send and Write-with-Imm both
 * consume peer RQ credit, but this engine exposes neither the peer's current
 * credit count nor a delivery acknowledgement. Before every TX publication,
 * including the first lap, the application protocol must ensure that the
 * peer's next expected receive WQE is posted; connect() validates only the
 * peer slot pitch and cannot prove that fill. A peer-slot release can serve
 * as that signal only when the application has independently established the
 * relevant receive-slot mapping, including compatible ring depths and an
 * aligned, loss-free sequence. This receiver publishes before re-arming the
 * consumed WQE; that does not make a local slot number identify the peer's
 * next WQE. A successful local CQE proves neither delivery nor remote RQ
 * credit. A lost UC Send can desynchronize an application's assumed
 * Send-to-RQ-slot sequence. Applications relying on slot correspondence need
 * request sequencing/timeouts and must treat a detected mismatch as a session
 * failure.
 *
 */
class CpuRoceTransceiver {
public:
    /**
     * Which payload rings a mode allocates, and which ring (if any) is the
     * outbound geometry for peer checks and inline negotiation.
     *
     *            rx_ring   tx_ring   outbound
     * Rx           yes        -         -
     * Tx            -        yes       tx
     * Duplex       yes       yes       tx
     * Forward      yes        -        rx
     * Unified      yes       yes       tx
     * CallerDriven  yes       yes       tx
     *
     * Forward transmits from RX storage and must not allocate a TxRing over
     * that span. Unified receives into RX storage and transmits callback
     * responses from the paired TX slot. CallerDriven uses the same rings, but
     * the caller thread drives them through rx_poll() and tx_publish().
     */
    enum class Mode {
        Rx,
        Tx,
        Duplex,
        Forward,
        Unified,
        CallerDriven,
    };

    enum class TxOperation {
        Send,
        WriteWithImm,
    };

    struct Peer {
        // Destination UC QP. Must fit 24 bits and be non-zero for any mode
        // that transmits. Rx mode permits 0, matching the HSB receive-only QP.
        std::uint32_t qp = 0;
        // Required IPv4 address used to construct the remote RoCE v2 GID.
        std::string ip;
        // Peer RX IOVA base; zero is valid and expected for HSB.
        std::uint64_t rx_base = 0;
        // Required when a TX-driving mode uses WriteWithImm.
        std::uint32_t rx_rkey = 0;
        // Peer receive-slot pitch. TX-driving modes require it to cover the
        // outbound payload so a Send or WriteWithImm cannot exceed the peer's slot.
        // Ordinary Send does not constrain the peer's slot count: SEND names
        // no destination slot, so a peer whose receive ring is deeper than
        // this one is a valid configuration.
        std::size_t rx_stride = 0;
        // Peer receive-ring depth. Required and non-zero for TX-driving
        // WriteWithImm so each local outbound slot has a remote slot. Ignored
        // for ordinary Send (zero is permitted).
        std::uint32_t rx_slot_count = 0;
    };

    /**
     * Borrowed host RX and TX spans must not overlap in Duplex, Unified, and
     * CallerDriven: those modes use both payload rings. The same-address and
     * same-fd overlap checks are best-effort; aliased mappings remain the
     * caller's duty.
     * Rx, Tx, and Forward may share a
     * span; Forward transmits from RX storage. Only rings used by the selected
     * mode are registered. The peer-visible RX MR is always REMOTE_WRITE.
     * Host RX also has a LOCAL_WRITE virtual-address alias. DMA-BUF and
     * GPU-pointer RX have no CPU-addressable alias and use credit-only
     * receive WQEs. On Linux, each
     * host registration accounts the RX pages against RLIMIT_MEMLOCK, so
     * budget approximately twice the host RX ring size plus the other
     * registered rings.
     */
    struct Config {
        // Required libibverbs device name.
        std::string ibv_name;
        // Required non-zero port number representable by ibv_qp_attr::port_num.
        std::uint32_t ibv_port = 1;
        // Optional IPv4 selector for the local RoCE v2 GID. Empty selects the
        // first IPv4 RoCE v2 GID on ibv_port.
        std::string local_ip;
        Peer peer;
        // RX and TX storage, slot pitch/count, and frame layout are
        // independent except that Unified and CallerDriven pair slot N across
        // both rings and require equal slot counts. Every used slot_count is
        // non-zero and at most 4096 (the modern HSB 12-bit IMM page field). A
        // TX ring must be a power of two. Tx, Duplex, and Unified map a cyclic
        // send sequence with that mask. CallerDriven publishes slots in any
        // order and does not use the mask; it still goes through TxRing, so it
        // has the same requirement. RX and Forward may use any count in that
        // range; placement is the CQE identity, not a wrap.
        // A ring's payload must fit its own stride. This is not an OS memory
        // page; borrowed RX alignment uses sysconf(_SC_PAGESIZE).
        RingConfig rx;
        RingConfig tx;
        // Payload rings and MRs are instantiated by role: Rx and Forward use
        // RX, Tx uses TX, and Duplex, Unified, and CallerDriven use both.
        // Unused ring fields are not validated. Forward sends a fixed
        // rx.frame_layout.payload_size
        // from its RX MR regardless of received length. Getters
        // return nullptr/zero for a missing ring. Tx mode implements ordinary
        // Send and WriteWithImm. Duplex, Forward, and Unified TX behavior
        // belongs to their monitor paths.
        Mode mode = Mode::Duplex;
        // Verb posted on TX. RX is independent: it accepts the operation
        // reported by each CQE, so a peer using WriteWithImm while this side
        // uses Send is supported.
        TxOperation tx_operation = TxOperation::Send;
        // Optional worker placement for Duplex only. Set both or neither; the
        // two CPUs must differ. Unset: workers inherit the spawning context's
        // OS mask (taskset or cgroup/cpuset). That is supported for
        // general deployment but is not a latency-qualified configuration.
        // Qualification requires distinct explicit RX and TX CPUs.
        // rx_cpu and tx_cpu are valid only for Duplex. Single-poller modes
        // reject them.
        std::optional<int> rx_cpu;
        std::optional<int> tx_cpu;
        // Optional CPU placement for Rx, Tx, Forward, and Unified. Applied
        // after blocking_monitor() admits a connected instance and restored
        // when that call returns. Duplex rejects this field because its
        // independent polling contexts use rx_cpu and tx_cpu. CallerDriven
        // rejects it too: the caller owns the only polling thread and pins
        // that thread itself.
        std::optional<int> poll_cpu;
        // When set, a failed affinity read or poll_cpu pin leaves the
        // polling thread on its current mask. Default off: that failure
        // throws before the monitor latch, which is what transmit and
        // CUDA-Q keep. Duplex and CallerDriven ignore this field.
        bool poll_cpu_best_effort = false;
        // Receive-only. Program IBV_MTU_4096 instead of the port active MTU.
        // Duplex and transmit keep the active MTU, which those paths need.
        // Default off, so existing callers are unchanged.
        bool rx_path_mtu_4096 = false;
    };

    enum class UnifiedDispatchDisposition {
        Send,
        Drop,
        Fatal,
    };

    struct UnifiedDispatchResult {
        UnifiedDispatchDisposition disposition = UnifiedDispatchDisposition::Fatal;
        std::size_t bytes = 0;
    };

    /**
     * Per-slot Unified callback. rx_payload_size is the declared valid RX
     * region, not a CQE byte count; the callback must validate and obey its
     * in-band framing. tx_capacity is the maximum response size and may differ
     * from rx_payload_size. The callback writes the response into tx_slot and
     * returns Send, Drop, or Fatal. Send with bytes > 0 transmits exactly
     * bytes. Send with bytes == 0 and Drop are the same on the wire, matching
     * CUDA-Q's zero-length result: no TX WQE, then RX re-arm. Drop with
     * bytes != 0 is rejected so those two dispositions stay distinguishable.
     * Fatal stops the monitor and reports an error to its caller. An exception
     * escaping the callback is treated as a terminal monitor failure and is
     * rethrown by blocking_monitor(); it is not a per-frame drop.
     *
     * Before invocation, the transceiver waits for the prior same-slot TX
     * completion; tx_slot is exclusively writable by the callback until it
     * returns. The callback executes synchronously on the Unified polling
     * context, must be quick, and must not call close().
     */
    using UnifiedDispatchFn = UnifiedDispatchResult (*)(
        void* context,
        const void* rx_slot,
        std::size_t rx_payload_size,
        void* tx_slot,
        std::size_t tx_capacity);

    explicit CpuRoceTransceiver(Config config);
    ~CpuRoceTransceiver() noexcept;

    CpuRoceTransceiver(const CpuRoceTransceiver&) = delete;
    CpuRoceTransceiver& operator=(const CpuRoceTransceiver&) = delete;
    CpuRoceTransceiver(CpuRoceTransceiver&&) = delete;
    CpuRoceTransceiver& operator=(CpuRoceTransceiver&&) = delete;

    /**
     * Create and register local rings, CQs, and a UC QP without connecting to
     * a peer. The QP is left in INIT with rx.slot_count receive WQEs posted
     * for every mode with an RX ring: Rx, Duplex, Forward, Unified, and
     * CallerDriven. Tx mode leaves the RQ empty.
     *
     * Invalid or unsupported configuration is reported as std::invalid_argument
     * before any network resources are created. RX metadata layout, DMA-BUF
     * payload, and a GPU pointer are accepted: the transport validates
     * the layout and delivers stride-sized slots; locating and deserializing
     * the trailer is the consumer's job. TX metadata remains unsupported
     * because no TX trailer contract is defined. Unified and CallerDriven
     * reject DMA-BUF and a GPU pointer because the caller reads and writes
     * slot bytes on the CPU. Calling setup() in Unified mode without first
     * setting a callback is also
     * std::invalid_argument. Calling setup() after close() or a monitor failure
     * is reported as std::logic_error. Device discovery,
     * GID selection, verbs allocation, QP creation, and receive-queue fill
     * failures are reported as std::runtime_error.
     */
    void setup();

    /**
     * Move the QP created by setup() to RTR/RTS using the supplied peer.
     *
     * Malformed peer fields are rejected with std::invalid_argument before a
     * QP transition is attempted. Path-MTU and other verbs failures before
     * ibv_modify_qp are reported as std::runtime_error; the QP remains in INIT
     * and connect() may be retried. A failed transition after its bounded
     * internal retry throws with the libibverbs error; the QP state is then
     * indeterminate and a later connect() throws std::logic_error. Calling
     * connect() after close() or a monitor failure is also std::logic_error.
     * Close and recreate the transceiver before trying another connection.
     */
    void connect(Peer peer);

    /** Convenience for setup() followed by connect() using config.peer. */
    void start();

    /**
     * Spin until close() or a fatal datapath error. Rx, Tx, Forward, and
     * Unified each use one polling context. Forward echoes a fixed
     * rx.frame_layout.payload_size
     * from RX storage (CUDA-Q forward_loop / GPU forward kernel); it does not
     * trim to CQE byte_len. Duplex uses independent RX and TX polling
     * contexts, so it consumes two busy polling cores.
     * Unified handles RX CQE -> callback -> optional TX -> RX re-arm
     * synchronously, without publishing ring flags. TX posts ordinary Sends
     * or WriteWithImm and uses inline data for eligible payloads when
     * supported.
     * CallerDriven starts no polling thread. blocking_monitor() returns
     * immediately once the instance is connected, and it may be called again.
     * The caller then serially drives rx_poll() and tx_publish(). Calls must
     * not overlap, but successive calls may run on different threads. The
     * caller orders those calls, for example by joining the previous thread
     * or holding a mutex.
     * CallerDriven rejects poll_cpu; the caller pins its own thread.
     * For Rx, Tx, Forward, and Unified, Config::poll_cpu is applied to the
     * single polling context, independently
     * of thread ownership, after this call's preconditions pass and before the
     * one-shot latch, so a placement failure is retryable and an invalid
     * lifecycle call does not change the caller's affinity.
     * poll_cpu_best_effort leaves the current mask in place after a failed
     * affinity read or pin and does not throw. On a normal return,
     * a restoration failure is reported. During exception unwinding,
     * restoration is best-effort so the original datapath exception is
     * preserved. Neither poll_cpu failure is a datapath failure or moves the
     * QP to ERROR. In contrast, a Duplex rx_cpu/tx_cpu application failure is
     * terminal because the peer polling context may already have entered verbs
     * operations.
     * The selected local GID is revalidated at TX start and after
     * a GID_CHANGE event; a changed selected entry is terminal.
     *
     * close() called from another thread makes this return normally. Once
     * monitoring has begun, an abnormal failure throws and leaves this
     * instance unusable (QP moved to ERROR, or destroyed if that fails); the
     * caller must close/destroy it and create a new instance. That teardown
     * publishes under the same lock the getters take, so get_qp_number() and
     * get_rkey() may be polled from another thread while monitoring.
     * Precondition errors (for example calling before connect(), or a mode
     * with no monitor datapath) throw without poisoning the instance. This function does not abort the process. If it is run
     * from a std::thread, catch inside the thread entry; an exception that
     * escapes the entry invokes std::terminate.
     */
    void blocking_monitor();

    /**
     * CallerDriven only. Non-blocking. Returns true and writes `*out_slot` when
     * a request is waiting in that RX slot. Returns false when the CQ is
     * empty, or when every completed slot still has a response outstanding:
     * tx_flag[slot] != 0, or this transceiver's previous tx_publish() of that
     * slot has not completed. Those slots stay queued and a later call
     * returns one after it clears. The caller reads the request at
     * get_rx_ring_data_addr() + slot * get_rx_ring_stride() and writes the
     * response at the paired TX address. This call does not publish or clear
     * either flag. It re-arms the consumed recv WQE when it returns the slot,
     * so a dropped request does not leak RQ credit.
     *
     * Calls to rx_poll() and tx_publish() must not overlap. Successive calls
     * may run on different threads; the caller orders them, for example by
     * joining the previous thread or holding a mutex. This call checks async
     * events under any load, including while every call returns a request.
     * The call that observes a bad CQE or fatal async event, including a
     * selected-GID change, throws and terminalizes the instance.
     * Later calls return false. A call before connect() returns false, so a
     * dispatcher may spin while connect() blocks. Wrong mode is
     * std::logic_error and does not terminalize. After close() returns,
     * further calls return false.
     * Stop rx_poll() and tx_publish() before calling close().
     *
     * The peer must not write slot N again until it has consumed the response
     * to the request this call returned. The recv WQE is already re-posted,
     * so an early rewrite lands in the buffer the caller is still reading.
     * That holds when the peer chooses the slot, as with WriteWithImm. With
     * ordinary Send, the receive queue chooses the slot, so a later request
     * can reuse a buffer the caller is still reading. Callers that need the
     * slot held until the response is consumed use WriteWithImm.
     */
    bool rx_poll(std::uint32_t* out_slot);

    /**
     * CallerDriven only. Post TX slot `slot` to the peer. The SGE length is
     * tx.frame_layout.payload_size, the same fixed length the Tx monitor
     * sends, using the configured Send or WriteWithImm verb and inline data
     * when the QP granted it. Slots may be published in any order. Does not
     * read or store tx_flag; the caller owns that flag.
     *
     * The payload must already be stable in the TX slot. Inline data is
     * copied before this returns. Non-inline data must remain stable until
     * the send completes; rx_poll() will not return that slot again until
     * then. A second publish of a slot that is still in flight is
     * std::logic_error. A post failure throws and terminalizes the instance.
     * Wrong mode or an out-of-range slot does not terminalize. A call before
     * connect(), after close(), or after a failure that already threw
     * returns false.
     *
     * Stop rx_poll() and tx_publish() before calling close().
     * Fatal async events are detected by rx_poll(); keep calling it while
     * the instance is in use.
     */
    bool tx_publish(std::uint32_t slot);

    /**
     * Idempotently set the exit flag, wait for blocking_monitor(), and release
     * resources.
     *
     * Safe to call from a control thread while blocking_monitor() is running.
     * Concurrent close() waits until active monitoring has returned and rings
     * are released. Terminal for this instance: a later setup() fails.
     *
     * Must not be called from the active monitor execution context: that
     * context clears the state close() waits for, so it would wait for itself.
     * This is reported as std::logic_error, after the exit flag is set, rather
     * than deadlocking. Return from the monitor context and close from its
     * control context. By extension, do not destroy the transceiver from the
     * monitor context; the destructor closes.
     */
    void close();

    /**
     * Ask blocking_monitor() to return without releasing the queue pair or
     * memory regions. The caller joins that thread. close() still releases
     * the resources. Safe before the monitor starts. In CallerDriven mode
     * this call has no effect, because that mode has no monitor thread.
     * Stop calling rx_poll() and tx_publish(), then call close().
     */
    void request_exit() noexcept;

    /**
     * Set the unified-mode callback and caller-owned context.
     *
     * Must be called with a non-null callback before setup() or start() in
     * Unified mode. Calling it in any other mode, or after setup(), close(),
     * or a monitor failure, is reported as std::logic_error. The callback
     * executes synchronously on the Unified polling context and must be quick.
     * The caller retains ownership of `context`; it must outlive
     * blocking_monitor().
     */
    void set_unified_dispatch(UnifiedDispatchFn fn, void* context);

    /**
     * Override Config::local_ip before setup().
     *
     * Supports interfaces that configure optional fields between separate
     * create and setup calls. Empty restores the default of the first IPv4
     * RoCE v2 GID on ibv_port. A malformed address is reported as
     * std::invalid_argument here rather than at setup(). Calling it after
     * setup(), close(), or a monitor failure is std::logic_error.
     */
    void set_local_ip(std::string local_ip);

    /** Local QP number, or 0 before setup(), after close(), or after a
     *  monitor-failure fallback that destroyed the QP. */
    std::uint32_t get_qp_number() const;
    /**
     * Payload bytes of the path MTU programmed on the queue pair.
     * Zero until connect() has reached RTS, and zero after close() or after
     * the queue pair is destroyed. A failed RTS transition leaves it zero.
     */
    std::uint32_t get_path_mtu() const;
    /**
     * Peer-visible receive-region rkey.
     *
     * A host RX ring's separate virtual-address mapping is never exported.
     * DMA-BUF and GPU-pointer RX use a zero-based mapping. Handled receive
     * may ask for metadata trailers in host memory. When that request is
     * accepted, this is the indirect key for the same zero-based addresses.
     * Other modes keep the payload mapping.
     * Returns 0 before setup(),
     * after close(), after a monitor-failure fallback that destroyed the QP,
     * or in Tx mode (no RX MR).
     */
    std::uint32_t get_rkey() const;
    /** Peer-visible IOVA. Always 0; this engine does not advertise another base. */
    std::uint64_t external_frame_memory() const;

    /**
     * Acquire-load a ring flag.
     *
     * Use this rather than a plain load: the transceiver accesses the same
     * location atomically, C++17 has no std::atomic_ref, and casting this
     * storage to std::atomic<std::uint64_t>* does not create atomic objects.
     * Defined here so a spin-poll can compile to one load without LTO.
     * __atomic_load_n is a GCC/Clang builtin; this header is not for MSVC.
     */
    [[nodiscard]] static std::uint64_t load_flag_acquire(
        const volatile std::uint64_t* flag) noexcept
    {
        return __atomic_load_n(flag, __ATOMIC_ACQUIRE);
    }

    /**
     * Release-store a ring flag, ordering it after earlier payload writes.
     *
     * RX consumers store zero to release a slot; TX producers store the slot
     * address to publish one. Same compiler-builtin constraint as
     * load_flag_acquire().
     */
    static void store_flag_release(
        volatile std::uint64_t* flag, std::uint64_t value) noexcept
    {
        __atomic_store_n(flag, value, __ATOMIC_RELEASE);
    }

    // Ring accessors. Valid between setup() and close(); after close() they
    // return nullptr/0 and any previously returned pointer is invalid. Scalar
    // resource getters are serialized against close() and monitor
    // terminalization; get_rx_collision_count() is an independent relaxed
    // atomic diagnostic. Ring-pointer getters obtain a data-race-free
    // snapshot, but do not extend the returned storage lifetime. The caller
    // must stop and join every pointer user before close().
    // Stop and join external RX/TX flag threads before close(); they do not
    // observe the transceiver's private exit flag and must use their own
    // cancellation signal.
    //
    // Every flag access concurrent with the transceiver must go through
    // load_flag_acquire()/store_flag_release(); a plain C++ load or store is a
    // data race. Exactly one releasing RX consumer and one TX producer may
    // operate on each slot; atomics provide visibility, not multi-owner
    // arbitration. RX consumers load the flag before reading payload and store
    // zero after consumption. Publication is an atomic release-store of the
    // exact slot address once payload writes are done; no wait-for-zero or
    // producer CAS is required. An application may wait for zero to avoid
    // coalescing, but zero proves only that no unclaimed publication exists
    // (free or already claimed) and never authorizes payload reuse. All
    // payload reuse is the application protocol's responsibility, not
    // something this flag can signal, for both inline and non-inline sends.
    // Inline payload must remain stable until ibv_post_send() has copied it;
    // the flag exposes claim, not that copy. Non-inline payload must remain
    // stable through the send CQE, which this flag also does not expose. At
    // most one unclaimed publication may exist per slot.
    //
    // RX flags match GpuRoceTransceiver: a non-zero flag is the address of
    // fresh slot data; the consumer releases the slot by storing zero.
    //
    // The TX doorbell and cyclic-order rules below apply to Tx, Duplex,
    // Forward, and Unified. CallerDriven does not use them: tx_publish()
    // neither reads nor stores tx_flag, and the caller chooses the slot.
    //
    // TX flags are doorbells: the exact slot address is an unclaimed
    // publication and zero means none (free or already claimed). The
    // transport clears the flag with a release-store of zero at claim.
    // Completion advances only the private cyclic post/completion window and
    // never writes the public flag. A rejected post is not retried and
    // terminalizes the instance. The consumed publication is not restored. A
    // newer concurrent publication, if present, is left untouched but will
    // not be consumed by this terminal instance. The final flag value is not
    // a recovery indication. Zero is therefore not a completion,
    // payload-reuse, or inline-copy acknowledgement of any kind; see the
    // payload-reuse note above.
    // An out-of-contract same-address republication before claim is
    // indistinguishable and may coalesce; replacing the load/store with a
    // claim CAS cannot detect that ABA case. This two-state protocol has no
    // generation.
    //
    // Both TX operations consume published slots in cyclic order beginning
    // with slot 0. A producer may publish a later slot early, but it remains
    // pending until every preceding slot is published. Omitting an expected
    // slot blocks all later transmissions. This is the local producer/ring
    // contract; it does not imply that local and remote slot numbers match.
    //
    // An ordinary SEND carries no remote slot identity, so the peer's
    // receive-WQE order determines placement. For Tx and Duplex, connect()
    // validates only that the outbound payload fits peer.rx_stride; it does
    // not constrain the peer's slot count and does not prove the peer has
    // posted receive WQEs.
    //
    // The delivery, credit, and slot-correspondence guarantees this engine
    // does and does not provide are stated once, at the class level above;
    // they apply identically to SEND and WriteWithImm and are not repeated
    // here.
    //
    // With TxOperation::WriteWithImm, the immediate names the remote slot and
    // the write address is peer.rx_base + slot * peer.rx_stride. For every
    // TX-driving mode, connect() requires a non-zero peer rkey and
    // rx_slot_count, outbound slot_count <= peer.rx_slot_count, and that the
    // last addressed remote byte is representable. Those checks prove
    // addressability only; the supplied rkey is not verified to cover the
    // declared geometry. Before publishing local slot N, the application
    // protocol must still ensure remote slot N is free and the peer has
    // receive-WQE credit for the immediate — an RDMA Write can overwrite an
    // unread remote slot before the receiver reports a collision.
    //
    // Geometry getters return each live ring's independent geometry between
    // setup() and close(), and zero before setup() or after close(). A missing
    // ring is get_*_ring_data_addr() == nullptr, not a zero stride.
    std::size_t get_rx_ring_stride() const;
    std::uint32_t get_rx_ring_slot_count() const;
    std::size_t get_tx_ring_stride() const;
    std::uint32_t get_tx_ring_slot_count() const;

    /**
     * Number of RX CQEs that first encountered an occupied public flag.
     * Relaxed diagnostic counter; remains readable after monitor failure or
     * close() until the transceiver is destroyed.
     */
    std::uint64_t get_rx_collision_count() const noexcept;

    std::uint8_t* get_rx_ring_data_addr() const;
    std::uint64_t* get_rx_ring_flag_addr() const;
    std::uint8_t* get_tx_ring_data_addr() const;
    std::uint64_t* get_tx_ring_flag_addr() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    // Caller-post and handled-receive stay in the internal header. Neither
    // is a public mode.
    friend class detail::CpuRoceCallerTx;
    friend class detail::CpuRoceRxCompletionHandler;
    friend struct detail::CpuRoceMonitorTestAccess;
};

} // namespace hololink::transport::roce

#endif /* SRC_HOLOLINK_TRANSPORT_ROCE_CPU_ROCE_TRANSCEIVER_HPP */
