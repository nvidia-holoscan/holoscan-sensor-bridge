/* SPDX-License-Identifier: MIT */
#ifndef HOLOLINK_PCIE_UAPI_H
#define HOLOLINK_PCIE_UAPI_H

#include <linux/ioctl.h>
#include <linux/types.h>

#define HL_PCIE_DEVICE_NAME "hololink_pcie"
#define HL_PCIE_MAX_MSIX_VECTORS 64

/*
 * mmap layout: page offset HL_PCIE_DMA_MMAP_BASE_PGOFF + handle maps the
 * complete DMA allocation returned by HL_PCIE_IOC_DMA_ALLOC. Endpoint BARs
 * are intentionally accessible only through the 32-bit BAR0 ioctls below.
 */
#define HL_PCIE_DMA_MMAP_BASE_PGOFF 0x100000UL

struct hl_pcie_rw {
	__u32 bar;          /* must be 0 */
	__u32 offset;
	__u32 width;        /* must be 4 */
	__u32 count;        /* must be 1 */
	__u64 data;
};

struct hl_pcie_dma_buf {
	__u64 size;
	__u64 iova;
	__u64 handle;
};

/*
 * DMA-BUF import.
 *
 * The FPGA data plane is programmed with a raw device address (BUF_BASE_HI + BUF_START),
 * so a GPU VRAM frame buffer has to be mapped into *this endpoint's* IOMMU domain before
 * it can be a DMA destination. Userspace exports the allocation with
 * cuMemGetHandleForAddressRange(..., CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, ...) and hands
 * the fd here; the driver attaches it to the PCI device and returns the resulting address.
 *
 * This is the PCIe equivalent of what ibv_reg_dmabuf_mr() does for the RoCE data path,
 * except that no rebasing to IOVA 0 happens: the address returned is what the FPGA is
 * programmed with, so the constraints of its address generator apply. The driver enforces
 * them at import time rather than letting them fail silently at stream time:
 *   - the mapping must be a single contiguous range (the FPGA has a start + increment
 *     address generator, not a scatter-gather descriptor list),
 *   - it must be 128-byte aligned (BUF_START ignores the low 7 bits), and
 *   - it must not cross a 4 GiB boundary (BUF_BASE_HI is a fixed upper 32 bits).
 *
 * Imports are owned by the file descriptor that created them and are released when it
 * closes, matching HL_PCIE_IOC_DMA_ALLOC.
 */
struct hl_pcie_dmabuf {
	__s32 fd;           /* in: dma-buf file descriptor */
	__u32 flags;        /* in: must be 0 */
	__u64 size;         /* out: mapped length in bytes */
	__u64 iova;         /* out: device address to program into the FPGA */
	__u64 handle;       /* in for RELEASE, out for IMPORT */
};

struct hl_pcie_msix_info {
	__u32 vector;       /* in: vector index; out: same */
	__u64 msg_addr;     /* out: kernel MSI-X message address */
	__u32 msg_data;     /* out: kernel MSI-X message data dword */
	__u32 fpga_data;    /* out: legacy/expected one-hot payload */
	__u32 irq_count;    /* out: driver IRQ handler count for vector */
	__u32 num_vectors;  /* out: vectors allocated on this device */
	__u32 enabled;      /* out: 1 if MSI-X vectors are active */
	__u32 reset_count;  /* in: nonzero clears irq_count before read, and enables the vector
	                     *     for this fd as HL_PCIE_IOC_MSIX_CTRL would (a caller that resets
	                     *     the count is about to poll it) */
};

/*
 * Block until an MSI-X vector fires.
 *
 * Without this the only way to notice an interrupt is to poll HL_PCIE_IOC_MSIX_INFO, which
 * costs an ioctl per poll interval and floors frame latency at the polling period. A
 * receiver waiting for one frame-completion interrupt wants to sleep, not spin.
 *
 * `last_count` is the irq_count the caller last observed; the call returns as soon as the
 * current count differs, so an interrupt that arrives between the previous return and this
 * call is not missed. Returns 0 with the new count in `irq_count`, -ETIMEDOUT if the
 * timeout expires, or -ERESTARTSYS if a signal arrives.
 *
 * Waiting on a vector implicitly enables it for the calling fd (see HL_PCIE_IOC_MSIX_CTRL),
 * so a caller that only ever waits never has to enable anything explicitly.
 */
struct hl_pcie_msix_wait {
	__u32 vector;       /* in: vector index */
	__u32 timeout_ms;   /* in: 0 polls, HL_PCIE_MSIX_WAIT_FOREVER blocks indefinitely */
	__u32 last_count;   /* in: count the caller last observed */
	__u32 irq_count;    /* out: current count */
};

#define HL_PCIE_MSIX_WAIT_FOREVER 0xffffffffU

/*
 * Enable or disable delivery of one MSI-X vector, HL_PCIE_IOC_MSIX_CTRL.
 *
 * The kernel owns the MSI-X table: the driver requests every vector disabled at probe and
 * flips it with enable_irq() / disable_irq(), which write the table entry's mask bit through
 * the kernel's own mapping and keep its shadow state consistent. Userspace never writes the
 * FPGA's MSIX_TABLE words; a write there behind the kernel's back is at best a no-op and at
 * worst undone by the next affinity change.
 *
 * Enables are counted per fd and per vector: an fd holds a vector enabled at most once,
 * however many times it asks; the vector is enabled in hardware while at least one fd holds
 * it, and disabled when the last holder disables it or closes. disable waits for a running
 * handler to finish before returning, so after it no completion for this vector can wake a
 * waiter. `enabled_refs` reports the holder count after the call.
 *
 * sizeof(struct hl_pcie_msix_ctrl) == 32 is part of the ioctl number -- append into
 * reserved[], never resize.
 */
struct hl_pcie_msix_ctrl {
	__u32 vector;        /* in: vector index */
	__u32 enable;        /* in: 1 enables the vector for this fd, 0 disables it */
	__u32 flags;         /* in: must be 0 */
	__u32 enabled_refs;  /* out: fds holding the vector enabled after this call */
	__u32 reserved[4];   /* in: must be 0; out: zero */
};

/*
 * Device identity, served by HL_PCIE_IOC_IDENTITY.
 *
 * Everything the driver already learned at probe (BAR0 identity registers, MSI-X vector
 * count, BAR0 length, the dma_address_bits it was loaded with) plus the PCI address, so a
 * consumer holding only the character device can key the board on its BDF and read its
 * UUID without parsing dmesg or walking sysfs. The UUID is re-read from BAR0 on every
 * open() and on every IDENTITY call, so a board provisioned after the driver bound reports
 * its new value without a reload; hsb_version/hsb_date/hsb_status are the probe-time values.
 *
 * uuid[] byte order: the FPGA holds the 128-bit UUID in four 32-bit registers,
 * HSB_UUID3 (BAR0+0xB8) most significant. identity.py renders it as
 * f"{uuid3:08x}{uuid2:08x}{uuid1:08x}{uuid0:08x}" and msix_pcie.hpp packs each register
 * big-endian into the byte array, so uuid[0] = HSB_UUID3 >> 24 ... uuid[3] = HSB_UUID3 & 0xff,
 * uuid[4..7] = HSB_UUID2, uuid[8..11] = HSB_UUID1, uuid[12..15] = HSB_UUID0. Printing the
 * bytes in array order as 8-4-4-4-12 lowercase hex therefore yields the canonical text the
 * reference tooling shows (and what the `uuid` sysfs attribute emits).
 *
 * Field widths follow the PCIe quantities they carry (domain 16 bits, bus/devfn 8 bits,
 * requester id = bus:devfn 16 bits) so the struct is exactly 64 bytes with 8 bytes of
 * reserved room and no implicit padding. sizeof(struct hl_pcie_identity_info) == 64 is
 * part of the ioctl number (_IOR encodes the size) -- append into reserved[], never resize.
 */
struct hl_pcie_identity_info {
	__u16 domain;              /* out: PCI segment (pci_domain_nr) */
	__u8 bus;                  /* out: PCI bus number */
	__u8 devfn;                /* out: PCI_DEVFN(slot, func) */
	__u16 requester_id;        /* out: (bus << 8) | devfn, as programmed into PCIE_DP_REQUESTER_ID */
	__u16 minor;               /* out: minor of this /dev/hololink_pcieN */
	__u32 hsb_version;         /* out: BAR0 HSB_VER as read at probe */
	__u32 hsb_date;            /* out: BAR0 HSB_DATE as read at probe */
	__u32 hsb_status;          /* out: BAR0 HSB_STAT as read at probe */
	__u32 flags;               /* out: HL_PCIE_ID_F_* */
	__u8 uuid[16];             /* out: board UUID, byte order documented above */
	__u32 num_msix_vectors;    /* out: MSI-X vectors allocated on this device */
	__u32 dma_address_bits;    /* out: DMA mask width the driver was loaded with */
	__u64 bar0_len;            /* out: BAR0 length in bytes */
	__u32 reserved[2];         /* out: zero; future additive fields live here */
};
/* sizeof(struct hl_pcie_identity_info) must stay 64; the driver BUILD_BUG_ONs this. */

/* hl_pcie_identity_info.flags -- also the values the driver logs at probe. */
#define HL_PCIE_ID_F_VERSION_VALID          0x00000001U /* HSB_VER is neither 0 nor all-ones */
#define HL_PCIE_ID_F_EXPECTED_VERSION_MATCH 0x00000002U /* HSB_VER matched expected_hsb_version */
#define HL_PCIE_ID_F_CHECK_SKIPPED          0x00000004U /* bound with skip_identity_check=1 */
#define HL_PCIE_ID_F_MAGIC_VALID            0x00000008U /* identity magic "HLPC" present */
#define HL_PCIE_ID_F_UUID_VALID             0x00000010U /* UUID is neither all-zero nor all-ones */

#define HL_PCIE_IOC_MAGIC 'H'

#define HL_PCIE_IOC_READ        _IOWR(HL_PCIE_IOC_MAGIC, 2, struct hl_pcie_rw)
#define HL_PCIE_IOC_WRITE       _IOW(HL_PCIE_IOC_MAGIC, 3, struct hl_pcie_rw)
#define HL_PCIE_IOC_DMA_ALLOC   _IOWR(HL_PCIE_IOC_MAGIC, 4, struct hl_pcie_dma_buf)
#define HL_PCIE_IOC_DMA_FREE    _IOW(HL_PCIE_IOC_MAGIC, 5, struct hl_pcie_dma_buf)
#define HL_PCIE_IOC_MSIX_INFO   _IOWR(HL_PCIE_IOC_MAGIC, 9, struct hl_pcie_msix_info)
#define HL_PCIE_IOC_DMABUF_IMPORT  _IOWR(HL_PCIE_IOC_MAGIC, 10, struct hl_pcie_dmabuf)
#define HL_PCIE_IOC_DMABUF_RELEASE _IOW(HL_PCIE_IOC_MAGIC, 11, struct hl_pcie_dmabuf)
#define HL_PCIE_IOC_MSIX_WAIT   _IOWR(HL_PCIE_IOC_MAGIC, 12, struct hl_pcie_msix_wait)
#define HL_PCIE_IOC_IDENTITY    _IOR(HL_PCIE_IOC_MAGIC, 13, struct hl_pcie_identity_info)
#define HL_PCIE_IOC_MSIX_CTRL   _IOWR(HL_PCIE_IOC_MAGIC, 14, struct hl_pcie_msix_ctrl)

/* FPGA data-plane address generator constraints, enforced by HL_PCIE_IOC_DMABUF_IMPORT. */
#define HL_PCIE_DMABUF_ALIGNMENT 128UL

/*
 * The following two IOCTLs:
 *  - HL_PCIE_IOC_GET_SYNCPOINT
 *  - HL_PCIE_IOC_ARM
 * are specifically for the Tegra SoC, where a host1x syncpoint is associated
 * for a frame completion event.
 * Both ioctls below return -ENODEV on platforms with no host1x.
 */

/*
 * HL_PCIE_IOC_GET_SYNCPOINT: get this vector's syncpoint id.
 * One syncpoint per vector, shared device-wide (not per-fd/caller).
 *
 * The first caller to hit this ioctl for a given vector allocates it,
 * every caller after that (same or different fd) just gets back the same id.
 *
 * The IRQ handler increments it on each completion. The syncpoint lives for
 * the device's lifetime, not any fd's -- freed only on driver teardown.
 */
struct hl_pcie_get_syncpoint {
	__u32 sensor;        /* in: MSI-X vector index */
	__u32 syncpoint_id;  /* out: host1x syncpoint id for this vector */
	__u32 current_value; /* out: syncpoint value at time of this call */
	__u32 reserved;      /* in: must be 0; out: zero */
};

/*
 * HL_PCIE_IOC_ARM: requires GET_SYNCPOINT to have been called first for this
 * vector (-ENODEV otherwise).
 * Returns the wait point (id, base_value) for one capture request.
 * The caller waits for the syncpoint to reach base_value + 1.
 */
struct hl_pcie_arm {
	__u32 sensor;       /* in: MSI-X vector index */
	__u32 syncpoint_id; /* out */
	__u32 base_value;   /* out: syncpoint value at arm time */
	__u32 reserved;     /* in: must be 0; out: zero */
};

#define HL_PCIE_IOC_GET_SYNCPOINT _IOWR(HL_PCIE_IOC_MAGIC, 15, struct hl_pcie_get_syncpoint)
#define HL_PCIE_IOC_ARM           _IOWR(HL_PCIE_IOC_MAGIC, 16, struct hl_pcie_arm)

#endif /* HOLOLINK_PCIE_UAPI_H */
