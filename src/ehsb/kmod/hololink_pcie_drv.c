// SPDX-License-Identifier: GPL-2.0
#include <linux/bitmap.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/etherdevice.h>
#include <linux/if_ether.h>
#include <linux/in.h>
#include <linux/ip.h>
#include <linux/jiffies.h>
#include <linux/netdevice.h>
#include <linux/skbuff.h>
#include <linux/string.h>
#include <linux/udp.h>
#include <linux/workqueue.h>
#include <linux/dma-buf.h>
#include <linux/dma-mapping.h>
#include <linux/dma-resv.h>
#include <linux/fs.h>
#include <linux/interrupt.h>
#include <linux/io.h>
#include <linux/iopoll.h>
#include <linux/irq.h>
#include <linux/kernel.h>
#include <linux/list.h>
#include <linux/kref.h>
#include <linux/math64.h>
#include <linux/module.h>
#include <linux/msi.h>
#include <linux/mutex.h>
#include <linux/of.h>
#include <linux/pci.h>
#include <linux/ptp_clock_kernel.h>
#include <linux/time64.h>
#include <linux/refcount.h>
#include <linux/scatterlist.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <linux/version.h>
#if LINUX_VERSION_CODE >= KERNEL_VERSION(6, 12, 0)
#include <linux/unaligned.h>
#else
#include <asm/unaligned.h>
#endif
#if IS_ENABLED(CONFIG_INET)
#include <net/ip.h>
#include <net/route.h>
#include <net/flow.h>
#include <net/net_namespace.h>
#endif

/*
 * Designed and tested against 6.x kernels. 5.18 is the hard minimum: netif_rx() callable from
 * process context (the bootp publisher runs on a workqueue), IRQF_NO_AUTOEN, and the dma-buf
 * dynamic-attach API. Fail at compile time rather than misbehave at load.
 */
#if LINUX_VERSION_CODE < KERNEL_VERSION(5, 18, 0)
#error "hololink_pcie requires Linux 5.18 or later (designed for 6.x)"
#endif

#include <linux/host1x.h>
#include <linux/of_platform.h>
#include <linux/platform_device.h>

#include "include/hololink_pcie_bootp.h"
#include "include/hololink_pcie_regs.h"
#include "include/hololink_pcie_uapi.h"

#define HL_PCIE_MAX_MINORS 16
#define HL_PCIE_REQUIRED_MSIX_VECTORS 33
#define HL_PCIE_HSB_MAGIC            0x484c5043 /* "HLPC" */
/* HL_PCIE_ID_F_* identity flags are shared with userspace; see hololink_pcie_uapi.h. */

static bool keep_pci_enabled = true;
module_param(keep_pci_enabled, bool, 0644);
MODULE_PARM_DESC(keep_pci_enabled, "leave PCI memory decoding enabled during remove; bus mastering is always cleared");

static bool skip_identity_check;
module_param(skip_identity_check, bool, 0644);
MODULE_PARM_DESC(skip_identity_check, "bind without validating the Hololink BAR identity");

static bool require_identity_magic;
module_param(require_identity_magic, bool, 0644);
MODULE_PARM_DESC(require_identity_magic, "require BAR0 Hololink identity magic before binding");

static uint expected_hsb_version = HL_PCIE_REG_MAP_VERSION;
static bool quiesce_on_release = true;
module_param(quiesce_on_release, bool, 0644);
MODULE_PARM_DESC(quiesce_on_release,
	"stop the endpoint writing before freeing DMA memory on fd close. Set 0 to isolate the endpoint from this driver's only unsolicited BAR0 writes");

static bool bootp_announce = true;
module_param(bootp_announce, bool, 0644);
MODULE_PARM_DESC(bootp_announce,
	"publish a bootp announcement for every bound endpoint once per second on the loopback interface, broadcast to 127.255.255.255:12267 (the loopback subnet broadcast, delivered to every socket bound to port 12267 on INADDR_ANY rather than to the single sharer a unicast to 127.0.0.1 would pick) and sourced from the endpoint's BDF packed as an IPv4 address ((domain << 16) | (bus << 8) | devfn, the ciaddr value), so the hololink_module Adapter discovers PCIe endpoints through its bootp listener (bool, default 1)");

/*
 * The endpoint's PTP clock (BAR0 bar_ptp block, 0x100..0x1FF) is opt-in twice over: a bitfile
 * without the block stubs the window with pready=0, and a read there can hang on the
 * interconnect watchdog rather than return all-ones, so nothing touches it unless asked. When
 * asked, STATUS is read first and a block that is unreadable or not PRESENT is remembered as
 * absent for the life of the binding; CONTROL is never written on such a block.
 */
static bool bootp_ptp;
module_param(bootp_ptp, bool, 0644);
MODULE_PARM_DESC(bootp_ptp,
	"carry a snapshot of the endpoint's PTP clock in each bootp announcement (vendor bytes 45..54, as an Ethernet HSB does) by running the bar_ptp SNAPSHOT handshake (CONTROL 0x100 / STATUS 0x19C / TIME 0x180..0x188) before sending. Off by default: bitfiles without the PTP block stub 0x100..0x1FF and a read there can stall the interconnect; when off the timestamp bytes are 0 and status bit0 is 0 (bool, default 0)");

static bool ptp_clock;
module_param(ptp_clock, bool, 0644);
MODULE_PARM_DESC(ptp_clock,
	"register the endpoint's PTP clock as a PTP hardware clock (/dev/ptpN, one per endpoint) so phc2sys can sync it to CLOCK_REALTIME or to another PHC. Off by default for the same reason as bootp_ptp; read at probe (bool, default 0)");

static unsigned int ptp_clk_hz = 125000000;
module_param(ptp_clk_hz, uint, 0644);
MODULE_PARM_DESC(ptp_clk_hz,
	"PTP timer clock in Hz; scales adjfine only (default_inc = 1e9 * 2^24 / Hz). 125000000 on the CertusPro-NX Versa (uint, default 125000000)");

/*
 * The timer applies inc = default_inc + FREQ_ADJ with no saturation, so an over-wide pull
 * range lets a servo slow or stall the counter instead of steering it. Keep this far below
 * 1e9 ppb; the observed board error is single-digit ppm.
 */
static unsigned int ptp_max_adj_ppb = 1000000;
module_param(ptp_max_adj_ppb, uint, 0444);
MODULE_PARM_DESC(ptp_max_adj_ppb, "adjfine pull range in ppb (uint, default 1000000 = 1000 ppm)");

static unsigned int dma_address_bits = 64;
module_param(dma_address_bits, uint, 0644);
MODULE_PARM_DESC(dma_address_bits,
	"DMA address width to request. 64 (default) allows GPU VRAM mappings above 4 GiB. Set 32 for bitfiles predating FIXED_MSB address-generation support, which silently truncate to the low 32 bits");

module_param(expected_hsb_version, uint, 0644);
MODULE_PARM_DESC(expected_hsb_version, "expected BAR0 HSB version; 0 accepts any nonzero/non-all-ones version");

struct hl_pcie_dma_alloc {
	struct list_head list;
	struct hl_pcie_dev *hdev;
	struct hl_pcie_file_ctx *owner;
	refcount_t refs;
	u64 handle;
	size_t size;
	void *cpu_addr;
	dma_addr_t dma_addr;
};

/*
 * A GPU (or other peer) allocation mapped into this endpoint's IOMMU domain.
 *
 * Unlike hl_pcie_dma_alloc the memory is not ours: we hold a reference to the exporter's
 * dma_buf and an attachment for the lifetime of the import, and the pages stay pinned so
 * the device address we handed userspace remains valid until release.
 */
/*
 * Per-MSI-X-vector state. The IRQ handler needs more than a counter now -- it has to wake
 * whoever is blocked in HL_PCIE_IOC_MSIX_WAIT -- so each vector gets its own context that
 * doubles as the request_irq() cookie.
 */
struct hl_pcie_vector {
	atomic_t count;
	wait_queue_head_t wait;
	/*
	 * Number of open fds holding this vector enabled (HL_PCIE_IOC_MSIX_CTRL or an implicit
	 * enable from HL_PCIE_IOC_MSIX_WAIT). The IRQ is enabled in the kernel while this is
	 * non-zero. Guarded by hdev->lock.
	 */
	unsigned int enable_refs;
	/*
	 * This vector's syncpoint, lazily allocated on the first GET_SYNCPOINT
	 * call (see hl_pcie_ioctl_get_syncpoint). NULL until then. Written with
	 * WRITE_ONCE under syncpt_lock, read with READ_ONCE from the lockless
	 * hard-irq handler.
	 */
	struct host1x_syncpt *sensor_syncpt;
	/* Back-pointer: the IRQ handler's cookie is this vector, not hdev. */
	struct hl_pcie_dev *hdev;
};

struct hl_pcie_dmabuf_import {
	struct list_head list;
	struct hl_pcie_dev *hdev;
	struct hl_pcie_file_ctx *owner;
	u64 handle;
	struct dma_buf *dmabuf;
	struct dma_buf_attachment *attach;
	struct sg_table *sgt;
	dma_addr_t iova;
	size_t size;
};

static void hl_pcie_dmabuf_release(struct hl_pcie_dmabuf_import *import);

struct hl_pcie_identity {
	u32 flags;
	u32 hsb_version;
	u32 hsb_date;
	u32 hsb_status;
	u32 magic;
	u64 uuid_low;
	u64 uuid_high;
};

struct hl_pcie_dev {
	struct pci_dev *pdev;
	struct cdev cdev;
	struct device *device;
	int minor;
	/*
	 * Lifetime. probe takes the first reference and remove drops it; every open fd, DMA
	 * allocation and dma-buf import holds one as well (hl_pcie_dev_get / hl_pcie_dev_put),
	 * so a file context, a VMA of a DMA allocation or an attachment that outlives the PCI
	 * device (sysfs unbind, surprise removal) still points at live memory. `dead` is set by
	 * remove under hl_pcie_devices_lock and hdev->lock before the IRQs are freed and BAR0 is
	 * unmapped; the file entry points return -ENODEV once it is set.
	 */
	struct kref ref;
	bool dead;
	void __iomem *bar0;
	resource_size_t bar0_len;
	struct mutex lock;
	struct list_head dma_allocs;
	struct list_head dmabuf_imports;
	/* Shared across both handle spaces so a handle is never ambiguous. */
	u64 next_dma_handle;
	int num_vectors;
	struct hl_pcie_vector vectors[HL_PCIE_MAX_MSIX_VECTORS];
	struct hl_pcie_identity identity;
	/* PCIe Device Serial Number capability (EUI-64), read once at probe; 0 = none. */
	u64 dsn;
	/*
	 * host1x instance found at probe (hl_pcie_find_host1x()); host1x is
	 * Tegra SoC-specific, so this is NULL on any other platform, and
	 * GET_SYNCPOINT/ARM then return -ENODEV. Guards lazy per-vector
	 * sensor_syncpt allocation in hl_pcie_ioctl_get_syncpoint().
	 */
	struct host1x *host1x;
	struct mutex syncpt_lock;
	/* bootp publisher bookkeeping; touched only under hl_pcie_devices_lock. */
	unsigned long bootp_start_jiffies;
	u32 bootp_count;
	/*
	 * The bar_ptp block. ptp_lock serializes every command handshake -- the bootp snapshot
	 * (delayed work) and the PHC operations (ptp_clock ioctls) -- and guards ptp_state.
	 */
	struct mutex ptp_lock;
	enum { HL_PCIE_PTP_UNKNOWN = 0, HL_PCIE_PTP_PRESENT, HL_PCIE_PTP_ABSENT } ptp_state;
	u64 ptp_default_inc; /* 1e9 * 2^24 / ptp_clk_hz: the increment FREQ_ADJ is relative to */
	struct ptp_clock_info ptp_info;
	struct ptp_clock *ptp_clock;
};

struct hl_pcie_file_ctx {
	struct hl_pcie_dev *hdev;
	/* Vectors this fd holds enabled; each is released on close. Guarded by hdev->lock. */
	DECLARE_BITMAP(enabled_vectors, HL_PCIE_MAX_MSIX_VECTORS);
};

static void hl_pcie_dev_release(struct kref *ref)
{
	struct hl_pcie_dev *hdev = container_of(ref, struct hl_pcie_dev, ref);

	pci_dev_put(hdev->pdev);
	kfree(hdev);
}

static struct hl_pcie_dev *hl_pcie_dev_get(struct hl_pcie_dev *hdev)
{
	kref_get(&hdev->ref);
	return hdev;
}

static void hl_pcie_dev_put(struct hl_pcie_dev *hdev)
{
	kref_put(&hdev->ref, hl_pcie_dev_release);
}

/*
 * Mask every entry of the FPGA's MSI-X table by writing BAR0 directly.
 *
 * Runs before the kernel has enabled MSI-X on the function (probe) and after it has torn it
 * down (remove), i.e. only while the kernel holds no state about the table. In between, the
 * table belongs to the kernel: vectors are requested disabled and are enabled / disabled
 * with enable_irq() / disable_irq() (hl_pcie_vector_enable / hl_pcie_vector_disable), which
 * write the mask bit through the kernel's own mapping and keep its shadow consistent. Nothing
 * in this driver writes the table behind the kernel's back once IRQs are set up.
 */
static void hl_pcie_mask_all_fpga_msix(struct hl_pcie_dev *hdev)
{
	void __iomem *bar0;
	resource_size_t required;
	int vector;

	bar0 = hdev->bar0;
	required = HL_PCIE_REG_MSIX_TABLE_END + 1;
	if (!bar0 || hdev->bar0_len < required)
		return;

	for (vector = 0; vector < HL_PCIE_MAX_MSIX_VECTORS; vector++)
		iowrite32(HL_PCIE_REG_MSIX_VECTOR_MASK,
			  bar0 + HL_PCIE_REG_MSIX_TABLE_BASE +
			  vector * HL_PCIE_REG_MSIX_ENTRY_STRIDE +
			  HL_PCIE_REG_MSIX_VECTOR_CONTROL);

	/* Flush posted BAR writes before bus mastering can be enabled. */
	ioread32(bar0 + HL_PCIE_REG_MSIX_TABLE_BASE +
		 (HL_PCIE_MAX_MSIX_VECTORS - 1) *
		 HL_PCIE_REG_MSIX_ENTRY_STRIDE +
		 HL_PCIE_REG_MSIX_VECTOR_CONTROL);
}

static dev_t hl_pcie_devt;
static struct class *hl_pcie_class;
static DEFINE_MUTEX(hl_pcie_devices_lock);
static struct hl_pcie_dev *hl_pcie_devices[HL_PCIE_MAX_MINORS];

/*
 * The bootp publisher: one delayed work on the system workqueue that, once per second,
 * announces every bound endpoint (hl_pcie_bootp_work_fn below). Scheduled from probe, re-armed
 * by itself while any device exists, cancelled synchronously in remove and exit.
 */
static void hl_pcie_bootp_work_fn(struct work_struct *work);
static DECLARE_DELAYED_WORK(hl_pcie_bootp_work, hl_pcie_bootp_work_fn);
#define HL_PCIE_BOOTP_PERIOD msecs_to_jiffies(HL_PCIE_BOOTP_INTERVAL_MS)

static const struct pci_device_id hl_pcie_ids[] = {
	{ PCI_DEVICE(0x1204, 0x9c25) },
	{ PCI_DEVICE(0x1204, 0x9c1d) },
	{ PCI_DEVICE(0x1204, 0x0e00c) },
	{ PCI_DEVICE(0x19aa, 0x9c25) },
	{ 0, }
};
MODULE_DEVICE_TABLE(pci, hl_pcie_ids);

static struct hl_pcie_dma_alloc *hl_pcie_find_dma_locked(struct hl_pcie_dev *hdev,
							 u64 handle)
{
	struct hl_pcie_dma_alloc *alloc;

	list_for_each_entry(alloc, &hdev->dma_allocs, list) {
		if (alloc->handle == handle)
			return alloc;
	}

	return NULL;
}

static void hl_pcie_dma_release(struct hl_pcie_dma_alloc *alloc)
{
	struct hl_pcie_dev *hdev = alloc->hdev;

	/*
	 * Can run after hl_pcie_remove() when a VMA of this allocation outlived the device;
	 * the allocation's reference keeps hdev and pdev valid for the free.
	 */
	dma_free_attrs(&hdev->pdev->dev, alloc->size, alloc->cpu_addr,
		       alloc->dma_addr, 0);
	kfree(alloc);
	hl_pcie_dev_put(hdev);
}

static void hl_pcie_dma_put(struct hl_pcie_dma_alloc *alloc)
{
	if (refcount_dec_and_test(&alloc->refs))
		hl_pcie_dma_release(alloc);
}

static irqreturn_t hl_pcie_irq_handler(int irq, void *data)
{
	struct hl_pcie_vector *vector = data;

	atomic_inc(&vector->count);
	/* If GET_SYNCPOINT was ever called for this vector, bump its syncpoint.
	 * CPU-driven incr for a client-managed syncpoint; safe in hard-irq context.
	 */
	if (READ_ONCE(vector->sensor_syncpt))
		host1x_syncpt_incr(vector->sensor_syncpt);
	/*
	 * Publish the count before waking: a waiter that is already running must not compare
	 * against a stale value and go back to sleep.
	 */
	smp_wmb();
	wake_up_interruptible(&vector->wait);
	return IRQ_HANDLED;
}

static bool hl_pcie_uuid_valid(u64 uuid_low, u64 uuid_high)
{
	return (uuid_low || uuid_high) &&
	       uuid_low != 0xffffffffffffffffULL &&
	       uuid_high != 0xffffffffffffffffULL;
}

/*
 * Re-read only the UUID registers into hdev->identity.
 *
 * The UUID is the one identity word userspace changes at runtime (the provisioning helper
 * programs it once per board), so a board provisioned after this driver bound would
 * otherwise keep reporting the probe-time value until a reload. The version/magic
 * checks stay as taken at probe: they gate binding, and re-running them here could
 * only ever unbind a device that is already in use.
 */
static void hl_pcie_refresh_uuid(struct hl_pcie_dev *hdev)
{
	struct hl_pcie_identity *id = &hdev->identity;
	u32 uuid0, uuid1, uuid2, uuid3;

	/*
	 * bar0 under the lock: hl_pcie_remove() marks the device dead and later clears bar0
	 * under hdev->lock, so a check made before taking it could be stale by the ioread.
	 */
	mutex_lock(&hdev->lock);
	if (hdev->dead || !hdev->bar0 ||
	    hdev->bar0_len < HL_PCIE_REG_HSB_UUID3 + sizeof(u32)) {
		mutex_unlock(&hdev->lock);
		return;
	}
	uuid0 = ioread32(hdev->bar0 + HL_PCIE_REG_HSB_UUID0);
	uuid1 = ioread32(hdev->bar0 + HL_PCIE_REG_HSB_UUID1);
	uuid2 = ioread32(hdev->bar0 + HL_PCIE_REG_HSB_UUID2);
	uuid3 = ioread32(hdev->bar0 + HL_PCIE_REG_HSB_UUID3);
	id->uuid_low = ((u64)uuid1 << 32) | uuid0;
	id->uuid_high = ((u64)uuid3 << 32) | uuid2;
	id->flags &= ~HL_PCIE_ID_F_UUID_VALID;
	if (hl_pcie_uuid_valid(id->uuid_low, id->uuid_high))
		id->flags |= HL_PCIE_ID_F_UUID_VALID;
	mutex_unlock(&hdev->lock);
}

/*
 * Serialize the UUID into the byte order documented on struct hl_pcie_identity_info:
 * HSB_UUID3 first, each register big-endian. uuid_high = UUID3:UUID2 and
 * uuid_low = UUID1:UUID0, so this is simply both halves big-endian, high half first.
 * Caller holds hdev->lock.
 */
static void hl_pcie_identity_uuid_bytes(const struct hl_pcie_identity *id,
					u8 uuid[16])
{
	put_unaligned_be64(id->uuid_high, uuid);
	put_unaligned_be64(id->uuid_low, uuid + 8);
}

static u16 hl_pcie_requester_id(const struct hl_pcie_dev *hdev)
{
	return ((u16)hdev->pdev->bus->number << 8) | (u16)hdev->pdev->devfn;
}

/*
 * Per-fd, refcounted MSI-X vector enable.
 *
 * The kernel owns the MSI-X table (see hl_pcie_mask_all_fpga_msix). A vector is enabled in
 * hardware while at least one fd holds it; an fd holds it at most once no matter how often it
 * asks. enable_irq() on a vector requested with IRQF_NO_AUTOEN performs the deferred startup
 * (unmask); disable_irq() masks eagerly (IRQ_DISABLE_UNLAZY) and waits for a running handler,
 * so once it returns no completion on this vector can wake a waiter. Both are called under
 * hdev->lock, which the IRQ handler never takes.
 */
/*
 * Rewrite one table entry's message address and data from the kernel's cached MSI message.
 *
 * The kernel wrote these words when it enabled MSI-X and does not expect them to change
 * underneath it -- but a SYS_RST of the FPGA design (HLNK_CTRL_0 bit 3, which the hololink
 * reset() path issues) wipes the table while the kernel's shadow stays as it was. Writing the
 * kernel's own values back is invisible to it and is what makes a vector fire again after a
 * reset. Called only while the vector is disabled (masked), as the spec requires for changing
 * an entry's address.
 */
static void hl_pcie_restore_msix_entry(struct hl_pcie_dev *hdev, u32 vector)
{
	void __iomem *entry;
	struct msi_msg msg;

	if (!hdev->bar0 || hdev->bar0_len < HL_PCIE_REG_MSIX_TABLE_END + 1)
		return;
	get_cached_msi_msg(pci_irq_vector(hdev->pdev, vector), &msg);
	entry = hdev->bar0 + HL_PCIE_REG_MSIX_TABLE_BASE +
		vector * HL_PCIE_REG_MSIX_ENTRY_STRIDE;
	iowrite32(msg.address_lo, entry + 0x0);
	iowrite32(msg.address_hi, entry + 0x4);
	iowrite32(msg.data, entry + 0x8);
	/* Flush before the unmask that follows. */
	ioread32(entry + HL_PCIE_REG_MSIX_VECTOR_CONTROL);
}

static int hl_pcie_vector_enable(struct hl_pcie_dev *hdev, struct hl_pcie_file_ctx *ctx,
				 u32 vector, u32 *refs)
{
	struct hl_pcie_vector *v;

	if (vector >= (u32)hdev->num_vectors)
		return -EINVAL;
	v = &hdev->vectors[vector];

	mutex_lock(&hdev->lock);
	if (hdev->dead) {
		mutex_unlock(&hdev->lock);
		return -ENODEV;
	}
	if (!test_and_set_bit(vector, ctx->enabled_vectors)) {
		if (v->enable_refs++ == 0) {
			hl_pcie_restore_msix_entry(hdev, vector);
			enable_irq(pci_irq_vector(hdev->pdev, vector));
		}
	}
	if (refs)
		*refs = v->enable_refs;
	mutex_unlock(&hdev->lock);
	return 0;
}

static int hl_pcie_vector_disable(struct hl_pcie_dev *hdev, struct hl_pcie_file_ctx *ctx,
				  u32 vector, u32 *refs)
{
	struct hl_pcie_vector *v;

	if (vector >= (u32)hdev->num_vectors)
		return -EINVAL;
	v = &hdev->vectors[vector];

	mutex_lock(&hdev->lock);
	if (test_and_clear_bit(vector, ctx->enabled_vectors)) {
		/* After remove the IRQs are already freed; just drop the bookkeeping. */
		if (--v->enable_refs == 0 && !hdev->dead)
			disable_irq(pci_irq_vector(hdev->pdev, vector));
	}
	if (refs)
		*refs = v->enable_refs;
	mutex_unlock(&hdev->lock);
	return 0;
}

/* Drop every vector this fd holds enabled. Called from release, after the DMA quiesce. */
static void hl_pcie_release_vectors(struct hl_pcie_dev *hdev, struct hl_pcie_file_ctx *ctx)
{
	unsigned int vector;

	for_each_set_bit(vector, ctx->enabled_vectors, HL_PCIE_MAX_MSIX_VECTORS)
		hl_pcie_vector_disable(hdev, ctx, vector, NULL);
}

static int hl_pcie_open(struct inode *inode, struct file *filp)
{
	struct hl_pcie_file_ctx *ctx;
	struct hl_pcie_dev *hdev = NULL;
	unsigned int minor = iminor(inode);

	/*
	 * Through the minor table rather than container_of(inode->i_cdev): the cdev is embedded
	 * in the hdev, and an open racing hl_pcie_remove() must not touch memory that remove
	 * may already have released. remove unlinks the slot and sets `dead` under this lock,
	 * and the reference taken here keeps the hdev alive for the life of the fd.
	 */
	mutex_lock(&hl_pcie_devices_lock);
	if (minor < HL_PCIE_MAX_MINORS && hl_pcie_devices[minor] &&
	    !hl_pcie_devices[minor]->dead)
		hdev = hl_pcie_dev_get(hl_pcie_devices[minor]);
	mutex_unlock(&hl_pcie_devices_lock);
	if (!hdev)
		return -ENODEV;

	ctx = kzalloc(sizeof(*ctx), GFP_KERNEL);
	if (!ctx) {
		hl_pcie_dev_put(hdev);
		return -ENOMEM;
	}
	ctx->hdev = hdev;
	filp->private_data = ctx;
	/* Pick up a UUID provisioned since probe (or since the last open). */
	hl_pcie_refresh_uuid(hdev);
	return 0;
}

/*
 * Stop the endpoint writing to memory this fd owns, before that memory is freed.
 *
 * Without this, a host process that dies between arming the ring and disarming it leaves the
 * FPGA streaming into an IOVA the kernel then reclaims and hands to something else. On IGX
 * the endpoint shares IOMMU group 22 with the GPU and the CX-7 NIC, so "something else" is a
 * device whose memory gets silently corrupted. That is not hypothetical: it took out the GPU
 * once and the NIC once during bring-up, both after a Ctrl-C interrupted teardown.
 *
 * Writes go directly to BAR0 rather than over the control plane, because the daemon may be
 * gone by the time we get here. Best effort: if the endpoint is already unreachable the
 * writes land nowhere, which is no worse than not trying.
 */
/* True if this fd owns any DMA memory the endpoint could still be writing to. */
static bool hl_pcie_ctx_owns_dma(struct hl_pcie_dev *hdev, struct hl_pcie_file_ctx *ctx)
{
	struct hl_pcie_dmabuf_import *import;
	struct hl_pcie_dma_alloc *alloc;
	bool owns = false;

	mutex_lock(&hdev->lock);
	list_for_each_entry(alloc, &hdev->dma_allocs, list) {
		if (alloc->owner == ctx) {
			owns = true;
			goto out;
		}
	}
	list_for_each_entry(import, &hdev->dmabuf_imports, list) {
		if (import->owner == ctx) {
			owns = true;
			goto out;
		}
	}
out:
	mutex_unlock(&hdev->lock);
	return owns;
}

static void hl_pcie_quiesce(struct hl_pcie_dev *hdev)
{
	if (!hdev->bar0)
		return;

	/* Disarm the packetizer: one write stops every sensor's DMA. */
	iowrite32(0, hdev->bar0 + HL_PCIE_REG_DP_PKT_BASE + HL_PCIE_REG_DP_PKT_VIP_MASK);
	/* Flush the posted write before the caller frees anything. */
	ioread32(hdev->bar0 + HL_PCIE_REG_DP_PKT_BASE + HL_PCIE_REG_DP_PKT_VIP_MASK);
}

static int hl_pcie_release(struct inode *inode, struct file *filp)
{
	struct hl_pcie_file_ctx *ctx = filp->private_data;
	struct hl_pcie_dmabuf_import *import;
	struct hl_pcie_dmabuf_import *import_tmp;
	struct hl_pcie_dma_alloc *alloc;
	struct hl_pcie_dma_alloc *tmp;
	struct hl_pcie_dev *hdev;
	LIST_HEAD(orphans);

	if (!ctx)
		return 0;
	hdev = ctx->hdev;

	/*
	 * Order matters: quiesce before anything is unmapped or freed. Doing it afterwards
	 * leaves a window in which the endpoint is still writing to reclaimed memory.
	 */
	if (quiesce_on_release && hl_pcie_ctx_owns_dma(hdev, ctx)) {
		/* Under hdev->lock: remove sets `dead` there before it unmaps BAR0. */
		mutex_lock(&hdev->lock);
		if (!hdev->dead)
			hl_pcie_quiesce(hdev);
		mutex_unlock(&hdev->lock);
	}
	/*
	 * Then stop this fd's completion interrupts: disable_irq() waits for a handler in
	 * flight, so no wakeup for a vector this fd owned can arrive after this point.
	 */
	hl_pcie_release_vectors(hdev, ctx);

	mutex_lock(&hdev->lock);
	list_for_each_entry_safe(alloc, tmp, &hdev->dma_allocs, list) {
		if (alloc->owner != ctx)
			continue;
		list_del(&alloc->list);
		hl_pcie_dma_put(alloc);
	}
	/*
	 * Unlink first, then release outside hdev->lock: hl_pcie_dmabuf_release() takes the
	 * exporter's reservation lock, and nesting that under ours would invert the order
	 * move_notify uses.
	 */
	list_for_each_entry_safe(import, import_tmp, &hdev->dmabuf_imports, list) {
		if (import->owner != ctx)
			continue;
		list_move_tail(&import->list, &orphans);
	}
	mutex_unlock(&hdev->lock);

	list_for_each_entry_safe(import, import_tmp, &orphans, list) {
		list_del(&import->list);
		hl_pcie_dmabuf_release(import);
	}
	kfree(ctx);
	filp->private_data = NULL;
	hl_pcie_dev_put(hdev);
	return 0;
}

static int hl_pcie_validate_rw(struct hl_pcie_dev *hdev, struct hl_pcie_rw *rw)
{
	if (rw->bar != 0 || !hdev->bar0)
		return -EINVAL;
	if (rw->width != sizeof(u32) || rw->count != 1)
		return -EINVAL;
	if (rw->offset & (sizeof(u32) - 1))
		return -EINVAL;
	if (hdev->bar0_len < sizeof(u32) ||
	    rw->offset > hdev->bar0_len - sizeof(u32))
		return -EINVAL;
	return 0;
}

/*
 * The two raw BAR0 accessors run under hdev->lock: hl_pcie_remove() sets `dead` and clears
 * bar0 under that lock, so neither can ioread/iowrite through a mapping that is being torn
 * down. The dispatcher's unlocked `dead` test is only a fast path.
 */
static int hl_pcie_ioctl_read(struct hl_pcie_dev *hdev, unsigned long arg)
{
	struct hl_pcie_rw rw;
	void __iomem *addr;
	int ret;

	if (copy_from_user(&rw, (void __user *)arg, sizeof(rw)))
		return -EFAULT;

	mutex_lock(&hdev->lock);
	if (hdev->dead) {
		mutex_unlock(&hdev->lock);
		return -ENODEV;
	}
	ret = hl_pcie_validate_rw(hdev, &rw);
	if (ret) {
		mutex_unlock(&hdev->lock);
		return ret;
	}
	addr = hdev->bar0 + rw.offset;
	rw.data = ioread32(addr);
	mutex_unlock(&hdev->lock);

	if (copy_to_user((void __user *)arg, &rw, sizeof(rw)))
		return -EFAULT;
	return 0;
}

static int hl_pcie_ioctl_write(struct hl_pcie_dev *hdev, unsigned long arg)
{
	struct hl_pcie_rw rw;
	void __iomem *addr;
	int ret;

	if (copy_from_user(&rw, (void __user *)arg, sizeof(rw)))
		return -EFAULT;

	mutex_lock(&hdev->lock);
	if (hdev->dead) {
		mutex_unlock(&hdev->lock);
		return -ENODEV;
	}
	ret = hl_pcie_validate_rw(hdev, &rw);
	if (ret) {
		mutex_unlock(&hdev->lock);
		return ret;
	}
	addr = hdev->bar0 + rw.offset;
	iowrite32((u32)rw.data, addr);
	mutex_unlock(&hdev->lock);

	return 0;
}

static int hl_pcie_ioctl_dma_alloc(struct hl_pcie_file_ctx *ctx,
				   unsigned long arg)
{
	struct hl_pcie_dma_alloc *alloc;
	struct hl_pcie_dev *hdev = ctx->hdev;
	struct hl_pcie_dma_buf user;

	if (copy_from_user(&user, (void __user *)arg, sizeof(user)))
		return -EFAULT;
	if (user.size == 0)
		return -EINVAL;

	alloc = kzalloc(sizeof(*alloc), GFP_KERNEL);
	if (!alloc)
		return -ENOMEM;

	alloc->hdev = hdev;
	alloc->owner = ctx;
	refcount_set(&alloc->refs, 1);
	alloc->size = PAGE_ALIGN(user.size);
	alloc->cpu_addr = dma_alloc_attrs(&hdev->pdev->dev, alloc->size,
					   &alloc->dma_addr, GFP_KERNEL, 0);
	if (!alloc->cpu_addr) {
		kfree(alloc);
		return -ENOMEM;
	}
	/* Dropped by hl_pcie_dma_release(), after the last VMA and the list are gone. */
	hl_pcie_dev_get(hdev);

	mutex_lock(&hdev->lock);
	alloc->handle = hdev->next_dma_handle++;
	if (hdev->next_dma_handle == 0)
		hdev->next_dma_handle = 1;
	list_add_tail(&alloc->list, &hdev->dma_allocs);
	mutex_unlock(&hdev->lock);

	user.size = alloc->size;
	user.iova = alloc->dma_addr;
	user.handle = alloc->handle;
	if (copy_to_user((void __user *)arg, &user, sizeof(user))) {
		mutex_lock(&hdev->lock);
		list_del(&alloc->list);
		mutex_unlock(&hdev->lock);
		hl_pcie_dma_put(alloc);
		return -EFAULT;
	}

	return 0;
}

static int hl_pcie_ioctl_dma_free(struct hl_pcie_file_ctx *ctx,
				  unsigned long arg)
{
	struct hl_pcie_dma_alloc *alloc;
	struct hl_pcie_dev *hdev = ctx->hdev;
	struct hl_pcie_dma_buf user;

	if (copy_from_user(&user, (void __user *)arg, sizeof(user)))
		return -EFAULT;

	mutex_lock(&hdev->lock);
	alloc = hl_pcie_find_dma_locked(hdev, user.handle);
	if (!alloc || alloc->owner != ctx) {
		mutex_unlock(&hdev->lock);
		return -ENOENT;
	}
	list_del(&alloc->list);
	mutex_unlock(&hdev->lock);

	hl_pcie_dma_put(alloc);
	return 0;
}

/*
 * A pinned attachment must never be moved, so this should be unreachable. Match the RDMA
 * subsystem and complain loudly rather than silently letting the FPGA write to an address
 * the exporter has invalidated.
 */
static void hl_pcie_dmabuf_move_notify(struct dma_buf_attachment *attach)
{
	struct hl_pcie_dmabuf_import *import = attach->importer_priv;

	dev_err(&import->hdev->pdev->dev,
		"dma-buf move_notify on a pinned import (handle %llu, iova 0x%llx); "
		"the endpoint may now be writing to invalid memory\n",
		import->handle, (unsigned long long)import->iova);
}

static const struct dma_buf_attach_ops hl_pcie_dmabuf_attach_ops = {
	/*
	 * Required for GPU VRAM: without it the exporter must migrate the buffer to system
	 * memory, which defeats the point. Set here rather than on the attachment because
	 * dma_buf_dynamic_attach() copies it into attach->peer2peer.
	 */
	.allow_peer2peer = true,
	.move_notify = hl_pcie_dmabuf_move_notify,
};

static void hl_pcie_dmabuf_release(struct hl_pcie_dmabuf_import *import)
{
	struct hl_pcie_dev *hdev = import->hdev;

	dma_resv_lock(import->dmabuf->resv, NULL);
	if (import->sgt)
		dma_buf_unmap_attachment(import->attach, import->sgt, DMA_FROM_DEVICE);
	dma_buf_unpin(import->attach);
	dma_resv_unlock(import->dmabuf->resv);

	dma_buf_detach(import->dmabuf, import->attach);
	dma_buf_put(import->dmabuf);
	kfree(import);
	hl_pcie_dev_put(hdev);
}

/*
 * Validate the mapping against the FPGA data-plane address generator. Rejecting here turns
 * what would otherwise be silent corruption at stream time into a clear failure at setup.
 */
static int hl_pcie_dmabuf_check(struct hl_pcie_dev *hdev, struct sg_table *sgt,
				dma_addr_t *iova, size_t *size)
{
	struct scatterlist *sg;
	dma_addr_t address = 0;
	dma_addr_t expected = 0;
	size_t length = 0;
	unsigned int i;

	/*
	 * What the endpoint needs is one contiguous address range, which is not the same as
	 * one scatter-gather entry. A dma-buf export of GPU VRAM is split into many entries --
	 * 317 for a 20 MiB allocation -- while still describing a single linear range, because
	 * the exporter chunks at its own granularity rather than reporting real fragmentation.
	 * Rejecting on nents made every ring above roughly 2 MiB fail for no reason.
	 *
	 * Walk the list instead and require each segment to start where the previous one
	 * ended. When that holds, one base address plus a length is sufficient, which is
	 * exactly what BUF_START and BUF_LEN express.
	 */
	for_each_sg(sgt->sgl, sg, sgt->nents, i) {
		dma_addr_t seg_address = sg_dma_address(sg);
		unsigned int seg_length = sg_dma_len(sg);

		if (!seg_length)
			continue;
		if (!length) {
			address = seg_address;
		} else if (seg_address != expected) {
			dev_err(&hdev->pdev->dev,
				"dma-buf segment %u starts at 0x%llx, not 0x%llx; the mapping "
				"is not one contiguous range and the endpoint address "
				"generator cannot express it\n",
				i, (unsigned long long)seg_address,
				(unsigned long long)expected);
			return -EOPNOTSUPP;
		}
		expected = seg_address + seg_length;
		length += seg_length;
	}
	if (!length)
		return -EINVAL;

	/* BUF_START ignores the low 7 bits of the address. */
	if (address & (HL_PCIE_DMABUF_ALIGNMENT - 1)) {
		dev_err(&hdev->pdev->dev,
			"dma-buf mapped at 0x%llx, which is not %lu-byte aligned\n",
			(unsigned long long)address, HL_PCIE_DMABUF_ALIGNMENT);
		return -EINVAL;
	}

	/*
	 * Reject anything above 4 GiB outright. Measured on hardware: PCIE_DP_BUF_BASE_HI
	 * (FIXED_MSB) is read by the endpoint's address validity check but is never applied
	 * to the address in the outbound TLPs, which carry only the low 32 bits from
	 * PCIE_DP_BUF_START. Programming BUF_BASE_HI=0x28 against a buffer at 0x28_da000000
	 * made the endpoint write to 0xda000000.
	 *
	 * The crossing check below cannot catch that: a buffer sitting entirely above 4 GiB
	 * never crosses a boundary, so it passed and the endpoint silently wrote to the
	 * truncated address. Because this device shares an IOMMU domain with the GPU and the
	 * NIC on IGX, those writes land in someone else's address space -- faulting when the
	 * IOVA is unmapped, corrupting silently when it is not.
	 *
	 * Lifts automatically once the RTL wires FIXED_MSB into address generation and
	 * dma_address_bits is raised to 64.
	 */
	if (dma_address_bits <= 32 && upper_32_bits(address)) {
		dev_err(&hdev->pdev->dev,
			"dma-buf mapped at 0x%llx, above the 4 GiB the endpoint can address; "
			"it would truncate to 0x%08x and write to memory it does not own\n",
			(unsigned long long)address, lower_32_bits(address));
		return -ERANGE;
	}

	/* BUF_BASE_HI is a fixed upper 32 bits, so the whole range shares one 4 GiB window. */
	if ((address >> 32) != ((address + length - 1) >> 32)) {
		dev_err(&hdev->pdev->dev,
			"dma-buf range 0x%llx..0x%llx crosses a 4 GiB boundary, which the "
			"endpoint's fixed-MSB address generator cannot represent\n",
			(unsigned long long)address,
			(unsigned long long)(address + length - 1));
		return -ERANGE;
	}

	*iova = address;
	*size = length;
	return 0;
}

static int hl_pcie_ioctl_dmabuf_import(struct hl_pcie_file_ctx *ctx, unsigned long arg)
{
	struct hl_pcie_dev *hdev = ctx->hdev;
	struct hl_pcie_dmabuf_import *import;
	struct hl_pcie_dmabuf user;
	struct sg_table *sgt;
	int ret;

	if (copy_from_user(&user, (void __user *)arg, sizeof(user)))
		return -EFAULT;
	if (user.flags || user.fd < 0)
		return -EINVAL;

	import = kzalloc(sizeof(*import), GFP_KERNEL);
	if (!import)
		return -ENOMEM;
	import->hdev = hdev;
	import->owner = ctx;

	import->dmabuf = dma_buf_get(user.fd);
	if (IS_ERR(import->dmabuf)) {
		ret = PTR_ERR(import->dmabuf);
		goto err_free;
	}

	/*
	 * Dynamic attach so allow_peer2peer reaches the exporter; then pin, so the mapping
	 * cannot move underneath the FPGA and move_notify never fires.
	 */
	import->attach = dma_buf_dynamic_attach(import->dmabuf, &hdev->pdev->dev,
						&hl_pcie_dmabuf_attach_ops, import);
	if (IS_ERR(import->attach)) {
		ret = PTR_ERR(import->attach);
		dev_err(&hdev->pdev->dev, "dma_buf_dynamic_attach failed: %d\n", ret);
		goto err_put;
	}

	dma_resv_lock(import->dmabuf->resv, NULL);
	ret = dma_buf_pin(import->attach);
	if (ret) {
		dma_resv_unlock(import->dmabuf->resv);
		dev_err(&hdev->pdev->dev, "dma_buf_pin failed: %d\n", ret);
		goto err_detach;
	}
	sgt = dma_buf_map_attachment(import->attach, DMA_FROM_DEVICE);
	if (IS_ERR(sgt)) {
		ret = PTR_ERR(sgt);
		dma_buf_unpin(import->attach);
		dma_resv_unlock(import->dmabuf->resv);
		dev_err(&hdev->pdev->dev, "dma_buf_map_attachment failed: %d\n", ret);
		goto err_detach;
	}
	dma_resv_unlock(import->dmabuf->resv);
	import->sgt = sgt;

	ret = hl_pcie_dmabuf_check(hdev, sgt, &import->iova, &import->size);
	if (ret)
		goto err_unmap;

	/* Dropped by hl_pcie_dmabuf_release(). */
	hl_pcie_dev_get(hdev);
	mutex_lock(&hdev->lock);
	import->handle = hdev->next_dma_handle++;
	if (hdev->next_dma_handle == 0)
		hdev->next_dma_handle = 1;
	list_add_tail(&import->list, &hdev->dmabuf_imports);
	mutex_unlock(&hdev->lock);

	user.size = import->size;
	user.iova = import->iova;
	user.handle = import->handle;
	if (copy_to_user((void __user *)arg, &user, sizeof(user))) {
		mutex_lock(&hdev->lock);
		list_del(&import->list);
		mutex_unlock(&hdev->lock);
		hl_pcie_dmabuf_release(import);
		return -EFAULT;
	}

	dev_info(&hdev->pdev->dev,
		 "imported dma-buf fd %d as handle %llu: iova 0x%llx size 0x%zx\n",
		 user.fd, import->handle, (unsigned long long)import->iova, import->size);
	return 0;

err_unmap:
	dma_resv_lock(import->dmabuf->resv, NULL);
	dma_buf_unmap_attachment(import->attach, import->sgt, DMA_FROM_DEVICE);
	dma_buf_unpin(import->attach);
	dma_resv_unlock(import->dmabuf->resv);
err_detach:
	dma_buf_detach(import->dmabuf, import->attach);
err_put:
	dma_buf_put(import->dmabuf);
err_free:
	kfree(import);
	return ret;
}

static int hl_pcie_ioctl_dmabuf_release(struct hl_pcie_file_ctx *ctx, unsigned long arg)
{
	struct hl_pcie_dev *hdev = ctx->hdev;
	struct hl_pcie_dmabuf_import *import;
	struct hl_pcie_dmabuf user;
	bool found = false;

	if (copy_from_user(&user, (void __user *)arg, sizeof(user)))
		return -EFAULT;

	mutex_lock(&hdev->lock);
	list_for_each_entry(import, &hdev->dmabuf_imports, list) {
		if (import->handle == user.handle && import->owner == ctx) {
			list_del(&import->list);
			found = true;
			break;
		}
	}
	mutex_unlock(&hdev->lock);

	if (!found)
		return -ENOENT;

	hl_pcie_dmabuf_release(import);
	return 0;
}

static int hl_pcie_ioctl_msix_info(struct hl_pcie_file_ctx *ctx, unsigned long arg)
{
	struct hl_pcie_dev *hdev = ctx->hdev;
	struct hl_pcie_msix_info info;
	struct msi_msg msg;
	int vector;
	int ret;

	if (copy_from_user(&info, (void __user *)arg, sizeof(info)))
		return -EFAULT;

	vector = info.vector;
	if (vector < 0 || vector >= hdev->num_vectors)
		return -EINVAL;

	if (info.reset_count) {
		/*
		 * reset_count is how the pre-MSIX_CTRL receivers (the reference Python player,
		 * the legacy DmaRxRing) say "I am about to wait on this vector" before polling
		 * irq_count. They never call MSIX_WAIT or MSIX_CTRL and unmask the table entry
		 * themselves -- which the kernel, owning the mask, treats as a disabled IRQ:
		 * the flow handler marks it pending and re-masks without running the handler,
		 * so irq_count never moves and they time out. Enable the vector for this fd
		 * here, exactly as MSIX_WAIT does, so that path keeps working.
		 */
		/*
		 * Zero first, enable second: a completion that lands between the unmask and a
		 * reset done afterwards would be erased, and the poller would wait for a frame
		 * that has already finished.
		 */
		atomic_set(&hdev->vectors[vector].count, 0);
		ret = hl_pcie_vector_enable(hdev, ctx, vector, NULL);
		if (ret)
			return ret;
	}

	memset(&msg, 0, sizeof(msg));
	get_cached_msi_msg(pci_irq_vector(hdev->pdev, vector), &msg);
	info.msg_addr = ((u64)msg.address_hi << 32) | msg.address_lo;
	info.msg_data = msg.data;
	info.fpga_data = vector < 32 ? BIT(vector) : 0;
	info.irq_count = atomic_read(&hdev->vectors[vector].count);
	info.num_vectors = hdev->num_vectors;
	info.enabled = hdev->num_vectors > 0 ? 1 : 0;
	info.reset_count = 0;

	if (copy_to_user((void __user *)arg, &info, sizeof(info)))
		return -EFAULT;
	return 0;
}

static int hl_pcie_ioctl_msix_wait(struct hl_pcie_file_ctx *ctx, unsigned long arg)
{
	struct hl_pcie_dev *hdev = ctx->hdev;
	struct hl_pcie_msix_wait wait;
	struct hl_pcie_vector *vector;
	long timeout;
	int ret = 0;

	if (copy_from_user(&wait, (void __user *)arg, sizeof(wait)))
		return -EFAULT;
	if (wait.vector >= (u32)hdev->num_vectors)
		return -EINVAL;

	/*
	 * Waiting implies wanting the interrupt: enable the vector for this fd if it is not
	 * already. This is what keeps a caller that never heard of HL_PCIE_IOC_MSIX_CTRL --
	 * one that still writes the FPGA table itself -- receiving completions now that the
	 * kernel owns the mask.
	 */
	ret = hl_pcie_vector_enable(hdev, ctx, wait.vector, NULL);
	if (ret)
		return ret;

	vector = &hdev->vectors[wait.vector];

	/*
	 * `dead` is part of the condition: hl_pcie_free_irqs() wakes every waiter on the remove
	 * path, and without it a FOREVER waiter would just go back to sleep on a count that can
	 * no longer change.
	 */
	if (wait.timeout_ms == HL_PCIE_MSIX_WAIT_FOREVER) {
		ret = wait_event_interruptible(vector->wait,
			atomic_read(&vector->count) != wait.last_count ||
			READ_ONCE(hdev->dead));
	} else if (wait.timeout_ms == 0) {
		/* Poll: report the count without sleeping. */
		if (atomic_read(&vector->count) == wait.last_count)
			ret = -ETIMEDOUT;
	} else {
		timeout = wait_event_interruptible_timeout(vector->wait,
			atomic_read(&vector->count) != wait.last_count ||
			READ_ONCE(hdev->dead),
			msecs_to_jiffies(wait.timeout_ms));
		if (timeout == 0)
			ret = -ETIMEDOUT;
		else if (timeout < 0)
			ret = timeout;
	}
	if (READ_ONCE(hdev->dead))
		return -ENODEV;

	/* Report the count even on timeout so the caller can resynchronize. */
	wait.irq_count = atomic_read(&vector->count);
	if (copy_to_user((void __user *)arg, &wait, sizeof(wait)))
		return -EFAULT;
	return ret;
}

static int hl_pcie_ioctl_msix_ctrl(struct hl_pcie_file_ctx *ctx, unsigned long arg)
{
	struct hl_pcie_dev *hdev = ctx->hdev;
	struct hl_pcie_msix_ctrl ctrl;
	int ret;

	if (copy_from_user(&ctrl, (void __user *)arg, sizeof(ctrl)))
		return -EFAULT;
	if (ctrl.flags || ctrl.reserved[0] || ctrl.reserved[1] || ctrl.reserved[2] ||
	    ctrl.reserved[3])
		return -EINVAL;
	if (ctrl.enable > 1)
		return -EINVAL;

	if (ctrl.enable)
		ret = hl_pcie_vector_enable(hdev, ctx, ctrl.vector, &ctrl.enabled_refs);
	else
		ret = hl_pcie_vector_disable(hdev, ctx, ctrl.vector, &ctrl.enabled_refs);
	if (ret)
		return ret;

	if (copy_to_user((void __user *)arg, &ctrl, sizeof(ctrl)))
		return -EFAULT;
	return 0;
}

static long hl_pcie_ioctl_identity(struct hl_pcie_dev *hdev, unsigned long arg)
{
	struct hl_pcie_identity_info info;
	struct pci_dev *pdev = hdev->pdev;

	hl_pcie_refresh_uuid(hdev);

	memset(&info, 0, sizeof(info));
	info.domain = pci_domain_nr(pdev->bus);
	info.bus = pdev->bus->number;
	info.devfn = pdev->devfn;
	info.requester_id = hl_pcie_requester_id(hdev);
	info.minor = hdev->minor;
	mutex_lock(&hdev->lock);
	info.hsb_version = hdev->identity.hsb_version;
	info.hsb_date = hdev->identity.hsb_date;
	info.hsb_status = hdev->identity.hsb_status;
	info.flags = hdev->identity.flags;
	hl_pcie_identity_uuid_bytes(&hdev->identity, info.uuid);
	mutex_unlock(&hdev->lock);
	info.num_msix_vectors = hdev->num_vectors;
	info.dma_address_bits = dma_address_bits;
	info.bar0_len = hdev->bar0_len;

	if (copy_to_user((void __user *)arg, &info, sizeof(info)))
		return -EFAULT;
	return 0;
}

/*
 * HL_PCIE_IOC_GET_SYNCPOINT: lazily allocate, or return if already allocated,
 * the syncpoint tracking one MSI-X vector's completions.
 *
 * @hdev: device the vector belongs to
 * @arg:  user pointer to struct hl_pcie_get_syncpoint (sensor in; syncpoint_id,
 *        current_value out)
 */
static int hl_pcie_ioctl_get_syncpoint(struct hl_pcie_dev *hdev, unsigned long arg)
{
	struct hl_pcie_get_syncpoint info;
	struct hl_pcie_vector *v;
	struct host1x_syncpt *sp;

	if (copy_from_user(&info, (void __user *)arg, sizeof(info)))
		return -EFAULT;
	if (info.reserved)
		return -EINVAL;
	if (info.sensor >= (u32)hdev->num_vectors)
		return -EINVAL;
	if (!hdev->host1x)
		return -ENODEV;

	v = &hdev->vectors[info.sensor];

	mutex_lock(&hdev->syncpt_lock);
	sp = v->sensor_syncpt;
	if (!sp) {
		sp = host1x_syncpt_alloc(hdev->host1x, HOST1X_SYNCPT_CLIENT_MANAGED,
					 HL_PCIE_DEVICE_NAME);
		if (IS_ERR_OR_NULL(sp)) {
			mutex_unlock(&hdev->syncpt_lock);
			return sp ? PTR_ERR(sp) : -ENOSPC;
		}
		/* Published for hl_pcie_irq_handler(); see struct hl_pcie_vector::sensor_syncpt. */
		WRITE_ONCE(v->sensor_syncpt, sp);
	}
	mutex_unlock(&hdev->syncpt_lock);

	info.syncpoint_id = host1x_syncpt_id(sp);
	info.current_value = host1x_syncpt_read(sp);
	info.reserved = 0;
	if (copy_to_user((void __user *)arg, &info, sizeof(info)))
		return -EFAULT;
	return 0;
}

/*
 * HL_PCIE_IOC_ARM: return the wait point (id + current value) to arm a
 * capture request against -- the caller waits for the syncpoint to reach
 * base_value + 1. Enabling the vector's completion interrupt
 * (HL_PCIE_IOC_MSIX_CTRL / HL_PCIE_IOC_MSIX_WAIT) stays the caller's job.
 *
 * @hdev: device the vector belongs to
 * @arg:  user pointer to struct hl_pcie_arm (sensor in; syncpoint_id,
 *        base_value out)
 */
static int hl_pcie_ioctl_arm(struct hl_pcie_dev *hdev, unsigned long arg)
{
	struct hl_pcie_arm info;
	struct host1x_syncpt *sp;

	if (copy_from_user(&info, (void __user *)arg, sizeof(info)))
		return -EFAULT;
	if (info.reserved)
		return -EINVAL;
	if (info.sensor >= (u32)hdev->num_vectors)
		return -EINVAL;

	/* READ_ONCE: races the WRITE_ONCE publish in hl_pcie_ioctl_get_syncpoint(). */
	sp = READ_ONCE(hdev->vectors[info.sensor].sensor_syncpt);
	if (!sp)
		return -ENODEV;

	info.syncpoint_id = host1x_syncpt_id(sp);
	info.base_value = host1x_syncpt_read(sp);
	info.reserved = 0;
	if (copy_to_user((void __user *)arg, &info, sizeof(info)))
		return -EFAULT;
	return 0;
}

static long hl_pcie_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
	struct hl_pcie_file_ctx *ctx = filp->private_data;
	struct hl_pcie_dev *hdev = ctx->hdev;

	/* The PCI device is gone (unbind / removal): BAR0 is unmapped and the IRQs freed. */
	if (READ_ONCE(hdev->dead))
		return -ENODEV;

	switch (cmd) {
	case HL_PCIE_IOC_READ:
		return hl_pcie_ioctl_read(hdev, arg);
	case HL_PCIE_IOC_WRITE:
		return hl_pcie_ioctl_write(hdev, arg);
	case HL_PCIE_IOC_DMA_ALLOC:
		return hl_pcie_ioctl_dma_alloc(ctx, arg);
	case HL_PCIE_IOC_DMA_FREE:
		return hl_pcie_ioctl_dma_free(ctx, arg);
	case HL_PCIE_IOC_MSIX_INFO:
		return hl_pcie_ioctl_msix_info(ctx, arg);
	case HL_PCIE_IOC_MSIX_WAIT:
		return hl_pcie_ioctl_msix_wait(ctx, arg);
	case HL_PCIE_IOC_MSIX_CTRL:
		return hl_pcie_ioctl_msix_ctrl(ctx, arg);
	case HL_PCIE_IOC_DMABUF_IMPORT:
		return hl_pcie_ioctl_dmabuf_import(ctx, arg);
	case HL_PCIE_IOC_DMABUF_RELEASE:
		return hl_pcie_ioctl_dmabuf_release(ctx, arg);
	case HL_PCIE_IOC_IDENTITY:
		return hl_pcie_ioctl_identity(hdev, arg);
	case HL_PCIE_IOC_GET_SYNCPOINT:
		return hl_pcie_ioctl_get_syncpoint(hdev, arg);
	case HL_PCIE_IOC_ARM:
		return hl_pcie_ioctl_arm(hdev, arg);
	default:
		return -ENOTTY;
	}
}

static void hl_pcie_dma_vma_open(struct vm_area_struct *vma)
{
	struct hl_pcie_dma_alloc *alloc = vma->vm_private_data;

	if (alloc)
		refcount_inc(&alloc->refs);
}

static void hl_pcie_dma_vma_close(struct vm_area_struct *vma)
{
	struct hl_pcie_dma_alloc *alloc = vma->vm_private_data;

	if (alloc)
		hl_pcie_dma_put(alloc);
}

static const struct vm_operations_struct hl_pcie_dma_vm_ops = {
	.open = hl_pcie_dma_vma_open,
	.close = hl_pcie_dma_vma_close,
};

static int hl_pcie_mmap_dma(struct hl_pcie_file_ctx *ctx,
			    struct vm_area_struct *vma, u64 handle)
{
	struct hl_pcie_dma_alloc *alloc;
	struct hl_pcie_dev *hdev = ctx->hdev;
	unsigned long size;
	unsigned long user_pgoff;
	int ret;

	size = vma->vm_end - vma->vm_start;
	mutex_lock(&hdev->lock);
	alloc = hl_pcie_find_dma_locked(hdev, handle);
	if (!alloc || alloc->owner != ctx) {
		mutex_unlock(&hdev->lock);
		return -ENOENT;
	}
	if (size > alloc->size) {
		mutex_unlock(&hdev->lock);
		return -EINVAL;
	}
	refcount_inc(&alloc->refs);
	mutex_unlock(&hdev->lock);

	/*
	 * Userspace uses vm_pgoff as an allocation selector:
	 *   HL_PCIE_DMA_MMAP_BASE_PGOFF + handle
	 * dma_mmap_attrs() interprets vm_pgoff as an offset into the DMA
	 * allocation, so clear it for the actual full-buffer mapping.
	 */
	user_pgoff = vma->vm_pgoff;
	vma->vm_pgoff = 0;
	ret = dma_mmap_attrs(&hdev->pdev->dev, vma, alloc->cpu_addr,
			     alloc->dma_addr, alloc->size, 0);
	vma->vm_pgoff = user_pgoff;
	if (ret) {
		hl_pcie_dma_put(alloc);
		return ret;
	}
	vma->vm_private_data = alloc;
	vma->vm_ops = &hl_pcie_dma_vm_ops;
	return ret;
}

static int hl_pcie_mmap(struct file *filp, struct vm_area_struct *vma)
{
	struct hl_pcie_file_ctx *ctx = filp->private_data;
	unsigned long pgoff;

	if (READ_ONCE(ctx->hdev->dead))
		return -ENODEV;

	pgoff = vma->vm_pgoff;
	if (pgoff >= HL_PCIE_DMA_MMAP_BASE_PGOFF)
		return hl_pcie_mmap_dma(ctx, vma,
					 pgoff - HL_PCIE_DMA_MMAP_BASE_PGOFF);

	return -EINVAL;
}

/*
 * Read-only identity attributes on the class device,
 * /sys/class/hololink_pcie/hololink_pcieN/{bdf,uuid,hsb_version,hsb_date,requester_id,
 * msix_vectors}: enumeration and udev rules can key a node on its BDF and UUID without
 * opening it. Same sources and formats as HL_PCIE_IOC_IDENTITY. device_create() stores
 * hdev as the device's drvdata.
 */
static ssize_t bdf_show(struct device *dev, struct device_attribute *attr,
			char *buf)
{
	struct hl_pcie_dev *hdev = dev_get_drvdata(dev);
	struct pci_dev *pdev = hdev->pdev;

	return sysfs_emit(buf, "%04x:%02x:%02x.%d\n", pci_domain_nr(pdev->bus),
			  pdev->bus->number, PCI_SLOT(pdev->devfn),
			  PCI_FUNC(pdev->devfn));
}
static DEVICE_ATTR_RO(bdf);

static ssize_t uuid_show(struct device *dev, struct device_attribute *attr,
			 char *buf)
{
	struct hl_pcie_dev *hdev = dev_get_drvdata(dev);
	u8 uuid[16];

	hl_pcie_refresh_uuid(hdev);
	mutex_lock(&hdev->lock);
	hl_pcie_identity_uuid_bytes(&hdev->identity, uuid);
	mutex_unlock(&hdev->lock);
	/* %pUb: bytes in array order, 8-4-4-4-12 lowercase -- the canonical text. */
	return sysfs_emit(buf, "%pUb\n", uuid);
}
static DEVICE_ATTR_RO(uuid);

static ssize_t hsb_version_show(struct device *dev,
				struct device_attribute *attr, char *buf)
{
	struct hl_pcie_dev *hdev = dev_get_drvdata(dev);

	return sysfs_emit(buf, "0x%08x\n", hdev->identity.hsb_version);
}
static DEVICE_ATTR_RO(hsb_version);

static ssize_t hsb_date_show(struct device *dev, struct device_attribute *attr,
			     char *buf)
{
	struct hl_pcie_dev *hdev = dev_get_drvdata(dev);

	return sysfs_emit(buf, "0x%08x\n", hdev->identity.hsb_date);
}
static DEVICE_ATTR_RO(hsb_date);

static ssize_t requester_id_show(struct device *dev,
				 struct device_attribute *attr, char *buf)
{
	struct hl_pcie_dev *hdev = dev_get_drvdata(dev);

	return sysfs_emit(buf, "0x%04x\n", hl_pcie_requester_id(hdev));
}
static DEVICE_ATTR_RO(requester_id);

static ssize_t msix_vectors_show(struct device *dev,
				 struct device_attribute *attr, char *buf)
{
	struct hl_pcie_dev *hdev = dev_get_drvdata(dev);

	return sysfs_emit(buf, "%d\n", hdev->num_vectors);
}
static DEVICE_ATTR_RO(msix_vectors);

static struct attribute *hl_pcie_dev_attrs[] = {
	&dev_attr_bdf.attr,
	&dev_attr_uuid.attr,
	&dev_attr_hsb_version.attr,
	&dev_attr_hsb_date.attr,
	&dev_attr_requester_id.attr,
	&dev_attr_msix_vectors.attr,
	NULL,
};
ATTRIBUTE_GROUPS(hl_pcie_dev);

static const struct file_operations hl_pcie_fops = {
	.owner = THIS_MODULE,
	.open = hl_pcie_open,
	.release = hl_pcie_release,
	.unlocked_ioctl = hl_pcie_ioctl,
	.mmap = hl_pcie_mmap,
};

static int hl_pcie_alloc_minor(struct hl_pcie_dev *hdev)
{
	int minor;

	mutex_lock(&hl_pcie_devices_lock);
	for (minor = 0; minor < HL_PCIE_MAX_MINORS; minor++) {
		if (!hl_pcie_devices[minor]) {
			hl_pcie_devices[minor] = hdev;
			hdev->minor = minor;
			mutex_unlock(&hl_pcie_devices_lock);
			return 0;
		}
	}
	mutex_unlock(&hl_pcie_devices_lock);
	return -ENOSPC;
}

/*
 * Unlink the device and, in the same critical section, re-arm the bootp publisher if any other
 * device remains. Callers on the remove path have already cancel_delayed_work_sync()'d the work
 * (outside this lock: the work fn blocks on it), so between the cancel and this re-arm no job
 * can observe the departing hdev.
 */
static void hl_pcie_free_minor(struct hl_pcie_dev *hdev)
{
	bool any = false;
	int minor;

	mutex_lock(&hl_pcie_devices_lock);
	if (hdev->minor >= 0 && hdev->minor < HL_PCIE_MAX_MINORS &&
	    hl_pcie_devices[hdev->minor] == hdev)
		hl_pcie_devices[hdev->minor] = NULL;
	for (minor = 0; minor < HL_PCIE_MAX_MINORS; minor++)
		if (hl_pcie_devices[minor])
			any = true;
	if (any)
		schedule_delayed_work(&hl_pcie_bootp_work, HL_PCIE_BOOTP_PERIOD);
	mutex_unlock(&hl_pcie_devices_lock);
}

/*
 * bar_ptp: the endpoint's PTP clock.
 *
 * Every command is an edge-triggered handshake against STATUS.COMMAND_ACK, which mirrors
 * CONTROL[3:0] across the clock-domain crossing: clear the bit and wait for its echo to drop
 * (rearm), write it and wait for the echo (assert), do the work, write CONTROL back to 0 and
 * wait for the echo to drop again (clear). SNAPSHOT latches a coherent (seconds, nanoseconds)
 * pair into the TIME registers, so the three reads need no ordering of their own. Sequence and
 * timeouts follow the bring-up tree's hololink_bar_ptp.c; everything runs in process context
 * (the bootp delayed work, PHC ioctls) under ptp_lock and may sleep.
 *
 * The block is probed lazily and remembered: a bitfile without it stubs the window, and the
 * one thing software cannot bound is an MMIO read that never completes, which is why both
 * users are opt-in module parameters and STATUS is consulted before CONTROL is ever written.
 */
#define HL_PCIE_PTP_ACK_TIMEOUT_US 20000
#define HL_PCIE_PTP_ADJFINE_DENOM 65536000000LL

static bool hl_pcie_ptp_mapped(const struct hl_pcie_dev *hdev)
{
	return hdev->bar0 && hdev->bar0_len >= HL_PCIE_REG_PTP_STATUS + sizeof(u32);
}

static u32 hl_pcie_ptp_read(struct hl_pcie_dev *hdev, u32 offset)
{
	return ioread32(hdev->bar0 + offset);
}

static void hl_pcie_ptp_write(struct hl_pcie_dev *hdev, u32 offset, u32 value)
{
	iowrite32(value, hdev->bar0 + offset);
}

/* Wait for the STATUS bits in `mask` to all read `asserted`; -EIO when STATUS reads all-ones. */
static int hl_pcie_ptp_wait_status(struct hl_pcie_dev *hdev, u32 mask, bool asserted)
{
	u32 status;
	int ret;

	ret = read_poll_timeout(ioread32, status,
				status == U32_MAX ||
				(asserted ? (status & mask) == mask : (status & mask) == 0),
				10, HL_PCIE_PTP_ACK_TIMEOUT_US, false,
				hdev->bar0 + HL_PCIE_REG_PTP_STATUS);
	if (ret)
		return ret;
	return status == U32_MAX ? -EIO : 0;
}

/* First contact with the block, remembered for the life of the binding. ptp_lock held. */
static int hl_pcie_ptp_probe_block_locked(struct hl_pcie_dev *hdev)
{
	u32 status;

	if (hdev->ptp_state == HL_PCIE_PTP_PRESENT)
		return 0;
	if (hdev->ptp_state == HL_PCIE_PTP_ABSENT)
		return -ENODEV;
	if (!hl_pcie_ptp_mapped(hdev)) {
		hdev->ptp_state = HL_PCIE_PTP_ABSENT;
		dev_info(&hdev->pdev->dev, "BAR0 too small for a PTP block at 0x%x; PTP disabled\n",
			 HL_PCIE_REG_PTP_BASE);
		return -ENODEV;
	}
	status = hl_pcie_ptp_read(hdev, HL_PCIE_REG_PTP_STATUS);
	if (status == U32_MAX || !(status & HL_PCIE_PTP_STATUS_PRESENT)) {
		hdev->ptp_state = HL_PCIE_PTP_ABSENT;
		dev_info(&hdev->pdev->dev,
			 "PTP block at BAR0 0x%x %s (STATUS 0x%08x); PTP disabled for this binding\n",
			 HL_PCIE_REG_PTP_BASE,
			 status == U32_MAX ? "unreadable" : "not present", status);
		return -ENODEV;
	}
	hdev->ptp_state = HL_PCIE_PTP_PRESENT;
	return 0;
}

static int hl_pcie_ptp_rearm(struct hl_pcie_dev *hdev, u32 cmd)
{
	u32 ctrl = hl_pcie_ptp_read(hdev, HL_PCIE_REG_PTP_CONTROL);

	if (ctrl == U32_MAX)
		return -EIO;
	hl_pcie_ptp_write(hdev, HL_PCIE_REG_PTP_CONTROL, ctrl & ~cmd);
	return hl_pcie_ptp_wait_status(hdev, cmd << HL_PCIE_PTP_STATUS_ACK_SHIFT, false);
}

static int hl_pcie_ptp_assert(struct hl_pcie_dev *hdev, u32 cmd, bool need_time_valid)
{
	u32 mask = cmd << HL_PCIE_PTP_STATUS_ACK_SHIFT;

	if (need_time_valid)
		mask |= HL_PCIE_PTP_STATUS_TIME_VALID;
	hl_pcie_ptp_write(hdev, HL_PCIE_REG_PTP_CONTROL, cmd);
	return hl_pcie_ptp_wait_status(hdev, mask, true);
}

static int hl_pcie_ptp_clear(struct hl_pcie_dev *hdev, u32 cmd)
{
	hl_pcie_ptp_write(hdev, HL_PCIE_REG_PTP_CONTROL, 0);
	return hl_pcie_ptp_wait_status(hdev, cmd << HL_PCIE_PTP_STATUS_ACK_SHIFT, false);
}

/* SNAPSHOT and read the latched time. ptp_lock held. `sts` may be NULL. */
static int hl_pcie_ptp_gettime_locked(struct hl_pcie_dev *hdev, struct timespec64 *ts,
				      struct ptp_system_timestamp *sts)
{
	u32 sec_lo, sec_hi, nsec;
	int clear_ret;
	int ret;

	ret = hl_pcie_ptp_probe_block_locked(hdev);
	if (ret)
		return ret;
	ret = hl_pcie_ptp_rearm(hdev, HL_PCIE_PTP_CMD_SNAPSHOT);
	if (ret)
		return ret;

	ptp_read_system_prets(sts);
	ret = hl_pcie_ptp_assert(hdev, HL_PCIE_PTP_CMD_SNAPSHOT, true);
	if (ret)
		goto clear;
	sec_lo = hl_pcie_ptp_read(hdev, HL_PCIE_REG_PTP_TIME_SEC_LO);
	sec_hi = hl_pcie_ptp_read(hdev, HL_PCIE_REG_PTP_TIME_SEC_HI);
	nsec = hl_pcie_ptp_read(hdev, HL_PCIE_REG_PTP_TIME_NSEC);
	ptp_read_system_postts(sts);
	if (sec_lo == U32_MAX || sec_hi == U32_MAX || nsec == U32_MAX) {
		ret = -EIO;
		goto clear;
	}
	if (nsec >= NSEC_PER_SEC) {
		ret = -ERANGE;
		goto clear;
	}
	ts->tv_sec = ((u64)(sec_hi & HL_PCIE_PTP_SEC_HI_MASK) << 32) | sec_lo;
	ts->tv_nsec = nsec;

clear:
	clear_ret = hl_pcie_ptp_clear(hdev, HL_PCIE_PTP_CMD_SNAPSHOT);
	return ret ? ret : clear_ret;
}

/* Load a new time. ptp_lock held. */
static int hl_pcie_ptp_settime_locked(struct hl_pcie_dev *hdev, const struct timespec64 *ts)
{
	int clear_ret;
	int ret;
	u64 sec;

	if (ts->tv_sec < 0 || ts->tv_sec >= (1LL << 48) || ts->tv_nsec < 0 ||
	    ts->tv_nsec >= NSEC_PER_SEC)
		return -ERANGE;
	sec = ts->tv_sec;

	ret = hl_pcie_ptp_probe_block_locked(hdev);
	if (ret)
		return ret;
	ret = hl_pcie_ptp_rearm(hdev, HL_PCIE_PTP_CMD_SET_TIME);
	if (ret)
		return ret;
	hl_pcie_ptp_write(hdev, HL_PCIE_REG_PTP_SET_SEC_LO, lower_32_bits(sec));
	hl_pcie_ptp_write(hdev, HL_PCIE_REG_PTP_SET_SEC_HI, upper_32_bits(sec) & HL_PCIE_PTP_SEC_HI_MASK);
	hl_pcie_ptp_write(hdev, HL_PCIE_REG_PTP_SET_NSEC, (u32)ts->tv_nsec);
	ret = hl_pcie_ptp_assert(hdev, HL_PCIE_PTP_CMD_SET_TIME, false);
	clear_ret = hl_pcie_ptp_clear(hdev, HL_PCIE_PTP_CMD_SET_TIME);
	return ret ? ret : clear_ret;
}

/* Program FREQ_ADJ (units of 2^-24 ns per clock, relative to default_inc). ptp_lock held. */
static int hl_pcie_ptp_apply_freq_locked(struct hl_pcie_dev *hdev, s32 adjustment)
{
	int clear_ret;
	int ret;

	ret = hl_pcie_ptp_probe_block_locked(hdev);
	if (ret)
		return ret;
	ret = hl_pcie_ptp_rearm(hdev, HL_PCIE_PTP_CMD_APPLY_FREQ);
	if (ret)
		return ret;
	hl_pcie_ptp_write(hdev, HL_PCIE_REG_PTP_FREQ_ADJ, (u32)adjustment);
	ret = hl_pcie_ptp_assert(hdev, HL_PCIE_PTP_CMD_APPLY_FREQ, false);
	clear_ret = hl_pcie_ptp_clear(hdev, HL_PCIE_PTP_CMD_APPLY_FREQ);
	return ret ? ret : clear_ret;
}

/* The bootp announcement's use of the clock: one snapshot, no system timestamps. */
static int hl_pcie_bootp_read_ptp(struct hl_pcie_dev *hdev, u64 *sec, u32 *nsec)
{
	struct timespec64 ts;
	int ret;

	mutex_lock(&hdev->ptp_lock);
	ret = hl_pcie_ptp_gettime_locked(hdev, &ts, NULL);
	mutex_unlock(&hdev->ptp_lock);
	if (ret)
		return ret;
	*sec = ts.tv_sec;
	*nsec = ts.tv_nsec;
	return 0;
}

#if IS_REACHABLE(CONFIG_PTP_1588_CLOCK)
/*
 * PTP hardware clock (/dev/ptpN) over the same handshake, for phc2sys: the HSB clock is the
 * slave (-c /dev/ptpN) and CLOCK_REALTIME or another PHC (a ConnectX's /dev/ptp0) the source,
 * e.g. `phc2sys -s CLOCK_REALTIME -c /dev/ptpN -O 0 -S 0.001 -m`. That path needs gettimex64
 * (offset measurement with system timestamps around the read), settime64 / adjtime (steps
 * beyond the -S threshold; the first one is typically whole seconds and takes the get+set
 * path) and adjfine (the steady-state frequency pull). External timestamps and periodic outputs
 * are not offered.
 */
static struct hl_pcie_dev *hl_pcie_ptp_dev(struct ptp_clock_info *info)
{
	return container_of(info, struct hl_pcie_dev, ptp_info);
}

static int hl_pcie_ptp_op_gettimex64(struct ptp_clock_info *info, struct timespec64 *ts,
				     struct ptp_system_timestamp *sts)
{
	struct hl_pcie_dev *hdev = hl_pcie_ptp_dev(info);
	int ret;

	mutex_lock(&hdev->ptp_lock);
	ret = hl_pcie_ptp_gettime_locked(hdev, ts, sts);
	mutex_unlock(&hdev->ptp_lock);
	return ret;
}

static int hl_pcie_ptp_op_gettime64(struct ptp_clock_info *info, struct timespec64 *ts)
{
	return hl_pcie_ptp_op_gettimex64(info, ts, NULL);
}

static int hl_pcie_ptp_op_settime64(struct ptp_clock_info *info, const struct timespec64 *ts)
{
	struct hl_pcie_dev *hdev = hl_pcie_ptp_dev(info);
	int ret;

	mutex_lock(&hdev->ptp_lock);
	ret = hl_pcie_ptp_settime_locked(hdev, ts);
	mutex_unlock(&hdev->ptp_lock);
	return ret;
}

static int hl_pcie_ptp_op_adjtime(struct ptp_clock_info *info, s64 delta)
{
	struct hl_pcie_dev *hdev = hl_pcie_ptp_dev(info);
	struct timespec64 ts;
	int clear_ret;
	int ret;

	mutex_lock(&hdev->ptp_lock);
	if (delta <= -NSEC_PER_SEC || delta >= NSEC_PER_SEC) {
		/* ADJ_NSEC carries less than a second; step through get + set instead. */
		ret = hl_pcie_ptp_gettime_locked(hdev, &ts, NULL);
		if (!ret) {
			set_normalized_timespec64(&ts, ts.tv_sec, ts.tv_nsec + delta);
			ret = hl_pcie_ptp_settime_locked(hdev, &ts);
		}
		goto out;
	}
	ret = hl_pcie_ptp_probe_block_locked(hdev);
	if (ret)
		goto out;
	ret = hl_pcie_ptp_rearm(hdev, HL_PCIE_PTP_CMD_ADJ_TIME);
	if (ret)
		goto out;
	hl_pcie_ptp_write(hdev, HL_PCIE_REG_PTP_ADJ_NSEC, (u32)(s32)delta);
	ret = hl_pcie_ptp_assert(hdev, HL_PCIE_PTP_CMD_ADJ_TIME, false);
	clear_ret = hl_pcie_ptp_clear(hdev, HL_PCIE_PTP_CMD_ADJ_TIME);
	if (ret == -ETIMEDOUT) {
		/*
		 * The bring-up notes say ADJ_NSEC is not implemented in every timer build yet; a
		 * command that is never acknowledged is stepped through get + set instead.
		 */
		ret = hl_pcie_ptp_gettime_locked(hdev, &ts, NULL);
		if (!ret) {
			set_normalized_timespec64(&ts, ts.tv_sec, ts.tv_nsec + delta);
			ret = hl_pcie_ptp_settime_locked(hdev, &ts);
		}
	} else if (!ret) {
		ret = clear_ret;
	}
out:
	mutex_unlock(&hdev->ptp_lock);
	return ret;
}

static int hl_pcie_ptp_op_adjfine(struct ptp_clock_info *info, long scaled_ppm)
{
	struct hl_pcie_dev *hdev = hl_pcie_ptp_dev(info);
	s64 adjustment;
	long ppb;
	int ret;

	ppb = scaled_ppm_to_ppb(scaled_ppm);
	if (ppb < -info->max_adj || ppb > info->max_adj)
		return -ERANGE;
	adjustment = div64_s64((s64)hdev->ptp_default_inc * (s64)scaled_ppm,
			       HL_PCIE_PTP_ADJFINE_DENOM);
	if (adjustment < S32_MIN || adjustment > S32_MAX)
		return -ERANGE;
	if ((s64)hdev->ptp_default_inc + adjustment <= 0)
		return -ERANGE;

	mutex_lock(&hdev->ptp_lock);
	ret = hl_pcie_ptp_apply_freq_locked(hdev, (s32)adjustment);
	mutex_unlock(&hdev->ptp_lock);
	return ret;
}

static int hl_pcie_ptp_op_enable(struct ptp_clock_info *info, struct ptp_clock_request *request,
				 int on)
{
	return -EOPNOTSUPP;
}

/* Register the PHC when asked and the block answers; never fails the probe. */
static void hl_pcie_ptp_register(struct hl_pcie_dev *hdev)
{
	struct ptp_clock *clock;
	int ret;

	if (!ptp_clock)
		return;
	if (!ptp_clk_hz || !ptp_max_adj_ppb || ptp_max_adj_ppb >= NSEC_PER_SEC) {
		dev_err(&hdev->pdev->dev, "ptp_clk_hz=%u / ptp_max_adj_ppb=%u out of range; no PHC\n",
			ptp_clk_hz, ptp_max_adj_ppb);
		return;
	}
	hdev->ptp_default_inc = div_u64((u64)NSEC_PER_SEC << 24, ptp_clk_hz);

	/*
	 * FREQ_ADJ survives a module reload, so a stale value from a previous session keeps the
	 * counter off-rate. Restore the nominal increment before any servo reads the clock. This
	 * is also the first contact with the block: an absent one is remembered here.
	 */
	mutex_lock(&hdev->ptp_lock);
	ret = hl_pcie_ptp_apply_freq_locked(hdev, 0);
	mutex_unlock(&hdev->ptp_lock);
	if (ret) {
		if (ret != -ENODEV)
			dev_err(&hdev->pdev->dev, "PTP: restoring the nominal increment failed: %d; no PHC\n", ret);
		return;
	}

	hdev->ptp_info.owner = THIS_MODULE;
	snprintf(hdev->ptp_info.name, sizeof(hdev->ptp_info.name), "%s%d", HL_PCIE_DEVICE_NAME, hdev->minor);
	hdev->ptp_info.max_adj = ptp_max_adj_ppb;
	hdev->ptp_info.gettime64 = hl_pcie_ptp_op_gettime64;
	hdev->ptp_info.gettimex64 = hl_pcie_ptp_op_gettimex64;
	hdev->ptp_info.settime64 = hl_pcie_ptp_op_settime64;
	hdev->ptp_info.adjtime = hl_pcie_ptp_op_adjtime;
	hdev->ptp_info.adjfine = hl_pcie_ptp_op_adjfine;
	hdev->ptp_info.enable = hl_pcie_ptp_op_enable;

	clock = ptp_clock_register(&hdev->ptp_info, &hdev->pdev->dev);
	if (IS_ERR_OR_NULL(clock)) {
		dev_err(&hdev->pdev->dev, "ptp_clock_register failed: %ld; no PHC\n",
			clock ? PTR_ERR(clock) : -ENODEV);
		return;
	}
	hdev->ptp_clock = clock;
	dev_info(&hdev->pdev->dev, "registered PHC /dev/ptp%d ptp_clk_hz=%u default_inc=%llu\n",
		 ptp_clock_index(clock), ptp_clk_hz, (unsigned long long)hdev->ptp_default_inc);
}

static void hl_pcie_ptp_unregister(struct hl_pcie_dev *hdev)
{
	if (hdev->ptp_clock) {
		ptp_clock_unregister(hdev->ptp_clock);
		hdev->ptp_clock = NULL;
	}
}
#else
static void hl_pcie_ptp_register(struct hl_pcie_dev *hdev)
{
	if (ptp_clock)
		dev_info(&hdev->pdev->dev,
			 "ptp_clock=1 ignored: this kernel has no PTP clock support (CONFIG_PTP_1588_CLOCK)\n");
}

static void hl_pcie_ptp_unregister(struct hl_pcie_dev *hdev)
{
}
#endif /* IS_REACHABLE(CONFIG_PTP_1588_CLOCK) */

/*
 * bootp publisher.
 *
 * The announcement is BROADCAST on the loopback subnet, <packed BDF>:12268 ->
 * 127.255.255.255:12267, not unicast to 127.0.0.1. A UDP unicast is delivered to exactly ONE of
 * the sockets sharing a port under SO_REUSEADDR/SO_REUSEPORT -- the most specific bind wins, and
 * among equals the kernel picks one -- so a process that bound 0.0.0.0:12267 after the discovery
 * daemon would silently take every announcement away from it. A datagram whose attached route
 * carries RTCF_BROADCAST takes __udp4_lib_mcast_deliver() in __udp4_lib_rcv() instead and is
 * handed to EVERY socket bound to the port on INADDR_ANY, exactly as an Ethernet board's
 * broadcast reaches every listener on the host. (A socket bound to 127.0.0.1 itself does not
 * match a 127.255.255.255 destination and never sees the announcement; receivers bind
 * INADDR_ANY.)
 *
 * A hand-built loopback skb handed to netif_rx() with no route attached is dropped by
 * ip_route_input_slow() as a martian destination (net.ipv4.conf.*.route_localnet is 0
 * everywhere), and with route_localnet=1 it then fails fib_validate_source(). Real loopback
 * traffic passes only because loopback_xmit() keeps the *output* route attached, which makes
 * ip_rcv_finish_core() skip input routing. So this does what loopback_xmit() relies on: look up
 * the output route and skb_dst_set() it before netif_rx(). The local routing table carries
 * "broadcast 127.255.255.255 dev lo" for as long as lo has its 127.0.0.1/8 address (the kernel
 * adds it with the address; no configuration puts it there), so ip_route_output_key() returns
 * an RTN_BROADCAST route flagged RTCF_BROADCAST|RTCF_LOCAL, whose input hook is
 * ip_local_deliver(). With that dst pre-attached no input routing runs and the UDP layer sees
 * the broadcast flag. Nothing depends on a sysctl.
 *
 * The IP source is not 127.0.0.1 but the endpoint's BDF packed exactly as ciaddr is,
 * (domain << 16) | (bus << 8) | devfn. The host's bootp parsers take peer_ip from the datagram's
 * source, not from any bootp field, so this makes every endpoint its own peer ("0.5.1.0" for
 * 0005:01:00.0) even to a raw listener such as hololink-enumerate, instead of every endpoint
 * collapsing to 127.0.0.1; the hsb_pcie module then rewrites peer_ip to the textual BDF and
 * cross-checks the source against ciaddr. That source lies in 0.0.0.0/8: ip_route_output_key()
 * refuses a zeronet source and ip_route_input_slow() would drop it as a martian source, which is
 * why the route is still looked up with the loopback source and only the header carries the
 * BDF -- with the dst pre-attached, neither check runs on receive.
 *
 * Ethernet dst/src stay all-zero although the IP destination is a broadcast: lo's dev_addr is all
 * zeros, so eth_type_trans() classifies the frame PACKET_HOST (an all-ones dst would be
 * PACKET_MULTICAST on lo, anything else PACKET_OTHERHOST, which ip_rcv_core() drops), which
 * ip_rcv() accepts; the broadcast fan-out is decided by the attached route's rt_flags, not by
 * pkt_type. The UDP checksum is 0, which IPv4 permits.
 */
#if IS_ENABLED(CONFIG_INET)
static int hl_pcie_bootp_send(struct net_device *lo, const struct hl_pcie_bootp_pkt *pkt,
			      __be32 saddr)
{
	const unsigned int udp_len = sizeof(struct udphdr) + HL_PCIE_BOOTP_LEN;
	const unsigned int ip_len = sizeof(struct iphdr) + udp_len;
	struct flowi4 fl4;
	struct rtable *rt;
	struct sk_buff *skb;
	struct ethhdr *eth;
	struct iphdr *iph;
	struct udphdr *uh;

	if (!netif_running(lo))
		return -ENETDOWN;

	memset(&fl4, 0, sizeof(fl4));
	fl4.daddr = htonl(HL_PCIE_BOOTP_DADDR);
	fl4.saddr = htonl(INADDR_LOOPBACK);
	fl4.flowi4_proto = IPPROTO_UDP;
	fl4.flowi4_oif = lo->ifindex;
	rt = ip_route_output_key(&init_net, &fl4);
	if (IS_ERR(rt))
		return PTR_ERR(rt);

	skb = alloc_skb(NET_IP_ALIGN + ETH_HLEN + ip_len, GFP_KERNEL);
	if (!skb) {
		ip_rt_put(rt);
		return -ENOMEM;
	}
	skb_reserve(skb, NET_IP_ALIGN);

	eth = skb_put_zero(skb, ETH_HLEN);
	eth->h_proto = htons(ETH_P_IP);

	iph = skb_put_zero(skb, sizeof(*iph));
	iph->version = 4;
	iph->ihl = 5;
	iph->tot_len = htons(ip_len);
	iph->frag_off = htons(IP_DF);
	iph->ttl = 64;
	iph->protocol = IPPROTO_UDP;
	iph->saddr = saddr; /* the packed BDF, see above; the route was looked up from 127.0.0.1 */
	iph->daddr = htonl(HL_PCIE_BOOTP_DADDR); /* the loopback subnet broadcast, see above */
	ip_send_check(iph);

	uh = skb_put_zero(skb, sizeof(*uh));
	uh->source = htons(HL_PCIE_BOOTP_SPORT);
	uh->dest = htons(HL_PCIE_BOOTP_DPORT);
	uh->len = htons(udp_len);
	uh->check = 0;

	skb_put_data(skb, pkt, HL_PCIE_BOOTP_LEN);

	skb->ip_summed = CHECKSUM_UNNECESSARY;
	skb_reset_mac_header(skb);
	skb->protocol = eth_type_trans(skb, lo); /* sets skb->dev and pkt_type, pulls the eth header */
	skb_reset_network_header(skb);
	skb_set_transport_header(skb, sizeof(*iph));
	skb_dst_set(skb, &rt->dst); /* the skb now owns the route reference */

	netif_rx(skb); /* consumes the skb, also on NET_RX_DROP */
	return 0;
}
#else
static int hl_pcie_bootp_send(struct net_device *lo, const struct hl_pcie_bootp_pkt *pkt,
			      __be32 saddr)
{
	return -EOPNOTSUPP;
}
#endif

static void hl_pcie_bootp_work_fn(struct work_struct *work)
{
	struct net_device *lo = init_net.loopback_dev;
	struct hl_pcie_bootp_pkt pkt;
	struct hl_pcie_bootp_id id;
	bool any = false;
	int minor;

	mutex_lock(&hl_pcie_devices_lock);
	for (minor = 0; minor < HL_PCIE_MAX_MINORS; minor++) {
		struct hl_pcie_dev *hdev = hl_pcie_devices[minor];
		unsigned long secs;
		int ret;

		/* Slot claimed but device_create() not yet done, or probe unwinding. */
		if (!hdev || !hdev->device)
			continue;
		any = true;
		if (!bootp_announce || !lo)
			continue;
		/*
		 * Only an endpoint with a board UUID names a module; announce nothing else. Re-read
		 * the registers each tick rather than trust the cached identity: a UUID provisioned
		 * after probe would otherwise stay unannounced until something open()s the node,
		 * and it is this announcement that leads to that open() in the first place.
		 */
		hl_pcie_refresh_uuid(hdev);
		if (!(hdev->identity.flags & HL_PCIE_ID_F_UUID_VALID))
			continue;

		memset(&id, 0, sizeof(id));
		id.domain = (u16)pci_domain_nr(hdev->pdev->bus);
		id.bus = hdev->pdev->bus->number;
		id.devfn = hdev->pdev->devfn;
		mutex_lock(&hdev->lock); /* identity.uuid_* is rewritten under hdev->lock */
		hl_pcie_identity_uuid_bytes(&hdev->identity, id.uuid);
		id.hsb_ip_version = (u16)(hdev->identity.hsb_version & 0xFFFFU);
		mutex_unlock(&hdev->lock);
		id.dsn = hdev->dsn; /* probe-time constant; the filler falls back to the placeholder */
		secs = jiffies_to_msecs(jiffies - hdev->bootp_start_jiffies) / 1000;
		id.secs = (u16)min_t(unsigned long, secs, 0xFFFFUL);
		id.pkt_count = hdev->bootp_count;
		if (bootp_ptp) {
			ret = hl_pcie_bootp_read_ptp(hdev, &id.ptp_sec, &id.ptp_nsec);
			if (!ret)
				id.ptp_valid = 1;
			else
				dev_dbg_ratelimited(&hdev->pdev->dev,
						    "bootp: PTP snapshot not carried: %d\n", ret);
		}

		hl_pcie_bootp_fill(&pkt, &id);
		ret = hl_pcie_bootp_send(lo, &pkt,
					 htonl(hl_pcie_bootp_bdf32(id.domain, id.bus, id.devfn)));
		if (ret)
			dev_dbg_ratelimited(&hdev->pdev->dev, "bootp announcement not sent: %d\n", ret);
		else
			hdev->bootp_count++;
	}
	if (any)
		schedule_delayed_work(&hl_pcie_bootp_work, HL_PCIE_BOOTP_PERIOD);
	mutex_unlock(&hl_pcie_devices_lock);
}

static void hl_pcie_free_all_dma(struct hl_pcie_dev *hdev)
{
	struct hl_pcie_dma_alloc *alloc;
	struct hl_pcie_dma_alloc *tmp;

	mutex_lock(&hdev->lock);
	list_for_each_entry_safe(alloc, tmp, &hdev->dma_allocs, list) {
		list_del(&alloc->list);
		hl_pcie_dma_put(alloc);
	}
	mutex_unlock(&hdev->lock);
}

static void hl_pcie_free_all_dmabufs(struct hl_pcie_dev *hdev)
{
	struct hl_pcie_dmabuf_import *import;
	struct hl_pcie_dmabuf_import *tmp;
	LIST_HEAD(orphans);

	/* See hl_pcie_release() for why the release runs outside hdev->lock. */
	mutex_lock(&hdev->lock);
	list_splice_init(&hdev->dmabuf_imports, &orphans);
	mutex_unlock(&hdev->lock);

	list_for_each_entry_safe(import, tmp, &orphans, list) {
		list_del(&import->list);
		hl_pcie_dmabuf_release(import);
	}
}

static int hl_pcie_setup_irqs(struct hl_pcie_dev *hdev)
{
	int available;
	int requested;
	int ret;
	int i;

	available = pci_msix_vec_count(hdev->pdev);
	if (available < HL_PCIE_REQUIRED_MSIX_VECTORS) {
		dev_err(&hdev->pdev->dev,
			"device exposes %d MSI-X vectors; at least %d are required\n",
			available, HL_PCIE_REQUIRED_MSIX_VECTORS);
		return available < 0 ? available : -ENOSPC;
	}
	requested = min_t(int, available, HL_PCIE_MAX_MSIX_VECTORS);

	ret = pci_alloc_irq_vectors(hdev->pdev, HL_PCIE_REQUIRED_MSIX_VECTORS,
				    requested, PCI_IRQ_MSIX);
	if (ret < 0) {
		dev_err(&hdev->pdev->dev, "MSI-X allocation failed: %d\n", ret);
		return ret;
	}

	hdev->num_vectors = ret;
	dev_info(&hdev->pdev->dev,
		 "allocated %d MSI-X vectors (device advertises %d)\n",
		 hdev->num_vectors, available);
	for (i = 0; i < hdev->num_vectors; i++) {
		int irq = pci_irq_vector(hdev->pdev, i);

		atomic_set(&hdev->vectors[i].count, 0);
		init_waitqueue_head(&hdev->vectors[i].wait);
		hdev->vectors[i].enable_refs = 0;
		hdev->vectors[i].sensor_syncpt = NULL;
		hdev->vectors[i].hdev = hdev;
		/*
		 * Every vector starts disabled and is enabled per fd through
		 * HL_PCIE_IOC_MSIX_CTRL / HL_PCIE_IOC_MSIX_WAIT (hl_pcie_vector_enable). Disable
		 * eagerly: without IRQ_DISABLE_UNLAZY the kernel defers the mask write until the
		 * next interrupt actually arrives, and this driver wants the table entry masked
		 * the moment the last holder lets go.
		 */
		irq_set_status_flags(irq, IRQ_DISABLE_UNLAZY);
#ifdef IRQF_NO_AUTOEN
		ret = request_irq(irq, hl_pcie_irq_handler, IRQF_NO_AUTOEN,
				  HL_PCIE_DEVICE_NAME, &hdev->vectors[i]);
#else
		ret = request_irq(irq, hl_pcie_irq_handler, 0, HL_PCIE_DEVICE_NAME,
				  &hdev->vectors[i]);
		if (!ret)
			disable_irq(irq);
#endif
		if (ret) {
			dev_err(&hdev->pdev->dev, "request_irq[%d] failed: %d\n",
				 i, ret);
			while (--i >= 0)
				free_irq(pci_irq_vector(hdev->pdev, i),
					 &hdev->vectors[i]);
			pci_free_irq_vectors(hdev->pdev);
			hdev->num_vectors = 0;
			return ret;
		}
	}

	return 0;
}

static void hl_pcie_free_irqs(struct hl_pcie_dev *hdev)
{
	int i;

	for (i = 0; i < hdev->num_vectors; i++) {
		/* Release anything still blocked in HL_PCIE_IOC_MSIX_WAIT before the IRQ goes. */
		wake_up_all(&hdev->vectors[i].wait);
		free_irq(pci_irq_vector(hdev->pdev, i), &hdev->vectors[i]);
	}
	if (hdev->num_vectors)
		pci_free_irq_vectors(hdev->pdev);
	hdev->num_vectors = 0;
}

static int hl_pcie_read_identity(struct hl_pcie_dev *hdev)
{
	struct hl_pcie_identity *id = &hdev->identity;
	u32 uuid0 = 0;
	u32 uuid1 = 0;
	u32 uuid2 = 0;
	u32 uuid3 = 0;

	memset(id, 0, sizeof(*id));
	if (!hdev->bar0 ||
	    hdev->bar0_len < HL_PCIE_REG_HSB_STATUS + sizeof(u32))
		return -ENODEV;

	id->hsb_version = ioread32(hdev->bar0 + HL_PCIE_REG_HSB_VERSION);
	id->hsb_date = ioread32(hdev->bar0 + HL_PCIE_REG_HSB_DATE);
	id->hsb_status = ioread32(hdev->bar0 + HL_PCIE_REG_HSB_STATUS);
	if (hdev->bar0_len >= HL_PCIE_REG_HSB_UUID3 + sizeof(u32)) {
		/*
		 * Deliberately the same register as hsb_status: the identity magic ("HLPC") is
		 * what HSB_STAT reads on the bring-up bitfiles, there is no separate magic word in
		 * the BAR0 map (hololink_pcie_regs.h), and the userspace side (pcie_regs.hpp
		 * HSB_MAGIC = HSB_STAT) and the reference driver make the same comparison.
		 */
		id->magic = ioread32(hdev->bar0 + HL_PCIE_REG_HSB_STATUS);
		uuid0 = ioread32(hdev->bar0 + HL_PCIE_REG_HSB_UUID0);
		uuid1 = ioread32(hdev->bar0 + HL_PCIE_REG_HSB_UUID1);
		uuid2 = ioread32(hdev->bar0 + HL_PCIE_REG_HSB_UUID2);
		uuid3 = ioread32(hdev->bar0 + HL_PCIE_REG_HSB_UUID3);
		id->uuid_low = ((u64)uuid1 << 32) | uuid0;
		id->uuid_high = ((u64)uuid3 << 32) | uuid2;
	}

	if (id->magic == HL_PCIE_HSB_MAGIC)
		id->flags |= HL_PCIE_ID_F_MAGIC_VALID;
	else if (require_identity_magic)
		return -ENODEV;

	if (hl_pcie_uuid_valid(id->uuid_low, id->uuid_high))
		id->flags |= HL_PCIE_ID_F_UUID_VALID;

	if (skip_identity_check) {
		id->flags |= HL_PCIE_ID_F_CHECK_SKIPPED;
		return 0;
	}

	if (id->hsb_version == 0 || id->hsb_version == 0xffffffff)
		return -ENODEV;

	id->flags |= HL_PCIE_ID_F_VERSION_VALID;
	if (expected_hsb_version == 0 ||
	    id->hsb_version == expected_hsb_version) {
		id->flags |= HL_PCIE_ID_F_EXPECTED_VERSION_MATCH;
		return 0;
	}

	return -ENODEV;
}

/*
 * Find this board's host1x instance by device-tree compatible-string match
 *
 * @return: host1x instance, or NULL if it hasn't probed (or doesn't exist on
 *          this platform) -- callers must treat that as "no syncpoint/fence
 *          support", not a hard failure.
 */
static struct host1x *hl_pcie_find_host1x(void)
{
	static const struct of_device_id host1x_match[] = {
		{ .compatible = "nvidia,tegra234-host1x", },
		{ .compatible = "nvidia,tegra264-host1x", },
		{},
	};
	struct device_node *np;
	struct platform_device *host1x_pdev;
	struct host1x *host1x = NULL;

	for_each_matching_node(np, host1x_match) {
		host1x_pdev = of_find_device_by_node(np);
		if (host1x_pdev) {
			host1x = platform_get_drvdata(host1x_pdev);
			put_device(&host1x_pdev->dev);
			if (host1x) {
				of_node_put(np);
				break;
			}
		}
	}

	return host1x;
}

static void hl_pcie_setup_syncpoint(struct hl_pcie_dev *hdev)
{
	mutex_init(&hdev->syncpt_lock);
	hdev->host1x = hl_pcie_find_host1x();
	if (hdev->host1x)
		dev_info(&hdev->pdev->dev, "host1x found; HL_PCIE_IOC_GET_SYNCPOINT/ARM enabled\n");
	else
		dev_info(&hdev->pdev->dev,
			 "no host1x device found; HL_PCIE_IOC_GET_SYNCPOINT/ARM disabled\n");
}

/*
 * Caller must call hl_pcie_free_irqs(hdev) first -- otherwise the IRQ
 * handler's lockless READ_ONCE(sensor_syncpt) could race this function's
 * host1x_syncpt_put() and increment an already-freed syncpoint.
 */
static void hl_pcie_teardown_syncpoint(struct hl_pcie_dev *hdev)
{
	int i;

	for (i = 0; i < HL_PCIE_MAX_MSIX_VECTORS; i++) {
		if (hdev->vectors[i].sensor_syncpt)
			host1x_syncpt_put(hdev->vectors[i].sensor_syncpt);
		/* WRITE_ONCE for symmetry with the publish in hl_pcie_ioctl_get_syncpoint(). */
		WRITE_ONCE(hdev->vectors[i].sensor_syncpt, NULL);
	}
	hdev->host1x = NULL;
}

static int hl_pcie_probe(struct pci_dev *pdev, const struct pci_device_id *id)
{
	struct hl_pcie_dev *hdev;
	dev_t devt;
	int ret;

	ret = pci_enable_device_mem(pdev);
	if (ret)
		return ret;
	pci_clear_master(pdev);

	ret = pci_request_regions(pdev, HL_PCIE_DEVICE_NAME);
	if (ret)
		goto err_disable;

	/*
	 * 64 bits, now that the RTL applies PCIE_DP_BUF_BASE_HI (FIXED_MSB) to the outbound
	 * address and selects the 3DW or 4DW TLP header from the address itself. GPU VRAM
	 * dma-buf exports on IGX land in the PCIe host-bridge aperture well above 4 GiB
	 * (observed at 0x28_02600000 and 0x28_01000000), so a 32-bit mask cannot reach them.
	 *
	 * Older bitfiles truncate to the low 32 bits instead of rejecting the address, which
	 * is silent from the host's point of view and lands writes in whatever else shares
	 * this IOMMU domain. Load with dma_address_bits=32 against those; the >4 GiB check in
	 * hl_pcie_dmabuf_check() keys off this value.
	 */
	ret = dma_set_mask_and_coherent(&pdev->dev, DMA_BIT_MASK(dma_address_bits));
	if (ret) {
		dev_err(&pdev->dev, "dma_set_mask_and_coherent(%u) failed: %d\n",
			dma_address_bits, ret);
		goto err_regions;
	}

	hdev = kzalloc(sizeof(*hdev), GFP_KERNEL);
	if (!hdev) {
		ret = -ENOMEM;
		goto err_regions;
	}

	kref_init(&hdev->ref);
	hdev->pdev = pci_dev_get(pdev);
	hdev->minor = -1;
	hdev->next_dma_handle = 1;
	mutex_init(&hdev->lock);
	mutex_init(&hdev->ptp_lock);
	INIT_LIST_HEAD(&hdev->dma_allocs);
	INIT_LIST_HEAD(&hdev->dmabuf_imports);
	pci_set_drvdata(pdev, hdev);

	hdev->bar0_len = pci_resource_len(pdev, 0);
	if (!hdev->bar0_len ||
	    !(pci_resource_flags(pdev, 0) & IORESOURCE_MEM)) {
		ret = -ENODEV;
		dev_err(&pdev->dev, "BAR0 is not a memory resource\n");
		goto err_free;
	}
	hdev->bar0 = pci_ioremap_bar(pdev, 0);
	if (!hdev->bar0) {
		ret = -ENOMEM;
		dev_err(&pdev->dev, "BAR0 ioremap failed\n");
		goto err_free;
	}
	hl_pcie_mask_all_fpga_msix(hdev);

	ret = hl_pcie_read_identity(hdev);
	if (ret) {
		dev_err(&pdev->dev,
			"rejecting device: unsupported Hololink BAR identity "
			"(BAR0+0x%x expected 0x%08x, read 0x%08x; use skip_identity_check=1 for bring-up)\n",
			HL_PCIE_REG_HSB_VERSION, expected_hsb_version,
			hdev->identity.hsb_version);
		goto err_unmap;
	}
	/* The Device Serial Number extended capability, when the endpoint implements one. */
	hdev->dsn = pci_get_dsn(pdev);
	dev_info(&pdev->dev,
		 "PCI IDs %04x:%04x sub %04x:%04x, Hololink HSB version=0x%08x date=0x%08x magic=0x%08x uuid=%016llx%016llx flags=0x%08x dsn=%016llx\n",
		 pdev->vendor, pdev->device, pdev->subsystem_vendor,
		 pdev->subsystem_device, hdev->identity.hsb_version,
		 hdev->identity.hsb_date, hdev->identity.magic,
		 (unsigned long long)hdev->identity.uuid_high,
		 (unsigned long long)hdev->identity.uuid_low,
		 hdev->identity.flags, (unsigned long long)hdev->dsn);

	ret = hl_pcie_setup_irqs(hdev);
	if (ret)
		goto err_unmap;

	hl_pcie_setup_syncpoint(hdev);

	pci_set_master(pdev);

	ret = hl_pcie_alloc_minor(hdev);
	if (ret)
		goto err_master;

	devt = MKDEV(MAJOR(hl_pcie_devt), hdev->minor);
	cdev_init(&hdev->cdev, &hl_pcie_fops);
	hdev->cdev.owner = THIS_MODULE;
	ret = cdev_add(&hdev->cdev, devt, 1);
	if (ret)
		goto err_minor;

	hdev->device = device_create(hl_pcie_class, &pdev->dev, devt, hdev,
				     HL_PCIE_DEVICE_NAME "%d", hdev->minor);
	if (IS_ERR(hdev->device)) {
		ret = PTR_ERR(hdev->device);
		hdev->device = NULL;
		goto err_cdev;
	}

	/* Start announcing now; the first bootp for a hot-plugged board goes out immediately. */
	mutex_lock(&hl_pcie_devices_lock);
	hdev->bootp_start_jiffies = jiffies;
	hdev->bootp_count = 0;
	mutex_unlock(&hl_pcie_devices_lock);
	schedule_delayed_work(&hl_pcie_bootp_work, 0);

	/* Last, and never fatal: the endpoint is usable without its clock. */
	hl_pcie_ptp_register(hdev);

	dev_info(&pdev->dev, "Hololink PCIe device /dev/%s%d ready\n",
		 HL_PCIE_DEVICE_NAME, hdev->minor);
	return 0;

err_cdev:
	cdev_del(&hdev->cdev);
err_minor:
	hl_pcie_free_minor(hdev);
err_master:
	pci_clear_master(pdev);
	hl_pcie_free_irqs(hdev);
	hl_pcie_teardown_syncpoint(hdev);
err_unmap:
	pci_iounmap(pdev, hdev->bar0);
err_free:
	hl_pcie_dev_put(hdev);
err_regions:
	pci_release_regions(pdev);
err_disable:
	if (!keep_pci_enabled)
		pci_disable_device(pdev);
	return ret;
}

static void hl_pcie_remove(struct pci_dev *pdev)
{
	struct hl_pcie_dev *hdev;

	hdev = pci_get_drvdata(pdev);
	if (!hdev)
		return;

	/*
	 * Stop the bootp publisher before anything is torn down, and outside hl_pcie_devices_lock
	 * (the work fn blocks on it). After this returns no job is running or queued, so the hdev
	 * can be unlinked and freed safely; hl_pcie_free_minor() re-arms for the devices that
	 * remain. This is also what makes rmmod / insmod cycles safe.
	 */
	cancel_delayed_work_sync(&hl_pcie_bootp_work);
	/* No bootp snapshot is in flight now; take the PHC down before BAR0 is unmapped. */
	hl_pcie_ptp_unregister(hdev);

	/*
	 * From here on every file entry point fails with -ENODEV and open() no longer finds the
	 * device. Under both locks: open() tests the flag under hl_pcie_devices_lock, release()
	 * and the vector paths under hdev->lock, so none of them can slip in between the test
	 * and a BAR0 access that follows it once this has been set.
	 */
	mutex_lock(&hl_pcie_devices_lock);
	mutex_lock(&hdev->lock);
	hdev->dead = true;
	mutex_unlock(&hdev->lock);
	mutex_unlock(&hl_pcie_devices_lock);

	pci_clear_master(pdev);
	hl_pcie_mask_all_fpga_msix(hdev);
	hl_pcie_free_irqs(hdev);
	hl_pcie_teardown_syncpoint(hdev);
	hl_pcie_free_all_dmabufs(hdev);
	hl_pcie_free_all_dma(hdev);
	if (hdev->device)
		device_destroy(hl_pcie_class, MKDEV(MAJOR(hl_pcie_devt),
						     hdev->minor));
	cdev_del(&hdev->cdev);
	hl_pcie_free_minor(hdev);

	mutex_lock(&hdev->lock);
	pci_iounmap(pdev, hdev->bar0);
	hdev->bar0 = NULL;
	mutex_unlock(&hdev->lock);

	pci_release_regions(pdev);
	if (!keep_pci_enabled)
		pci_disable_device(pdev);
	/*
	 * The probe reference. An fd still open, a VMA of a DMA allocation still mapped or a
	 * dma-buf still attached keeps the hdev (and, through pci_dev_get, the pdev) around
	 * until the last of them goes.
	 */
	hl_pcie_dev_put(hdev);
}

static struct pci_driver hl_pcie_driver = {
	.name = HL_PCIE_DEVICE_NAME,
	.id_table = hl_pcie_ids,
	.probe = hl_pcie_probe,
	.remove = hl_pcie_remove,
};

static int __init hl_pcie_init(void)
{
	int ret;

	BUILD_BUG_ON(HL_PCIE_REG_MSIX_TABLE_ENTRIES !=
		     HL_PCIE_MAX_MSIX_VECTORS);
	/* The struct size is baked into HL_PCIE_IOC_IDENTITY; see hololink_pcie_uapi.h. */
	BUILD_BUG_ON(sizeof(struct hl_pcie_identity_info) != 64);
	/* Likewise HL_PCIE_IOC_MSIX_CTRL. */
	BUILD_BUG_ON(sizeof(struct hl_pcie_msix_ctrl) != 32);
	/* The bootp announcement is a fixed 300-byte RFC 951 payload. */
	BUILD_BUG_ON(sizeof(struct hl_pcie_bootp_pkt) != HL_PCIE_BOOTP_LEN);

	ret = alloc_chrdev_region(&hl_pcie_devt, 0, HL_PCIE_MAX_MINORS,
				  HL_PCIE_DEVICE_NAME);
	if (ret)
		return ret;

	/*
	 * class_create() dropped its owner argument in v6.4 (commit 1aaba11da9aa).
	 * The reference bring-up tree targets older IGX kernels, so keep both.
	 */
#if LINUX_VERSION_CODE >= KERNEL_VERSION(6, 4, 0)
	hl_pcie_class = class_create(HL_PCIE_DEVICE_NAME);
#else
	hl_pcie_class = class_create(THIS_MODULE, HL_PCIE_DEVICE_NAME);
#endif
	if (IS_ERR(hl_pcie_class)) {
		ret = PTR_ERR(hl_pcie_class);
		unregister_chrdev_region(hl_pcie_devt, HL_PCIE_MAX_MINORS);
		return ret;
	}
	/* Applied by device_create() in probe, which cannot run before pci_register_driver(). */
	hl_pcie_class->dev_groups = hl_pcie_dev_groups;

	ret = pci_register_driver(&hl_pcie_driver);
	if (ret) {
		class_destroy(hl_pcie_class);
		unregister_chrdev_region(hl_pcie_devt, HL_PCIE_MAX_MINORS);
		return ret;
	}

	return 0;
}

static void __exit hl_pcie_exit(void)
{
	pci_unregister_driver(&hl_pcie_driver); /* runs hl_pcie_remove() for every bound device */
	/*
	 * Each remove cancelled the publisher and re-armed it for the devices left; two removes
	 * racing can leave one final re-arm queued against an empty array. Cancel it before the
	 * module text goes away.
	 */
	cancel_delayed_work_sync(&hl_pcie_bootp_work);
	class_destroy(hl_pcie_class);
	unregister_chrdev_region(hl_pcie_devt, HL_PCIE_MAX_MINORS);
}

module_init(hl_pcie_init);
module_exit(hl_pcie_exit);

/*
 * dma_buf_* live in the DMA_BUF symbol namespace. The macro took a bare token until v6.13
 * (commit "module: Convert symbol namespace to string literal") and a string literal after,
 * so spell both -- the reference bring-up tree targets older IGX kernels.
 */
#if LINUX_VERSION_CODE >= KERNEL_VERSION(6, 13, 0)
MODULE_IMPORT_NS("DMA_BUF");
#else
MODULE_IMPORT_NS(DMA_BUF);
#endif

MODULE_DESCRIPTION("Hololink PCIe userspace test driver with DMA mmap support");
MODULE_AUTHOR("NVIDIA");
MODULE_LICENSE("GPL");
/*
 * The hololink release this driver ships with. The CMake build passes the repository's
 * VERSION file through kbuild (HL_PCIE_DRV_VERSION, see Makefile); a bare
 * `make -C src/ehsb/kmod` gets the fallback below, which is kept at the same value.
 * Besides /sys/module/hololink_pcie_drv/version, declaring a version makes modpost emit
 * srcversion on every kernel, not only those built with CONFIG_MODULE_SRCVERSION_ALL
 * (the Tegra 6.8 kernels are not), so tooling can match a loaded module to its .ko.
 */
#ifndef HL_PCIE_DRV_VERSION
#define HL_PCIE_DRV_VERSION "2.7.0"
#endif
MODULE_VERSION(HL_PCIE_DRV_VERSION);
