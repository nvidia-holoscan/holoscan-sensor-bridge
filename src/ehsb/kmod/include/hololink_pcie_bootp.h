/* SPDX-License-Identifier: GPL-2.0 WITH Linux-syscall-note */
/*
 * BOOTP announcement the hololink_pcie kernel driver publishes for every bound endpoint.
 *
 * The hololink_module Adapter discovers devices only through bootp on UDP port 12267. Ethernet
 * boards broadcast one of these every second; the driver does the same for PCIe endpoints,
 * injecting the datagram into the loopback interface as a broadcast on the loopback subnet
 * (<packed BDF>:12268 -> 127.255.255.255:12267), so the unchanged host listener sees an endpoint
 * exactly as it sees a board. Like an Ethernet board's broadcast it is delivered to EVERY socket
 * bound to port 12267 on INADDR_ANY -- a unicast to 127.0.0.1 would reach only one of the sockets
 * sharing the port -- while a socket bound to 127.0.0.1 itself does not match the broadcast
 * destination and never sees it: raw listeners bind INADDR_ANY.
 *
 * The 300-byte payload follows RFC 951 with the HSB vendor extension the host parsers expect
 * (hololink_module/host/src/bootp.cpp deserialize_bootp_request / deserialize_bootp_v2, and the
 * legacy src/hololink/core/enumerator.cpp). The header says what the carrier is:
 *   htype              HL_PCIE_BOOTP_HTYPE_PCIE (0x50, 'P'; not an IANA ARP hardware type,
 *                      which stop at 37). An Ethernet HSB sends 1. The hsb_pcie module and the
 *                      legacy enumerator select on it; the hololink_module parser reports it.
 *   hlen / chaddr      4 bytes: the PCI function address as (domain BE16, bus, devfn) -- the
 *                      same packing as ciaddr and as the datagram's IP source. The
 *                      hololink_module parser is transport-agnostic: for any htype, mac_id is
 *                      the colon-hex of chaddr ("00:05:01:00") and peer_ip the IPv4 source.
 *                      For htype 0x50 the hsb_pcie module (update_metadata, from the parsed
 *                      hardware_address; the legacy core enumerator does it in the parser)
 *                      renders it as the textual BDF ("0005:01:00.0") into BOTH mac_id (the
 *                      hardware address in its htype's convention) and peer_ip (the channel
 *                      key applications wait on).
 *   ciaddr             the BDF packed as (domain << 16) | (bus << 8) | devfn, rendered dotted-quad
 *                      in client_ip_address ("0.5.1.0"). The datagram's IPv4 source is the same
 *                      packed value, so client_ip_address also stands for the source; older
 *                      parsers derive the BDF from it.
 *   transaction_id     xid: (domain << 16) | requester_id.
 * And the vendor area carries what the Ethernet firmware carries, in the same positions
 * (docs/user_guide/ip_integration.mdx, Table 2):
 *   fpga_uuid          vend 10..25, the board UUID in the driver's byte order (HSB_UUID3 first,
 *                      each register big-endian), which names the module .so.
 *   serial_number      vend 30..36, 7 bytes rendered as 14 hex digits. The PCIe Device Serial
 *                      Number capability (EUI-64, pci_get_dsn()) when the endpoint has one: its
 *                      7 least-significant bytes, LSB first as the Ethernet firmware sends the
 *                      EEPROM serial (Table 2: little endian), dropping the top OUI byte.
 *                      Without a DSN the placeholder 'P','C','I','E', domain_lo, bus, devfn.
 *                      Status bit HL_PCIE_BOOTP_STATUS_SERIAL_DSN says which.
 *   hsb_ip_version     vend 37..38, LE16: HSB_VER & 0xFFFF.   data_plane 0, enum_version 2.
 *   ptp_timestamp_s/ns vend 45..50 BE48 seconds and 51..54 BE32 nanoseconds: a SNAPSHOT of the
 *                      endpoint's PTP clock taken just before the announcement, when the driver
 *                      is loaded with bootp_ptp=1 and the bitfile implements the bar_ptp block;
 *                      otherwise zero. Status bit HL_PCIE_BOOTP_STATUS_PTP_VALID says which.
 *   bootp_psn          vend 55..56 BE16, the announcement sequence number (pkt_count & 0xFFFF).
 *   bootp_status       vend 57: bit0 PTP_VALID (Ethernet: PTP enabled and a SYNC received),
 *                      bit1 SERIAL_DSN (PCIe only; an Ethernet record reads 0 there).
 *
 * Everything here is plain C over __u8 arrays: no packing attribute is needed (no field wider
 * than a byte, so no padding), the layout is endian-neutral, and the same inline filler compiles
 * in the kernel module and in a userspace test that pins the bytes to the host parser.
 */
#ifndef HOLOLINK_PCIE_BOOTP_H
#define HOLOLINK_PCIE_BOOTP_H

#include <linux/types.h>

#define HL_PCIE_BOOTP_SPORT         12268U /* the board-side "reply" port, as the emulator */
#define HL_PCIE_BOOTP_DPORT         12267U /* the host's bootp request port */
#define HL_PCIE_BOOTP_DADDR         0x7FFFFFFFu /* 127.255.255.255, the loopback subnet broadcast */
#define HL_PCIE_BOOTP_LEN           300U
#define HL_PCIE_BOOTP_VEND_TAG_LEN  56U    /* vend[1]: bytes following it, as the Ethernet HSB IP sends */
#define HL_PCIE_BOOTP_VEND_LEN      (HL_PCIE_BOOTP_VEND_TAG_LEN + 2U) /* bytes of vend[] that carry data */
#define HL_PCIE_BOOTP_INTERVAL_MS   1000U
#define HL_PCIE_BOOTP_ENUM_VERSION  2U
#define HL_PCIE_BOOTP_VENDOR_TAG    0xE0U
#define HL_PCIE_BOOTP_HTYPE_PCIE    0x50U  /* 'P': a PCIe HSB / eHSB endpoint; Ethernet boards send 1 */
#define HL_PCIE_BOOTP_HLEN          4U     /* chaddr = the BDF, (domain BE16, bus, devfn) */
#define HL_PCIE_BOOTP_SERIAL_LEN    7U

/* Vendor-area offsets of the fields past the enum_version 2 body (the Ethernet firmware's). */
#define HL_PCIE_BOOTP_VEND_SERIAL   30U
#define HL_PCIE_BOOTP_VEND_PTP_SEC  45U    /* BE48 */
#define HL_PCIE_BOOTP_VEND_PTP_NSEC 51U    /* BE32 */
#define HL_PCIE_BOOTP_VEND_PSN      55U    /* BE16 */
#define HL_PCIE_BOOTP_VEND_STATUS   57U
/* bootp_status bits. */
#define HL_PCIE_BOOTP_STATUS_PTP_VALID  0x01U /* the timestamp is a live snapshot of the endpoint's clock */
#define HL_PCIE_BOOTP_STATUS_SERIAL_DSN 0x02U /* serial[7] is the PCIe Device Serial Number, not the placeholder */

struct hl_pcie_bootp_pkt {
	__u8 op;          /* 1 = BOOTREQUEST */
	__u8 htype;       /* HL_PCIE_BOOTP_HTYPE_PCIE */
	__u8 hlen;        /* HL_PCIE_BOOTP_HLEN */
	__u8 hops;        /* 0 */
	__u8 xid[4];      /* BE: (domain << 16) | requester_id */
	__u8 secs[2];     /* BE: seconds since the endpoint was bound */
	__u8 flags[2];    /* 0 */
	__u8 ciaddr[4];   /* BE: the BDF, (domain << 16) | (bus << 8) | devfn */
	__u8 yiaddr[4];   /* 0 */
	__u8 siaddr[4];   /* 0 */
	__u8 giaddr[4];   /* 0 */
	__u8 chaddr[16];  /* [0..3] = dom_hi, dom_lo, bus, devfn (the same bytes as ciaddr), rest 0 */
	__u8 sname[64];   /* 0 */
	__u8 file[128];   /* 0 */
	/*
	 * vend[]: off 0 tag 0xE0 | 1 len 56 | 2 'N' 'V' 'D' 'A' | 6 data_plane | 7 enum_version |
	 *         8 compat_id LE16 | 10 uuid[16] | 26 pkt_count BE24 | 29 pad | 30 serial[7] |
	 *         37 hsb_ip_version LE16 | 39 fpga_crc LE16 | 41 reserved[4] | 45 ptp_sec BE48 |
	 *         51 ptp_nsec BE32 | 55 psn BE16 | 57 status | 58.. zero
	 * The len byte is the count of bytes following it; the parsers read the tail (41..57) only
	 * when it says at least 56, so an older announcement (39) still parses.
	 */
	__u8 vend[64];
};

/* Fails to compile when the struct is not exactly 300 bytes (both kernel and userspace). */
typedef char hl_pcie_bootp_pkt_is_300_bytes[(sizeof(struct hl_pcie_bootp_pkt) == HL_PCIE_BOOTP_LEN) ? 1 : -1];

/* Everything the filler needs; the driver snapshots it from its per-device state. */
struct hl_pcie_bootp_id {
	__u16 domain;
	__u8 bus;
	__u8 devfn;
	__u8 uuid[16];       /* driver byte order: HSB_UUID3 first, each register big-endian */
	__u16 hsb_ip_version; /* HSB_VER & 0xFFFF */
	__u16 compat_id;      /* 0: the module is built with NO_COMPAT_SUFFIX */
	__u16 fpga_crc;       /* 0 */
	__u8 data_plane;      /* 0 */
	__u32 pkt_count;      /* announcements sent so far (24 bits carried; low 16 as the PSN) */
	__u16 secs;           /* seconds since bind, clamped */
	__u64 dsn;            /* PCIe Device Serial Number (pci_get_dsn()); 0 or all-ones = none */
	__u8 ptp_valid;       /* nonzero: ptp_sec / ptp_nsec hold a snapshot of the endpoint's clock */
	__u64 ptp_sec;        /* 48 bits carried */
	__u32 ptp_nsec;
};

static inline __u32 hl_pcie_bootp_bdf32(__u16 domain, __u8 bus, __u8 devfn)
{
	return ((__u32)domain << 16) | ((__u32)bus << 8) | (__u32)devfn;
}

static inline __u16 hl_pcie_bootp_requester_id(__u8 bus, __u8 devfn)
{
	return ((__u16)bus << 8) | (__u16)devfn;
}

static inline void hl_pcie_bootp_put_be16(__u8 *p, __u16 v) { p[0] = (__u8)(v >> 8); p[1] = (__u8)v; }
static inline void hl_pcie_bootp_put_le16(__u8 *p, __u16 v) { p[0] = (__u8)v; p[1] = (__u8)(v >> 8); }
static inline void hl_pcie_bootp_put_be24(__u8 *p, __u32 v) { p[0] = (__u8)(v >> 16); p[1] = (__u8)(v >> 8); p[2] = (__u8)v; }
static inline void hl_pcie_bootp_put_be32(__u8 *p, __u32 v)
{
	p[0] = (__u8)(v >> 24); p[1] = (__u8)(v >> 16); p[2] = (__u8)(v >> 8); p[3] = (__u8)v;
}
static inline void hl_pcie_bootp_put_be48(__u8 *p, __u64 v)
{
	p[0] = (__u8)(v >> 40); p[1] = (__u8)(v >> 32); p[2] = (__u8)(v >> 24);
	p[3] = (__u8)(v >> 16); p[4] = (__u8)(v >> 8); p[5] = (__u8)v;
}

/* The 4-byte hardware address carried in chaddr for htype 0x50: the BDF, big-endian. */
static inline void hl_pcie_bootp_chaddr(__u8 chaddr[4], __u16 domain, __u8 bus, __u8 devfn)
{
	hl_pcie_bootp_put_be32(chaddr, hl_pcie_bootp_bdf32(domain, bus, devfn));
}

/* A DSN of 0 means the endpoint has no Device Serial Number capability; all-ones is unprogrammed. */
static inline int hl_pcie_bootp_dsn_valid(__u64 dsn)
{
	return dsn != 0 && dsn != ~(__u64)0;
}

/*
 * serial[7] from the 64-bit DSN: the 7 least-significant bytes, LSB first, matching the
 * little-endian order the Ethernet firmware uses for the EEPROM serial (ip_integration.mdx,
 * Table 2, bytes [36:30]). The byte dropped is the most significant one of the OUI, constant
 * for a vendor, so no per-board entropy is lost.
 */
static inline void hl_pcie_bootp_serial_from_dsn(__u8 serial[HL_PCIE_BOOTP_SERIAL_LEN], __u64 dsn)
{
	unsigned int i;

	for (i = 0; i < HL_PCIE_BOOTP_SERIAL_LEN; i++)
		serial[i] = (__u8)(dsn >> (8 * i));
}

/* The placeholder serial for an endpoint without a DSN: "PCIE" + domain low byte, bus, devfn. */
static inline void hl_pcie_bootp_serial_placeholder(__u8 serial[HL_PCIE_BOOTP_SERIAL_LEN],
						    __u16 domain, __u8 bus, __u8 devfn)
{
	serial[0] = 'P'; serial[1] = 'C'; serial[2] = 'I'; serial[3] = 'E';
	serial[4] = (__u8)domain; serial[5] = bus; serial[6] = devfn;
}

static inline void hl_pcie_bootp_fill(struct hl_pcie_bootp_pkt *pkt, const struct hl_pcie_bootp_id *id)
{
	__u8 *b = (__u8 *)pkt;
	__u8 *v = pkt->vend;
	__u8 status = 0;
	unsigned int i;

	for (i = 0; i < HL_PCIE_BOOTP_LEN; i++)
		b[i] = 0;

	pkt->op = 1;
	pkt->htype = HL_PCIE_BOOTP_HTYPE_PCIE;
	pkt->hlen = HL_PCIE_BOOTP_HLEN;
	hl_pcie_bootp_put_be32(pkt->xid,
			       ((__u32)id->domain << 16) | hl_pcie_bootp_requester_id(id->bus, id->devfn));
	hl_pcie_bootp_put_be16(pkt->secs, id->secs);
	hl_pcie_bootp_put_be32(pkt->ciaddr, hl_pcie_bootp_bdf32(id->domain, id->bus, id->devfn));
	hl_pcie_bootp_chaddr(pkt->chaddr, id->domain, id->bus, id->devfn);

	v[0] = HL_PCIE_BOOTP_VENDOR_TAG;
	v[1] = HL_PCIE_BOOTP_VEND_TAG_LEN;
	v[2] = 'N'; v[3] = 'V'; v[4] = 'D'; v[5] = 'A';
	v[6] = id->data_plane;
	v[7] = HL_PCIE_BOOTP_ENUM_VERSION;
	hl_pcie_bootp_put_le16(v + 8, id->compat_id);
	for (i = 0; i < 16; i++)
		v[10 + i] = id->uuid[i];
	hl_pcie_bootp_put_be24(v + 26, id->pkt_count & 0xFFFFFFU);
	v[29] = 0;
	if (hl_pcie_bootp_dsn_valid(id->dsn)) {
		hl_pcie_bootp_serial_from_dsn(v + HL_PCIE_BOOTP_VEND_SERIAL, id->dsn);
		status |= HL_PCIE_BOOTP_STATUS_SERIAL_DSN;
	} else {
		hl_pcie_bootp_serial_placeholder(v + HL_PCIE_BOOTP_VEND_SERIAL, id->domain, id->bus, id->devfn);
	}
	hl_pcie_bootp_put_le16(v + 37, id->hsb_ip_version);
	hl_pcie_bootp_put_le16(v + 39, id->fpga_crc);
	/* 41..44: reserved (the Ethernet firmware's eeprom_misc, tied to 0). */
	if (id->ptp_valid) {
		hl_pcie_bootp_put_be48(v + HL_PCIE_BOOTP_VEND_PTP_SEC, id->ptp_sec & 0xFFFFFFFFFFFFULL);
		hl_pcie_bootp_put_be32(v + HL_PCIE_BOOTP_VEND_PTP_NSEC, id->ptp_nsec);
		status |= HL_PCIE_BOOTP_STATUS_PTP_VALID;
	}
	hl_pcie_bootp_put_be16(v + HL_PCIE_BOOTP_VEND_PSN, (__u16)(id->pkt_count & 0xFFFFU));
	v[HL_PCIE_BOOTP_VEND_STATUS] = status;
}

#endif /* HOLOLINK_PCIE_BOOTP_H */
