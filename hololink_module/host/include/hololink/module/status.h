/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_STATUS_H
#define HOLOLINK_MODULE_STATUS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t hololink_module_status_t;

#define HOLOLINK_MODULE_OK                  ((hololink_module_status_t)0)
#define HOLOLINK_MODULE_INVALID_PARAMETER   ((hololink_module_status_t)1)
#define HOLOLINK_MODULE_NOT_FOUND           ((hololink_module_status_t)2)
#define HOLOLINK_MODULE_NETWORK_ERROR       ((hololink_module_status_t)3)
#define HOLOLINK_MODULE_TIMEOUT             ((hololink_module_status_t)4)
#define HOLOLINK_MODULE_ABI_MISMATCH        ((hololink_module_status_t)5)
#define HOLOLINK_MODULE_INIT_FAILED  ((hololink_module_status_t)6)

/* Returned by EnumerationInterfaceV1::update_metadata when the module
 * recognizes the device but declines to drive it (e.g. an FPGA whose
 * reported IP version is outside the range the module supports). This
 * is NOT an error: the Adapter suppresses the announcement to the
 * application (post-enrichment subscribers are not notified) but does
 * not throw or abort enumeration of other devices. */
#define HOLOLINK_MODULE_ENUMERATION_SKIPPED ((hololink_module_status_t)7)

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* HOLOLINK_MODULE_STATUS_H */
