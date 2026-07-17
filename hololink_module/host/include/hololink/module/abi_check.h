/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_ABI_CHECK_H
#define HOLOLINK_MODULE_ABI_CHECK_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Magic value identifying a valid hololink_module_abi_check_t payload.
 * The host rejects modules whose magic does not match. */
#define HOLOLINK_MODULE_ABI_MAGIC ((uint32_t)0x484C4143u)  /* 'HLAC' */

/* Layout fingerprint a module returns to the host before init.
 *
 * The host compares the returned struct against its own value computed
 * locally; mismatched fields reject the module with a diagnostic.
 * sizeof / alignof of every C++ type that crosses the .so boundary
 * is included so toolchain / STL drift is caught before any of those
 * types is constructed.
 *
 * Fields can be appended in future versions; struct_size lets the host
 * detect a smaller-than-expected payload from an older module without
 * reading past the end. */
typedef struct hololink_module_abi_check {
    uint32_t magic;        /* HOLOLINK_MODULE_ABI_MAGIC */
    uint32_t api_version;  /* HOLOLINK_MODULE_API_VERSION */
    uint32_t struct_size;  /* sizeof(hololink_module_abi_check) */

    /* C++ types that cross the boundary. Same toolchain on both
     * sides keeps these consistent; the check rejects mismatches
     * up-front. */
    uint32_t size_of_enumeration_metadata;
    uint32_t align_of_enumeration_metadata;
    uint32_t size_of_std_string;
    uint32_t align_of_std_string;
} hololink_module_abi_check_t;

/* API version bumped whenever the C ABI layout changes incompatibly. */
#define HOLOLINK_MODULE_API_VERSION ((uint32_t)1)

/* Forward declaration of the visibility-export macro from
 * service_locator.h; abi_check.h is allowed to be included on its
 * own. */
#if defined(__GNUC__) || defined(__clang__)
#ifndef HOLOLINK_MODULE_EXPORT
#define HOLOLINK_MODULE_EXPORT __attribute__((visibility("default")))
#endif
#else
#ifndef HOLOLINK_MODULE_EXPORT
#define HOLOLINK_MODULE_EXPORT
#endif
#endif

/* Symbol every module .so exports. The implementation lives in
 * hololink::module_runtime and is absorbed privately into each module. */
HOLOLINK_MODULE_EXPORT hololink_module_abi_check_t hololink_module_get_abi_check(void);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* HOLOLINK_MODULE_ABI_CHECK_H */
