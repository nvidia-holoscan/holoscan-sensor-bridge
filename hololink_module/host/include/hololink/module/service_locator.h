/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOLINK_MODULE_SERVICE_LOCATOR_H
#define HOLOLINK_MODULE_SERVICE_LOCATOR_H

#include <stdint.h>

#include "abi_check.h"
#include "status.h"

/* Mark a symbol with default visibility so dlsym() can resolve it
 * across the .so boundary even when the consuming binary is compiled
 * with -fvisibility=hidden (the module project default). */
#if defined(__GNUC__) || defined(__clang__)
#define HOLOLINK_MODULE_EXPORT __attribute__((visibility("default")))
#else
#define HOLOLINK_MODULE_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle to a service instance held by the publisher.
 * The host treats it as a token to feed back into release_service. */
typedef const void* hololink_module_service_t;

/* Look up a service by (instance_id, type_id). On a registry hit the
 * cached instance is returned; on a miss the peer invokes a
 * type_id-keyed default constructor (if registered) to produce a
 * fresh, cheap-constructed instance, publishes it under instance_id,
 * and returns it. Materialization of resources is the caller's
 * responsibility via ConfigurableService::configure(metadata) on the
 * returned handle. Returns NULL when neither path yields an entry. */
typedef hololink_module_service_t (*hololink_module_get_service)(
    const char* instance_id, const char* type_id);

/* Release one reference the host obtained from get_service. Each
 * get_service must be balanced by exactly one release_service.
 * Required; may never be NULL. */
typedef void (*hololink_module_release_service)(
    hololink_module_service_t instance);

/* Host -> module init payload. */
typedef struct hololink_module_init {
    uint32_t api_version;
    uint32_t reserved_;
    hololink_module_get_service get_service;          /* host -> module */
    hololink_module_release_service release_service;  /* host -> module */
} hololink_module_init_t;

/* Module -> host return from hololink_module_init. */
typedef struct hololink_module_services {
    hololink_module_status_t status;                  /* HOLOLINK_MODULE_OK or an error */
    hololink_module_get_service get_service;
    hololink_module_release_service release_service;
} hololink_module_services_t;

/* Symbol every module .so exports. */
HOLOLINK_MODULE_EXPORT hololink_module_services_t
hololink_module_init(const hololink_module_init_t* init);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* HOLOLINK_MODULE_SERVICE_LOCATOR_H */
