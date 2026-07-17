/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Test fixture: a module .so that returns a deliberately-wrong magic
 * from hololink_module_get_abi_check so the host rejects the load
 * before hololink_module_init is ever called. This file does NOT
 * link hololink::module_runtime — it provides its own abi_check + init
 * symbols so the override actually overrides.
 */

#include "hololink/module/abi_check.h"
#include "hololink/module/service_locator.h"

extern "C" hololink_module_abi_check_t hololink_module_get_abi_check(void)
{
    hololink_module_abi_check_t check;
    check.magic = 0xDEADBEEF; // not HOLOLINK_MODULE_ABI_MAGIC
    check.api_version = HOLOLINK_MODULE_API_VERSION;
    check.struct_size = sizeof(hololink_module_abi_check_t);
    return check;
}

extern "C" hololink_module_services_t
hololink_module_init(const hololink_module_init_t* /*init*/)
{
    // Should never be called — the host rejects the load on the bad
    // magic above. Returning an error keeps the symbol non-NULL so
    // dlsym() succeeds during the load attempt.
    hololink_module_services_t result {};
    result.status = HOLOLINK_MODULE_INIT_FAILED;
    return result;
}
