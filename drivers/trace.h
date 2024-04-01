/**
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

#include <linux/ioctl.h>

static int trace = 0;
module_param(trace, int, 0644);
MODULE_PARM_DESC(trace, "Set to 1 to enable trace-level messages");

static int debug = 1;
module_param(debug, int, 0644);
MODULE_PARM_DESC(debug, "Set to 0 to disable debug-level messages");

//
static inline char const * last_slash(char const * p)
{
    char const * r = p;
    while (true) {
        char q = *p;
        if (q == 0) {
            return r;
        }
        if (q == '/') {
            r = p+1;
        }
        p++;
    }
}
#define TRACE(fmt, ...) while (trace) { printk(KERN_INFO "TRACE %s:%d %s -- " fmt "\n", last_slash(__FILE__), __LINE__,__FUNCTION__, ##__VA_ARGS__); break; }
#define DEBUG(fmt, ...) while (debug) { printk(KERN_INFO "DEBUG %s:%d %s -- " fmt "\n", last_slash(__FILE__), __LINE__,__FUNCTION__, ##__VA_ARGS__); break; }
#define WARNING(fmt, ...) printk(KERN_WARNING "WARNING %s:%d %s -- " fmt "\n", last_slash(__FILE__), __LINE__,__FUNCTION__, ##__VA_ARGS__)

