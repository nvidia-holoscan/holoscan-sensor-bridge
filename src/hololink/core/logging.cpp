/**
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 *
 * See README.md for detailed information.
 */

#include "logging_internal.hpp"

#include <arpa/inet.h>
#include <ctype.h>
#include <net/if.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <sys/syscall.h>
#define gettid() syscall(SYS_gettid)

#include <fmt/core.h>

#define CONSOLE_LOG
#undef SOCKET_LOG

namespace hololink::logging {

HsbLogLevel hsb_log_level = HSB_LOG_LEVEL_INVALID;
static HsbLogLevel hsb_log_level_default = HSB_LOG_LEVEL_INFO;
// Allow "HOLOSCAN_LOG_LEVEL" to set our logging level (along with HSDK applications)
static const char* log_level_environment_variable = "HOLOSCAN_LOG_LEVEL";
#ifdef SOCKET_LOG
// Socket to send our data to.
static int logger_socket = -1;
#endif /* SOCKET_LOG */

static void _hsb_logger(char const* file, unsigned line, const char* function, HsbLogLevel level, const char* message)
{
    // Don't include all the directory information included in "file"
    const char* basename = strrchr(file, '/');
    if (basename) {
        basename++; // skip the '/'
    } else {
        basename = file;
    }
    const char* level_description = "INVALID";
    switch (level) {
    case HSB_LOG_LEVEL_TRACE:
        level_description = "TRACE";
        break;
    case HSB_LOG_LEVEL_DEBUG:
        level_description = "DEBUG";
        break;
    case HSB_LOG_LEVEL_INFO:
        level_description = "INFO";
        break;
    case HSB_LOG_LEVEL_WARN:
        level_description = "WARN";
        break;
    case HSB_LOG_LEVEL_ERROR:
        level_description = "ERROR";
        break;
    default:
        break;
    }
    pid_t thread_id = gettid();
    std::string msg = fmt::format("{} {}:{} {} tid={:#x} -- {}", level_description, basename, line, function, thread_id, message);

#ifdef CONSOLE_LOG
    fprintf(stderr, "%s\n", msg.c_str());
#endif /* CONSOLE_LOG */

#ifdef SOCKET_LOG
    int flags = 0;
    ssize_t r = send(logger_socket, msg.data(), msg.size(), flags);
    if (r <= 0) {
        throw std::runtime_error("hsb_log send failed");
    }
#endif /* SOCKET_LOG */
}

#ifdef SOCKET_LOG
static int create_logger_socket(const char* sender_ip, const char* destination_ip = "255.255.255.255")
{
    // Set up the socket.
    int s = socket(AF_INET, SOCK_DGRAM, 0);
    if (s == -1) {
        throw std::runtime_error(fmt::format("create_logger_socket failed to create socket, errno={}({})", errno, strerror(errno)));
    }
    // Allow us to send broadcast
    int enable = 1;
    int r = setsockopt(s, SOL_SOCKET, SO_BROADCAST, &enable, sizeof(enable));
    if (r == -1) {
        throw std::runtime_error(fmt::format("create_logger_socket, setsockopt failed, errno={}({})", errno, strerror(errno)));
    }
    // Send from this local address
    struct sockaddr_in address = {
        .sin_family = AF_INET,
    };
    r = inet_pton(AF_INET, sender_ip, (void*)&address.sin_addr);
    if (r != 1) {
        throw std::runtime_error(fmt::format("create_logger_socket, inet_pton({}) failed, errno={}({})", sender_ip, errno, strerror(errno)));
    }
    r = bind(s, (struct sockaddr*)&address, sizeof(address));
    if (r == -1) {
        throw std::runtime_error(fmt::format("create_logger_socket, bind failed, errno={}({})", errno, strerror(errno)));
    }
    // Use syslog's destination udp port to make wireshark show us the content
    uint16_t port = 514;
    address = {
        .sin_family = AF_INET,
        .sin_port = htons(port),
    };
    r = inet_pton(AF_INET, destination_ip, (void*)&address.sin_addr);
    if (r != 1) {
        throw std::runtime_error(fmt::format("create_logger_socket, inet_pton({}) failed, errno={}({})", destination_ip, errno, strerror(errno)));
    }
    r = connect(s, (struct sockaddr*)&address, sizeof(address));
    if (r == -1) {
        throw std::runtime_error(fmt::format("Failed to connect logger socket, errno={}({})", errno, strerror(errno)));
    }

    return s;
}
#endif /* SOCKET_LOG */

static void _initial_hsb_log(char const* file, unsigned line, const char* function, HsbLogLevel level, const char* msg)
{
    // If the application hasn't already rewritten this value, set it
    // to a reasonable default.
    if (hsb_log_level == HSB_LOG_LEVEL_INVALID) {
        hsb_log_level = hsb_log_level_default;
    }

    // Allow the environment to override that.
    char const* env_log_level = getenv(log_level_environment_variable);
    if (env_log_level) {
        if (strcasecmp(env_log_level, "TRACE") == 0) {
            hsb_log_level = HSB_LOG_LEVEL_TRACE;
        } else if (strcasecmp(env_log_level, "DEBUG") == 0) {
            hsb_log_level = HSB_LOG_LEVEL_DEBUG;
        } else if (strcasecmp(env_log_level, "INFO") == 0) {
            hsb_log_level = HSB_LOG_LEVEL_INFO;
        } else if (strcasecmp(env_log_level, "WARN") == 0) {
            hsb_log_level = HSB_LOG_LEVEL_WARN;
        } else if (strcasecmp(env_log_level, "ERROR") == 0) {
            hsb_log_level = HSB_LOG_LEVEL_ERROR;
        } else {
            throw std::runtime_error(fmt::format("Invalid environment setting in \"{}\".", log_level_environment_variable));
        }
    }

#ifdef SOCKET_LOG
    // Note that this requires that something listens to this;
    // so in another terminal, run 'sudo nc -lkup 514'
    logger_socket = create_logger_socket("127.0.0.1", "127.0.0.1");
#endif /* SOCKET_LOG */

    // We only need to be called once.
    hsb_logger = _hsb_logger;

    // Does this specific logger call still apply?
    if (level < hsb_log_level) {
        return;
    }

    // Then show it.
    hsb_logger(file, line, function, level, msg);
}

HsbLogger hsb_logger = _initial_hsb_log;

} // namespace hololink::logging
