/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 
#include <sys/stat.h>
#include <stdlib.h>
#include <errno.h>
#include <stdio.h>
#include <signal.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <sys/times.h>
#include <stdint.h>
#include <stddef.h>

#include "STM32/stm32_system.h"

char * get_uid(void)
{
    static char uid[12];
    static _Bool uid_set = 0;
    if (uid_set) {
        return uid;
    }
    uint32_t dword[3] = { HAL_GetUIDw0(), HAL_GetUIDw1(), HAL_GetUIDw2() };
    memcpy(uid, dword, sizeof(dword));
    uid_set = 1;
    return uid;
}

// stub for non maskable interrupt
void NMI_Handler(void)
{
    while (1)
    {
    }
}

// stub for hard fault
void HardFault_Handler(void)
{
    while (1)
    {
    }
}

// stub for memory management fault
void MemManage_Handler(void)
{
    while (1)
    {
    }
}

// stub for bus fault
void BusFault_Handler(void)
{
    while (1)
    {
    }
}

// stub for usage fault
void UsageFault_Handler(void)
{
    while (1)
    {
    }
}

// stub for system service call via SWI instruction
void SVC_Handler(void)
{
}

// stub for debug monitor
void DebugMon_Handler(void)
{

}

// stub for pendable request for system service
void PendSV_Handler(void)
{

}

// increment tick for system tick timer
void SysTick_Handler(void)
{
    HAL_IncTick();
}

// SystemClock_Config, MPU_Config, and HAL_MspInit live in the board-specific source (board.c).

// newlib stubs.
// NOTE: for picolibc review output from STM32CubeMx generated code
extern int __io_putchar(int ch) __attribute__((weak));
extern int __io_getchar(void) __attribute__((weak));

char *__env[1] = { 0 };
char **environ = __env;

void initialise_monitor_handles()
{
}

int _getpid(void)
{
    return 1;
}

int _kill(int pid, int sig)
{
    (void)pid;
    (void)sig;
    errno = EINVAL;
    return -1;
}

void _exit (int status)
{
    _kill(status, -1);
    while (1) {}    /* Make sure we hang here */
}

__attribute__((weak)) int _read(int file, char *ptr, int len)
{
    (void)file;
    int DataIdx;

    for (DataIdx = 0; DataIdx < len; DataIdx++)
    {
        *ptr++ = __io_getchar();
    }

    return len;
}

__attribute__((weak)) int _write(int file, char *ptr, int len)
{
    (void)file;
    int DataIdx;

    for (DataIdx = 0; DataIdx < len; DataIdx++)
    {
        __io_putchar(*ptr++);
    }
    return len;
}

int _close(int file)
{
    (void)file;
    return -1;
}


int _fstat(int file, struct stat *st)
{
    (void)file;
    st->st_mode = S_IFCHR;
    return 0;
}

int _isatty(int file)
{
    (void)file;
    return 1;
}

int _lseek(int file, int ptr, int dir)
{
    (void)file;
    (void)ptr;
    (void)dir;
    return 0;
}

int _open(char *path, int flags, ...)
{
    (void)path;
    (void)flags;
    return -1;
}

int _wait(int *status)
{
    (void)status;
    errno = ECHILD;
    return -1;
}

int _unlink(char *name)
{
    (void)name;
    errno = ENOENT;
    return -1;
}

clock_t _times(struct tms *buf)
{
    (void)buf;
    return -1;
}

int _stat(const char *file, struct stat *st)
{
    (void)file;
    st->st_mode = S_IFCHR;
    return 0;
}

int _link(char *old, char *new)
{
    (void)old;
    (void)new;
    errno = EMLINK;
    return -1;
}

int _fork(void)
{
    errno = EAGAIN;
    return -1;
}

int _execve(char *name, char **argv, char **env)
{
    (void)name;
    (void)argv;
    (void)env;
    errno = ENOMEM;
    return -1;
}

// sbrk implementation for heap management in newlib
static uint8_t *__sbrk_heap_end = NULL;

void *_sbrk(ptrdiff_t incr)
{
    // extern symbols defined in the linker script
    extern uint8_t _end;
    extern uint8_t _estack;
    extern uint32_t _Min_Stack_Size;

    
    const uint32_t stack_limit = (uint32_t)&_estack - (uint32_t)&_Min_Stack_Size;
    const uint8_t *max_heap = (uint8_t *)stack_limit;
    uint8_t *prev_heap_end;

    if (NULL == __sbrk_heap_end)
    {
        __sbrk_heap_end = &_end;
    }

    if (__sbrk_heap_end + incr > max_heap)
    {
        errno = ENOMEM;
        return (void *)-1;
    }

    prev_heap_end = __sbrk_heap_end;
    __sbrk_heap_end += incr;

    return (void *)prev_heap_end;
}
