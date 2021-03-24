/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * See COPYRIGHT.txt for license information
 */

#ifndef _DEBUG_H_
#define _DEBUG_H_

#include <pthread.h>
#include "util_debug.h"

extern int nvshmem_debug_level;
extern uint64_t nvshmem_debug_mask;
extern pthread_mutex_t nvshmem_debug_output_lock;
extern FILE *nvshmem_debug_file;

typedef enum {
    NVSHMEM_LOG_NONE = 0,
    NVSHMEM_LOG_VERSION = 1,
    NVSHMEM_LOG_WARN = 2,
    NVSHMEM_LOG_INFO = 3,
    NVSHMEM_LOG_ABORT = 4,
    NVSHMEM_LOG_TRACE = 5
} nvshmem_debug_log_level;
typedef enum {
    NVSHMEM_INIT = 1,
    NVSHMEM_COLL = 2,
    NVSHMEM_P2P = 4,
    NVSHMEM_PROXY = 8,
    NVSHMEM_TRANSPORT = 16,
    NVSHMEM_MEM = 32,
    NVSHMEM_BOOTSTRAP = 64,
    NVSHMEM_TOPO = 128,
    NVSHMEM_UTIL = 256,
    NVSHMEM_TEAM = 512,
    NVSHMEM_ALL = ~0
} nvshmem_debug_log_sub_sys;

static void nvshmem_debug_log(nvshmem_debug_log_level level, unsigned long flags,
                              const char *filefunc, int line, const char *fmt, ...);

#define WARN(...) nvshmem_debug_log(NVSHMEM_LOG_WARN, NVSHMEM_ALL, __FILE__, __LINE__, __VA_ARGS__)
#define INFO(FLAGS, ...) \
    nvshmem_debug_log(NVSHMEM_LOG_INFO, (FLAGS), __func__, __LINE__, __VA_ARGS__)

#ifdef ENABLE_TRACE
#include <chrono>
#define TRACE(FLAGS, ...) \
    nvshmem_debug_log(NVSHMEM_LOG_TRACE, (FLAGS), __func__, __LINE__, __VA_ARGS__)
extern std::chrono::high_resolution_clock::time_point nvshmem_epoch;
#else
#define TRACE(...)
#endif

#include <cctype>
static int strcmp_case_insensitive(const char *a, const char *b) {
    int ca, cb;
    do {
        ca = (unsigned char)*a++;
        cb = (unsigned char)*b++;
        ca = tolower(toupper(ca));
        cb = tolower(toupper(cb));
    } while (ca == cb && ca != '\0');
    return ca - cb;
}

#include <cstring>
#include <cstdio>
#include <sys/types.h>
#include <unistd.h>
#include <limits.h>

void init_debug();

#include <sys/syscall.h>
#define gettid() (pid_t) syscall(SYS_gettid)

#include <cstdarg>

static void nvshmem_debug_log(nvshmem_debug_log_level level, unsigned long flags,
                              const char *filefunc, int line, const char *fmt, ...) {
    if (nvshmem_debug_level <= NVSHMEM_LOG_NONE) {
        return;
    }

    char hostname[1024];
    getHostName(hostname, 1024);
    int cudaDev = -1;
    cudaGetDevice(&cudaDev);

    char buffer[1024];
    size_t len = 0;
    pthread_mutex_lock(&nvshmem_debug_output_lock);
    if (level == NVSHMEM_LOG_WARN && nvshmem_debug_level >= NVSHMEM_LOG_WARN)
        len = snprintf(buffer, sizeof(buffer), "\n%s:%d:%d [%d] %s:%d NVSHMEM WARN ", hostname,
                       getpid(), gettid(), cudaDev, filefunc, line);
    else if (level == NVSHMEM_LOG_INFO && nvshmem_debug_level >= NVSHMEM_LOG_INFO &&
             (flags & nvshmem_debug_mask))
        len = snprintf(buffer, sizeof(buffer), "%s:%d:%d [%d] NVSHMEM INFO ", hostname, getpid(),
                       gettid(), cudaDev);
#ifdef ENABLE_TRACE
    else if (level == NVSHMEM_LOG_TRACE && nvshmem_debug_level >= NVSHMEM_LOG_TRACE &&
             (flags & nvshmem_debug_mask)) {
        auto delta = std::chrono::high_resolution_clock::now() - nvshmem_epoch;
        double timestamp =
            std::chrono::duration_cast<std::chrono::duration<double>>(delta).count() * 1000;
        len = snprintf(buffer, sizeof(buffer), "%s:%d:%d [%d] %f %s:%d NVSHMEM TRACE ", hostname,
                       getpid(), gettid(), cudaDev, timestamp, filefunc, line);
    }
#endif
    if (len) {
        va_list vargs;
        va_start(vargs, fmt);
        (void)vsnprintf(buffer + len, sizeof(buffer) - len, fmt, vargs);
        va_end(vargs);
        fprintf(nvshmem_debug_file, "%s\n", buffer);
        fflush(nvshmem_debug_file);
    }
    pthread_mutex_unlock(&nvshmem_debug_output_lock);

    // If nvshmem_debug_level == NVSHMEM_LOG_ABORT then WARN() will also call abort()
    if (level == NVSHMEM_LOG_WARN && nvshmem_debug_level == NVSHMEM_LOG_ABORT) {
        fprintf(stderr, "\n%s:%d:%d [%d] %s:%d NVSHMEM ABORT\n", hostname, getpid(), gettid(),
                cudaDev, filefunc, line);
        abort();
    }
}

#endif
