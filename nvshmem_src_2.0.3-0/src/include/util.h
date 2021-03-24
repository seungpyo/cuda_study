/****
 * Copyright (c) 2016-2020, NVIDIA Corporation.  All rights reserved.
 *
 * Copyright 2011 Sandia Corporation. Under the terms of Contract
 * DE-AC04-94AL85000 with Sandia Corporation, the U.S.  Government
 * retains certain rights in this software.
 *
 * Copyright (c) 2017 Intel Corporation. All rights reserved.
 * This software is available to you under the BSD license.
 *
 * Portions of this file are derived from Sandia OpenSHMEM.
 *
 * See COPYRIGHT for license information
 ****/

#ifndef _UTIL_H
#define _UTIL_H

#include <stdio.h>
#include <stdint.h>
#include <cassert>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string.h>
#include <errno.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdbool.h>
#include <stddef.h>
#include <sstream>
#include <vector>
#include <inttypes.h>
#include "error_codes_internal.h"
#include "debug.h"
#include "nvshmem_internal.h"

#define likely(x)       __builtin_expect((x),1)
#define unlikely(x)     __builtin_expect((x),0)

#define ERROR_EXIT(...)                                                  \
    do {                                                                 \
        fprintf(stderr, "%s:%s:%d: ", __FILE__, __FUNCTION__, __LINE__); \
        fprintf(stderr, __VA_ARGS__);                                    \
        exit(-1);                                                        \
    } while (0)

#define ERROR_PRINT(...)                                                 \
    do {                                                                 \
        fprintf(stderr, "%s:%s:%d: ", __FILE__, __FUNCTION__, __LINE__); \
        fprintf(stderr, __VA_ARGS__);                                    \
    } while (0)

#define WARN_PRINT(...)               \
    do {                              \
        fprintf(stdout, "WARN: ");    \
        fprintf(stdout, __VA_ARGS__); \
    } while (0)

#define NULL_JMP(var, status, err, label)		               \
    do {                                                               \
        if (var == NULL) {                                             \
            status = err;                                              \
            goto label;                                                \
        }                                                              \
    } while (0)

#define ERROR_JMP(status, err, label, ...)                                          \
    do {                                                                            \
        fprintf(stderr, "%s:%d: non-zero status: %d ", __FILE__, __LINE__, err);    \
        fprintf(stderr, __VA_ARGS__);                                               \
        status = err;                                                               \
        goto label;                                                                 \
    } while (0)

#define NULL_ERROR_JMP(var, status, err, label, ...)                   \
    do {                                                               \
        if (unlikely(var == NULL)) {                                             \
            fprintf(stderr, "%s:%d: NULL value ", __FILE__, __LINE__); \
            fprintf(stderr, __VA_ARGS__);                              \
            status = err;                                              \
            goto label;                                                \
        }                                                              \
    } while (0)

#define EQ_ERROR_JMP(status, expected, err, label, ...)                              \
    do {                                                                             \
        if (unlikely(status == expected)) {                                                    \
            fprintf(stderr, "%s:%d: error status: %d ", __FILE__, __LINE__, status); \
            fprintf(stderr, __VA_ARGS__);                                            \
            status = err;                                                            \
            goto label;                                                              \
        }                                                                            \
    } while (0)

#define NE_ERROR_JMP(status, expected, err, label, ...)                                 \
    do {                                                                                \
        if (unlikely(status != expected)) {                                                       \
            fprintf(stderr, "%s:%d: non-zero status: %d ", __FILE__, __LINE__, status); \
            fprintf(stderr, __VA_ARGS__);                                               \
            status = err;                                                               \
            goto label;                                                                 \
        }                                                                               \
    } while (0)

#define NZ_DEBUG_JMP(status, err, label, ...)                                               \
    do {                                                                                    \
        if (unlikely(status != 0)) {                                                                  \
            if (nvshmem_debug_level >= NVSHMEM_LOG_TRACE) {                                 \
                fprintf(stderr, "%s:%d: non-zero status: %d ", __FILE__, __LINE__, status); \
                fprintf(stderr, __VA_ARGS__);                                               \
            }                                                                               \
            status = err;                                                                   \
            goto label;                                                                     \
        }                                                                                   \
    } while (0)

#define NZ_ERROR_JMP(status, err, label, ...)                                           \
    do {                                                                                \
        if (unlikely(status != 0)) {                                                              \
            fprintf(stderr, "%s:%d: non-zero status: %d ", __FILE__, __LINE__, status); \
            fprintf(stderr, __VA_ARGS__);                                               \
            status = err;                                                               \
            goto label;                                                                 \
        }                                                                               \
    } while (0)

#define NZ_JMP(status, label, ...)                                                      \
    do {                                                                                \
        if (unlikely(status != 0)) {                                                              \
            fprintf(stderr, "%s:%d: non-zero status: %d ", __FILE__, __LINE__, status); \
            fprintf(stderr, __VA_ARGS__);                                               \
            goto label;                                                                 \
        }                                                                               \
    } while (0)

#define NZ_EXIT(status, ...)                                                                   \
    do {                                                                                       \
        if (unlikely(status != 0)) {                                                                     \
            fprintf(stderr, "%s:%d: non-zero status: %d: %s, exiting... ", __FILE__, __LINE__, \
                    status, strerror(errno));                                                  \
            fprintf(stderr, __VA_ARGS__);                                                      \
            exit(-1);                                                                          \
        }                                                                                      \
    } while (0)

#define NULL_EXIT(ptr, ...)                                                                       \
    do {                                                                                          \
        if (unlikely(!ptr)) {                                                                               \
            fprintf(stderr, "%s:%d: null ptr, error string: %s, exiting... ", __FILE__, __LINE__, \
                    strerror(errno));                                                             \
            fprintf(stderr, __VA_ARGS__);                                                         \
            exit(-1);                                                                             \
        }                                                                                         \
    } while (0)

#define NE_EXIT(status, expected, ...)                                                      \
    do {                                                                                    \
        if (unlikely(status != expected)) {                                                           \
            fprintf(stderr, "%s:%d: error status: %d: %s, exiting... ", __FILE__, __LINE__, \
                    status, strerror(errno));                                               \
            fprintf(stderr, __VA_ARGS__);                                                   \
            exit(-1);                                                                       \
        }                                                                                   \
    } while (0)

#define CUDA_DRIVER_CHECK(cmd)                    \
    do {                                          \
        CUresult r = cmd;                         \
        cuGetErrorString(r, &p_err_str);          \
        if (unlikely(CUDA_SUCCESS != r)) {                  \
            WARN("Cuda failure '%s'", p_err_str); \
            return NVSHMEMI_UNHANDLED_CUDA_ERROR; \
        }                                         \
    } while (false)

#define CUDA_CHECK(stmt)                                                                      \
    do {                                                                                      \
        CUresult result = (stmt);                                                             \
        cuGetErrorString(result, &p_err_str);                                                 \
        if (unlikely(CUDA_SUCCESS != result)) {                                                         \
            fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, p_err_str); \
            exit(-1);                                                                         \
        }                                                                                     \
        assert(CUDA_SUCCESS == result);                                                       \
    } while (0)

#define CUDA_RUNTIME_CHECK(stmt)                                                  \
    do {                                                                          \
        cudaError_t result = (stmt);                                              \
        if (unlikely(cudaSuccess != result)) {                                              \
            fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, \
                    cudaGetErrorString(result));                                  \
            exit(-1);                                                             \
        }                                                                         \
        assert(cudaSuccess == result);                                            \
    } while (0)

#define NCCL_CHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,nccl_ftable.GetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define CUDA_RUNTIME_ERROR_STRING(result)                                         \
    do {                                                                          \
        if (unlikely(cudaSuccess != result)) {                                              \
            fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, \
                    cudaGetErrorString(result));                                  \
        }                                                                         \
    } while (0)

#define CUDA_DRIVER_ERROR_STRING(result)                                                      \
    do {                                                                                      \
        if (unlikely(CUDA_SUCCESS != result)) {                                                         \
            cuGetErrorString(result, &p_err_str);                                             \
            fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__, p_err_str); \
        }                                                                                     \
    } while (0)

#define NVSHMEMU_THREAD_CS_INIT nvshmemu_thread_cs_init
#define NVSHMEMU_THREAD_CS_ENTER nvshmemu_thread_cs_enter
#define NVSHMEMU_THREAD_CS_EXIT nvshmemu_thread_cs_exit
#define NVSHMEMU_THREAD_CS_FINALIZE nvshmemu_thread_cs_finalize

#define NVSHMEMU_MAPPED_PTR_TRANSLATE(toPtr, fromPtr, peer)          \
    toPtr = (void *)((char *)(nvshmemi_state->peer_heap_base[peer]) + \
                     ((char *)fromPtr - (char *)(nvshmemi_state->heap_base)));

#define NVSHMEMU_UNMAPPED_PTR_TRANSLATE(toPtr, fromPtr, peer)               \
    toPtr = (void *)((char *)(nvshmemi_state->peer_heap_base_actual[peer]) + \
                     ((char *)fromPtr - (char *)(nvshmemi_state->heap_base)));

void nvshmemu_thread_cs_init();
void nvshmemu_thread_cs_finalize();
void nvshmemu_thread_cs_enter();
void nvshmemu_thread_cs_exit();

int nvshmemu_get_num_gpus_per_node();
int cuCheck(CUresult res);
int cudaCheck(cudaError_t res);

uint64_t getHostHash();
void setup_sig_handler();
char * nvshmemu_hexdump(void *ptr, size_t len);

extern const char *p_err_str;

typedef int nvshmemi_env_int;
typedef long nvshmemi_env_long;
typedef size_t nvshmemi_env_size;
typedef bool nvshmemi_env_bool;
typedef const char* nvshmemi_env_string;


#define NVSHFMT_int(_v)    _v
#define NVSHFMT_long(_v)   _v
#define NVSHFMT_size(_v)   _v
#define NVSHFMT_bool(_v)  (_v) ? "true" : "false"
#define NVSHFMT_string(_v) _v

enum nvshmemi_env_categories {
    NVSHMEMI_ENV_CAT_OPENSHMEM, NVSHMEMI_ENV_CAT_OTHER,
    NVSHMEMI_ENV_CAT_COLLECTIVES, NVSHMEMI_ENV_CAT_TRANSPORT,
    NVSHMEMI_ENV_CAT_HIDDEN
};


struct nvshmemi_options_s {
#define NVSHMEMI_ENV_DEF(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC) \
  nvshmemi_env_##KIND NAME;
#include "env_defs.h"
#undef NVSHMEMI_ENV_DEF

#define NVSHMEMI_ENV_DEF(NAME, KIND, DEFAULT, CATEGORY, SHORT_DESC) \
  bool NAME##_provided;
#include "env_defs.h"
#undef NVSHMEMI_ENV_DEF
};


extern struct nvshmemi_options_s nvshmemi_options;

int  nvshmemi_options_init(void);
void nvshmemi_options_print(void);

#endif
