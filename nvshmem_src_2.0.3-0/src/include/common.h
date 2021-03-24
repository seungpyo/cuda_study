/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef __COMMON_H
#define __COMMON_H

#include <stddef.h>
#include <stdint.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include "nvshmemi_constants.h"

#define CHECK_RMA_PRESENT 1
#define CHECK_RMA_AMO_PRESENT 1

#define NVSHMEM_MEM_HANDLE_SIZE 64
#define NVSHMEM_EP_HANDLE_SIZE 128

#define MAX_P2P_ACCESSIBLE_GPUS 128

/*TODO:generates warning for P2P transport when arguments of this type are specified but not used*/
typedef void *nvshmemt_ep_t;

typedef struct nvshmem_mem_handle {
    char reserved[NVSHMEM_MEM_HANDLE_SIZE];
} nvshmem_mem_handle_t;

typedef struct {
    char reserved[NVSHMEM_EP_HANDLE_SIZE];
} nvshmemt_ep_handle_t;

enum {
    NO_NBI = 0,
    NBI,
};

enum {
    NO_ASYNC = 0,
    ASYNC,
};

enum {
    SRC_STRIDE_CONTIG = 1,
};

enum {
    DEST_STRIDE_CONTIG = 1,
};

enum {
    UINT = 0,
    ULONG,
    ULONGLONG,
    INT32,
    INT64,
    UINT32,
    UINT64,
    INT,
    LONG,
    LONGLONG,
    SIZE,
    PTRDIFF,
    FLOAT,
    DOUBLE
};

#define NOT_A_CUDA_STREAM ((cudaStream_t)0)

typedef struct rma_verb {
    nvshmemi_op_t desc;
    int is_nbi;
    int is_stream;
    cudaStream_t cstrm;
} rma_verb_t;

typedef struct rma_memdesc {
    void *ptr;
    nvshmem_mem_handle_t handle;
} rma_memdesc_t;

typedef struct rma_bytesdesc {
    size_t nelems;
    int elembytes;
    ptrdiff_t srcstride;
    ptrdiff_t deststride;
} rma_bytesdesc_t;

typedef enum amo_desc {
    INC,
    SET,
    ADD,
    AND,
    OR,
    XOR,
    FETCH,
    FETCH_INC,
    FETCH_ADD,
    FETCH_AND,
    FETCH_OR,
    FETCH_XOR,
    SWAP,
    COMPARE_SWAP,
    SIGNAL,
} amo_desc_t;

typedef struct amo_verb {
    nvshmemi_amo_t desc;
    int is_fetch;
    int is_val;
    int is_cmp;
} amo_verb_t;

typedef struct amo_memdesc {
    void *ptr;
    uint64_t retflag;
    void *retptr;
    void *valptr;
    void *cmpptr;
    uint64_t val;
    uint64_t cmp;
    nvshmem_mem_handle_t handle;
} amo_memdesc_t;

typedef struct amo_bytesdesc {
    int name_type;
    int elembytes;
} amo_bytesdesc_t;

typedef int (*rma_handle)(nvshmemt_ep_t tep, rma_verb_t verb, rma_memdesc_t dest, rma_memdesc_t src,
                          rma_bytesdesc_t bytesdesc);
typedef int (*amo_handle)(nvshmemt_ep_t, void *curetptr, amo_verb_t verb, amo_memdesc_t target,
                          amo_bytesdesc_t bytesdesc);
typedef int (*fence_handle)(nvshmemt_ep_t tep);
typedef int (*quiet_handle)(nvshmemt_ep_t tep);
typedef int (*wait_until_handle)(volatile unsigned int *ptr, unsigned int value, int cond,
                                 int flush);

#endif
