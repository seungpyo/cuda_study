/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef _INTERNAL_H
#define _INTERNAL_H

#include <cuda.h>
#include <cuda_runtime.h>
#include "bootstrap.h"
#include "transport.h"
#include "util.h"
#include "common.h"
#include "nvshmem_common.cuh"
#ifdef NVSHMEM_USE_NCCL
#include "nccl.h"
#endif /* NVSHMEM_USE_NCCL */

#define MAX_PEER_STREAMS 3

#define MAX_TRANSPORT_EP_COUNT 1

#define SYMMETRIC_SIZE_DEFAULT 1024 * 1024 * 1024

#define MAXPATHSIZE 1024
#define MAX_BUSID_SIZE 16

#define NUM_G_BUF_ELEMENTS 1024 * 1024

#define NVSHMEM_CHECK_STATE_AND_INIT()                                               \
    do {                                                                             \
        if (!nvshmemi_state) ERROR_EXIT("nvshmem API called before nvshmem_init \n"); \
        if (!nvshmemi_state->initialized) {                                           \
            if (nvshmemi_common_init(nvshmemi_state)) {                               \
                ERROR_EXIT("nvshmem initialization failed, exiting \n");             \
            }                                                                        \
        }                                                                            \
    } while (0)

typedef struct {
    int multi_processor_count;
    int cooperative_launch;
} cuda_device_attributes_t;

typedef struct {
    CUstream stream;
    CUevent begin_event;
    CUevent end_event;
} collective_launch_params_t;

typedef struct nvshmemi_state_dec {
    /*PE state*/
    int mype;
    int npes;
    int mype_node;
    int npes_node;
    /*environment info*/
    char *prefix;
    /*device state*/
    CUdevice cudevice;
    CUcontext cucontext;
    /*symmetric heap state*/
    size_t heap_size;
    int initialized;
    void *heap_base;
    void **peer_heap_base_actual;
    void **peer_heap_base;
    void *heap_mspace;
    struct nvshmem_mem_handle *handles;
    bootstrap_handle_t boot_handle;
    /*transport info*/
    int transport_count;
    int transport_bitmap;
    int *transport_map;
    struct nvshmem_transport_pe_info *pe_info;
    struct nvshmem_transport **transports;
    int *selected_transport_for_rma;
    int *selected_transport_for_amo;
    /*consolidated rma ops*/
    rma_handle *rma;
    amo_handle *amo;
    fence_handle *fence;
    quiet_handle *quiet;
    /*scratch space*/
    char *scratch_space;
    size_t scratch_size;
    int *scratch;
    cuda_device_attributes_t cu_dev_attrib;
    int *p2p_attrib_native_atomic_support;
    collective_launch_params_t claunch_params;
    CUstream my_stream;
    // proxy
    void *proxy;
    CUstream *custreams;
    CUevent *cuevents;
    CUdeviceptr *curets;
} nvshmemi_state_t;

typedef struct {
    volatile uint64_t data;
    volatile uint64_t flag;
} g_elem_t;

typedef struct {
    int error_checks;
} nvshmem_options_t;

extern nvshmemi_state_t *nvshmemi_state;
extern nvshmem_options_t nvshmem_options;
extern __device__ unsigned long long test_wait_any_start_idx_d;
extern int nvshmemi_job_connectivity;
extern int nvshmemi_use_nccl;
#ifdef NVSHMEM_USE_NCCL
/* Reduction operation types */
#define NCCL_REDOP_sum ncclSum
#define NCCL_REDOP_prod ncclProd
#define NCCL_REDOP_min ncclMin
#define NCCL_REDOP_max ncclMax
#define NCCL_REDOP_and -1
#define NCCL_REDOP_or -1
#define NCCL_REDOP_xor -1

/* Reduction datatypes */
#define NCCL_DT_char        ncclChar
#define NCCL_DT_schar       -1
#define NCCL_DT_short       -1
#define NCCL_DT_int         ncclInt
#define NCCL_DT_long        ncclInt64
#define NCCL_DT_longlong    ncclInt64
#define NCCL_DT_ptrdiff     ncclUint64
#define NCCL_DT_uchar       ncclUint8
#define NCCL_DT_ushort      -1
#define NCCL_DT_uint        ncclUint32
#define NCCL_DT_ulong       ncclUint64
#define NCCL_DT_ulonglong   ncclUint64
#define NCCL_DT_int8        ncclInt8
#define NCCL_DT_int16       -1
#define NCCL_DT_int32       ncclInt
#define NCCL_DT_int64       ncclInt64
#define NCCL_DT_uint8       ncclUint8
#define NCCL_DT_uint16       -1
#define NCCL_DT_uint32      ncclUint32
#define NCCL_DT_uint64      ncclUint64
#define NCCL_DT_size        ncclUint64
#define NCCL_DT_float       ncclFloat
#define NCCL_DT_double      ncclDouble
#define NCCL_DT_longdouble  -1
#define NCCL_DT_complexd    -1
#define NCCL_DT_complexf    -1

#else /* NVSHMEM_USE_NCCL */

/* Reduction operation types */
#define NCCL_REDOP_sum -1
#define NCCL_REDOP_prod -1
#define NCCL_REDOP_min -1
#define NCCL_REDOP_max -1
#define NCCL_REDOP_and -1
#define NCCL_REDOP_or -1
#define NCCL_REDOP_xor -1

/* Reduction datatypes */
#define NCCL_DT_char        -1
#define NCCL_DT_schar       -1
#define NCCL_DT_short       -1
#define NCCL_DT_int         -1
#define NCCL_DT_long        -1
#define NCCL_DT_longlong    -1
#define NCCL_DT_ptrdiff     -1
#define NCCL_DT_uchar       -1
#define NCCL_DT_ushort      -1
#define NCCL_DT_uint        -1
#define NCCL_DT_ulong       -1
#define NCCL_DT_ulonglong   -1
#define NCCL_DT_int8        -1
#define NCCL_DT_int16       -1
#define NCCL_DT_int32       -1
#define NCCL_DT_int64       -1
#define NCCL_DT_uint8       -1
#define NCCL_DT_uint16      -1
#define NCCL_DT_uint32      -1
#define NCCL_DT_uint64      -1
#define NCCL_DT_size        -1
#define NCCL_DT_float       -1
#define NCCL_DT_double      -1
#define NCCL_DT_longdouble  -1
#define NCCL_DT_complexd    -1
#define NCCL_DT_complexf    -1

typedef int ncclRedOp_t;
typedef int ncclDataType_t;
typedef int ncclComm_t;
typedef int ncclResult_t;
typedef int ncclUniqueId;
#define ncclSuccess 0

#endif /* NVSHMEM_USE_NCCL */

struct nccl_function_table {
    ncclResult_t (*GetVersion)(int *version);
    const char*  (*GetErrorString)(ncclResult_t result);
    ncclResult_t (*GetUniqueId)(ncclUniqueId* uniqueId);
    ncclResult_t (*CommInitRank)(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank);
    ncclResult_t (*CommDestroy)(ncclComm_t comm);
    ncclResult_t (*AllReduce)(const void* sendbuff, void* recvbuff, size_t count,
                              ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream);
    ncclResult_t (*Broadcast)(const void* sendbuff, void* recvbuff, size_t count,
                              ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream);
    ncclResult_t (*AllGather)(const void* sendbuff, void* recvbuff, size_t sendcount,
                              ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
};

extern struct nccl_function_table nccl_ftable;

#include "nvshmemi_team.h"

int nvshmemi_common_init(nvshmemi_state_t *state);
int nvshmemi_init_g_buffer();
int nvshmemi_init_device_state(nvshmemi_state_t *state);
int nvshmemi_setup_local_heap(nvshmemi_state_t *state);
int nvshmemi_setup_symmetric_heap(nvshmemi_state_t *state);
int nvshmemi_setup_connections(nvshmemi_state_t *state);
int nvshmemi_cleanup_symmetric_heap(nvshmemi_state_t *state);
int nvshmemi_setup_collective_launch(nvshmemi_state_t *state);
int nvshmemi_teardown_collective_launch(nvshmemi_state_t *state);
int nvshmemi_setup_mops_kernels(nvshmemi_state_t *state);
void *nvshmemi_malloc(size_t size);
void *nvshmemi_calloc(size_t count, size_t size);
void *nvshmemi_align(size_t alignment, size_t size);
void nvshmemi_free(void *ptr);

void nvshmemi_barrier_all();

int nvshmemi_proxy_init(nvshmemi_state_t *state);
int nvshmemi_proxy_finalize(nvshmemi_state_t *state);


#endif
