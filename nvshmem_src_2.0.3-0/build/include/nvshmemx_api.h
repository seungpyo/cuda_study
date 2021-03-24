/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * See COPYRIGHT.txt for license information
 */

#ifndef _NVSHMEMX_API_H_
#define _NVSHMEMX_API_H_

#include <stdint.h>
#include <stddef.h>
#include "nvshmem_types.h"
#include "nvshmem_constants.h"
#include "nvshmem_common.cuh"
#include "nvshmemx_coll_api.h"
#include "nvshmemx_error.h"

#define INIT_HANDLE_BYTES 128

#ifdef __cplusplus
extern "C" {
#endif

enum flags {
    NVSHMEMX_INIT_THREAD_PES = 1,
    NVSHMEMX_INIT_WITH_MPI_COMM = 1 << 1,
    NVSHMEMX_INIT_WITH_SHMEM = 1 << 2,
    NVSHMEMX_INIT_WITH_HANDLE = 1 << 3
};

typedef struct {
    char content[INIT_HANDLE_BYTES];
} nvshmemx_init_handle_t;

typedef struct {
    size_t heap_size;
    int num_threads;
    int n_pes;
    int my_pe;
    void *mpi_comm;
    nvshmemx_init_handle_t handle;
} nvshmemx_init_attr_t;

/* Renamed from nvshmemx_<FUNC> to nvshmem_<FUNC> */
int nvshmemx_init_thread(int requested, int *provided) __attribute__((deprecated));
void nvshmemx_query_thread(int *provided) __attribute__((deprecated));

int nvshmemx_init_attr(unsigned int flags, nvshmemx_init_attr_t *attributes);

/* Replaced by teams API */
typedef nvshmem_team_t nvshmemx_team_t;
NVSHMEMI_HOSTDEVICE_PREFIX int nvshmemx_my_pe(nvshmemx_team_t team) __attribute__((deprecated));
NVSHMEMI_HOSTDEVICE_PREFIX int nvshmemx_n_pes(nvshmemx_team_t team) __attribute__((deprecated));

int nvshmemx_get_init_handle(nvshmemx_init_handle_t *handle);
int nvshmemx_free_init_handle(nvshmemx_init_handle_t handle);

int nvshmemx_collective_launch(const void *func, dim3 gridDims, dim3 blockDims, void **args,
                               size_t sharedMem, cudaStream_t stream);
int nvshmemx_collective_launch_query_gridsize(const void *func, dim3 blockDims, void **args,
                                              size_t sharedMem, int *gridsize);

//////////////////// Put On Stream ////////////////////

#define NVSHMEMX_DECL_TYPE_P_ON_STREAM(NAME, TYPE)                              \
    void nvshmemx_##NAME##_p_on_stream(TYPE *dest, const TYPE value, int pe,    \
            cudaStream_t cstrm);

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_DECL_TYPE_P_ON_STREAM)
#undef NVSHMEMX_DECL_TYPE_P_ON_STREAM

#define NVSHMEMX_DECL_TYPE_PUT_ON_STREAM(NAME, TYPE)                            \
    void nvshmemx_##NAME##_put_on_stream(TYPE *dest, const TYPE *source,        \
            size_t nelems, int pe, cudaStream_t cstrm);

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_DECL_TYPE_PUT_ON_STREAM)
#undef NVSHMEMX_DECL_TYPE_PUT_ON_STREAM

#define NVSHMEMX_DECL_SIZE_PUT_ON_STREAM(NAME)                                  \
    void nvshmemx_put##NAME##_on_stream(void *dest, const void *source,         \
            size_t nelems, int pe, cudaStream_t cstrm);

NVSHMEMI_REPT_FOR_SIZES(NVSHMEMX_DECL_SIZE_PUT_ON_STREAM)
#undef NVSHMEMX_DECL_SIZE_PUT_ON_STREAM

void nvshmemx_putmem_on_stream(void *dest, const void *source, size_t bytes, int pe,
                               cudaStream_t cstrm);

#define NVSHMEMX_DECL_TYPE_IPUT_ON_STREAM(NAME, TYPE)                           \
    void nvshmemx_##NAME##_iput_on_stream(TYPE *dest, const TYPE *source,       \
            ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe, cudaStream_t cstrm);

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_DECL_TYPE_IPUT_ON_STREAM)
#undef NVSHMEMX_DECL_TYPE_IPUT_ON_STREAM

#define NVSHMEMX_DECL_SIZE_IPUT_ON_STREAM(NAME)                                 \
    void nvshmemx_iput##NAME##_on_stream(void *dest, const void *source,        \
            ptrdiff_t dst, ptrdiff_t sst, size_t nelems, int pe, cudaStream_t cstrm);

NVSHMEMI_REPT_FOR_SIZES(NVSHMEMX_DECL_SIZE_IPUT_ON_STREAM)
#undef NVSHMEMX_DECL_SIZE_IPUT_ON_STREAM

#define NVSHMEMX_DECL_TYPE_PUT_NBI_ON_STREAM(NAME, TYPE)                                    \
    void nvshmemx_##NAME##_put_nbi_on_stream(TYPE *dest, const TYPE *source, size_t nelems, \
                                             int pe, cudaStream_t cstrm);

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_DECL_TYPE_PUT_NBI_ON_STREAM)
#undef NVSHMEMX_DECL_TYPE_PUT_NBI_ON_STREAM

#define NVSHMEMX_DECL_SIZE_PUT_NBI_ON_STREAM(NAME)                                                 \
    void nvshmemx_put##NAME##_nbi_on_stream(void *dest, const void *source, size_t nelems, int pe, \
                                            cudaStream_t cstrm);

NVSHMEMI_REPT_FOR_SIZES(NVSHMEMX_DECL_SIZE_PUT_NBI_ON_STREAM)
#undef NVSHMEMX_DECL_SIZE_PUT_NBI_ON_STREAM

void nvshmemx_putmem_nbi_on_stream(void *dest, const void *source, size_t bytes, int pe,
                                   cudaStream_t cstrm);

//////////////////// Get On Stream ////////////////////

#define NVSHMEMX_DECL_TYPE_G_ON_STREAM(NAME, TYPE) \
    TYPE nvshmemx_##NAME##_g_on_stream(const TYPE *src, int pe, cudaStream_t cstrm);

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_DECL_TYPE_G_ON_STREAM)
#undef NVSHMEMX_DECL_TYPE_G_ON_STREAM

#define NVSHMEMX_DECL_TYPE_GET_ON_STREAM(NAME, TYPE)                                            \
    void nvshmemx_##NAME##_get_on_stream(TYPE *dest, const TYPE *source, size_t nelems, int pe, \
                                         cudaStream_t cstrm);

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_DECL_TYPE_GET_ON_STREAM)
#undef NVSHMEMX_DECL_TYPE_GET_ON_STREAM

#define NVSHMEMX_DECL_TYPE_IGET_ON_STREAM(NAME, TYPE)                                    \
    void nvshmemx_##NAME##_iget_on_stream(TYPE *dest, const TYPE *source, ptrdiff_t dst, \
                                          ptrdiff_t sst, size_t nelems, int pe,          \
                                          cudaStream_t cstrm);

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_DECL_TYPE_IGET_ON_STREAM)
#undef NVSHMEMX_DECL_TYPE_IGET_ON_STREAM

#define NVSHMEMX_DECL_SIZE_GET_ON_STREAM(NAME)                                                 \
    void nvshmemx_get##NAME##_on_stream(void *dest, const void *source, size_t nelems, int pe, \
                                        cudaStream_t cstrm);

NVSHMEMI_REPT_FOR_SIZES(NVSHMEMX_DECL_SIZE_GET_ON_STREAM)
#undef NVSHMEMX_DECL_SIZE_GET_ON_STREAM

void nvshmemx_getmem_on_stream(void *dest, const void *source, size_t bytes, int pe,
                               cudaStream_t cstrm);

#define NVSHMEMX_DECL_SIZE_IGET_ON_STREAM(NAME)                                         \
    void nvshmemx_iget##NAME##_on_stream(void *dest, const void *source, ptrdiff_t dst, \
                                         ptrdiff_t sst, size_t nelems, int pe, cudaStream_t cstrm);

NVSHMEMI_REPT_FOR_SIZES(NVSHMEMX_DECL_SIZE_IGET_ON_STREAM)
#undef NVSHMEMX_DECL_SIZE_IGET_ON_STREAM

#define NVSHMEMX_DECL_TYPE_GET_NBI_ON_STREAM(NAME, TYPE)                                    \
    void nvshmemx_##NAME##_get_nbi_on_stream(TYPE *dest, const TYPE *source, size_t nelems, \
                                             int pe, cudaStream_t cstrm);

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_DECL_TYPE_GET_NBI_ON_STREAM)
#undef NVSHMEMX_DECL_TYPE_GET_NBI_ON_STREAM

#define NVSHMEMX_DECL_SIZE_GET_NBI_ON_STREAM(NAME)                                                 \
    void nvshmemx_get##NAME##_nbi_on_stream(void *dest, const void *source, size_t nelems, int pe, \
                                            cudaStream_t cstrm);

NVSHMEMI_REPT_FOR_SIZES(NVSHMEMX_DECL_SIZE_GET_NBI_ON_STREAM)
#undef NVSHMEMX_DECL_SIZE_GET_NBI_ON_STREAM

void nvshmemx_getmem_nbi_on_stream(void *dest, const void *source, size_t bytes, int pe,
                                   cudaStream_t cstrm);

//////////////////// Synchronization On Stream ////////////////////

void nvshmemx_quiet_on_stream(cudaStream_t cstrm);
void nvshmemx_wait_on_stream(long *ivar, long cmp_value, cudaStream_t cstream);
void nvshmemx_wait_until_on_stream(long *ivar, int cmp, long cmp_value,
                                   cudaStream_t cstream);

#define NVSHMEMX_DECL_WAIT_ON_STREAM(NAME, Type) \
    void nvshmemx_##NAME##_wait_on_stream(Type *ivar, Type cmp_value, cudaStream_t cstream);

NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEMX_DECL_WAIT_ON_STREAM)
#undef NVSHMEMX_DECL_WAIT_ON_STREAM

#define NVSHMEMX_DECL_WAIT_UNTIL_ON_STREAM(NAME, Type)                                        \
    void nvshmemx_##NAME##_wait_until_on_stream(Type *ivar, int cmp, Type cmp_value, \
                                                cudaStream_t cstream);

NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEMX_DECL_WAIT_UNTIL_ON_STREAM)
#undef NVSHMEMX_DECL_WAIT_UNTIL_ON_STREAM

//////////////////// Put on Thread Group ////////////////////


#define NVSHMEMX_DECL_TYPE_PUT_THREADGROUP(NAME, TYPE)                                         \
    __device__ void nvshmemx_##NAME##_put_warp(TYPE *dest, const TYPE *source, size_t nelems,  \
                                               int pe);                                        \
    __device__ void nvshmemx_##NAME##_put_block(TYPE *dest, const TYPE *source, size_t nelems, \
                                                int pe);

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_DECL_TYPE_PUT_THREADGROUP)
#undef NVSHMEMX_DECL_TYPE_PUT_THREADGROUP

#define NVSHMEMX_DECL_SIZE_PUT_THREADGROUP(NAME)                                              \
    __device__ void nvshmemx_put##NAME##_warp(void *dest, const void *source, size_t nelems,  \
                                              int pe);                                        \
    __device__ void nvshmemx_put##NAME##_block(void *dest, const void *source, size_t nelems, \
                                               int pe);

NVSHMEMI_REPT_FOR_SIZES(NVSHMEMX_DECL_SIZE_PUT_THREADGROUP)
#undef NVSHMEMX_DECL_SIZE_PUT_THREADGROUP

__device__ void nvshmemx_putmem_warp(void *dest, const void *source, size_t bytes, int pe);
__device__ void nvshmemx_putmem_block(void *dest, const void *source, size_t bytes, int pe);

#define NVSHMEMX_DECL_TYPE_IPUT_THREADGROUP(NAME, TYPE)                                         \
    __device__ void nvshmemx_##NAME##_iput_warp(TYPE *dest, const TYPE *source, ptrdiff_t dst,  \
                                                ptrdiff_t sst, size_t nelems, int pe);          \
    __device__ void nvshmemx_##NAME##_iput_block(TYPE *dest, const TYPE *source, ptrdiff_t dst, \
                                                 ptrdiff_t sst, size_t nelems, int pe);

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_DECL_TYPE_IPUT_THREADGROUP)
#undef NVSHMEMX_DECL_TYPE_IPUT_THREADGROUP

#define NVSHMEMX_DECL_SIZE_IPUT_THREADGROUP(NAME)                                              \
    __device__ void nvshmemx_iput##NAME##_warp(void *dest, const void *source, ptrdiff_t dst,  \
                                               ptrdiff_t sst, size_t nelems, int pe);          \
    __device__ void nvshmemx_iput##NAME##_block(void *dest, const void *source, ptrdiff_t dst, \
                                                ptrdiff_t sst, size_t nelems, int pe);

NVSHMEMI_REPT_FOR_SIZES(NVSHMEMX_DECL_SIZE_IPUT_THREADGROUP)
#undef NVSHMEMX_DECL_SIZE_IPUT_THREADGROUP

#define NVSHMEMX_DECL_TYPE_PUT_NBI_THREADGROUP(NAME, TYPE)                                         \
    __device__ void nvshmemx_##NAME##_put_nbi_warp(TYPE *dest, const TYPE *source, size_t nelems,  \
                                                   int pe);                                        \
    __device__ void nvshmemx_##NAME##_put_nbi_block(TYPE *dest, const TYPE *source, size_t nelems, \
                                                    int pe);

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_DECL_TYPE_PUT_NBI_THREADGROUP)
#undef NVSHMEMX_DECL_TYPE_PUT_NBI_THREADGROUP

#define NVSHMEMX_DECL_SIZE_PUT_NBI_THREADGROUP(NAME)                                              \
    __device__ void nvshmemx_put##NAME##_nbi_warp(void *dest, const void *source, size_t nelems,  \
                                                  int pe);                                        \
    __device__ void nvshmemx_put##NAME##_nbi_block(void *dest, const void *source, size_t nelems, \
                                                   int pe);

NVSHMEMI_REPT_FOR_SIZES(NVSHMEMX_DECL_SIZE_PUT_NBI_THREADGROUP)
#undef NVSHMEMX_DECL_SIZE_PUT_NBI_THREADGROUP

__device__ void nvshmemx_putmem_nbi_warp(void *dest, const void *source, size_t bytes, int pe);
__device__ void nvshmemx_putmem_nbi_block(void *dest, const void *source, size_t bytes, int pe);

//////////////////// Get on Thread Group ////////////////////

#define NVSHMEMX_DECL_TYPE_GET_THREADGROUP(NAME, TYPE)                                         \
    __device__ void nvshmemx_##NAME##_get_warp(TYPE *dest, const TYPE *source, size_t nelems,  \
                                               int pe);                                        \
    __device__ void nvshmemx_##NAME##_get_block(TYPE *dest, const TYPE *source, size_t nelems, \
                                                int pe);

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_DECL_TYPE_GET_THREADGROUP)
#undef NVSHMEMX_DECL_TYPE_GET_THREADGROUP

#define NVSHMEMX_DECL_TYPE_IGET_THREADGROUP(NAME, TYPE)                                         \
    __device__ void nvshmemx_##NAME##_iget_warp(TYPE *dest, const TYPE *source, ptrdiff_t dst,  \
                                                ptrdiff_t sst, size_t nelems, int pe);          \
    __device__ void nvshmemx_##NAME##_iget_block(TYPE *dest, const TYPE *source, ptrdiff_t dst, \
                                                 ptrdiff_t sst, size_t nelems, int pe);

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_DECL_TYPE_IGET_THREADGROUP)
#undef NVSHMEMX_DECL_TYPE_IGET_THREADGROUP

#define NVSHMEMX_DECL_SIZE_GET_THREADGROUP(NAME)                                              \
    __device__ void nvshmemx_get##NAME##_warp(void *dest, const void *source, size_t nelems,  \
                                              int pe);                                        \
    __device__ void nvshmemx_get##NAME##_block(void *dest, const void *source, size_t nelems, \
                                               int pe);
NVSHMEMI_REPT_FOR_SIZES(NVSHMEMX_DECL_SIZE_GET_THREADGROUP)
#undef NVSHMEMX_DECL_SIZE_GET_THREADGROUP

__device__ void nvshmemx_getmem_warp(void *dest, const void *source, size_t bytes, int pe);
__device__ void nvshmemx_getmem_block(void *dest, const void *source, size_t bytes, int pe);

#define NVSHMEMX_DECL_SIZE_IGET_THREADGROUP(NAME)                                              \
    __device__ void nvshmemx_iget##NAME##_warp(void *dest, const void *source, ptrdiff_t dst,  \
                                               ptrdiff_t sst, size_t nelems, int pe);          \
    __device__ void nvshmemx_iget##NAME##_block(void *dest, const void *source, ptrdiff_t dst, \
                                                ptrdiff_t sst, size_t nelems, int pe);

NVSHMEMI_REPT_FOR_SIZES(NVSHMEMX_DECL_SIZE_IGET_THREADGROUP)
#undef NVSHMEMX_DECL_SIZE_IGET_THREADGROUP

#define NVSHMEMX_DECL_TYPE_GET_NBI_THREADGROUP(NAME, TYPE)                                         \
    __device__ void nvshmemx_##NAME##_get_nbi_warp(TYPE *dest, const TYPE *source, size_t nelems,  \
                                                   int pe);                                        \
    __device__ void nvshmemx_##NAME##_get_nbi_block(TYPE *dest, const TYPE *source, size_t nelems, \
                                                    int pe);

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMX_DECL_TYPE_GET_NBI_THREADGROUP)
#undef NVSHMEMX_DECL_TYPE_GET_NBI_THREADGROUP

#define NVSHMEMX_DECL_SIZE_GET_NBI_THREADGROUP(NAME)                                              \
    __device__ void nvshmemx_get##NAME##_nbi_warp(void *dest, const void *source, size_t nelems,  \
                                                  int pe);                                        \
    __device__ void nvshmemx_get##NAME##_nbi_block(void *dest, const void *source, size_t nelems, \
                                                   int pe);

NVSHMEMI_REPT_FOR_SIZES(NVSHMEMX_DECL_SIZE_GET_NBI_THREADGROUP)
#undef NVSHMEMX_DECL_SIZE_GET_NBI_THREADGROUP

__device__ void nvshmemx_getmem_nbi_warp(void *dest, const void *source, size_t bytes, int pe);
__device__ void nvshmemx_getmem_nbi_block(void *dest, const void *source, size_t bytes, int pe);


//////////////////// Signal ////////////////////

#define NVSHMEMX_DECL_TYPE_SIGNAL(NAME, TYPE)                                               \
    __device__ void nvshmemx_##NAME##_signal(TYPE *dest, const TYPE value, int pe);         \

NVSHMEMX_REPT_FOR_SIGNAL_TYPES(NVSHMEMX_DECL_TYPE_SIGNAL)
#undef NVSHMEMX_DECL_TYPE_SIGNAL

NVSHMEMI_HOSTDEVICE_PREFIX void nvshmemx_signal_op(uint64_t *sig_addr, uint64_t signal, \
                                                   int sig_op, int pe);


#ifdef __cplusplus
}
#endif

#endif /* _NVSHMEMX_API_H_ */
