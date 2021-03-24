/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmemx.h"
#include "gpu_coll.h"
#include "nvshmemi_coll.h"
#include <cstdio>
#include <cassert>

#ifdef __CUDA_ARCH__

#define NVSHMEMI_GPU_ALLTOALL_THREADGROUP_ALLPUSH(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE, dest, source, nelems, PE_start, \
                                                      PE_stride, PE_size, pSync)                  \
    do {                                                                                          \
        int stride = PE_stride;                                                                   \
        int next_rank;                                                                            \
        int src_offset;                                                                           \
        int dst_offset;                                                                           \
        int mype = nvshmemi_mype_d;                                                               \
        int my_idx_in_active_set = (mype - PE_start) / PE_stride;                                 \
                                                                                                  \
        for (int ii = 0; ii < PE_size; ii++) {                                                    \
            next_rank = PE_start + ((my_idx_in_active_set + ii) % PE_size) * stride;              \
            src_offset = nelems * ((next_rank - PE_start) / stride);                              \
            dst_offset = nelems * ((mype - PE_start) / stride);                                   \
            nvshmem##SC_PREFIX##_##TYPENAME##_put_nbi##SC_SUFFIX(dest + dst_offset,               \
                                                                 source + src_offset, nelems,     \
                                                                 next_rank);                      \
        }                                                                                         \
        nvshmem##SC_PREFIX##i_barrier##SC_SUFFIX(PE_start, PE_stride, PE_size, pSync, NULL);      \
    } while (0)

#define NVSHMEMI_GPU_ALLTOALL_THREADGROUP_P2P_ALLPUSH(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE, dest, source, nelems, PE_start,\
                                                  PE_stride, PE_size, pSync)                     \
    do {                                                                                         \
        int stride = PE_stride;                                                                  \
        int next_rank;                                                                           \
        int src_offset;                                                                          \
        int dst_offset;                                                                          \
        int mype = nvshmemi_mype_d;                                                              \
        int my_idx_in_active_set = (mype - PE_start) / PE_stride;                                \
        TYPE *dst_ptr;                                                                           \
        NVSHMEMI_DECL_THREAD_IDX_##SC();                                                         \
        NVSHMEMI_DECL_THREADGROUP_SIZE_##SC();                                                   \
                                                                                                 \
        for (int ii = 0; ii < PE_size; ii++) {                                                   \
            next_rank = PE_start + ((my_idx_in_active_set + ii) % PE_size) * stride;             \
            src_offset = nelems * ((next_rank - PE_start) / stride);                             \
            dst_offset = nelems * ((mype - PE_start) / stride);                                  \
            dst_ptr = (TYPE *)nvshmem_ptr((void *)(dest + dst_offset), next_rank);               \
            GPU_BITS_COPY_THREADGROUP_DIRECT(TYPENAME, TYPE, dst_ptr,                            \
                                             source + src_offset, nelems,                \
                                             myIdx, groupSize);                                  \
        }                                                                                        \
        nvshmem##SC_PREFIX##i_barrier##SC_SUFFIX(PE_start, PE_stride, PE_size, pSync, NULL);                 \
    } while (0)

#define NVSHMEMI_GPU_ALLTOALL_THREADGROUP(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE, dest, source, nelems, PE_start,    \
                                          PE_stride, PE_size, pSync)                                        \
    do {                                                                                                    \
        if (nvshmemi_job_connectivity_d <= NVSHMEMI_JOB_GPU_LDST)                                           \
            NVSHMEMI_GPU_ALLTOALL_THREADGROUP_P2P_ALLPUSH(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE, dest, source,      \
                                                          nelems, PE_start, PE_stride, PE_size, pSync);     \
        else                                                                                                \
            NVSHMEMI_GPU_ALLTOALL_THREADGROUP_ALLPUSH(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE, dest, source, nelems, PE_start,   \
                                                      PE_stride, PE_size, pSync);                           \
    } while (0)

#define DEFN_NVSHMEMXI_TYPENAME_ALLTOALL_THREADGROUP(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE)                   \
    __device__ void nvshmemxi_##TYPENAME##_alltoall##SC_SUFFIX(TYPE *dest, const TYPE *source,                   \
                                                         size_t nelems, int PE_start, int PE_stride, int PE_size,\
                                                         long *pSync) {                                          \
        NVSHMEMI_GPU_ALLTOALL_THREADGROUP(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE, dest, source, nelems,                   \
                                          PE_start, PE_stride, PE_size, pSync);                                  \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(DEFN_NVSHMEMXI_TYPENAME_ALLTOALL_THREADGROUP, warp, _warp, x)
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(DEFN_NVSHMEMXI_TYPENAME_ALLTOALL_THREADGROUP, block, _block, x)
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(DEFN_NVSHMEMXI_TYPENAME_ALLTOALL_THREADGROUP, thread, ,)
#undef DEFN_NVSHMEMXI_TYPENAME_ALLTOALL_THREADGROUP

#define DEFN_NVSHMEMX_TYPENAME_ALLTOALL_THREADGROUP(SC, SC_SUFFIX, SC_PREFIX, TYPENAME, TYPE)                         \
    __device__ int nvshmemx_##TYPENAME##_alltoall##SC_SUFFIX(nvshmem_team_t team, TYPE *dest, const TYPE *source,     \
                                                     size_t nelems) {                                                 \
        nvshmemi_team_t *teami = nvshmemi_team_pool_d[team];                                                          \
        nvshmem##SC_PREFIX##_barrier##SC_SUFFIX(team);                                                               \
        nvshmemxi_##TYPENAME##_alltoall##SC_SUFFIX(dest, source, nelems, teami->start, teami->stride, teami->size,    \
                                          nvshmemi_team_get_psync(teami, ALLTOALL));                                  \
        return 0;                                                                                                     \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(DEFN_NVSHMEMX_TYPENAME_ALLTOALL_THREADGROUP, warp, _warp, x)
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES_WITH_SCOPE2(DEFN_NVSHMEMX_TYPENAME_ALLTOALL_THREADGROUP, block, _block, x)
#undef DEFN_NVSHMEMX_TYPENAME_ALLTOALL_THREADGROUP


#define DEFN_NVSHMEM_TYPENAME_ALLTOALL(TYPENAME, TYPE)                                                                   \
    __device__ int nvshmem_##TYPENAME##_alltoall(nvshmem_team_t team, TYPE *dest, const TYPE *source, size_t nelems) {   \
        nvshmemi_team_t *teami = nvshmemi_team_pool_d[team];                                                             \
        nvshmem_barrier(team);                                                                                           \
        nvshmemxi_##TYPENAME##_alltoall(dest, source, nelems, teami->start, teami->stride, teami->size,                  \
                                        nvshmemi_team_get_psync(teami, ALLTOALL));                                       \
        return 0;                                                                                                        \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DEFN_NVSHMEM_TYPENAME_ALLTOALL)
#undef DEFN_NVSHMEM_TYPENAME_ALLTOALL

#endif

/* on-stream API implementation */
#define ALLTOALL_ON_STREAM_KERNEL(TYPENAME, TYPE)                                                  \
    __global__ void alltoall_##TYPENAME##_on_stream_kernel(                                        \
        TYPE *dest, const TYPE *source, size_t nelems, int PE_start, int PE_stride,                \
        int PE_size, long *pSync) {                                                                \
        if (!blockIdx.x)                                                                           \
            nvshmemxi_##TYPENAME##_alltoall_block(dest, source, nelems, PE_start, PE_stride, PE_size, \
                                            pSync);                                                \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(ALLTOALL_ON_STREAM_KERNEL)
#undef ALLTOALL_ON_STREAM_KERNEL

#define CALL_ALLTOALL_ON_STREAM(TYPENAME, TYPE)                                                    \
    extern "C" void call_##TYPENAME##_alltoall_on_stream_kern(                                     \
        TYPE *dest, const TYPE *source, size_t nelems, int PE_start, int PE_stride,                \
        int PE_size, long *pSync, cudaStream_t stream) {                                           \
        int num_threads_per_block = (MAX_THREADS_PER_CTA > nelems) ? nelems : MAX_THREADS_PER_CTA; \
        int num_blocks = 1;                                                                        \
        alltoall_##TYPENAME##_on_stream_kernel<<<num_blocks, num_threads_per_block, 0, stream>>>(  \
            dest, source, nelems, PE_start, PE_stride, PE_size, pSync);                            \
        CUDA_RUNTIME_CHECK(cudaGetLastError());                                                    \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(CALL_ALLTOALL_ON_STREAM)
#undef CALL_ALLTOALL_ON_STREAM
