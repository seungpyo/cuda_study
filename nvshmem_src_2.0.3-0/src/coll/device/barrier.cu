/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmemx.h"
#include "nvshmemi_util.h"
#include "nvshmemi_coll.h"
#include "gpu_coll.h"

extern __constant__ int barrier_tg_dissem_kval_d;

__global__ void barrier_on_stream_kernel(int start, int stride, int size, long *pSync, long *counter);
__global__ void barrier_on_stream_kernel_warp(int start, int stride, int size, long *pSync, long *counter);
__global__ void barrier_on_stream_kernel_block(int start, int stride, int size, long *pSync, long *counter);
__global__ void barrier_all_on_stream_kernel();
__global__ void barrier_all_on_stream_kernel_warp();
__global__ void barrier_all_on_stream_kernel_block();

__global__ void sync_on_stream_kernel(int start, int stride, int size, long *pSync, long *counter);
__global__ void sync_on_stream_kernel_warp(int start, int stride, int size, long *pSync, long *counter);
__global__ void sync_on_stream_kernel_block(int start, int stride, int size, long *pSync, long *counter);
__global__ void sync_all_on_stream_kernel();
__global__ void sync_all_on_stream_kernel_warp();
__global__ void sync_all_on_stream_kernel_block();


#ifdef __CUDA_ARCH__

#define NVSHMEMI_SYNC_DISSEM_POW2_ALGO(SC, SC_SUFFIX, SC_PREFIX)                         \
template<int k, int logk>                                                                \
__device__ static inline void sync_dissem_pow2##SC_SUFFIX(int start, int stride, int size,  \
                                                    volatile long *pSync,                \
                                                    volatile long *sync_counter)         \
    {                                                                                    \
        int my_idx_in_active_set = (nvshmemi_mype_d - start) / stride;                   \
        volatile long *sync_arr = NULL;                                                  \
        int shift;                                                                       \
        int to_nbr_idx, to_nbr;                                                          \
        int from_nbr_idx, from_nbr;                                                      \
        int temp = size - 1; /* used to calculate number of phases */                    \
        int myIdx = nvshmemi_thread_id_in_##SC();                                        \
        int groupSize = nvshmemi_##SC##_size();                                          \
        sync_arr = (volatile long *) pSync;                                              \
        int pow_k = 1;                                                                   \
        int phase_num = 0;                                                               \
        volatile long counter_val = NVSHMEMI_SYNC_VALUE + 1;                             \
        volatile long *counter;                                                          \
        if (sync_counter == NULL)                                                        \
            counter = &counter_val;                                                      \
        else                                                                             \
            counter = sync_counter;                                                      \
        while (temp) {                                                                   \
            /* notify neighbors */                                                       \
            for (int j = myIdx + 1; j <= k - 1; j += groupSize) {                        \
                shift = j << phase_num;                                                  \
                if (shift >= size) break;                                                \
                                                                                         \
                to_nbr_idx = my_idx_in_active_set + shift;                               \
                if (to_nbr_idx >= size) to_nbr_idx = to_nbr_idx - size;                  \
                to_nbr = start + to_nbr_idx * stride;                                    \
                                                                                         \
                nvshmemi_signal_for_barrier<long>                                        \
                (((long *)sync_arr + nvshmemi_mype_d), counter[0], to_nbr);              \
            }                                                                            \
                                                                                         \
            /* wait for neighbors notification */                                        \
            for (int j = myIdx + 1; j <= k - 1; j += groupSize) {                        \
                shift = j << phase_num;                                                  \
                if (shift >= size) break;                                                \
                                                                                         \
                from_nbr_idx = my_idx_in_active_set - shift;                             \
                if (from_nbr_idx < 0) from_nbr_idx = size + from_nbr_idx;                \
                from_nbr = start + from_nbr_idx * stride;                                \
                                                                                         \
                nvshmemi_wait_until_greater_than_equals<volatile long>(                  \
                                                        sync_arr + from_nbr, counter[0], \
                                                        NVSHMEMI_CALL_SITE_BARRIER_WARP);\
                if (sync_counter == NULL)                                                \
                    *(sync_arr + from_nbr) = NVSHMEMI_SYNC_VALUE;                        \
            }                                                                            \
            pow_k <<= logk;                                                              \
            temp >>= logk;                                                               \
            phase_num++;                                                                 \
            nvshmemi_##SC##_sync();                                                      \
        }                                                                                \
        if (sync_counter) {                                                              \
            if (!myIdx)                                                                  \
                sync_counter[0] += 1;                                                    \
            nvshmemi_##SC##_sync();                                                      \
        }                                                                                \
    }

NVSHMEMI_SYNC_DISSEM_POW2_ALGO(thread, , )
NVSHMEMI_SYNC_DISSEM_POW2_ALGO(warp, _warp, x)
NVSHMEMI_SYNC_DISSEM_POW2_ALGO(block, _block, x)
#undef NVSHMEMI_SYNC_DISSEM_POW2_ALGO


#define NVSHMEMI_SYNC_DISSEM_ALGO(SC, SC_SUFFIX, SC_PREFIX)                              \
__device__ static inline void sync_dissem##SC_SUFFIX(int start, int stride, int size,    \
                                               volatile long *pSync,                     \
                                               volatile long *sync_counter) {            \
        int num_phases = 0;                                                              \
        int k = min(barrier_tg_dissem_kval_d, size); /* radix for the dissemination      \
                                                        algorithm */                     \
        int my_idx_in_active_set = (nvshmemi_mype_d - start) / stride;                   \
        volatile long *sync_arr = NULL;                                                  \
        int shift;                                                                       \
        int to_nbr_idx, to_nbr;                                                          \
        int from_nbr_idx, from_nbr;                                                      \
        int temp = size - 1; /* used to calculate number of phases */                    \
        while (temp) {                                                                   \
            num_phases++;                                                                \
            temp /= k;                                                                   \
        }                                                                                \
        int myIdx = nvshmemi_thread_id_in_##SC();                                        \
        int groupSize = nvshmemi_##SC##_size();                                          \
        sync_arr = (volatile long *) pSync;                                              \
        int pow_k = 1;                                                                   \
        volatile long counter_val = NVSHMEMI_SYNC_VALUE + 1;                             \
        volatile long *counter;                                                          \
        if (sync_counter == NULL)                                                        \
            counter = &counter_val;                                                      \
        else                                                                             \
            counter = sync_counter;                                                      \
        for (int i = 0; i < num_phases; i++) {                                           \
            /* notify neighbors */                                                       \
            for (int j = myIdx + 1; j <= k - 1; j += groupSize) {                        \
                shift = j * pow_k;                                                       \
                if (shift >= size) break;                                                \
                to_nbr_idx = (my_idx_in_active_set + shift) % size;                      \
                to_nbr = start + to_nbr_idx * stride;                                    \
                                                                                         \
                nvshmemi_signal_for_barrier<long>                                        \
                (((long *)sync_arr + nvshmemi_mype_d), counter[0], to_nbr);              \
            }                                                                            \
                                                                                         \
            /* wait for neighbors notification */                                        \
            for (int j = myIdx + 1; j <= k - 1; j += groupSize) {                        \
                shift = j * pow_k;                                                       \
                if (shift >= size) break;                                                \
                                                                                         \
                from_nbr_idx = my_idx_in_active_set - shift;                             \
                if (from_nbr_idx < 0) from_nbr_idx = size + from_nbr_idx;                \
                from_nbr = start + from_nbr_idx * stride;                                \
                                                                                         \
                nvshmemi_wait_until_greater_than_equals<volatile long>(                  \
                                                        sync_arr + from_nbr, counter[0], \
                                                        NVSHMEMI_CALL_SITE_BARRIER_WARP);\
                if (sync_counter == NULL)                                                \
                    *(sync_arr + from_nbr) = NVSHMEMI_SYNC_VALUE;                        \
            }                                                                            \
            pow_k *= k;                                                                  \
            nvshmemi_##SC##_sync();                                                      \
        }                                                                                \
        if (sync_counter) {                                                              \
            if (!myIdx)                                                                  \
                sync_counter[0] += 1;                                                    \
            nvshmemi_##SC##_sync();                                                      \
        }                                                                                \
    }

NVSHMEMI_SYNC_DISSEM_ALGO(thread, , )
NVSHMEMI_SYNC_DISSEM_ALGO(warp, _warp, x)
NVSHMEMI_SYNC_DISSEM_ALGO(block, _block, x)
#undef NVSHMEMI_SYNC_DISSEM_ALGO

#define NVSHMEMI_BARRIER_THREADGROUP_ALGO(SC, SC_SUFFIX, SC_PREFIX)                                                   \
    __device__ inline void nvshmemi_sync_algo##SC_SUFFIX(int start, int stride, int size, volatile long *pSync,       \
                                                           volatile long *counter) {                                  \
        int k = min(barrier_tg_dissem_kval_d, size);                                                                  \
        k = max(k, 2);                                                                                                \
        switch(k) {                                                                                                   \
            case 2:                                                                                                   \
                sync_dissem_pow2##SC_SUFFIX<2, 1>(start, stride, size, pSync, counter);                               \
                break;                                                                                                \
            case 4:                                                                                                   \
                sync_dissem_pow2##SC_SUFFIX<4, 2>(start, stride, size, pSync, counter);                               \
                break;                                                                                                \
            case 8:                                                                                                   \
                sync_dissem_pow2##SC_SUFFIX<8, 3>(start, stride, size, pSync, counter);                               \
                break;                                                                                                \
            case 16:                                                                                                  \
                sync_dissem_pow2##SC_SUFFIX<16, 4>(start, stride, size, pSync, counter);                              \
                break;                                                                                                \
            case 32:                                                                                                  \
                sync_dissem_pow2##SC_SUFFIX<32, 5>(start, stride, size, pSync, counter);                              \
                break;                                                                                                \
            default:                                                                                                  \
                sync_dissem##SC_SUFFIX(start, stride, size, pSync, counter);                                          \
                break;                                                                                                \
        }                                                                                                             \
    }
NVSHMEMI_BARRIER_THREADGROUP_ALGO(thread, , )
NVSHMEMI_BARRIER_THREADGROUP_ALGO(warp, _warp, x)
NVSHMEMI_BARRIER_THREADGROUP_ALGO(block, _block, x) 

#define DEFN_NVSHMEMXI_BARRIER_SCOPE(SC, SC_SUFFIX, SC_PREFIX)                                              \
    __device__ void nvshmem##SC_PREFIX##i_barrier##SC_SUFFIX(int start, int stride, int size, long *pSync, long *counter) {   \
        int myIdx = nvshmemi_thread_id_in_##SC();                                                           \
                                                                                                            \
        NVSHMEMI_SYNC_##SC();                                                                               \
        if (!myIdx) nvshmem_quiet();                                                                        \
        NVSHMEMI_SYNC_##SC();                                                                               \
                                                                                                            \
        nvshmemi_sync_algo##SC_SUFFIX(start, stride, size, pSync, counter);                                 \
                                                                                                            \
        if (!myIdx) {                                                                                       \
            if (nvshmemi_job_connectivity_d > NVSHMEMI_JOB_GPU_PROXY)                                       \
                nvshmemi_proxy_enforce_consistency_at_target_no_membar();                                   \
        }                                                                                                   \
        NVSHMEMI_SYNC_##SC();                                                                               \
    }

DEFN_NVSHMEMXI_BARRIER_SCOPE(thread, , )
DEFN_NVSHMEMXI_BARRIER_SCOPE(warp, _warp, x)
DEFN_NVSHMEMXI_BARRIER_SCOPE(block, _block, x)
#undef DEFN_NVSHMEMXI_BARRIER_SCOPE

#define DEFN_NVSHMEMX_BARRIER_SCOPE(SC, SC_SUFFIX, SC_PREFIX)                            \
    __device__ int nvshmem##SC_PREFIX##_barrier##SC_SUFFIX(nvshmem_team_t team) {        \
        nvshmemi_team_t *teami = nvshmemi_team_pool_d[team];                             \
        nvshmem##SC_PREFIX##i_barrier##SC_SUFFIX(teami->start, teami->stride, teami->size,\
                               nvshmemi_team_get_psync(teami, SYNC),                     \
                               nvshmemi_team_get_sync_counter(teami));                   \
        return 0;                                                                        \
    }

DEFN_NVSHMEMX_BARRIER_SCOPE(thread, , )
DEFN_NVSHMEMX_BARRIER_SCOPE(warp, _warp, x)
DEFN_NVSHMEMX_BARRIER_SCOPE(block, _block, x)
#undef DEFN_NVSHMEMX_BARRIER_SCOPE

#define DEFN_NVSHMEMX_BARRIER_ALL_SCOPE(SC, SC_SUFFIX, SC_PREFIX)                        \
    __device__ void nvshmem##SC_PREFIX##_barrier_all##SC_SUFFIX() {                      \
        nvshmemi_team_t *teami = nvshmemi_team_pool_d[NVSHMEM_TEAM_WORLD];               \
        nvshmem##SC_PREFIX##i_barrier##SC_SUFFIX(teami->start, teami->stride, teami->size,           \
                                     nvshmemi_team_get_psync(teami, SYNC),               \
                                     nvshmemi_team_get_sync_counter(teami));             \
    }

DEFN_NVSHMEMX_BARRIER_ALL_SCOPE(thread, , )
DEFN_NVSHMEMX_BARRIER_ALL_SCOPE(warp, _warp, x)
DEFN_NVSHMEMX_BARRIER_ALL_SCOPE(block, _block, x)
#undef DEFN_NVSHMEMX_BARRIER_ALL_SCOPE


#define DEFN_NVSHMEMXI_SYNC_SCOPE(SC, SC_SUFFIX, SC_PREFIX)                                             \
    __device__ void nvshmem##SC_PREFIX##i_sync##SC_SUFFIX(int start, int stride, int size, long *pSync, long *counter) {  \
        int myidx = nvshmemi_thread_id_in_##SC();                                                       \
        NVSHMEMI_SYNC_##SC();                                                                           \
        nvshmemi_sync_algo##SC_SUFFIX(start, stride, size, pSync, counter);                             \
    }

DEFN_NVSHMEMXI_SYNC_SCOPE(thread, , )
DEFN_NVSHMEMXI_SYNC_SCOPE(warp, _warp, x)
DEFN_NVSHMEMXI_SYNC_SCOPE(block, _block, x)
#undef DEFN_NVSHMEMXI_SYNC_SCOPE

#define DEFN_NVSHMEMX_SYNC_SCOPE(SC, SC_SUFFIX, SC_PREFIX)                              \
    __device__ int  nvshmem##SC_PREFIX##_team_sync##SC_SUFFIX(nvshmem_team_t team) {    \
        nvshmemi_team_t *teami = nvshmemi_team_pool_d[team];                            \
        nvshmem##SC_PREFIX##i_sync##SC_SUFFIX(teami->start, teami->stride, teami->size, \
                                              nvshmemi_team_get_psync(teami, SYNC),     \
                                              nvshmemi_team_get_sync_counter(teami));   \
        return 0;                                                                       \
    }

DEFN_NVSHMEMX_SYNC_SCOPE(thread, , )
DEFN_NVSHMEMX_SYNC_SCOPE(warp, _warp, x)
DEFN_NVSHMEMX_SYNC_SCOPE(block, _block, x)
#undef DEFN_NVSHMEMX_SYNC_SCOPE

#define DEFN_NVSHMEMX_SYNC_ALL_SCOPE(SC, SC_SUFFIX, SC_PREFIX)                      \
    __device__ void nvshmem##SC_PREFIX##_sync_all##SC_SUFFIX() {                    \
        nvshmemi_team_t *teami = nvshmemi_team_pool_d[NVSHMEM_TEAM_WORLD];          \
        nvshmem##SC_PREFIX##i_sync##SC_SUFFIX(teami->start, teami->stride, teami->size,         \
                                  nvshmemi_team_get_psync(teami, SYNC),             \
                                  nvshmemi_team_get_sync_counter(teami));           \
    }

DEFN_NVSHMEMX_SYNC_ALL_SCOPE(thread, , )
DEFN_NVSHMEMX_SYNC_ALL_SCOPE(warp, _warp, x)
DEFN_NVSHMEMX_SYNC_ALL_SCOPE(block, _block, x)
#undef DEFN_NVSHMEMX_SYNC_ALL_SCOPE


#define BARRIER_ON_STREAM_KERNEL(SC, SC_SUFFIX, SC_PREFIX)                                                        \
    __global__ void barrier_on_stream_kernel##SC_SUFFIX (int start, int stride, int size, long *pSync, long *counter) {  \
        int myidx = nvshmemi_thread_id_in_##SC();                                                                 \
                                                                                                                  \
        if (nvshmemi_job_connectivity_d >= NVSHMEMI_JOB_GPU_PROXY) { 					                          \
            if (!myidx)                                                                                           \
                nvshmemi_proxy_quiet_no_membar();                                                                 \
            NVSHMEMI_SYNC_##SC();                                                                                 \
        }                                                                                                         \
                                                                                                                  \
        nvshmemi_sync_algo##SC_SUFFIX(start, stride, size, pSync, counter);                                       \
                                                                                                                  \
        if (!myidx) {                                                                                             \
            if (nvshmemi_job_connectivity_d > NVSHMEMI_JOB_GPU_PROXY)						                      \
                nvshmemi_proxy_enforce_consistency_at_target_no_membar();                                         \
        }                                                                                                         \
    }

BARRIER_ON_STREAM_KERNEL(thread, , );
BARRIER_ON_STREAM_KERNEL(warp, _warp, x);
BARRIER_ON_STREAM_KERNEL(block, _block, x);
#undef BARRIER_ON_STREAM_KERNEL

#define BARRIER_ALL_ON_STREAM_KERNEL(SCOPE, SC_SUFFIX, SC_PREFIX)                               \
    __global__ void barrier_all_on_stream_kernel##SC_SUFFIX () {                                \
        int myidx = nvshmemi_thread_id_in_##SCOPE();                                            \
                                                                                                \
        if (nvshmemi_job_connectivity_d >= NVSHMEMI_JOB_GPU_PROXY) {		                    \
            if (!myidx)                                                                         \
                nvshmemi_proxy_quiet_no_membar();                                               \
            NVSHMEMI_SYNC_##SCOPE();                                                            \
        }                                                                                       \
        nvshmemi_team_t *teami = nvshmemi_team_pool_d[NVSHMEM_TEAM_WORLD];                      \
        long *counter = nvshmemi_team_get_sync_counter(teami);                                  \
        nvshmemi_sync_algo##SC_SUFFIX(teami->start, teami->stride, teami->size,                 \
                                      nvshmemi_team_get_psync(teami, SYNC),                     \
                                      counter);                                                 \
        if (!myidx) {                                                                           \
            if (nvshmemi_job_connectivity_d > NVSHMEMI_JOB_GPU_PROXY)				            \
                nvshmemi_proxy_enforce_consistency_at_target_no_membar();                       \
        }                                                                                       \
    }

BARRIER_ALL_ON_STREAM_KERNEL(thread, , )
BARRIER_ALL_ON_STREAM_KERNEL(warp, _warp, x)
BARRIER_ALL_ON_STREAM_KERNEL(block, _block, x)
#undef BARRIER_ALL_ON_STREAM_KERNEL


#define SYNC_ON_STREAM_KERNEL(SCOPE, SC_SUFFIX, SC_PREFIX)                                                          \
    __global__ void sync_on_stream_kernel##SC_SUFFIX(int start, int stride, int size, long *pSync, long *counter) { \
        nvshmemi_sync_algo##SC_SUFFIX(start, stride, size, pSync, counter);                                         \
    }

SYNC_ON_STREAM_KERNEL(thread, , )
SYNC_ON_STREAM_KERNEL(warp, _warp, x)
SYNC_ON_STREAM_KERNEL(block, _block, x)
#undef SYNC_ON_STREAM_KERNEL

#define SYNC_ALL_ON_STREAM_KERNEL(SCOPE, SC_SUFFIX, SC_PREFIX)                                                      \
    __global__ void sync_all_on_stream_kernel##SC_SUFFIX() {                                                        \
        int myidx = nvshmemi_thread_id_in_##SCOPE();                                                                \
        nvshmemi_team_t *teami = nvshmemi_team_pool_d[NVSHMEM_TEAM_WORLD];                                          \
        long *counter = nvshmemi_team_get_sync_counter(teami);                                                      \
        nvshmemi_sync_algo##SC_SUFFIX(teami->start, teami->stride, teami->size,                                     \
                                      nvshmemi_team_get_psync(teami, SYNC),                                         \
                                      counter);                                                                     \
    }

SYNC_ALL_ON_STREAM_KERNEL(thread, , )
SYNC_ALL_ON_STREAM_KERNEL(warp, _warp, x)
SYNC_ALL_ON_STREAM_KERNEL(block, _block, x)
#undef SYNC_ALL_ON_STREAM_KERNEL

#endif


extern "C" int call_barrier_on_stream_kern(int start, int stride, int size, long *pSync,
                                           long *counter, cudaStream_t stream) {
    int num_blocks = 1;
    int num_threads_per_block;
    if (nvshmemi_job_connectivity <= NVSHMEMI_JOB_GPU_LDST) {
        num_threads_per_block = size - 1; // Have enough threads for alltoall algo
    } else {
        num_threads_per_block = nvshmemi_options.BARRIER_TG_DISSEM_KVAL;
    }

    if (num_threads_per_block <= 32) {
        barrier_on_stream_kernel_warp<<<num_blocks, 32, 0, stream>>>(
            start, stride, size, pSync, counter);
    }
    else {
        barrier_on_stream_kernel_block<<<num_blocks, num_threads_per_block, 0, stream>>>(
            start, stride, size, pSync, counter);
    }

    CUDA_RUNTIME_CHECK(cudaGetLastError());
    return 0;
}


extern "C" int call_sync_on_stream_kern(int start, int stride, int size, long *pSync,
                                        long *counter, cudaStream_t stream) {
    int num_blocks = 1;
    int num_threads_per_block;
    if (nvshmemi_job_connectivity <= NVSHMEMI_JOB_GPU_LDST) {
        num_threads_per_block = size - 1; // Have enough threads for alltoall algo
    } else {
        num_threads_per_block = nvshmemi_options.BARRIER_TG_DISSEM_KVAL;
    }

    if (num_threads_per_block <= 32) {
        sync_on_stream_kernel_warp<<<num_blocks, 32, 0, stream>>>(start, stride,
                                                                  size, pSync, counter);
    } else {
        sync_on_stream_kernel_block<<<num_blocks, num_threads_per_block, 0, stream>>>(start, stride,
                                                                                      size, pSync, counter);
    }
    CUDA_RUNTIME_CHECK(cudaGetLastError());
    return 0;
}
