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

#ifndef _NVSHMEMX_DEFINES_H_
#define _NVSHMEMX_DEFINES_H_

#include "nvshmemi_util.h"
#include "nvshmem_common.cuh"

#ifdef __CUDA_ARCH__
template <typename T>
__device__ inline void put_warp(T *dest, const T *source, size_t nelems, int pe) {
    nvshmemi_warp_sync();
    void *peer_base_addr = (void *)__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + pe);
    int myIdx = nvshmemi_thread_id_in_warp();
    int groupSize = nvshmemi_warp_size();
    if (peer_base_addr) {
        volatile T *dest_actual = (volatile T *)((char *)(peer_base_addr) +
                                                 ((char *)dest - (char *)(nvshmemi_heap_base_d)));
        for (size_t i = myIdx; i < nelems; i += groupSize) {
            *(dest_actual + i) = *(source + i);
        }
    } else {
        if (!myIdx) {
            nvshmemi_proxy_rma_nbi<NVSHMEMI_OP_PUT>((void *)dest, (void *)source, nelems * sizeof(T),
                                               pe);
            nvshmemi_proxy_quiet();
        }
    }
    nvshmemi_warp_sync();
}

template <typename T>
__device__ inline void put_threadblock(T *dest, const T *source, size_t nelems, int pe) {
    nvshmemi_block_sync();
    void *peer_base_addr = (void *)__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + pe);
    int myIdx = nvshmemi_thread_id_in_block();
    int groupSize = nvshmemi_block_size();
    if (peer_base_addr) {
        volatile T *dest_actual = (volatile T *)((char *)(peer_base_addr) +
                                                 ((char *)dest - (char *)(nvshmemi_heap_base_d)));
        for (size_t i = myIdx; i < nelems; i += groupSize) {
            *(dest_actual + i) = *(source + i);
        }
    } else {
        if (!myIdx) {
            nvshmemi_proxy_rma_nbi<NVSHMEMI_OP_PUT>((void *)dest, (void *)source, nelems * sizeof(T),
                                               pe);
            nvshmemi_proxy_quiet();
        }
    }
    nvshmemi_block_sync();
}

template <typename T>
__device__ inline void signal(T *dest, const T value, int pe) {
   const void *peer_base_addr =
       (void *)__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + pe);
   if (peer_base_addr != NULL) {
       volatile T *dest_actual = (volatile T *)((char *)(peer_base_addr) +
                              ((char *)dest - (char *)(nvshmemi_heap_base_d)));
       *dest_actual = value;
   } else {
       nvshmemi_proxy_amo_nonfetch<T>((void *)dest, value, pe, NVSHMEMI_AMO_SIGNAL);
   }
}

#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __CUDA_ARCH__

#define NVSHMEM_TYPE_PUT_THREADGROUP(Name, Type, Group)                                  \
    __device__ inline void nvshmemx_##Name##_put_##Group(Type *dest, const Type *source, \
                                                         size_t nelems, int pe) {        \
        NVSHMEMI_SYNC_##Group();                                                         \
        void *peer_base_addr =                                                           \
            (void *)__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + pe);    \
        NVSHMEMI_DECL_THREAD_IDX_##Group();                                              \
        NVSHMEMI_DECL_THREADGROUP_SIZE_##Group();                                        \
        if (peer_base_addr) {                                                            \
            volatile Type *dest_actual =                                                 \
                (volatile Type *)((char *)(peer_base_addr) +                             \
                                  ((char *)dest - (char *)(nvshmemi_heap_base_d)));       \
            for (size_t i = myIdx; i < nelems; i += groupSize) {                         \
                *(dest_actual + i) = *(source + i);                                      \
            }                                                                            \
        } else {                                                                         \
            if (!myIdx) {                                                                \
                nvshmemi_proxy_rma_nbi<NVSHMEMI_OP_PUT>((void *)dest, (void *)source,         \
                                                   nelems * sizeof(Type), pe);           \
                nvshmemi_proxy_quiet();                                                  \
            }                                                                            \
        }                                                                                \
        NVSHMEMI_SYNC_##Group();                                                         \
    }

#define DEFINE_NVSHMEM_TYPE_PUT_THREADGROUP(Name, Type) \
    NVSHMEM_TYPE_PUT_THREADGROUP(Name, Type, warp)      \
    NVSHMEM_TYPE_PUT_THREADGROUP(Name, Type, block)

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DEFINE_NVSHMEM_TYPE_PUT_THREADGROUP)
#undef DEFINE_NVSHMEM_TYPE_PUT_THREADGROUP

#define NVSHMEM_TYPE_GET_THREADGROUP(Name, Type, Group)                                  \
    __device__ inline void nvshmemx_##Name##_get_##Group(Type *dest, const Type *source, \
                                                         size_t nelems, int pe) {        \
        NVSHMEMI_SYNC_##Group();                                                         \
        void *peer_base_addr =                                                           \
            (void *)__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + pe);    \
        NVSHMEMI_DECL_THREAD_IDX_##Group();                                              \
        NVSHMEMI_DECL_THREADGROUP_SIZE_##Group();                                        \
        if (peer_base_addr) {                                                            \
            volatile Type *source_actual =                                               \
                (volatile Type *)((char *)(peer_base_addr) +                             \
                                  ((char *)source - (char *)(nvshmemi_heap_base_d)));     \
            for (size_t i = myIdx; i < nelems; i += groupSize) {                         \
                *(dest + i) = *(source_actual + i);                                      \
            }                                                                            \
        } else {                                                                         \
            if (!myIdx) {                                                                \
                nvshmemi_proxy_rma_nbi<NVSHMEMI_OP_GET>((void *)source, (void *)dest,         \
                                                   nelems * sizeof(Type), pe);           \
                nvshmemi_proxy_quiet();                                                  \
            }                                                                            \
        }                                                                                \
        NVSHMEMI_SYNC_##Group();                                                         \
    }

#define DEFINE_NVSHMEM_TYPE_GET(Name, Type)        \
    NVSHMEM_TYPE_GET_THREADGROUP(Name, Type, warp) \
    NVSHMEM_TYPE_GET_THREADGROUP(Name, Type, block)

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DEFINE_NVSHMEM_TYPE_GET)
#undef DEFINE_NVSHMEM_TYPE_GET

#define NVSHMEM_PUTSIZE_THREADGROUP(Name, Type, Group)                                  \
    __device__ inline void nvshmemx_put##Name##_##Group(void *dest, const void *source, \
                                                        size_t nelems, int pe) {        \
        NVSHMEMI_SYNC_##Group();                                                        \
        void *peer_base_addr =                                                          \
            (void *)__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + pe);   \
        NVSHMEMI_DECL_THREAD_IDX_##Group();                                             \
        NVSHMEMI_DECL_THREADGROUP_SIZE_##Group();                                       \
        if (peer_base_addr) {                                                           \
            volatile Type *dest_actual =                                                \
                (volatile Type *)((char *)(peer_base_addr) +                            \
                                  ((char *)dest - (char *)(nvshmemi_heap_base_d)));      \
            for (size_t i = myIdx; i < nelems; i += groupSize) {                        \
                *((Type *)dest_actual + i) = *((Type *)source + i);                     \
            }                                                                           \
        } else {                                                                        \
            if (!myIdx) {                                                               \
                nvshmemi_proxy_rma_nbi<NVSHMEMI_OP_PUT>((void *)dest, (void *)source,        \
                                                   nelems * sizeof(Type), pe);          \
                nvshmemi_proxy_quiet();                                                 \
            }                                                                           \
        }                                                                               \
        NVSHMEMI_SYNC_##Group();                                                        \
    }

#define DEFINE_NVSHMEM_PUTSIZE_THREADGROUP(Name, Type) \
    NVSHMEM_PUTSIZE_THREADGROUP(Name, Type, warp)      \
    NVSHMEM_PUTSIZE_THREADGROUP(Name, Type, block)

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(DEFINE_NVSHMEM_PUTSIZE_THREADGROUP)
#undef DEFINE_NVSHMEM_PUTSIZE_THREADGROUP

#define NVSHMEM_GETSIZE_THREADGROUP(Name, Type, Group)                                         \
    __device__ inline void nvshmemx_get##Name##_##Group(void *dest, const void *source,        \
                                                        size_t nelems, int pe) {               \
        NVSHMEMI_SYNC_##Group();                                                               \
        void *peer_base_addr =                                                                 \
            (void *)__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + pe);          \
        NVSHMEMI_DECL_THREAD_IDX_##Group();                                                    \
        NVSHMEMI_DECL_THREADGROUP_SIZE_##Group();                                              \
        if (peer_base_addr) {                                                                  \
            volatile char *source_actual =                                                     \
                ((char *)(peer_base_addr) + ((char *)source - (char *)(nvshmemi_heap_base_d))); \
            for (size_t i = myIdx; i < nelems; i += groupSize) {                               \
                *((Type *)dest + i) = *((Type *)source_actual + i);                            \
            }                                                                                  \
        } else {                                                                               \
            if (!myIdx) {                                                                      \
                nvshmemi_proxy_rma_nbi<NVSHMEMI_OP_GET>((void *)source, (void *)dest,               \
                                                   nelems * sizeof(Type), pe);                 \
                nvshmemi_proxy_quiet();                                                        \
            }                                                                                  \
        }                                                                                      \
        NVSHMEMI_SYNC_##Group();                                                               \
    }

#define DEFINE_NVSHMEM_GETSIZE_THREADGROUP(Name, Type) \
    NVSHMEM_GETSIZE_THREADGROUP(Name, Type, warp)      \
    NVSHMEM_GETSIZE_THREADGROUP(Name, Type, block)

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(DEFINE_NVSHMEM_GETSIZE_THREADGROUP)
#undef DEFINE_NVSHMEM_GETSIZE_THREADGROUP

#define DEFINE_NVSHMEM_PUTMEM_THREADGROUP(Group)                                                 \
    __device__ inline void nvshmemx_putmem_##Group(void *dest, const void *source, size_t bytes, \
                                                   int pe) {                                     \
        NVSHMEMI_SYNC_##Group();                                                                 \
        void *peer_base_addr =                                                                   \
            (void *)__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + pe);            \
        NVSHMEMI_DECL_THREAD_IDX_##Group();                                                      \
        NVSHMEMI_DECL_THREADGROUP_SIZE_##Group();                                                \
        if (peer_base_addr) {                                                                    \
            volatile char *dest_actual =                                                         \
                ((char *)(peer_base_addr) + ((char *)dest - (char *)(nvshmemi_heap_base_d)));     \
            for (size_t i = myIdx; i < bytes; i += groupSize) {                                  \
                *(dest_actual + i) = *((volatile char *)source + i);                             \
            }                                                                                    \
        } else {                                                                                 \
            if (!myIdx) {                                                                        \
                nvshmemi_proxy_rma_nbi<NVSHMEMI_OP_PUT>((void *)dest, (void *)source, bytes, pe);     \
                nvshmemi_proxy_quiet();                                                          \
            }                                                                                    \
        }                                                                                        \
        NVSHMEMI_SYNC_##Group();                                                                 \
    }

DEFINE_NVSHMEM_PUTMEM_THREADGROUP(warp)
DEFINE_NVSHMEM_PUTMEM_THREADGROUP(block)

#define DEFINE_NVSHMEM_GETMEM_THREADGROUP(Group)                                                 \
    __device__ inline void nvshmemx_getmem_##Group(void *dest, const void *source, size_t bytes, \
                                                   int pe) {                                     \
        NVSHMEMI_SYNC_##Group();                                                                 \
        void *peer_base_addr =                                                                   \
            (void *)__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + pe);            \
        NVSHMEMI_DECL_THREAD_IDX_##Group();                                                      \
        NVSHMEMI_DECL_THREADGROUP_SIZE_##Group();                                                \
        if (peer_base_addr) {                                                                    \
            volatile char *source_actual =                                                       \
                ((char *)(peer_base_addr) + ((char *)source - (char *)(nvshmemi_heap_base_d)));   \
            for (size_t i = myIdx; i < bytes; i += groupSize) {                                  \
                *((char *)dest + i) = *((char *)source_actual + i);                              \
            }                                                                                    \
        } else {                                                                                 \
            if (!myIdx) {                                                                        \
                nvshmemi_proxy_rma_nbi<NVSHMEMI_OP_GET>((void *)source, (void *)dest, bytes, pe);     \
                nvshmemi_proxy_quiet();                                                          \
            }                                                                                    \
        }                                                                                        \
        NVSHMEMI_SYNC_##Group();                                                                 \
    }

DEFINE_NVSHMEM_GETMEM_THREADGROUP(warp)
DEFINE_NVSHMEM_GETMEM_THREADGROUP(block)

#define NVSHMEM_TYPE_PUT_NBI_THREADGROUP(Name, Type, Group)                                  \
    __device__ inline void nvshmemx_##Name##_put_nbi_##Group(Type *dest, const Type *source, \
                                                             size_t nelems, int pe) {        \
        NVSHMEMI_SYNC_##Group();                                                             \
        NVSHMEMI_DECL_THREAD_IDX_##Group();                                                  \
        NVSHMEMI_DECL_THREADGROUP_SIZE_##Group();                                            \
        void *peer_base_addr =                                                               \
            (void *)__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + pe);        \
        if (peer_base_addr) {                                                                \
            volatile Type *dest_actual =                                                     \
                (volatile Type *)((char *)(peer_base_addr) +                                 \
                                  ((char *)dest - (char *)(nvshmemi_heap_base_d)));           \
            for (size_t i = myIdx; i < nelems; i += groupSize) {                             \
                *(dest_actual + i) = *((volatile Type *)source + i);                         \
            }                                                                                \
        } else {                                                                             \
            if (!myIdx) {                                                                    \
                nvshmemi_proxy_rma_nbi<NVSHMEMI_OP_PUT>((void *)dest, (void *)source,             \
                                                   nelems * sizeof(Type), pe);               \
            }                                                                                \
        }                                                                                    \
        NVSHMEMI_SYNC_##Group();                                                             \
    }

#define DEFINE_NVSHMEM_TYPE_PUT_NBI_THREADGROUP(Name, Type) \
    NVSHMEM_TYPE_PUT_NBI_THREADGROUP(Name, Type, warp)      \
    NVSHMEM_TYPE_PUT_NBI_THREADGROUP(Name, Type, block)

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DEFINE_NVSHMEM_TYPE_PUT_NBI_THREADGROUP)
#undef DEFINE_NVSHMEM_TYPE_PUT_NBI_THREADGROUP

#define NVSHMEM_TYPE_GET_NBI_THREADGROUP(Name, Type, Group)                                  \
    __device__ inline void nvshmemx_##Name##_get_nbi_##Group(Type *dest, const Type *source, \
                                                             size_t nelems, int pe) {        \
        NVSHMEMI_SYNC_##Group();                                                             \
        void *peer_base_addr =                                                               \
            (void *)__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + pe);        \
        NVSHMEMI_DECL_THREAD_IDX_##Group();                                                  \
        NVSHMEMI_DECL_THREADGROUP_SIZE_##Group();                                            \
        if (peer_base_addr) {                                                                \
            volatile Type *source_actual =                                                   \
                (volatile Type *)((char *)(peer_base_addr) +                                 \
                                  ((char *)source - (char *)(nvshmemi_heap_base_d)));         \
            for (size_t i = myIdx; i < nelems; i += groupSize) {                             \
                *(dest + i) = *(source_actual + i);                                          \
            }                                                                                \
        } else {                                                                             \
            if (!myIdx) {                                                                    \
                nvshmemi_proxy_rma_nbi<NVSHMEMI_OP_GET>((void *)source, (void *)dest,             \
                                                   nelems * sizeof(Type), pe);               \
            }                                                                                \
        }                                                                                    \
        NVSHMEMI_SYNC_##Group();                                                             \
    }

#define DEFINE_NVSHMEM_TYPE_GET_NBI_THREADGROUP(Name, Type) \
    NVSHMEM_TYPE_GET_NBI_THREADGROUP(Name, Type, warp)      \
    NVSHMEM_TYPE_GET_NBI_THREADGROUP(Name, Type, block)

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DEFINE_NVSHMEM_TYPE_GET_NBI_THREADGROUP)
#undef DEFINE_NVSHMEM_TYPE_GET_NBI_THREADGROUP

#define NVSHMEM_PUTSIZE_NBI_THREADGROUP(Name, Type, Group)                                  \
    __device__ inline void nvshmemx_put##Name##_nbi_##Group(void *dest, const void *source, \
                                                            size_t nelems, int pe) {        \
        NVSHMEMI_SYNC_##Group();                                                            \
        NVSHMEMI_DECL_THREAD_IDX_##Group();                                                 \
        NVSHMEMI_DECL_THREADGROUP_SIZE_##Group();                                           \
        void *peer_base_addr =                                                              \
            (void *)__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + pe);       \
        if (peer_base_addr) {                                                               \
            volatile Type *dest_actual =                                                    \
                (volatile Type *)((char *)(peer_base_addr) +                                \
                                  ((char *)dest - (char *)(nvshmemi_heap_base_d)));          \
            for (size_t i = myIdx; i < nelems; i += groupSize) {                            \
                *((Type *)dest_actual + i) = *((Type *)source + i);                         \
            }                                                                               \
        } else {                                                                            \
            if (!myIdx) {                                                                   \
                nvshmemi_proxy_rma_nbi<NVSHMEMI_OP_PUT>((void *)dest, (void *)source,            \
                                                   nelems * sizeof(Type), pe);              \
            }                                                                               \
        }                                                                                   \
        NVSHMEMI_SYNC_##Group();                                                            \
    }

#define DEFINE_NVSHMEM_PUTSIZE_NBI_THREADGROUP(Name, Type) \
    NVSHMEM_PUTSIZE_NBI_THREADGROUP(Name, Type, warp)      \
    NVSHMEM_PUTSIZE_NBI_THREADGROUP(Name, Type, block)

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(DEFINE_NVSHMEM_PUTSIZE_NBI_THREADGROUP)
#undef DEFINE_NVSHMEM_PUTSIZE_NBI_THREADGROUP

#define NVSHMEM_GETSIZE_NBI_THREADGROUP(Name, Type, Group)                                  \
    __device__ inline void nvshmemx_get##Name##_nbi_##Group(void *dest, const void *source, \
                                                            size_t nelems, int pe) {        \
        NVSHMEMI_SYNC_##Group();                                                            \
        void *peer_base_addr =                                                              \
            (void *)__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + pe);       \
        NVSHMEMI_DECL_THREAD_IDX_##Group();                                                 \
        NVSHMEMI_DECL_THREADGROUP_SIZE_##Group();                                           \
        if (peer_base_addr) {                                                               \
            volatile Type *source_actual =                                                  \
                (volatile Type *)((char *)(peer_base_addr) +                                \
                                  ((char *)source - (char *)(nvshmemi_heap_base_d)));        \
            for (size_t i = myIdx; i < nelems; i += groupSize) {                            \
                *((Type *)dest + i) = *((Type *)source_actual + i);                         \
            }                                                                               \
        } else {                                                                            \
            if (!myIdx) {                                                                   \
                nvshmemi_proxy_rma_nbi<NVSHMEMI_OP_GET>((void *)source, (void *)dest,            \
                                                   nelems * sizeof(Type), pe);              \
            }                                                                               \
        }                                                                                   \
        NVSHMEMI_SYNC_##Group();                                                            \
    }

#define DEFINE_NVSHMEM_GETSIZE_NBI_THREADGROUP(Name, Type) \
    NVSHMEM_GETSIZE_NBI_THREADGROUP(Name, Type, warp)      \
    NVSHMEM_GETSIZE_NBI_THREADGROUP(Name, Type, block)

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(DEFINE_NVSHMEM_GETSIZE_NBI_THREADGROUP)
#undef DEFINE_NVSHMEM_GETSIZE_NBI_THREADGROUP

#define DEFINE_NVSHMEM_PUTMEM_NBI_THREADGROUP(Group)                                         \
    __device__ inline void nvshmemx_putmem_nbi_##Group(void *dest, const void *source,       \
                                                       size_t bytes, int pe) {               \
        NVSHMEMI_SYNC_##Group();                                                             \
        NVSHMEMI_DECL_THREAD_IDX_##Group();                                                  \
        NVSHMEMI_DECL_THREADGROUP_SIZE_##Group();                                            \
        void *peer_base_addr =                                                               \
            (void *)__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + pe);        \
        if (peer_base_addr) {                                                                \
            volatile char *dest_actual =                                                     \
                (volatile char *)((char *)(peer_base_addr) +                                 \
                                  ((char *)dest - (char *)(nvshmemi_heap_base_d)));           \
            for (size_t i = myIdx; i < bytes; i += groupSize) {                              \
                *(dest_actual + i) = *((volatile char *)source + i);                         \
            }                                                                                \
        } else {                                                                             \
            if (!myIdx) {                                                                    \
                nvshmemi_proxy_rma_nbi<NVSHMEMI_OP_PUT>((void *)dest, (void *)source, bytes, pe); \
            }                                                                                \
        }                                                                                    \
        NVSHMEMI_SYNC_##Group();                                                             \
    }

DEFINE_NVSHMEM_PUTMEM_NBI_THREADGROUP(warp)
DEFINE_NVSHMEM_PUTMEM_NBI_THREADGROUP(block)

#define DEFINE_NVSHMEM_GETMEM_NBI_THREADGROUP(Group)                                           \
    __device__ inline void nvshmemx_getmem_nbi_##Group(void *dest, const void *source,         \
                                                       size_t bytes, int pe) {                 \
        NVSHMEMI_SYNC_##Group();                                                               \
        void *peer_base_addr =                                                                 \
            (void *)__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + pe);          \
        NVSHMEMI_DECL_THREAD_IDX_##Group();                                                    \
        NVSHMEMI_DECL_THREADGROUP_SIZE_##Group();                                              \
        if (peer_base_addr) {                                                                  \
            volatile char *source_actual =                                                     \
                ((char *)(peer_base_addr) + ((char *)source - (char *)(nvshmemi_heap_base_d))); \
            for (size_t i = myIdx; i < bytes; i += groupSize) {                                \
                *((char *)dest + i) = *((char *)source_actual + i);                            \
            }                                                                                  \
        } else {                                                                               \
            if (!myIdx) {                                                                      \
                nvshmemi_proxy_rma_nbi<NVSHMEMI_OP_GET>((void *)source, (void *)dest, bytes, pe);   \
            }                                                                                  \
        }                                                                                      \
        NVSHMEMI_SYNC_##Group();                                                               \
    }

DEFINE_NVSHMEM_GETMEM_NBI_THREADGROUP(warp)
DEFINE_NVSHMEM_GETMEM_NBI_THREADGROUP(block)

/* __device__ nvshmem_signal */
#define DEFINE_NVSHMEMX_TYPE_SIGNAL(TYPENAME, TYPE)                                             \
    __device__ inline void nvshmemx_##TYPENAME##_signal(TYPE *dest, const TYPE value, int pe) { \
        signal<TYPE>(dest, value, pe);                                                          \
}

NVSHMEMX_REPT_FOR_SIGNAL_TYPES(DEFINE_NVSHMEMX_TYPE_SIGNAL)
#undef DEFINE_NVSHMEMX_TYPE_SIGNAL

__device__ inline void nvshmemx_signal_op(uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) {
   const void *peer_base_addr =
       (void *)__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + pe);
   if (sig_op == NVSHMEM_SIGNAL_SET && peer_base_addr != NULL) {
       volatile uint64_t *dest_actual = (volatile uint64_t *)((char *)(peer_base_addr) +
                              ((char *)sig_addr - (char *)(nvshmemi_heap_base_d)));
       *dest_actual = signal;
   }
   else if (nvshmemi_job_connectivity_d <= NVSHMEMI_JOB_GPU_LDST_ATOMICS) {
       volatile uint64_t *dest_actual = (volatile uint64_t *)((char *)(peer_base_addr) +
                              ((char *)sig_addr - (char *)(nvshmemi_heap_base_d)));
        /* sig_op == NVSHMEM_SIGNAL_ADD */
        atomicAdd((unsigned long long *)dest_actual, signal);
   } else {
        nvshmemi_proxy_amo_nonfetch<uint64_t>((void *)sig_addr, signal, pe, (nvshmemi_amo_t)sig_op);
   }
}


#endif /* __CUDA_ARCH__ */

#ifdef __cplusplus
}
#endif
#endif
