/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmemx.h"
#include "nvshmem_internal.h"

#ifdef __CUDA_ARCH__

#define NVSHMEM_TYPE_IGET_THREADGROUP(Name, Type, Group)                                          \
    __device__ void nvshmemx_##Name##_iget_##Group(Type *dest, const Type *source, ptrdiff_t dst, \
                                                   ptrdiff_t sst, size_t nelems, int pe) {        \
        NVSHMEMI_SYNC_##Group();                                                                  \
        void *peer_base_addr =                                                                    \
            (void *)__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + pe);             \
        if (peer_base_addr) {                                                                     \
            NVSHMEMI_DECL_THREAD_IDX_##Group();                                                   \
            NVSHMEMI_DECL_THREADGROUP_SIZE_##Group();                                             \
            volatile Type *source_actual;                                                         \
            source_actual = (volatile Type *)((char *)(peer_base_addr) +                          \
                                              ((char *)source - (char *)(nvshmemi_heap_base_d)));  \
            int i;                                                                                \
            for (i = myIdx; i < nelems; i += groupSize) {                                         \
                *(dest + i * dst) = *(source_actual + i * sst);                                   \
            }                                                                                     \
            NVSHMEMI_SYNC_##Group();                                                              \
        } else {                                                                                  \
            printf("nvshmemx_" #Name "_iget_" #Group " not implemented over IB\n");               \
            assert(0);                                                                            \
        }                                                                                         \
    }

#define DEFINE_NVSHMEM_TYPE_IGET_THREADGROUP(Name, Type) \
    NVSHMEM_TYPE_IGET_THREADGROUP(Name, Type, warp)      \
    NVSHMEM_TYPE_IGET_THREADGROUP(Name, Type, block)

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DEFINE_NVSHMEM_TYPE_IGET_THREADGROUP)
#undef DEFINE_NVSHMEM_TYPE_IGET_THREADGROUP

#define NVSHMEM_IGETSIZE_THREADGROUP(Name, Type, Group)                                          \
    __device__ void nvshmemx_iget##Name##_##Group(void *dest, const void *source, ptrdiff_t dst, \
                                                  ptrdiff_t sst, size_t nelems, int pe) {        \
        NVSHMEMI_SYNC_##Group();                                                                 \
        void *peer_base_addr =                                                                   \
            (void *)__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + pe);            \
        if (peer_base_addr) {                                                                    \
            NVSHMEMI_DECL_THREAD_IDX_##Group();                                                  \
            NVSHMEMI_DECL_THREADGROUP_SIZE_##Group();                                            \
            volatile char *source_actual;                                                        \
            source_actual =                                                                      \
                ((char *)(peer_base_addr) + ((char *)source - (char *)(nvshmemi_heap_base_d)));   \
            int i;                                                                               \
            for (i = myIdx; i < nelems; i += groupSize) {                                        \
                *((Type *)dest + i * dst) = *((Type *)source_actual + i * sst);                  \
            }                                                                                    \
            NVSHMEMI_SYNC_##Group();                                                             \
        } else {                                                                                 \
            printf("nvshmemx_iget" #Name "_" #Group " not implemented over IB\n");               \
            assert(0);                                                                           \
        }                                                                                        \
    }

#define DEFINE_NVSHMEM_IGETSIZE_THREADGROUP(Name, Type) \
    NVSHMEM_IGETSIZE_THREADGROUP(Name, Type, warp)      \
    NVSHMEM_IGETSIZE_THREADGROUP(Name, Type, block)

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(DEFINE_NVSHMEM_IGETSIZE_THREADGROUP)
#undef DEFINE_NVSHMEM_IGETSIZE_THREADGROUP

#endif
