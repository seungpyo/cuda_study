/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmemx.h"
#include "nvshmem_internal.h"

#ifdef __CUDA_ARCH__

#define NVSHMEM_TYPE_IPUT_THREADGROUP(Name, Type, Group)                                          \
    __device__ void nvshmemx_##Name##_iput_##Group(Type *dest, const Type *source, ptrdiff_t dst, \
                                                   ptrdiff_t sst, size_t nelems, int pe) {        \
        NVSHMEMI_SYNC_##Group();                                                                  \
        void *peer_base_addr =                                                                    \
            (void *)__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + pe);             \
        if (peer_base_addr) {                                                                     \
            NVSHMEMI_DECL_THREAD_IDX_##Group();                                                   \
            NVSHMEMI_DECL_THREADGROUP_SIZE_##Group();                                             \
            volatile Type *dest_actual;                                                           \
            dest_actual = (volatile Type *)((char *)(peer_base_addr) +                            \
                                            ((char *)dest - (char *)(nvshmemi_heap_base_d)));      \
            int i;                                                                                \
            for (i = myIdx; i < nelems; i += groupSize) {                                         \
                *(dest_actual + i * dst) = *((volatile Type *)source + i * sst);                  \
            }                                                                                     \
            NVSHMEMI_SYNC_##Group();                                                              \
        } else {                                                                                  \
            printf("nvshmemx_" #Name "_iput_" #Group " not implemented over IB\n");               \
            assert(0);                                                                            \
        }                                                                                         \
    }

#define DEFINE_NVSHMEM_TYPE_IPUT_THREADGROUP(Name, Type) \
    NVSHMEM_TYPE_IPUT_THREADGROUP(Name, Type, warp)      \
    NVSHMEM_TYPE_IPUT_THREADGROUP(Name, Type, block)

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DEFINE_NVSHMEM_TYPE_IPUT_THREADGROUP)
#undef DEFINE_NVSHMEM_TYPE_IPUT_THREADGROUP

#define NVSHMEM_IPUTSIZE_THREADGROUP(Name, Type, Group)                                          \
    __device__ void nvshmemx_iput##Name##_##Group(void *dest, const void *source, ptrdiff_t dst, \
                                                  ptrdiff_t sst, size_t nelems, int pe) {        \
        NVSHMEMI_SYNC_##Group();                                                                 \
        void *peer_base_addr =                                                                   \
            (void *)__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + pe);            \
        if (peer_base_addr) {                                                                    \
            NVSHMEMI_DECL_THREAD_IDX_##Group();                                                  \
            NVSHMEMI_DECL_THREADGROUP_SIZE_##Group();                                            \
            volatile Type *dest_actual;                                                          \
            dest_actual = (volatile Type *)((char *)(peer_base_addr) +                           \
                                            ((char *)dest - (char *)(nvshmemi_heap_base_d)));     \
            int i;                                                                               \
            for (i = myIdx; i < nelems; i += groupSize) {                                        \
                *((Type *)dest_actual + i * dst) = *((Type *)source + i * sst);                  \
            }                                                                                    \
            NVSHMEMI_SYNC_##Group();                                                             \
        } else {                                                                                 \
            printf("nvshmemx_iput" #Name "_" #Group " not implemented over IB\n");               \
            assert(0);                                                                           \
        }                                                                                        \
    }

#define DEFINE_NVSHMEM_IPUTSIZE_THREADGROUP(Name, Type) \
    NVSHMEM_IPUTSIZE_THREADGROUP(Name, Type, warp)      \
    NVSHMEM_IPUTSIZE_THREADGROUP(Name, Type, block)

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(DEFINE_NVSHMEM_IPUTSIZE_THREADGROUP)
#undef DEFINE_NVSHMEM_IPUTSIZE_THREADGROUP

#endif
