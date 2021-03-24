/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmem_internal.h"

#ifdef __CUDA_ARCH__

#define NVSHMEM_TYPE_IPUT(NAME, TYPE)                                                        \
    __device__ void nvshmem_##NAME##_iput(TYPE *dest, const TYPE *source, ptrdiff_t dst,     \
                                          ptrdiff_t sst, size_t nelems, int pe) {            \
        void *peer_base_addr =                                                               \
            (void *)__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + pe);        \
        if (peer_base_addr) {                                                                \
            volatile TYPE *dest_actual;                                                      \
            dest_actual = (volatile TYPE *)((char *)(peer_base_addr) +                       \
                                            ((char *)dest - (char *)(nvshmemi_heap_base_d))); \
            int i;                                                                           \
            for (i = 0; i < nelems; i++) {                                                   \
                *(dest_actual + i * dst) = *(source + i * sst);                              \
            }                                                                                \
        } else {                                                                             \
            printf("nvshmem_" #NAME "_iput not implemented over IB\n");                      \
            assert(0);                                                                       \
        }                                                                                    \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEM_TYPE_IPUT)
#undef NVSHMEM_TYPE_IPUT

#define NVSHMEM_IPUTSIZE(NAME, type)                                                         \
    __device__ void nvshmem_iput##NAME(void *dest, const void *source, ptrdiff_t dst,        \
                                       ptrdiff_t sst, size_t nelems, int pe) {               \
        void *peer_base_addr =                                                               \
            (void *)__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + pe);        \
        if (peer_base_addr) {                                                                \
            volatile char *dest_actual;                                                      \
            dest_actual =                                                                    \
                ((char *)(peer_base_addr) + ((char *)dest - (char *)(nvshmemi_heap_base_d))); \
            int i;                                                                           \
            for (i = 0; i < nelems; i++) {                                                   \
                *((type *)dest_actual + i * dst) = *((type *)source + i * sst);              \
            }                                                                                \
        } else {                                                                             \
            printf("nvshmem_iput" #NAME " not implemented over IB\n");                       \
            assert(0);                                                                       \
        }                                                                                    \
    }

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(NVSHMEM_IPUTSIZE)
#undef NVSHMEM_IPUTSIZE

#endif
