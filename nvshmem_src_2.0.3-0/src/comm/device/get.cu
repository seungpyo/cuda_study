/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmem_internal.h"

#ifdef __CUDA_ARCH__

#define NVSHMEM_TYPE_IGET(Name, TYPE)                                                            \
    __device__ void nvshmem_##Name##_iget(TYPE *dest, const TYPE *source, ptrdiff_t dst,         \
                                          ptrdiff_t sst, size_t nelems, int pe) {                \
        void *peer_base_addr =                                                                   \
            (void *)__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + pe);            \
        if (peer_base_addr) {                                                                    \
            volatile TYPE *source_actual;                                                        \
            source_actual = (volatile TYPE *)((char *)(peer_base_addr) +                         \
                                              ((char *)source - (char *)(nvshmemi_heap_base_d))); \
            int i;                                                                               \
            for (i = 0; i < nelems; i++) {                                                       \
                *(dest + i * dst) = *(source_actual + i * sst);                                  \
            }                                                                                    \
        } else {                                                                                 \
            printf("nvshmem_" #Name "_iget not implemented over IB\n");                          \
            assert(0);                                                                           \
        }                                                                                        \
    }

NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEM_TYPE_IGET)
#undef NVSHMEM_TYPE_IGET

#define NVSHMEM_IGETSIZE(Name, TYPE)                                                           \
    __device__ void nvshmem_iget##Name(void *dest, const void *source, ptrdiff_t dst,          \
                                       ptrdiff_t sst, size_t nelems, int pe) {                 \
        void *peer_base_addr =                                                                 \
            (void *)__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + pe);          \
        if (peer_base_addr) {                                                                  \
            volatile char *source_actual;                                                      \
            source_actual =                                                                    \
                ((char *)(peer_base_addr) + ((char *)source - (char *)(nvshmemi_heap_base_d))); \
            int i;                                                                             \
            for (i = 0; i < nelems; i++) {                                                     \
                *((TYPE *)dest + i * dst) = *((TYPE *)source_actual + i * sst);                \
            }                                                                                  \
        } else {                                                                               \
            printf("nvshmem_iget" #Name " not implemented over IB\n");                         \
            assert(0);                                                                         \
        }                                                                                      \
    }

NVSHMEMI_REPT_FOR_SIZES_WITH_TYPE(NVSHMEM_IGETSIZE)
#undef NVSHMEM_IGETSIZE

#endif
