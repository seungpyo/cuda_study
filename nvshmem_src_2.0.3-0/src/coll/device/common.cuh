/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef _GPU_COMMON_H_
#define _GPU_COMMON_H_

#define GPU_BITS_COPY_DIRECT(TYPENAME, TYPE, dest, src, nelems)               \
    do {                                                                      \
        int i;                                                                \
        for (i = 0; i < nelems; i++) {                                        \
            *((TYPE *)dest + i) = *((TYPE *)src + i);                         \
        }                                                                     \
    } while (0)

#define GPU_BITS_COPY_THREADGROUP_DIRECT(TYPENAME, TYPE, dest, src, nelems, myIdx, groupSize) \
    do {                                                                              \
        int i;                                                                        \
        for (i = myIdx; i < nelems; i += groupSize) {                                 \
            *((TYPE *)dest + i) = *((TYPE *)src + i);                                 \
        }                                                                             \
    } while (0)

#define GPU_BITS_REG_COPY(SUFFIX, dest, src, nelems)            \
    do {                                                        \
        int i;                                                  \
        uint64_t tmp;                                           \
        uint32_t *header = NULL;                                \
        TYPE *payload = (TYPE *)(&tmp);                         \
        header = ((uint32_t *)payload + 1);                     \
        *header = 1;                                            \
        for (i = 0; i < nelems; i++) {                          \
            *payload = *((TYPE *)src + i);                      \
            *((uint64_t *)dest + i) = tmp;                      \
        }                                                       \
    } while (0)

#define GPU_BITS_REG_CHECK(TYPENAME, TYPE, dest, src, nelems)    \
    do {                                                 \
        int i;                                           \
        assert(sizeof(TYPE) >= 4);                       \
        int subelems = ((sizeof(TYPE) * 8) / 32);        \
        volatile uint32_t *header = NULL;                \
        uint32_t *payload = NULL;                        \
        for (i = 0; i < nelems * subelems; i++) {        \
            payload = (uint32_t *)((uint64_t *)src + i); \
            header = (uint32_t *)payload + 1;            \
            while (1 != *header)                         \
                ;                                        \
            *((uint32_t *)dest + i) = *payload;          \
            *((uint64_t *)src + i) = NVSHMEM_SYNC_VALUE; \
        }                                                \
    } while (0)

#define GPU_BITS_REG_CHECK_THREADGROUP(TYPENAME, TYPE, dest, src, nelems, myIdx, groupSize) \
    do {                                                                                    \
        int i;                                                                              \
        int subelems = ((sizeof(TYPE) * 8) / 32);                                           \
        int total_bytes = nelems * sizeof(TYPE);                                            \
        int num_signals = (total_bytes + sizeof(uint32_t) - 1) / sizeof(uint32_t);          \
        volatile uint32_t *header = NULL;                                                   \
        uint32_t *payload = NULL;                                                           \
        for (i = myIdx; i < num_signals - 1; i += groupSize) {                              \
            payload = (uint32_t *)((uint64_t *)src + i);                                    \
            header = (uint32_t *)payload + 1;                                               \
            while (1 != *header)                                                            \
                ;                                                                           \
            *((uint32_t *)dest + i) = *payload;                                             \
            *((uint64_t *)src + i) = NVSHMEM_SYNC_VALUE;                                    \
        }                                                                                   \
        if (!myIdx) {                                                                       \
            header = (uint32_t *)((uint64_t *) src + (num_signals - 1)) + 1;                \
            while (1 != *header)                                                            \
                ;                                                                           \
            memcpy((uint32_t *)dest + (num_signals - 1),                                    \
                   (uint64_t *)src + (num_signals - 1),                                     \
                   total_bytes - (num_signals - 1) * sizeof(uint32_t));                     \
            *((uint64_t *)src + (num_signals - 1)) = NVSHMEM_SYNC_VALUE;                    \
        }                                                                                   \
    } while (0)

#define perform_gpu_rd_sum(result, op1, op2) result = op1 + op2
#define perform_gpu_rd_prod(result, op1, op2) result = op1 * op2
#define perform_gpu_rd_and(result, op1, op2) result = op1 & op2
#define perform_gpu_rd_or(result, op1, op2) result = op1 | op2
#define perform_gpu_rd_xor(result, op1, op2) result = op1 ^ op2
#define perform_gpu_rd_min(result, op1, op2) result = (op1 > op2) ? op2 : op1
#define perform_gpu_rd_max(result, op1, op2) result = (op1 > op2) ? op1 : op2

#endif
