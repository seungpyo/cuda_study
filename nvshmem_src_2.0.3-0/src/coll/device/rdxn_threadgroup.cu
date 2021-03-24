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

#define GPU_HEAD_CHECK_OP_THREADGROUP(TYPENAME, TYPE, OP, dest, src, actual_src, nelems, myIdx, groupSize) \
    do {                                                                                         \
        int i, k;                                                                                \
        int subelems = sizeof(TYPE) / sizeof(uint32_t);                                          \
        volatile uint32_t *header = NULL;                                                        \
        TYPE tmp;                                                                                \
        uint32_t *tmp_ptr = (uint32_t *)&tmp;                                                    \
        uint32_t *payload = NULL;                                                                \
        for (i = myIdx; i < nelems; i += groupSize) {                                            \
            for (k = 0; k < subelems; k++) {                                                     \
                payload = (uint32_t *)((uint64_t *)src + (i * subelems) + k);                    \
                header = (uint32_t *)payload + 1;                                                \
                while (1 != *header)                                                             \
                    ;                                                                            \
                *header = 0;                                                                     \
                *(tmp_ptr + k) = *payload;                                                       \
            }                                                                                    \
            perform_gpu_rd_##OP(*((TYPE *)dest + i), *((TYPE *)actual_src + i), tmp);            \
        }                                                                                        \
    } while (0)

#define GPU_HEAD_CHECKALL_OP_THREADGROUP(TYPENAME, TYPE, OP, dest, src, actual_src, nelems, start, \
                                         stride, size, myIdx, groupSize)                    \
    do {                                                                                    \
        int i, j, k;                                                                        \
        int subelems = sizeof(TYPE) / sizeof(uint32_t);                                     \
        volatile uint32_t *header = NULL;                                                   \
        TYPE tmp;                                                                           \
        uint32_t *tmp_ptr = (uint32_t *)&tmp;                                               \
        uint32_t *payload = NULL;                                                           \
        int my_active_set_pe = ((nvshmemi_mype_d - start) / stride);                        \
        TYPE *src_ptr = (TYPE *)actual_src;                                                 \
                                                                                            \
        for (j = (my_active_set_pe - 1); j >= 0; j--) {                                     \
            for (i = myIdx; i < nelems; i += groupSize) {                                   \
                for (k = 0; k < subelems; k++) {                                            \
                    payload = (uint32_t *)((uint64_t *)src + (i * subelems) + k +           \
                                           (nelems * subelems * j));                        \
                    header = (uint32_t *)payload + 1;                                       \
                    while (1 != *header)                                                    \
                        ;                                                                   \
                    *header = 0;                                                            \
                    *(tmp_ptr + k) = *payload;                                              \
                }                                                                           \
                perform_gpu_rd_##OP(*((TYPE *)dest + i), *((TYPE *)src_ptr + i), tmp);      \
            }                                                                               \
            src_ptr = dest;                                                                 \
        }                                                                                   \
        for (j = size - 1; j > my_active_set_pe; j--) {                                  \
            for (i = myIdx; i < nelems; i += groupSize) {                                   \
                for (k = 0; k < subelems; k++) {                                            \
                    payload = (uint32_t *)((uint64_t *)src + (i * subelems) + k +           \
                                           (nelems * subelems * j));                        \
                    header = (uint32_t *)payload + 1;                                       \
                    while (1 != *header)                                                    \
                        ;                                                                   \
                    *header = 0;                                                            \
                    *(tmp_ptr + k) = *payload;                                              \
                }                                                                           \
                perform_gpu_rd_##OP(*((TYPE *)dest + i), *((TYPE *)src_ptr + i), tmp);      \
            }                                                                               \
            src_ptr = dest;                                                                 \
        }                                                                                   \
    } while (0)

#define GPU_LINEAR_REDUCE_THREADGROUP(SC, TYPENAME, TYPE, OP, x, y, z, nelems, myIdx, groupSize)     \
    do {                                                                               \
        int i;                                                                         \
        for (i = myIdx; i < nelems; i += groupSize) {                                  \
            perform_gpu_rd_##OP(*((TYPE *)z + i), *((TYPE *)x + i), *((TYPE *)y + i)); \
        }                                                                              \
    } while (0)

#define GPU_LINEAR_REDUCE_THREADGROUP_P2P_GET(SC, TYPENAME, TYPE, OP, x, y, next_rank, z, nelems, myIdx,   \
                                              groupSize, start, stride, size, pWrk, pSync) \
    do {                                                                                         \
        int i;                                                                                   \
        int group_nelems = ((nelems / groupSize) * groupSize);                                   \
        int excess = nelems - group_nelems;                                                      \
        long counter = 1;                                                                        \
        for (i = myIdx; i < group_nelems; i += groupSize) {                                      \
            nvshmem_##TYPENAME##_get((TYPE *)((TYPE *)pWrk + myIdx), (TYPE *)((TYPE *)y + i), \
                                 1, next_rank);                                                  \
            nvshmemxi_barrier_##SC(start, stride, size, pSync, &counter);                        \
            perform_gpu_rd_##OP(*((TYPE *)z + i), *((TYPE *)x + i),                              \
                                *((TYPE *)pWrk + myIdx));                                 \
        }                                                                                        \
        if (excess) {                                                                            \
            if (i < nelems) {                                                                    \
                nvshmem_##TYPENAME##_get((TYPE *)((TYPE *)pWrk + myIdx),                  \
                                     (TYPE *)((TYPE *)y + i), 1, next_rank);                     \
            }                                                                                    \
            nvshmemxi_barrier_##SC(start, stride, size, pSync, &counter);                        \
            if (i < nelems) {                                                                    \
                perform_gpu_rd_##OP(*((TYPE *)z + i), *((TYPE *)x + i),                          \
                                    *((TYPE *)pWrk + myIdx));                             \
            }                                                                                    \
        }                                                                                        \
        NVSHMEMI_SYNC_##SC();                                                                     \
        int end = start + stride * size;                                                         \
        for (i = myIdx; i < end; i += groupSize)                                                 \
            pSync[i] = NVSHMEMI_SYNC_VALUE;                                                       \
        NVSHMEMI_SYNC_##SC();                                                                     \
    } while (0)

#define GPU_LINEAR_REDUCE_THREADGROUP_P2P_PUT(SC, TYPENAME, TYPE, OP, x, y, next_rank, z, offset, nelems,  \
                                              myIdx, groupSize, start, stride, size, pWrk, pSync) \
    do {                                                                                         \
        int i;                                                                                   \
        int group_nelems = ((nelems / groupSize) * groupSize);                                   \
        int excess = nelems - group_nelems;                                                      \
        long counter = 1;                                                                        \
        for (i = myIdx; i < group_nelems; i += groupSize) {                                      \
            nvshmem_##TYPENAME##_put_nbi(                                                        \
                (TYPE *)((TYPE *)pWrk + (myIdx + ((offset & 1) * groupSize))),                   \
                (const TYPE *)((TYPE *)y + i), 1, next_rank);                                    \
            nvshmemxi_barrier_##SC(start, stride, size, pSync, &counter);                        \
            perform_gpu_rd_##OP(*((TYPE *)z + i), *((TYPE *)x + i),                              \
                                *((TYPE *)pWrk + (myIdx + ((offset & 1) * groupSize))));         \
        }                                                                                        \
        offset++;                                                                                \
        if (excess) {                                                                            \
            if (i < nelems) {                                                                    \
                nvshmem_##TYPENAME##_put_nbi(                                                    \
                    (TYPE *)((TYPE *)pWrk + (myIdx + ((offset & 1) * groupSize))),               \
                    (const TYPE *)((TYPE *)y + i), 1, next_rank);                                \
            }                                                                                    \
            nvshmemxi_barrier_##SC(start, stride, size, pSync, &counter);                        \
            if (i < nelems) {                                                                    \
                perform_gpu_rd_##OP(                                                             \
                    *((TYPE *)z + i), *((TYPE *)x + i),                                          \
                    *((TYPE *)pWrk + (myIdx + ((offset & 1) * groupSize))));                     \
            }                                                                                    \
        }                                                                                        \
        offset++;                                                                                \
        int end = start + stride * size;                                                         \
        NVSHMEMI_SYNC_##SC();                                                                     \
        for (i = myIdx; i < end; i += groupSize)                                                 \
            pSync[i] = NVSHMEMI_SYNC_VALUE;                                                      \
        NVSHMEMI_SYNC_##SC();                                                                     \
    } while (0)

#define GPU_LINEAR_REDUCE_THREADGROUP_P2P_PUT_DIRECT(SC, TYPENAME, TYPE, OP, x, y, next_rank, z, offset,  \
                                                     nelems, myIdx, groupSize, start,           \
                                                     stride, size, pWrk, pSync)                 \
    do {                                                                                        \
        int i;                                                                                  \
        int group_nelems = ((nelems / groupSize) * groupSize);                                  \
        int excess = nelems - group_nelems;                                                     \
        long counter = 1;                                                                       \
        for (i = myIdx; i < group_nelems; i += groupSize) {                                     \
            *((TYPE *)((TYPE *)pWrk + (myIdx + ((offset & 1) * groupSize)))) =                  \
                *((TYPE *)((TYPE *)y + i));                                                     \
            nvshmemxi_barrier_##SC(start, stride, size, pSync, &counter);                       \
            perform_gpu_rd_##OP(*((TYPE *)z + i), *((TYPE *)x + i),                             \
                                *((TYPE *)pWrk + (myIdx + ((offset & 1) * groupSize))));        \
        }                                                                                       \
        offset++;                                                                               \
        if (excess) {                                                                           \
            if (i < nelems) {                                                                   \
                *((TYPE *)((TYPE *)pWrk + (myIdx + ((offset & 1) * groupSize)))) =              \
                    *((TYPE *)((TYPE *)y + i));                                                 \
            }                                                                                   \
            nvshmemxi_barrier_##SC(start, stride, size, pSync, &counter);                       \
            if (i < nelems) {                                                                   \
                perform_gpu_rd_##OP(                                                            \
                    *((TYPE *)z + i), *((TYPE *)x + i),                                         \
                    *((TYPE *)pWrk + (myIdx + ((offset & 1) * groupSize))));                    \
            }                                                                                   \
        }                                                                                       \
        offset++;                                                                               \
        int end = start + stride * size;                                                        \
        NVSHMEMI_SYNC_##SC();                                                                     \
        for (i = myIdx; i < end; i += groupSize)                                                \
            pSync[i] = NVSHMEMI_SYNC_VALUE;                                                     \
        NVSHMEMI_SYNC_##SC();                                                                     \
    } while (0)

#define NVSHMEMXI_GPU_RDXN_THREADGROUP_ZCOPY_GET_BAR_DIRECT(                                       \
    SC, TYPENAME, TYPE, OP, dest, source, nreduce, start, stride, size, pWrk, pSync)             \
    do {                                                                                           \
        int next_rank = -1;                                                                        \
        int src_offset = -1;                                                                       \
        int next_offset = -1;                                                                      \
        char *base = NULL;                                                                         \
        char *peer_base = NULL;                                                                    \
        char *peer_source = NULL;                                                                  \
        NVSHMEMI_DECL_THREAD_IDX_##SC();                                                           \
        NVSHMEMI_DECL_THREADGROUP_SIZE_##SC();                                                     \
        int i;                                                                                     \
        int my_active_set_pe = ((nvshmemi_mype_d - start) / stride);                               \
                                                                                                   \
        base = (char *)((void *)__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d +      \
                                      nvshmemi_mype_d));                                           \
        src_offset = ((char *)source - base);                                                      \
                                                                                                   \
        next_rank = start + ((my_active_set_pe + 1) % size) * stride;                              \
        next_offset = src_offset;                                                                  \
        peer_base = (char *)((void *)__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d +  \
                                           next_rank));                                             \
        peer_source = peer_base + next_offset;                                                      \
        GPU_LINEAR_REDUCE_THREADGROUP(SC, TYPENAME, TYPE, OP, (void *)source, peer_source, dest, nreduce, myIdx, \
                                      groupSize);                                                   \
                                                                                                    \
        for (i = 2; i < size; i++) {                                                                \
            next_rank = start + ((my_active_set_pe + i) % size) * stride;                           \
            next_offset = src_offset;                                                               \
            peer_base = (char *)((void *)__ldg(                                                     \
                (const long long unsigned *)nvshmemi_peer_heap_base_d + next_rank));                \
            peer_source = peer_base + next_offset;                                                  \
            GPU_LINEAR_REDUCE_THREADGROUP(SC, TYPENAME, TYPE, OP, dest, peer_source, dest, nreduce, myIdx,       \
                                          groupSize);                                              \
        }                                                                                          \
        nvshmemxi_barrier_##SC(start, stride, size, pSync, NULL);                                  \
    } while (0)

// barrier limited
#define NVSHMEMXI_GPU_RDXN_THREADGROUP_GET_BAR(SC, TYPENAME, TYPE, OP, dest, source, nreduce, start,    \
                                               stride, size, pWrk, pSync)                        \
    do {                                                                                         \
        int next_rank = -1;                                                                      \
        NVSHMEMI_DECL_THREAD_IDX_##SC();                                                         \
        NVSHMEMI_DECL_THREADGROUP_SIZE_##SC();                                                   \
        int i;                                                                                   \
        int my_active_set_pe = ((nvshmemi_mype_d - start) / stride);                             \
                                                                                                 \
        next_rank = start + ((my_active_set_pe + 1) % size) * stride;                            \
        GPU_LINEAR_REDUCE_THREADGROUP_P2P_GET(SC, TYPENAME, TYPE, OP, source, source, next_rank, dest,     \
                                              nreduce, myIdx, groupSize, start, stride,        \
                                              size, pWrk, pSync);                              \
                                                                                               \
        for (i = 2; i < size; i++) {                                                           \
            next_rank = start + ((my_active_set_pe + i) % size) * stride;                      \
            GPU_LINEAR_REDUCE_THREADGROUP_P2P_GET(SC, TYPENAME, TYPE, OP, dest, source, next_rank, dest,   \
                                                  nreduce, myIdx, groupSize, start,              \
                                                  stride, size, pWrk, pSync);                    \
        }                                                                                        \
        NVSHMEMI_SYNC_##SC();                                                                    \
    } while (0)

// barrier limited
#define NVSHMEMXI_GPU_RDXN_THREADGROUP_PUT_BAR(SC, TYPENAME, TYPE, OP, dest, source, nreduce, start,   \
                                               stride, size, pWrk, pSync)              \
    do {                                                                                        \
        int next_rank = -1;                                                                     \
        int counter = 0;                                                                        \
        NVSHMEMI_DECL_THREAD_IDX_##SC();                                                        \
        NVSHMEMI_DECL_THREADGROUP_SIZE_##SC();                                                  \
        int i;                                                                                  \
        int my_active_set_pe = ((nvshmemi_mype_d - start) / stride);                            \
                                                                                                \
        next_rank = start + ((my_active_set_pe + 1) % size) * stride;                           \
        GPU_LINEAR_REDUCE_THREADGROUP_P2P_PUT(SC, TYPE, OP, source, source, next_rank, dest,    \
                                              counter, nreduce, myIdx, groupSize, start,        \
                                              stride, size, pSync);                             \
                                                                                                \
        for (i = 2; i < size; i++) {                                                            \
            next_rank = start + ((my_active_set_pe + i) % size) * stride;                       \
            GPU_LINEAR_REDUCE_THREADGROUP_P2P_PUT(SC, TYPENAME, TYPE, OP, dest, source, next_rank, dest,  \
                                                  counter, nreduce, myIdx, groupSize, start,    \
                                                  stride, size, pSync);                         \
        }                                                                                       \
        NVSHMEMI_SYNC_##SC();                                                                   \
    } while (0)

#define GPU_RDXN_SEGMENT_THREADGROUP(SC, TYPENAME, TYPE, OP, dest, source, nelems, start, stride,  \
                                     size, pWrk, pSync)                                        \
    do {                                                                                          \
        int type_size = sizeof(TYPE);                                                             \
        int msg_len = nelems * type_size;                                                         \
        int next_rank = -1;                                                                       \
        NVSHMEMI_DECL_THREAD_IDX_##SC();                                                          \
        NVSHMEMI_DECL_THREADGROUP_SIZE_##SC();                                                    \
        TYPE *op1 = NULL, *op2 = NULL;                                                            \
        int i, j;                                                                                 \
        volatile TYPE *tmp_operand;                                                               \
        int remainder = 0;                                                                        \
        int rnds_floor = 0;                                                                       \
        int offset = 0;                                                                           \
        int pe_offset = 0;                                                                        \
        int pes_per_round = 0;                                                                    \
        int round = 0;                                                                            \
        int exchange_size = 0;                                                                    \
        int my_active_set_pe = ((nvshmemi_mype_d - start) / stride);                              \
        int nvshm_gpu_rdxn_seg_size = NVSHMEMI_REDUCE_MIN_WRKDATA_SIZE;                           \
                                                                                                  \
        tmp_operand = (TYPE *)pWrk;                                                               \
        nvshmemx_##TYPENAME##_put_nbi_##SC((TYPE *)dest, (const TYPE *)source, nelems, nvshmemi_mype_d);     \
                                                                                                  \
        rnds_floor = msg_len / nvshm_gpu_rdxn_seg_size;                                           \
        remainder = msg_len % nvshm_gpu_rdxn_seg_size;                                            \
                                                                                                  \
        long sync_counter = NVSHMEMI_SYNC_VALUE + 1;                                              \
        for (j = 0; j < rnds_floor; j++) {                                                        \
            exchange_size = nvshm_gpu_rdxn_seg_size;                                              \
            for (i = 1; i < size; i++) {                                                          \
                next_rank = start + ((my_active_set_pe + i) % size) * stride;                     \
                nvshmemx_##TYPENAME##_put_nbi_##SC((TYPE *)tmp_operand, (const TYPE *)source + offset,      \
                                               (exchange_size / sizeof(TYPE)), next_rank);        \
                nvshmemxi_barrier_##SC(start, stride, size, pSync, &sync_counter);                \
                op1 = (TYPE *)dest + offset;                                                      \
                op2 = (TYPE *)tmp_operand;                                                        \
                GPU_LINEAR_REDUCE_THREADGROUP(SC, TYPENAME, TYPE, OP, op1, op2, op1,              \
                                              (exchange_size / sizeof(TYPE)), myIdx, groupSize);  \
                nvshmemxi_sync_##SC(start, stride, size, pSync, &sync_counter);                   \
            }                                                                                     \
            offset += (exchange_size / sizeof(TYPE));                                             \
        }                                                                                         \
        if (remainder != 0) {                                                                     \
            exchange_size = remainder;                                                            \
            pes_per_round = nvshm_gpu_rdxn_seg_size / remainder;                                  \
            pe_offset = 1;                                                                        \
            do {                                                                                  \
                round = 0;                                                                        \
                for (i = pe_offset; ((round < pes_per_round) && (i < size)); i++) {            \
                    next_rank = start + ((my_active_set_pe + i) % size) * stride;                 \
                    nvshmemx_##TYPENAME##_put_nbi_##SC(                                           \
                        (TYPE *)((TYPE *)tmp_operand + (round * (exchange_size / sizeof(TYPE)))), \
                        (TYPE *)source + offset, (exchange_size / sizeof(TYPE)), next_rank);      \
                    round++;                                                                      \
                    pe_offset++;                                                                  \
                }                                                                                 \
                nvshmemxi_barrier_##SC(start, stride, size, pSync, &sync_counter);                \
                for (i = 0; i < round; i++) {                                                     \
                    op1 = (TYPE *)dest + offset;                                                  \
                    op2 = (TYPE *)((TYPE *)tmp_operand + (i * (exchange_size / sizeof(TYPE))));   \
                    GPU_LINEAR_REDUCE_THREADGROUP(SC, TYPENAME, TYPE, OP, op1, op2, op1,          \
                                                  (exchange_size / sizeof(TYPE)), myIdx,          \
                                                  groupSize);                                     \
                }                                                                                 \
                nvshmemxi_sync_##SC(start, stride, size, pSync, &sync_counter);                   \
            } while (pe_offset < size);                                                           \
        }                                                                                         \
        NVSHMEMI_SYNC_##SC();                                                                     \
        for (i = myIdx; i < size; i += groupSize) {                                               \
            pSync[start + i * stride] = NVSHMEMI_SYNC_VALUE;                                      \
        }                                                                                         \
        NVSHMEMI_SYNC_##SC();                                                                     \
    } while (0)

// barrier limited
#define NVSHMEMXI_GPU_RDXN_THREADGROUP_PUT_BAR_DIRECT(                                             \
    SC, TYPENAME, TYPE, OP, dest, source, nreduce, start, stride, size, pWrk, pSync)             \
    do {                                                                                           \
        int next_rank = -1;                                                                        \
        int src_offset = -1;                                                                       \
        int next_offset = -1;                                                                      \
        char *base = NULL;                                                                         \
        char *peer_base = NULL;                                                                    \
        char *peer_source = NULL;                                                                  \
        int counter = 0;                                                                           \
        int my_active_set_pe = ((nvshmemi_mype_d - start) / stride);                              \
        NVSHMEMI_DECL_THREAD_IDX_##SC();                                                           \
        NVSHMEMI_DECL_THREADGROUP_SIZE_##SC();                                                     \
        int i;                                                                                     \
                                                                                                   \
        base = (char *)((void *)__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d +       \
                                      nvshmemi_mype_d));                                            \
        src_offset = ((char *)source - base);                                                      \
                                                                                                   \
        next_rank = start + ((my_active_set_pe + 1) % size) * stride;                              \
        next_offset = src_offset;                                                                  \
        peer_base = (char *)((void *)__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d +  \
                                           next_rank));                                            \
        peer_source = peer_base + next_offset;                                                     \
        GPU_LINEAR_REDUCE_THREADGROUP_P2P_PUT_DIRECT(SC, TYPENAME, TYPE, OP, source, peer_source, next_rank, \
                                                     dest, counter, nreduce, myIdx, groupSize,     \
                                                     start, stride, size, pWrk, pSync);              \
                                                                                                   \
        for (i = 2; i < size; i++) {                                                            \
            next_rank = start + ((my_active_set_pe + i) % size) * stride;                         \
            next_offset = src_offset;                                                              \
            peer_base = (char *)((void *)__ldg(                                                    \
                (const long long unsigned *)nvshmemi_peer_heap_base_d + next_rank));                \
            peer_source = peer_base + next_offset;                                                 \
            GPU_LINEAR_REDUCE_THREADGROUP_P2P_PUT_DIRECT(                                          \
                SC, TYPENAME, TYPE, OP, dest, peer_source, next_rank, dest, counter, nreduce, myIdx,         \
                groupSize, start, stride, size, pWrk, pSync);                                      \
        }                                                                                          \
        NVSHMEMI_SYNC_##SC();                                                                      \
    } while (0)

#define NVSHMEMXI_GPU_RDXN_THREADGROUP_PUTRING(SC, TYPENAME, TYPE, OP, dest, source, nelems, start,       \
                                               stride, size, pWrk, pSync)                 \
    do {                                                                                           \
        int next_rank = -1;                                                                        \
        int prev_rank = -1;                                                                        \
        int i, j;                                                                                  \
        int PE_end = start + (stride * size);                                                      \
        uint32_t tmp[2];                                                                           \
        long *tmp_rdxn = (long *)pWrk;                                                             \
        int *tmp_int_rdxn = (int *)((long *)&tmp_rdxn[1]);                                         \
        uint32_t payld;                                                                            \
        NVSHMEMI_DECL_THREAD_IDX_##SC();                                                           \
        NVSHMEMI_DECL_THREADGROUP_SIZE_##SC();                                                     \
        volatile uint32_t *my_notif_ptr = NULL;                                                    \
        int subelems = sizeof(TYPE) / sizeof(uint32_t);                                            \
        tmp[1] = 1;                                                                                \
        next_rank = (nvshmemi_mype_d != (PE_end - stride)) ? (nvshmemi_mype_d + stride) : start;  \
        prev_rank = (nvshmemi_mype_d != start) ? (nvshmemi_mype_d - stride) : (PE_end - stride);  \
        my_notif_ptr = (uint32_t *)((uint32_t *)((uint64_t *)pWrk + (nelems * subelems)) + myIdx); \
                                                                                                   \
        for (j = myIdx; j < nelems * subelems; j += groupSize) {                                   \
            payld = *((uint32_t *)source + j);                                                     \
            tmp[0] = payld;                                                                        \
            *tmp_rdxn = *((long *)tmp);                                                            \
            nvshmemx_long_signal(((long *)pWrk + j), *tmp_rdxn, next_rank);                      \
        }                                                                                          \
        GPU_HEAD_CHECK_OP_THREADGROUP(TYPENAME, TYPE, OP, dest, pWrk, source, nelems, myIdx, groupSize);     \
        /* sync needed on volta (intermittent hangs seen otherwise) */                             \
        NVSHMEMI_SYNC_##SC();                                                                      \
                                                                                                   \
        for (i = 1; i < (size - 1); i++) {                                                      \
            __threadfence_system();                                                                \
            /* Don't want notification to overtake data transfer */                                \
            *tmp_int_rdxn = 1;                                                                     \
            nvshmemx_int_signal(((int *)((int *)((long *)pWrk + (nelems * subelems)) + myIdx)),    \
                                *tmp_int_rdxn, prev_rank);                                         \
            while (1 != *my_notif_ptr)                                                             \
                ;                                                                                  \
            *my_notif_ptr = 0;                                                                     \
            for (j = myIdx; j < nelems * subelems; j += groupSize) {                               \
                payld = *((uint32_t *)dest + j);                                                   \
                tmp[0] = payld;                                                                    \
                *tmp_rdxn = *((long *)tmp);                                                        \
                nvshmemx_long_signal(((long *)pWrk + j), *tmp_rdxn, next_rank);                    \
            }                                                                                      \
            GPU_HEAD_CHECK_OP_THREADGROUP(TYPENAME, TYPE, OP, dest, pWrk, source, nelems, myIdx, groupSize); \
            NVSHMEMI_SYNC_##SC();                                                                  \
        }                                                                                          \
    } while (0)

#define NVSHMEMXI_GPU_RDXN_THREADGROUP_PUTRING_DIRECT(                                             \
    SC, TYPENAME, TYPE, OP, dest, source, nelems, start, stride, size, pWrk, pSync)              \
    do {                                                                                           \
        int offset;                                                                                \
        char *notif_pwrk_dest;                                                                     \
        char *round_pwrk_dest;                                                                     \
        int next_rank = -1;                                                                        \
        int prev_rank = -1;                                                                        \
        int i, j;                                                                                  \
        int PE_end = start + (stride * size);                                                \
        uint32_t tmp[2];                                                                           \
        uint32_t payld;                                                                            \
        uint32_t *notif_ptr = NULL;                                                                \
        volatile uint32_t *my_notif_ptr = NULL;                                                    \
        NVSHMEMI_DECL_THREAD_IDX_##SC();                                                           \
        NVSHMEMI_DECL_THREADGROUP_SIZE_##SC();                                                     \
        int subelems = sizeof(TYPE) / sizeof(uint32_t);                                            \
        tmp[1] = 1;                                                                                \
        offset =                                                                                   \
            (char *)pWrk - (char *)(__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d +   \
                                          nvshmemi_mype_d));                                        \
        my_notif_ptr = (uint32_t *)((uint32_t *)((uint64_t *)pWrk + (nelems * subelems)) + myIdx); \
        next_rank = (nvshmemi_mype_d != (PE_end - stride)) ? (nvshmemi_mype_d + stride) : start;  \
        round_pwrk_dest =                                                                          \
            (char *)(__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + next_rank)) +    \
            offset;                                                                                \
        prev_rank = (nvshmemi_mype_d != start) ? (nvshmemi_mype_d - stride) : (PE_end - stride);  \
        notif_pwrk_dest =                                                                          \
            (char *)(__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + prev_rank)) +    \
            offset;                                                                                \
        notif_ptr =                                                                                \
            (uint32_t *)((uint32_t *)((uint64_t *)notif_pwrk_dest + (nelems * subelems)) + myIdx); \
                                                                                                   \
        for (j = myIdx; j < nelems * subelems; j += groupSize) {                                   \
            payld = *((uint32_t *)source + j);                                                     \
            tmp[0] = payld;                                                                        \
            *((uint64_t *)round_pwrk_dest + j) = *((uint64_t *)tmp);                               \
        }                                                                                          \
        GPU_HEAD_CHECK_OP_THREADGROUP(TYPENAME, TYPE, OP, dest, pWrk, source, nelems, myIdx, groupSize);     \
        /* sync needed on volta (intermittent hangs seen otherwise) */                             \
        NVSHMEMI_SYNC_##SC();                                                                      \
                                                                                                   \
        for (i = 1; i < (size - 1); i++) {                                                      \
            __threadfence_system();                                                                \
            /* Don't want notification to overtake data transfer */                                \
            *notif_ptr = 1;                                                                        \
            while (1 != *my_notif_ptr)                                                             \
                ;                                                                                  \
            *my_notif_ptr = 0;                                                                     \
            for (j = myIdx; j < nelems * subelems; j += groupSize) {                               \
                payld = *((uint32_t *)dest + j);                                                   \
                tmp[0] = payld;                                                                    \
                *((uint64_t *)round_pwrk_dest + j) = *((uint64_t *)tmp);                           \
            }                                                                                      \
            GPU_HEAD_CHECK_OP_THREADGROUP(TYPENAME, TYPE, OP, dest, pWrk, source, nelems, myIdx, groupSize); \
            NVSHMEMI_SYNC_##SC();                                                                  \
        }                                                                                          \
    } while (0)

#define NVSHMEMXI_GPU_RDXN_THREADGROUP_PUTALL(SC, TYPENAME, TYPE, OP, dest, source, nelems, start,       \
                                              stride, size, pWrk, pSync)                          \
    do {                                                                                          \
        int i, j;                                                                                 \
        int PE_end = start + (stride * size);                                                     \
        NVSHMEMI_DECL_THREAD_IDX_##SC();                                                          \
        NVSHMEMI_DECL_THREADGROUP_SIZE_##SC();                                                    \
        long *tmp_rdxn = (long *)pWrk;                                                            \
        uint32_t tmp[2];                                                                          \
        uint32_t payld;                                                                           \
        int subelems = sizeof(TYPE) / sizeof(uint32_t);                                           \
        int my_active_set_pe = ((nvshmemi_mype_d - start) / stride);                              \
        tmp[1] = 1;                                                                               \
                                                                                                  \
        for (j = myIdx; j < nelems * subelems; j += groupSize) {                                  \
            payld = *((uint32_t *)source + j);                                                    \
            tmp[0] = payld;                                                                       \
            *tmp_rdxn = *((long *)tmp);                                                           \
            for (i = start; i < nvshmemi_mype_d; i += stride) {                                 \
                nvshmemx_long_signal(((long *)pWrk + j + (nelems * subelems * my_active_set_pe)), \
                                     *tmp_rdxn, i);                                               \
            }                                                                                     \
            for (i = nvshmemi_mype_d + stride; i < PE_end; i += stride) {                          \
                nvshmemx_long_signal(((long *)pWrk + j + (nelems * subelems * my_active_set_pe)), \
                                     *tmp_rdxn, i);                                               \
            }                                                                                     \
        }                                                                                         \
        GPU_HEAD_CHECKALL_OP_THREADGROUP(TYPENAME, TYPE, OP, dest, pWrk, source, nelems, start, stride,  \
                                         size, myIdx, groupSize);                              \
        __threadfence();                                                                          \
        NVSHMEMI_SYNC_##SC();                                                                     \
    } while (0)

#define NVSHMEMXI_GPU_RDXN_THREADGROUP_PUTALL_DIRECT(SC, TYPENAME, TYPE, OP, dest, source, nelems, start, \
                                                     stride, size, pWrk, pSync)                     \
    do {                                                                                           \
        int offset;                                                                                \
        char *round_pwrk_dest;                                                                     \
        int i, j;                                                                                  \
        int PE_end = start + (stride * size);                                                       \
        NVSHMEMI_DECL_THREAD_IDX_##SC();                                                           \
        NVSHMEMI_DECL_THREADGROUP_SIZE_##SC();                                                     \
        uint32_t tmp[2];                                                                           \
        uint32_t payld;                                                                            \
        int subelems = sizeof(TYPE) / sizeof(uint32_t);                                            \
        int my_active_set_pe = ((nvshmemi_mype_d - start) / stride);                                \
        tmp[1] = 1;                                                                                 \
        offset =                                                                                    \
            (char *)pWrk - (char *)(__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d +   \
                                          nvshmemi_mype_d));                                        \
                                                                                                   \
        for (j = myIdx; j < nelems * subelems; j += groupSize) {                                   \
            payld = *((uint32_t *)source + j);                                                     \
            tmp[0] = payld;                                                                        \
            for (i = nvshmemi_mype_d + stride; i < PE_end; i += stride) {                           \
                round_pwrk_dest =                                                                   \
                    (char *)(__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + i)) +    \
                    offset;                                                                        \
                *((uint64_t *)round_pwrk_dest + j + (nelems * subelems * my_active_set_pe)) =      \
                    *((uint64_t *)tmp);                                                            \
            }                                                                                      \
            for (i = start; i < nvshmemi_mype_d; i += stride) {                                     \
                round_pwrk_dest =                                                                   \
                    (char *)(__ldg((const long long unsigned *)nvshmemi_peer_heap_base_d + i)) +    \
                    offset;                                                                        \
                *((uint64_t *)round_pwrk_dest + j + (nelems * subelems * my_active_set_pe)) =      \
                    *((uint64_t *)tmp);                                                            \
            }                                                                                      \
        }                                                                                          \
        GPU_HEAD_CHECKALL_OP_THREADGROUP(TYPENAME, TYPE, OP, dest, pWrk, source, nelems, start, stride,   \
                                         size, myIdx, groupSize);                                   \
        __threadfence();                                                                            \
        NVSHMEMI_SYNC_##SC();                                                                       \
    } while (0)

#ifdef NVSHMEM_GPU_COLL_USE_LDST
#ifdef NVSHMEM_DISABLE_COLL_POLL
#define NVSHMEMXI_GPU_RDXN_THREADGROUP(SC, TYPENAME, TYPE, OP, dest, source, nelems, start, stride, \
                                       size, pWrk, pSync)                                           \
    do {                                                                                            \
        NVSHMEMXI_GPU_RDXN_THREADGROUP_ZCOPY_GET_BAR_DIRECT(                                        \
            SC, TYPENAME, TYPE, OP, dest, source, nreduce, start, stride, size, pWrk, pSync);       \
    } while (0)
#else
#define NVSHMEMXI_GPU_RDXN_THREADGROUP(SC, TYPENAME, TYPE, OP, dest, source, nelems, start, stride, \
                                       size, pWrk, pSync)                                           \
    do {                                                                                            \
        int subelems = sizeof(TYPE) / sizeof(uint32_t);                                             \
        int pwrk_req_sz_allgather = ((subelems * nelems) * sizeof(uint64_t)) * size;                \
        int wrk_size = NVSHMEMI_REDUCE_MIN_WRKDATA_SIZE * sizeof(TYPE);                             \
        if (subelems && (pwrk_req_sz_allgather <= wrk_size)) {                                      \
            NVSHMEMXI_GPU_RDXN_THREADGROUP_PUTALL_DIRECT(SC, TYPENAME, TYPE, OP, dest, source, nreduce,      \
                                                         start, stride, size, pWrk,                 \
                                                         pSync);                                    \
        } else {                                                                                    \
            NVSHMEMXI_GPU_RDXN_THREADGROUP_ZCOPY_GET_BAR_DIRECT(SC, TYPENAME, TYPE, OP, dest, source,        \
                                                                nreduce, start, stride,             \
                                                                size, pWrk, pSync);                 \
        }                                                                                           \
    } while (0)
#endif
#else
#ifdef NVSHMEM_DISABLE_COLL_POLL
#define NVSHMEMXI_GPU_RDXN_THREADGROUP(SC, TYPENAME, TYPE, OP, dest, source, nelems, start, stride, \
                                       size, pWrk, pSync)                                           \
    do {                                                                                            \
        GPU_RDXN_SEGMENT_THREADGROUP(SC, TYPENAME, TYPE, OP, dest, source, nelems, start, stride,   \
                                     size, pWrk, pSync);                                            \
    } while (0)
#else
#define NVSHMEMXI_GPU_RDXN_THREADGROUP(SC, TYPENAME, TYPE, OP, dest, source, nelems, start, stride, \
                                       size, pWrk, pSync)                                           \
    do {                                                                                           \
        int subelems = sizeof(TYPE) / sizeof(uint32_t);                                            \
        int pwrk_req_sz_allgather = ((subelems * nelems) * sizeof(uint64_t)) * size;                \
        NVSHMEMI_DECL_THREADGROUP_SIZE_##SC();                                                     \
        int pwrk_req_sz_ring =                                                                     \
            ((subelems * nelems) * sizeof(uint64_t)) + (groupSize * sizeof(uint32_t));             \
        int wrk_size = NVSHMEMI_REDUCE_MIN_WRKDATA_SIZE * sizeof(TYPE);                            \
        if (subelems && pwrk_req_sz_allgather <= wrk_size) {                                       \
            NVSHMEMXI_GPU_RDXN_THREADGROUP_PUTALL(SC, TYPENAME, TYPE, OP, dest, source, nreduce, start,   \
                                                  stride, size, pWrk, pSync);             \
        } else if (subelems && pwrk_req_sz_ring <= wrk_size) {                                     \
            NVSHMEMXI_GPU_RDXN_THREADGROUP_PUTRING(SC, TYPENAME, TYPE, OP, dest, source, nreduce, start,  \
                                                   stride, size, pWrk, pSync);            \
        } else {                                                                                   \
            NVSHMEMXI_GPU_RDXN_THREADGROUP_PUT_BAR(SC, TYPENAME, TYPE, OP, dest, source, nreduce, start,  \
                                                   stride, size, pWrk, pSync);            \
        }                                                                                          \
    } while (0)
#endif
#endif


#define DEFN_NVSHMEMXI_TYPENAME_OP_REDUCE_THREADGROUP(SC, TYPENAME, TYPE, OP)                     \
    __device__ void nvshmemxi_##TYPENAME##_##OP##_reduce_##SC(TYPE * dest, const TYPE *source,    \
          int nreduce, int start, int stride, int size, TYPE *pWrk, long *pSync) {                \
        NVSHMEMXI_GPU_RDXN_THREADGROUP(SC, TYPENAME, TYPE, OP, dest, source, nreduce, start,      \
                                       stride, size, pWrk, pSync);                                \
    }

#define DEFN_NVSHMEMI_REDUCE_THREADGROUP(SC)                                                                 \
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE(DEFN_NVSHMEMXI_TYPENAME_OP_REDUCE_THREADGROUP, SC, and)    \
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE(DEFN_NVSHMEMXI_TYPENAME_OP_REDUCE_THREADGROUP, SC, or)     \
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE(DEFN_NVSHMEMXI_TYPENAME_OP_REDUCE_THREADGROUP, SC, xor)    \
                                                                                                             \
NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES_WITH_SCOPE(DEFN_NVSHMEMXI_TYPENAME_OP_REDUCE_THREADGROUP, SC, max)   \
NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES_WITH_SCOPE(DEFN_NVSHMEMXI_TYPENAME_OP_REDUCE_THREADGROUP, SC, min)   \
                                                                                                             \
NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES_WITH_SCOPE(DEFN_NVSHMEMXI_TYPENAME_OP_REDUCE_THREADGROUP, SC, sum)      \
NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES_WITH_SCOPE(DEFN_NVSHMEMXI_TYPENAME_OP_REDUCE_THREADGROUP, SC, prod)

DEFN_NVSHMEMI_REDUCE_THREADGROUP(warp);
DEFN_NVSHMEMI_REDUCE_THREADGROUP(block);
#undef DEFN_NVSHMEMXI_TYPENAME_OP_REDUCE_THREADGROUP
#undef DEFN_NVSHMEMI_REDUCE_THREADGROUP

#define DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_THREADGROUP(SC, TYPENAME, TYPE, OP)                      \
    __device__ int nvshmemx_##TYPENAME##_##OP##_reduce_##SC(nvshmem_team_t team, TYPE *dest,      \
                                                           const TYPE *source, int nreduce) {       \
        nvshmemi_team_t *teami = nvshmemi_team_pool_d[team];                                      \
        TYPE *pWrk = (TYPE *) nvshmemi_team_get_psync(teami, REDUCE);                             \
        long *pSync = (long *) ((long *)pWrk + NVSHMEMI_REDUCE_MIN_WRKDATA_SIZE);                 \
        nvshmemx_barrier_##SC(team);                                                              \
        nvshmemxi_##TYPENAME##_##OP##_reduce_##SC(dest, source, nreduce, teami->start,            \
                                       teami->stride, teami->size, pWrk, pSync);                  \
        return 0;                                                                                 \
    }

#define DEFN_NVSHMEM_REDUCE_THREADGROUP(SC)                                                                 \
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE(DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_THREADGROUP, SC, and)    \
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE(DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_THREADGROUP, SC, or)     \
NVSHMEMI_REPT_FOR_BITWISE_REDUCE_TYPES_WITH_SCOPE(DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_THREADGROUP, SC, xor)    \
                                                                                                            \
NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES_WITH_SCOPE(DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_THREADGROUP, SC, max)   \
NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES_WITH_SCOPE(DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_THREADGROUP, SC, min)   \
                                                                                                            \
NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES_WITH_SCOPE(DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_THREADGROUP, SC, sum)      \
NVSHMEMI_REPT_FOR_ARITH_REDUCE_TYPES_WITH_SCOPE(DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_THREADGROUP, SC, prod)

DEFN_NVSHMEM_REDUCE_THREADGROUP(warp);
DEFN_NVSHMEM_REDUCE_THREADGROUP(block);
#undef DEFN_NVSHMEMX_TYPENAME_OP_REDUCE_THREADGROUP
#undef DEFN_NVSHMEM_REDUCE_THREADGROUP

#endif

#define RDXN_ON_STREAM_KERNEL(TYPENAME, TYPE, OP)                                                \
    __global__ void rdxn_##TYPENAME##_##OP##_on_stream_kernel(                                   \
        TYPE *dest, const TYPE *source, int nreduce, int start, int stride, int size,            \
        TYPE *pWrk, long *pSync) {                                                               \
        if (!blockIdx.x)                                                                         \
            nvshmemxi_##TYPENAME##_##OP##_reduce_block(dest, source, nreduce, start, stride,     \
                                                  size, pWrk, pSync);                            \
    }

NVSHMEMI_REPT_TYPES_AND_OPS_FOR_REDUCE(RDXN_ON_STREAM_KERNEL)
#undef RDXN_ON_STREAM_KERNEL

#define CALL_RDXN_ON_STREAM(TYPENAME, TYPE, OP)                                                   \
    extern "C" void call_rdxn_##TYPENAME##_##OP##_on_stream_kern(                                 \
        TYPE *dest, const TYPE *source, int nreduce, int start, int stride, int size,             \
        TYPE *pWrk, long *pSync, cudaStream_t stream) {                                           \
        int num_threads_per_block =                                                               \
            (MAX_THREADS_PER_CTA > nreduce) ? nreduce : MAX_THREADS_PER_CTA;                      \
        int num_blocks = 1;                                                                       \
        rdxn_##TYPENAME##_##OP##_on_stream_kernel<<<num_blocks, num_threads_per_block, 0, stream>>>(  \
            dest, source, nreduce, start, stride, size, pWrk, pSync);                             \
        CUDA_RUNTIME_CHECK(cudaGetLastError());                                                   \
    }

NVSHMEMI_REPT_TYPES_AND_OPS_FOR_REDUCE(CALL_RDXN_ON_STREAM)
#undef CALL_RDXN_ON_STREAM
