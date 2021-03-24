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

#include "coll_test.h"
#define LARGEST_DT uint64_t

#define RUN_RDXN(TYPENAME, TYPE, OP, team, d_source, h_source, d_dest, h_dest, num_elems, stream)       \
    do {                                                                                          \
        int iters = MAX_ITERS;                                                                    \
        int skip = MAX_SKIP;                                                                      \
        struct timeval t_start, t_stop;                                                           \
        for (iters = 0; iters < MAX_ITERS + skip; iters++) {                                      \
            CUDA_CHECK(cudaMemcpyAsync(d_source, h_source, (sizeof(TYPE) * num_elems),            \
                                       cudaMemcpyHostToDevice, stream));                          \
            CUDA_CHECK(cudaMemcpyAsync(d_dest, h_dest, (sizeof(TYPE) * num_elems),                \
                                       cudaMemcpyHostToDevice, stream));                          \
                                                                                                  \
            CUDA_CHECK(cudaStreamSynchronize(stream));                                            \
            nvshmem_barrier_all();                                                                \
                                                                                                  \
            if (iters >= skip) gettimeofday(&t_start, NULL);                                      \
                                                                                                  \
            nvshmem_##TYPENAME##_##OP##_reduce(team, (TYPE *)d_dest, (const TYPE *)d_source, num_elems);\
                                                                                                  \
            if (iters >= skip) {                                                                  \
                gettimeofday(&t_stop, NULL);                                                      \
                latency += ((t_stop.tv_usec - t_start.tv_usec) +                                  \
                            (1e+6 * (t_stop.tv_sec - t_start.tv_sec)));                           \
            }                                                                                     \
                                                                                                  \
            nvshmem_barrier_all();                                                                \
                                                                                                  \
            CUDA_CHECK(cudaMemcpyAsync(h_source, d_source, (sizeof(TYPE) * num_elems),            \
                                       cudaMemcpyDeviceToHost, stream));                          \
            CUDA_CHECK(cudaMemcpyAsync(h_dest, d_dest, (sizeof(TYPE) * num_elems),                \
                                       cudaMemcpyDeviceToHost, stream));                          \
            CUDA_CHECK(cudaStreamSynchronize(stream));                                            \
        }                                                                                         \
    } while (0)

#define RUN_RDXN_ITERS(TYPENAME, TYPE, team, d_source, h_source, d_dest, h_dest, num_elems, stream, mype, \
                       size, usec_sum, usec_prod, usec_and, usec_or, usec_xor, usec_min, usec_max)      \
    do {                                                                                           \
        double latency = 0;                                                                        \
        size = num_elems * sizeof(TYPE);                                                           \
        RUN_RDXN(TYPENAME, TYPE, sum, team, d_source, h_source, d_dest, h_dest, num_elems, stream);\
        usec_sum = latency / MAX_ITERS;                                                            \
        latency = 0;                                                                               \
        RUN_RDXN(TYPENAME, TYPE, prod, team, d_source, h_source, d_dest, h_dest, num_elems, stream);  \
        usec_prod = latency / MAX_ITERS;                                                           \
        latency = 0;                                                                               \
        RUN_RDXN(TYPENAME, TYPE, and, team, d_source, h_source, d_dest, h_dest, num_elems, stream);      \
        usec_and = latency / MAX_ITERS;                                                            \
        latency = 0;                                                                               \
        RUN_RDXN(TYPENAME, TYPE, or, team, d_source, h_source, d_dest, h_dest, num_elems, stream);       \
        usec_or = latency / MAX_ITERS;                                                             \
        latency = 0;                                                                               \
        RUN_RDXN(TYPENAME, TYPE, xor, team, d_source, h_source, d_dest, h_dest, num_elems, stream);      \
        usec_xor = latency / MAX_ITERS;                                                            \
        latency = 0;                                                                               \
        RUN_RDXN(TYPENAME, TYPE, min, team, d_source, h_source, d_dest, h_dest, num_elems, stream);      \
        usec_min = latency / MAX_ITERS;                                                            \
        latency = 0;                                                                               \
        RUN_RDXN(TYPENAME, TYPE, max, team, d_source, h_source, d_dest, h_dest, num_elems, stream);\
        usec_max = latency / MAX_ITERS;                                                            \
    } while (0)

int main(int argc, char **argv) {
    int status = 0;
    int mype, npes;
    int i = 0;
    size_t size = (MAX_ELEMS * 8) * sizeof(LARGEST_DT);
    size_t alloc_size;
    int num_elems;
    LARGEST_DT *h_buffer = NULL;
    LARGEST_DT *d_buffer = NULL;
    LARGEST_DT *d_source, *d_dest;
    LARGEST_DT *h_source, *h_dest;
    char size_string[100];
    uint64_t size_array[MAX_ELEMS_LOG];
    double sum_latency_array[MAX_ELEMS_LOG];
    double prod_latency_array[MAX_ELEMS_LOG];
    double and_latency_array[MAX_ELEMS_LOG];
    double or_latency_array[MAX_ELEMS_LOG];
    double xor_latency_array[MAX_ELEMS_LOG];
    double min_latency_array[MAX_ELEMS_LOG];
    double max_latency_array[MAX_ELEMS_LOG];
    cudaStream_t stream;

    DEBUG_PRINT("symmetric size requested %lu\n", size);
    sprintf(size_string, "%lu", size);

    status = setenv("NVSHMEM_SYMMETRIC_SIZE", size_string, 1);
    if (status) {
        fprintf(stderr, "setenv failed \n");
        status = -1;
        goto out;
    }

    init_wrapper(&argc, &argv);

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    num_elems = MAX_ELEMS / 2;
    alloc_size = (num_elems * 2) * sizeof(long);

    CUDA_CHECK(cudaHostAlloc(&h_buffer, alloc_size, cudaHostAllocDefault));
    h_source = (LARGEST_DT *)h_buffer;
    h_dest = (LARGEST_DT *)&h_source[num_elems];

    d_buffer = (LARGEST_DT *)nvshmem_malloc(alloc_size);
    if (!d_buffer) {
        fprintf(stderr, "nvshmem_malloc failed \n");
        status = -1;
        goto out;
    }

    d_source = (LARGEST_DT *)d_buffer;
    d_dest = (LARGEST_DT *)&d_source[num_elems];

    i = 0;
    for (num_elems = 1; num_elems < (MAX_ELEMS / 2); num_elems *= 2) {
        RUN_RDXN_ITERS(int32, int32_t, NVSHMEM_TEAM_WORLD, (int *)d_source, (int *)h_source,
                       (int *)d_dest, (int *)h_dest, num_elems, stream, mype, size_array[i], sum_latency_array[i], prod_latency_array[i],
                       and_latency_array[i], or_latency_array[i], xor_latency_array[i], min_latency_array[i],
                       max_latency_array[i]);
        i++;
    }

    if (!mype) {
        print_table("reduction", "int-sum", "size (Bytes)", "latency", "us", '-', size_array, sum_latency_array, i);
        print_table("reduction", "int-prod", "size (Bytes)", "latency", "us", '-', size_array, prod_latency_array, i);
        print_table("reduction", "int-and", "size (Bytes)", "latency", "us", '-', size_array, and_latency_array, i);
        print_table("reduction", "int-or", "size (Bytes)", "latency", "us", '-', size_array, or_latency_array, i);
        print_table("reduction", "int-xor", "size (Bytes)", "latency", "us", '-', size_array, xor_latency_array, i);
        print_table("reduction", "int-min", "size (Bytes)", "latency", "us", '-', size_array, min_latency_array, i);
        print_table("reduction", "int-max", "size (Bytes)", "latency", "us", '-', size_array, max_latency_array, i);
    }

    i = 0;
    for (num_elems = 1; num_elems < (MAX_ELEMS / 2); num_elems *= 2) {
        RUN_RDXN_ITERS(int64, int64_t, NVSHMEM_TEAM_WORLD, d_source, h_source,
                       d_dest, h_dest, num_elems, stream, mype, size_array[i],
                       sum_latency_array[i], prod_latency_array[i], and_latency_array[i], or_latency_array[i],
                       xor_latency_array[i], min_latency_array[i], max_latency_array[i]);
        i++;
    }

    if (!mype) {
        print_table("reduction", "int64-sum", "size (Bytes)", "latency", "us", '-', size_array, sum_latency_array, i);
        print_table("reduction", "int64-prod", "size (Bytes)", "latency", "us", '-', size_array, prod_latency_array, i);
        print_table("reduction", "int64-and", "size (Bytes)", "latency", "us", '-', size_array, and_latency_array, i);
        print_table("reduction", "int64-or", "size (Bytes)", "latency", "us", '-', size_array, or_latency_array, i);
        print_table("reduction", "int64-xor", "size (Bytes)", "latency", "us", '-', size_array, xor_latency_array, i);
        print_table("reduction", "int64-min", "size (Bytes)", "latency", "us", '-', size_array, min_latency_array, i);
        print_table("reduction", "int64-max", "size (Bytes)", "latency", "us", '-', size_array, max_latency_array, i);
    }

    nvshmem_barrier_all();

    CUDA_CHECK(cudaFreeHost(h_buffer));
    nvshmem_free(d_buffer);

    CUDA_CHECK(cudaStreamDestroy(stream));

    finalize_wrapper();

out:
    return status;
}
