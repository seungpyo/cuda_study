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
#define LARGEST_DT int64_t

#ifdef MAX_ITERS
#undef MAX_ITERS
#endif
#define MAX_ITERS 50

#define CALL_RDXN_BLOCK(TYPENAME, TYPE, OP, usec)                                          \
    do {                                                                                   \
        double time;                                                                       \
        int i;                                                                             \
        int iter = MAX_ITERS;                                                              \
        int skip = MAX_SKIP;                                                               \
        long long int start = 0, stop = 0;                                                 \
                                                                                           \
        time = 0;                                                                          \
        for (i = 0; i < (iter + skip); i++) {                                              \
            nvshmemx_barrier_all_block();                                                  \
            if (i > skip) start = clock64();                                               \
            nvshmemx_##TYPENAME##_##OP##_reduce_block(team, dest, source, nelems);         \
            if (i > skip) stop = clock64();                                                \
            time += (stop - start);                                                        \
        }                                                                                  \
        nvshmemx_barrier_all_block();                                                      \
        if (!threadIdx.x && !mype) {                                                       \
            time = time / iter;                                                            \
            *usec = time * 1000 / clockrate;                                               \
        }                                                                                  \
    } while (0)

#define CALL_RDXN_WARP(TYPENAME, TYPE, OP, usec)                                                   \
    do {                                                                                           \
        double time;                                                                               \
        int i;                                                                                     \
        int iter = MAX_ITERS;                                                                      \
        int skip = MAX_SKIP;                                                                       \
        long long int start = 0, stop = 0;                                                         \
                                                                                                   \
        time = 0;                                                                                  \
        for (i = 0; i < (iter + skip); i++) {                                                      \
            nvshmemx_barrier_all_warp();                                                           \
            if (i > skip) start = clock64();                                                       \
            nvshmemx_##TYPENAME##_##OP##_reduce_warp(team, dest, source, nelems);                  \
            if (i > skip) stop = clock64();                                                        \
            time += (stop - start);                                                                \
        }                                                                                          \
        nvshmemx_barrier_all_warp();                                                               \
        if (!threadIdx.x && !mype) {                                                               \
            time = time / iter;                                                                    \
            *usec = time * 1000 / clockrate;                                                       \
        }                                                                                          \
    } while (0)

#define CALL_RDXN_THREAD(TYPENAME, TYPE, OP, usec)                                           \
    do {                                                                                     \
        double time;                                                                         \
        int i;                                                                               \
        int iter = MAX_ITERS;                                                                \
        int skip = MAX_SKIP;                                                                 \
        long long int start = 0, stop = 0;                                                   \
                                                                                             \
        time = 0;                                                                            \
        for (i = 0; i < (iter + skip); i++) {                                                \
            nvshmem_barrier_all();                                                           \
            if (i > skip) start = clock64();                                                 \
            nvshmem_##TYPENAME##_##OP##_reduce(team, dest, source, nelems);                  \
            if (i > skip) stop = clock64();                                                  \
            time += (stop - start);                                                          \
        }                                                                                    \
        nvshmem_barrier_all();                                                               \
        if (!threadIdx.x && !mype) {                                                         \
            time = time / iter;                                                              \
            *usec = time * 1000 / clockrate;                                                 \
        }                                                                                    \
    } while (0)

#define DEFN_RDXN_BLOCK_FXNS(TYPENAME, TYPE)                                                     \
    __global__ void test_##TYPENAME##_call_kern_block(                                           \
        nvshmem_team_t team, TYPE *dest, const TYPE *source, size_t nelems, int mype, double *d_time_avg, \
        double *usec_sum, double *usec_prod, double *usec_and, double *usec_or,                   \
        double *usec_xor, double *usec_min, double *usec_max) {                                   \
                                                                                                 \
        if (!blockIdx.x) nvshmemx_barrier_all_block();                                           \
        if (!blockIdx.x && nelems < 65536) {                                                     \
            CALL_RDXN_BLOCK(TYPENAME, TYPE, sum, usec_sum);                                      \
            CALL_RDXN_BLOCK(TYPENAME, TYPE, prod, usec_prod);                                    \
            CALL_RDXN_BLOCK(TYPENAME, TYPE, and, usec_and);                                      \
            CALL_RDXN_BLOCK(TYPENAME, TYPE, or, usec_or);                                        \
            CALL_RDXN_BLOCK(TYPENAME, TYPE, xor, usec_xor);                                      \
            CALL_RDXN_BLOCK(TYPENAME, TYPE, min, usec_min);                                      \
            CALL_RDXN_BLOCK(TYPENAME, TYPE, max, usec_max);                                      \
        }                                                                                        \
        if (!blockIdx.x) nvshmemx_barrier_all_block();                                           \
    }

#define DEFN_RDXN_WARP_FXNS(TYPENAME, TYPE)                                                       \
    __global__ void test_##TYPENAME##_call_kern_warp(                                             \
        nvshmem_team_t team, TYPE *dest, const TYPE *source, size_t nelems, int mype, double *d_time_avg, \
        double *usec_sum, double *usec_prod, double *usec_and, double *usec_or,                   \
        double *usec_xor, double *usec_min, double *usec_max) {                                   \
                                                                                                  \
        if (!blockIdx.x) nvshmemx_barrier_all_block();                                            \
        if (!blockIdx.x && !(threadIdx.x / warpSize) && nelems < 4096) {                          \
            CALL_RDXN_WARP(TYPENAME, TYPE, sum, usec_sum);                                        \
            CALL_RDXN_WARP(TYPENAME, TYPE, prod, usec_prod);                                      \
            CALL_RDXN_WARP(TYPENAME, TYPE, and, usec_and);                                        \
            CALL_RDXN_WARP(TYPENAME, TYPE, or, usec_or);                                          \
            CALL_RDXN_WARP(TYPENAME, TYPE, xor, usec_xor);                                        \
            CALL_RDXN_WARP(TYPENAME, TYPE, min, usec_min);                                        \
            CALL_RDXN_WARP(TYPENAME, TYPE, max, usec_max);                                        \
        }                                                                                         \
        __syncthreads();                                                                          \
        if (!blockIdx.x) nvshmemx_barrier_all_block();                                            \
    }

#define DEFN_RDXN_THREAD_FXNS(TYPENAME, TYPE)                                                    \
    __global__ void test_##TYPENAME##_call_kern_thread(                                          \
        nvshmem_team_t team, TYPE *dest, const TYPE *source, size_t nelems, int mype, double *d_time_avg,          \
        double *usec_sum, double *usec_prod, double *usec_and, double *usec_or,                  \
        double *usec_xor, double *usec_min, double *usec_max) {                                  \
                                                                                                 \
        if (!blockIdx.x) nvshmemx_barrier_all_block();                                           \
        if (!blockIdx.x && !threadIdx.x && nelems < 512) {                                       \
            CALL_RDXN_THREAD(TYPENAME, TYPE, sum, usec_sum);                                     \
            CALL_RDXN_THREAD(TYPENAME, TYPE, prod, usec_prod);                                   \
            CALL_RDXN_THREAD(TYPENAME, TYPE, and, usec_and);                                     \
            CALL_RDXN_THREAD(TYPENAME, TYPE, or, usec_or);                                       \
            CALL_RDXN_THREAD(TYPENAME, TYPE, xor, usec_xor);                                     \
            CALL_RDXN_THREAD(TYPENAME, TYPE, min, usec_min);                                     \
            CALL_RDXN_THREAD(TYPENAME, TYPE, max, usec_max);                                     \
        }                                                                                        \
        __syncthreads();                                                                         \
        if (!blockIdx.x) nvshmemx_barrier_all_block();                                           \
    }

DEFN_RDXN_THREAD_FXNS(int32, int32_t);
DEFN_RDXN_THREAD_FXNS(int64, int64_t);
DEFN_RDXN_WARP_FXNS(int32, int32_t);
DEFN_RDXN_WARP_FXNS(int64, int64_t);
DEFN_RDXN_BLOCK_FXNS(int32, int32_t);
DEFN_RDXN_BLOCK_FXNS(int64, int64_t);

#define RUN_ITERS(TYPENAME, TYPE, GROUP)                                                           \
    do {                                                                                           \
        nvshmem_barrier_all();                                                                     \
        j = 0;                                                                                     \
        for (num_elems = 1; num_elems < max_elems; num_elems *= 2) {                               \
            size_arr[j] = num_elems * sizeof(TYPE);                                                \
            nvshmem_barrier_all();                                                                 \
            test_##TYPENAME##_call_kern_##GROUP<<<num_blocks, nvshm_test_num_tpb, 0, stream>>>(    \
                team, (TYPE *)dest, (const TYPE *)source, num_elems, mype, d_time_avg, &h_sum_lat[j], &h_prod_lat[j],          \
                &h_and_lat[j], &h_or_lat[j], &h_xor_lat[j], &h_min_lat[j], &h_max_lat[j]);         \
            cuda_check_error();                                                                    \
            CUDA_CHECK(cudaStreamSynchronize(stream));                                             \
            j++;                                                                                   \
        }                                                                                          \
    } while (0)

int rdxn_calling_kernel(nvshmem_team_t team, void *dest, const void *source, int mype, int max_elems, cudaStream_t stream,
                        double *d_time_avg, run_opt_t run_options, void **h_tables) {
    int status = 0;
    int nvshm_test_num_tpb = TEST_NUM_TPB_BLOCK;
    int num_blocks = 1;
    int num_elems = 1;
    int j;
    uint64_t *size_arr = (uint64_t *)h_tables[0];
    double *h_sum_lat = (double *)h_tables[1];
    double *h_prod_lat = (double *)h_tables[2];
    double *h_and_lat = (double *)h_tables[3];
    double *h_or_lat = (double *)h_tables[4];
    double *h_xor_lat = (double *)h_tables[5];
    double *h_min_lat = (double *)h_tables[6];
    double *h_max_lat = (double *)h_tables[7];

    // if (!mype) printf("Transfer size in bytes and latency of thread/warp/block variants of all
    // operations of reduction API in us\n");
    if (run_options.run_thread) {

        RUN_ITERS(int32, int32_t, thread);
        if (!mype) {
            print_table("device_reduction", "int32-sum-t", "size (Bytes)", "latency", "us", '-', size_arr, h_sum_lat, j);
            print_table("device_reduction", "int32-prod-t", "size (Bytes)", "latency", "us", '-', size_arr, h_prod_lat, j);
            print_table("device_reduction", "int32-and-t", "size (Bytes)", "latency", "us", '-', size_arr, h_and_lat, j);
            print_table("device_reduction", "int32-or-t", "size (Bytes)", "latency", "us", '-', size_arr, h_or_lat, j);
            print_table("device_reduction", "int32-xor-t", "size (Bytes)", "latency", "us", '-', size_arr, h_xor_lat, j);
            print_table("device_reduction", "int32-min-t", "size (Bytes)", "latency", "us", '-', size_arr, h_min_lat, j);
            print_table("device_reduction", "int32-max-t", "size (Bytes)", "latency", "us", '-', size_arr, h_max_lat, j);
        }

        RUN_ITERS(int64, int64_t, thread);
        if (!mype) {
            print_table("device_reduction", "int64-sum-t", "size (Bytes)", "latency", "us", '-', size_arr, h_sum_lat, j);
            print_table("device_reduction", "int64-prod-t", "size (Bytes)", "latency", "us", '-', size_arr, h_prod_lat, j);
            print_table("device_reduction", "int64-and-t", "size (Bytes)", "latency", "us", '-', size_arr, h_and_lat, j);
            print_table("device_reduction", "int64-or-t", "size (Bytes)", "latency", "us", '-', size_arr, h_or_lat, j);
            print_table("device_reduction", "int64-xor-t", "size (Bytes)", "latency", "us", '-', size_arr, h_xor_lat, j);
            print_table("device_reduction", "int64-min-t", "size (Bytes)", "latency", "us", '-', size_arr, h_min_lat, j);
            print_table("device_reduction", "int64-max-t", "size (Bytes)", "latency", "us", '-', size_arr, h_max_lat, j);
        }
    }

    if (run_options.run_warp) {

        RUN_ITERS(int32, int32_t, warp);
        if (!mype) {
            print_table("device_reduction", "int32-sum-w", "size (Bytes)", "latency", "us", '-', size_arr, h_sum_lat, j);
            print_table("device_reduction", "int32-prod-w", "size (Bytes)", "latency", "us", '-', size_arr, h_prod_lat, j);
            print_table("device_reduction", "int32-and-w", "size (Bytes)", "latency", "us", '-', size_arr, h_and_lat, j);
            print_table("device_reduction", "int32-or-w", "size (Bytes)", "latency", "us", '-', size_arr, h_or_lat, j);
            print_table("device_reduction", "int32-xor-w", "size (Bytes)", "latency", "us", '-', size_arr, h_xor_lat, j);
            print_table("device_reduction", "int32-min-w", "size (Bytes)", "latency", "us", '-', size_arr, h_min_lat, j);
            print_table("device_reduction", "int32-max-w", "size (Bytes)", "latency", "us", '-', size_arr, h_max_lat, j);
        }

        RUN_ITERS(int64, int64_t, warp);
        if (!mype) {
            print_table("device_reduction", "int64-sum-w", "size (Bytes)", "latency", "us", '-', size_arr, h_sum_lat, j);
            print_table("device_reduction", "int64-prod-w", "size (Bytes)", "latency", "us", '-', size_arr, h_prod_lat, j);
            print_table("device_reduction", "int64-and-w", "size (Bytes)", "latency", "us", '-', size_arr, h_and_lat, j);
            print_table("device_reduction", "int64-or-w", "size (Bytes)", "latency", "us", '-', size_arr, h_or_lat, j);
            print_table("device_reduction", "int64-xor-w", "size (Bytes)", "latency", "us", '-', size_arr, h_xor_lat, j);
            print_table("device_reduction", "int64-min-w", "size (Bytes)", "latency", "us", '-', size_arr, h_min_lat, j);
            print_table("device_reduction", "int64-max-w", "size (Bytes)", "latency", "us", '-', size_arr, h_max_lat, j);
        }
    }

    if (run_options.run_block) {

        RUN_ITERS(int32, int32_t, block);
        if (!mype) {
            print_table("device_reduction", "int32-sum-b", "size (Bytes)", "latency", "us", '-', size_arr, h_sum_lat, j);
            print_table("device_reduction", "int32-prod-b", "size (Bytes)", "latency", "us", '-', size_arr, h_prod_lat, j);
            print_table("device_reduction", "int32-and-b", "size (Bytes)", "latency", "us", '-', size_arr, h_and_lat, j);
            print_table("device_reduction", "int32-or-b", "size (Bytes)", "latency", "us", '-', size_arr, h_or_lat, j);
            print_table("device_reduction", "int32-xor-b", "size (Bytes)", "latency", "us", '-', size_arr, h_xor_lat, j);
            print_table("device_reduction", "int32-min-b", "size (Bytes)", "latency", "us", '-', size_arr, h_min_lat, j);
            print_table("device_reduction", "int32-max-b", "size (Bytes)", "latency", "us", '-', size_arr, h_max_lat, j);
        }

        RUN_ITERS(int64, int64_t, block);
        if (!mype) {
            print_table("device_reduction", "int64-sum-b", "size (Bytes)", "latency", "us", '-', size_arr, h_sum_lat, j);
            print_table("device_reduction", "int64-prod-b", "size (Bytes)", "latency", "us", '-', size_arr, h_prod_lat, j);
            print_table("device_reduction", "int64-and-b", "size (Bytes)", "latency", "us", '-', size_arr, h_and_lat, j);
            print_table("device_reduction", "int64-or-b", "size (Bytes)", "latency", "us", '-', size_arr, h_or_lat, j);
            print_table("device_reduction", "int64-xor-b", "size (Bytes)", "latency", "us", '-', size_arr, h_xor_lat, j);
            print_table("device_reduction", "int64-min-b", "size (Bytes)", "latency", "us", '-', size_arr, h_min_lat, j);
            print_table("device_reduction", "int64-max-b", "size (Bytes)", "latency", "us", '-', size_arr, h_max_lat, j);
        }
    }

    return status;
}

int main(int argc, char **argv) {
    int status = 0;
    int mype, array_size;
    size_t size = 0;
    size_t alloc_size;
    int num_elems;
    char *value = NULL;
    int max_elems = (MAX_ELEMS / 2);
    int *h_buffer = NULL;
    int *d_source, *d_dest;
    int *h_source, *h_dest;
    char size_string[100];
    double *d_time_avg;
    cudaStream_t cstrm;
    run_opt_t run_options;
    void **h_tables;

    PROCESS_OPTS(run_options);

    size = page_size_roundoff((MAX_ELEMS) * sizeof(LARGEST_DT));   // send buf
    size += page_size_roundoff((MAX_ELEMS) * sizeof(LARGEST_DT));  // recv buf

    DEBUG_PRINT("symmetric size requested %lu\n", size);
    sprintf(size_string, "%lu", size);

    status = setenv("NVSHMEM_SYMMETRIC_SIZE", size_string, 1);
    if (status) {
        fprintf(stderr, "setenv failed \n");
        status = -1;
        goto out;
    }

    value = getenv("NVSHMEM_PERF_COLL_MAX_ELEMS");

    if (NULL != value) {
        max_elems = atoi(value);
        if (0 == max_elems) {
            fprintf(stderr, "Warning: min max elem size = 1\n");
            max_elems = 1;
        }
    }

    array_size = floor(log2((float)max_elems)) + 1;

    init_wrapper(&argc, &argv);
    alloc_tables(&h_tables, 8, array_size);

    mype = nvshmem_my_pe();

    CUDA_CHECK(cudaStreamCreateWithFlags(&cstrm, cudaStreamNonBlocking));

    d_time_avg = (double *)nvshmem_align(getpagesize(), sizeof(double) * 2);

    num_elems = MAX_ELEMS / 2;
    alloc_size = (num_elems * 2) * sizeof(long);

    CUDA_CHECK(cudaHostAlloc(&h_buffer, alloc_size, cudaHostAllocDefault));
    h_source = (int32_t *)h_buffer;
    h_dest = (int32_t *)&h_source[num_elems];

    d_source = (int32_t *)nvshmem_align(getpagesize(), num_elems * sizeof(LARGEST_DT));
    d_dest = (int32_t *)nvshmem_align(getpagesize(), num_elems * sizeof(LARGEST_DT));

    CUDA_CHECK(cudaMemcpyAsync(d_source, h_source, (sizeof(LARGEST_DT) * num_elems),
                               cudaMemcpyHostToDevice, cstrm));
    CUDA_CHECK(cudaMemcpyAsync(d_dest, h_dest, (sizeof(LARGEST_DT) * num_elems),
                               cudaMemcpyHostToDevice, cstrm));

    rdxn_calling_kernel(NVSHMEM_TEAM_WORLD, d_dest, d_source, mype, max_elems, cstrm, d_time_avg, run_options, h_tables);

    DEBUG_PRINT("last error = %s\n", cudaGetErrorString(cudaGetLastError()));

    CUDA_CHECK(cudaMemcpyAsync(h_source, d_source, (sizeof(LARGEST_DT) * num_elems),
                               cudaMemcpyDeviceToHost, cstrm));
    CUDA_CHECK(cudaMemcpyAsync(h_dest, d_dest, (sizeof(LARGEST_DT) * num_elems),
                               cudaMemcpyDeviceToHost, cstrm));

    nvshmem_barrier_all();

    CUDA_CHECK(cudaFreeHost(h_buffer));
    nvshmem_free(d_source);
    nvshmem_free(d_dest);
    nvshmem_free(d_time_avg);

    CUDA_CHECK(cudaStreamDestroy(cstrm));

    finalize_wrapper();

out:
    return 0;
}
