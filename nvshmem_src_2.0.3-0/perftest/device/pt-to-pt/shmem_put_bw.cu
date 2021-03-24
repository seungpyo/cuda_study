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

#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <getopt.h>
#include "utils.h"

#define MAX_MSG_SIZE (32 * 1024 * 1024)

#define MAX_ITERS 200
#define MAX_SKIP 20
#define BLOCKS 4
#define THREADS_PER_BLOCK 1024

__global__ void bw(volatile double *data_d, volatile unsigned int *counter_d, int len, int pe,
                   int iter, int skip, double *bw_result) {
    int i, peer;
    unsigned int counter;
    int tid = (threadIdx.x * blockDim.y * blockDim.z + threadIdx.y * blockDim.z + threadIdx.z);
    int bid = blockIdx.x;
    int nblocks = gridDim.x;
    long long int start = 0, stop = 0;
    double time = 0;

    peer = !pe;
    for (i = 0; i < (iter + skip); i++) {
        if (i == skip) {
            nvshmem_quiet();
            start = clock64();
        }

        nvshmemx_double_put_nbi_block((double *)data_d + (bid * (len / nblocks)),
                                      (double *)data_d + (bid * (len / nblocks)), len / nblocks,
                                      peer);

        // synchronizing across blocks
        __syncthreads();
        if (!tid) {
            __threadfence();
            counter = atomicInc((unsigned int *)counter_d, UINT_MAX);
            if (counter == (gridDim.x * (i + 1) - 1)) {
                *(counter_d + 1) += 1;
            }
            while (*(counter_d + 1) != i + 1)
                ;
        }
        __syncthreads();
    }
    
    // synchronize and call nvshme_quiet
    __syncthreads();
    if (!tid) {
        __threadfence();
        counter = atomicInc((unsigned int *)counter_d, UINT_MAX);
        if (counter == (gridDim.x * (i + 1) - 1)) {
            nvshmem_quiet();
            *(counter_d + 1) += 1;
        }
        while (*(counter_d + 1) != i + 1)
            ;
    }
    __syncthreads();

    stop = clock64();
    time = (stop - start);

    if (!tid && !bid) {
        *bw_result = ((float)iter * (float)len * sizeof(double) * clockrate) / ((time / 1000) * 1024 * 1024 * 1024);
    }
}

int main(int argc, char *argv[]) {
    int mype, npes;
    double *data_d = NULL;
    unsigned int *counter_d;
    int max_blocks = BLOCKS, max_threads = THREADS_PER_BLOCK;

    int array_size, i;
    void **h_tables;
    uint64_t *h_size_arr;
    double *h_bw;

    int iter = MAX_ITERS;
    int skip = MAX_SKIP;

    init_wrapper(&argc, &argv);

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();

    if (npes != 2) {
        fprintf(stderr, "This test requires exactly two processes \n");
        goto finalize;
    }

    while (1) {
        int c;
        c = getopt(argc, argv, "c:t:h");
        if (c == -1) break;

        switch (c) {
            case 'c':
                max_blocks = strtol(optarg, NULL, 0);
                break;
            case 't':
                max_threads = strtol(optarg, NULL, 0);
                break;
            default:
            case 'h':
                printf("-c [CTAs] -t [THREADS] \n");
                goto finalize;
        }
    }

    array_size = floor(log2((float)MAX_MSG_SIZE)) + 1;
    alloc_tables(&h_tables, 2, array_size);
    h_size_arr = (uint64_t *)h_tables[0];
    h_bw = (double *)h_tables[1];

    data_d = (double *)nvshmem_malloc(MAX_MSG_SIZE);
    CUDA_CHECK(cudaMemset(data_d, 0, MAX_MSG_SIZE));

    CUDA_CHECK(cudaMalloc((void **)&counter_d, sizeof(unsigned int) * 2));
    CUDA_CHECK(cudaMemset(counter_d, 0, sizeof(unsigned int) * 2));

    CUDA_CHECK(cudaDeviceSynchronize());

    if (mype == 0) {
        i = 0;
        for (int size = 1024; size <= MAX_MSG_SIZE; size *= 2) {
            h_size_arr[i] = size;
            CUDA_CHECK(cudaMemset(counter_d, 0, sizeof(unsigned int) * 2));
            bw<<<max_blocks, max_threads>>>(data_d, counter_d, size / sizeof(double), mype, iter,
                                            skip, &h_bw[i]);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            nvshmem_barrier_all();
            i++;
        }
    } else {
        for (int size = 1024; size <= MAX_MSG_SIZE; size *= 2) {
            nvshmem_barrier_all();
        }
    }

    if (mype == 0) {
        print_table("shmem_put_bw", "None", "size (Bytes)", "BW", "GB/sec", '+', h_size_arr, h_bw, i);
    }

finalize:

    if (data_d) nvshmem_free(data_d);
    free_tables(h_tables, 2);
    finalize_wrapper();

    return 0;
}
