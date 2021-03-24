/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.  All rights reserved.
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
#include <unistd.h>
#include "utils.h"

#define THREADS 512
#define MAX_MSG_SIZE 64 * 1024
#define UNROLL 8

__global__ void p_latency(volatile int *data_d, volatile int *flag_d, int len, int pe, int iter,
                          int skip, double *lat_result) {
    long long int start, stop;
    double time;
    int i, j, tid, peer;

    peer = !pe;
    tid = threadIdx.x;

    for (i = 0; i < (iter + skip); i++) {
        if (i == skip) start = clock64();

        for (j = tid; j < len; j += THREADS) {
            nvshmem_int_p((int *)data_d + j, *(data_d + j), peer);
        }
        __syncthreads();
        if (!tid) {
            nvshmem_quiet();
        }
        __syncthreads();
    }
    stop = clock64();

    if (!tid) {
        time = (stop - start) / iter;
        *lat_result = time * 1000 / clockrate;
    }
}

int main(int c, char *v[]) {
    int mype, npes, size;
    int *flag_d = NULL, *data_d = NULL;

    int iter = 50;
    int skip = 5;
    int max_msg_size = MAX_MSG_SIZE;

    int array_size, i;
    void **h_tables;
    uint64_t *h_size_arr;
    double *h_lat;

    init_wrapper(&c, &v);

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();

    if (npes != 2) {
        fprintf(stderr, "This test requires exactly two processes \n");
        goto finalize;
    }

    array_size = floor(log2((float)max_msg_size)) + 1;
    alloc_tables(&h_tables, 2, array_size);
    h_size_arr = (uint64_t *)h_tables[0];
    h_lat = (double *)h_tables[1];

    data_d = (int *)nvshmem_malloc(max_msg_size);
    flag_d = (int *)nvshmem_malloc(sizeof(int));
    CUDA_CHECK(cudaMemset(data_d, 0, max_msg_size));
    CUDA_CHECK(cudaMemset(flag_d, 0, sizeof(int)));

    nvshmem_barrier_all();

    CUDA_CHECK(cudaDeviceSynchronize());

    i = 0;
    for (size = sizeof(int); size <= max_msg_size; size *= 2) {
        if (!mype) {
            int nelems;
            h_size_arr[i] = size;
            nelems = size / sizeof(int);

            p_latency<<<1, THREADS>>>(data_d, flag_d, nelems, mype, iter, skip, &h_lat[i]);

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            i++;
        }

        nvshmem_barrier_all();
    }

    if (mype == 0) {
        print_table("shmem_p_latency", "None", "size (Bytes)", "latency", "us", '-', h_size_arr, h_lat, i);
    }

finalize:

    if (data_d) nvshmem_free(data_d);
    if (flag_d) nvshmem_free(flag_d);
    free_tables(h_tables, 2);
    finalize_wrapper();

    return 0;
}
