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

#define MAX_MSG_SIZE 1 * 1024 * 1024
#define UNROLL 8

__global__ void ping_pong(volatile int *data_d, volatile int *flag_d, volatile int *flag_d_local,
                          int len, int pe, int iter, int skip, int *hflag, double *lat_result) {
    long long int start, stop;
    double time;
    int i, tid, peer;

    peer = !pe;
    tid = threadIdx.x;

    for (i = 0; i < (iter + skip); i++) {
        if (i == skip) start = clock64();

        if (pe) {
            nvshmem_int_wait_until((int *)flag_d, NVSHMEM_CMP_EQ, (i + 1));

            nvshmem_int_put_nbi((int *)data_d, (int *)data_d, len, peer);

            nvshmem_fence();

            nvshmemx_int_signal((int *)flag_d, i + 1, peer);
        } else {
            nvshmem_int_put_nbi((int *)data_d, (int *)data_d, len, peer);

            nvshmem_fence();

            nvshmemx_int_signal((int *)flag_d, i + 1, peer);

            nvshmem_int_wait_until((int *)flag_d, NVSHMEM_CMP_EQ, (i + 1));
        }
    }
    stop = clock64();
    nvshmem_quiet();
    *hflag = 1;

    if ((pe == 0) && !tid) {
        time = (stop - start) / iter;
        *lat_result = time * 1000 / clockrate;
    }
}

int main(int c, char *v[]) {
    int mype, npes, size;
    int *flag_d = NULL, *data_d = NULL, *flag_d_local = NULL;
    cudaStream_t stream;

    int iter = 500;
    int skip = 50;
    int max_msg_size = MAX_MSG_SIZE;

    int array_size, i;
    void **h_tables;
    uint64_t *h_size_arr;
    double *h_lat;
    double *cur_lat;

    init_wrapper(&c, &v);

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();

    if (npes != 2) {
        fprintf(stderr, "This test requires exactly two processes \n");
        goto finalize;
    }

    data_d = (int *)nvshmem_malloc(max_msg_size);
    flag_d = (int *)nvshmem_malloc(sizeof(int));
    flag_d_local = (int *)nvshmem_malloc(sizeof(int));
    CUDA_CHECK(cudaMemset(data_d, 0, max_msg_size));
    CUDA_CHECK(cudaMemset(flag_d, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(flag_d_local, 0, sizeof(int)));

    array_size = floor(log2((float)max_msg_size)) + 1;
    alloc_tables(&h_tables, 2, array_size);
    h_size_arr = (uint64_t *)h_tables[0];
    h_lat = (double *)h_tables[1];

    int *hflag, *hflag_d;
    CUDA_CHECK(cudaHostAlloc((void **)&hflag, sizeof(int), 0));
    *hflag = 0;
    CUDA_CHECK(cudaHostGetDevicePointer(&hflag_d, hflag, 0));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    nvshmem_barrier_all();

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    if (mype == 0) {
        printf("Note: This test measures full round-trip latency\n");
    }

    i = 0;
    for (size = sizeof(int); size <= max_msg_size; size *= 2) {
        int nelems, status = 0;
        nelems = size / sizeof(int);
        h_size_arr[i] = size;
        cur_lat = &h_lat[i];
        void *args[] = {&data_d, &flag_d, &flag_d_local, &nelems, &mype, &iter, &skip, &hflag_d, &cur_lat};

        CUDA_CHECK(cudaMemset(flag_d, 0, sizeof(int)));
        CUDA_CHECK(cudaDeviceSynchronize());
        nvshmem_barrier_all();

        *hflag = 0;
        status = nvshmemx_collective_launch((const void *)ping_pong, 1, 1, args, 0, stream);
        if (status != NVSHMEMX_SUCCESS) {
            fprintf(stderr, "shmemx_collective_launch failed %d \n", status);
            exit(-1);
        }

        while (*((volatile int *)hflag) != 1)
            ;

        nvshmem_barrier_all();
        i++;
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    if (mype == 0) {
        print_table("shmem_put_ping_lat", "None", "size (Bytes)", "latency", "us", '-', h_size_arr, h_lat, i);
    }
finalize:

    if (data_d) nvshmem_free(data_d);
    if (flag_d) nvshmem_free(flag_d);
    if (flag_d_local) nvshmem_free(flag_d_local);
    free_tables(h_tables, 2);
    finalize_wrapper();

    return 0;
}
