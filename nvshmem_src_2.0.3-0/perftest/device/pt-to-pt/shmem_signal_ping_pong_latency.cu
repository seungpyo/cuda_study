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

__global__ void ping_pong(uint64_t *flag_d,
                          int pe, int iter, int skip, int *hflag, double *lat_result) {
    long long int start, stop;
    double time;
    int i, tid, peer;

    peer = !pe;
    tid = threadIdx.x;

    for (i = 0; i < (iter + skip); i++) {
        if (i == skip) start = clock64();

        if (pe) {
            nvshmem_uint64_wait_until(flag_d, NVSHMEM_CMP_EQ, (i + 1));

            nvshmemx_uint64_signal(flag_d, (i + 1), peer);
        } else {
            nvshmemx_uint64_signal(flag_d, (i + 1), peer);

            nvshmem_uint64_wait_until(flag_d, NVSHMEM_CMP_EQ, (i + 1));
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
    int mype, npes;
    uint64_t *flag_d = NULL;
    cudaStream_t stream;

    int iter = 500;
    int skip = 50;

    void **h_tables;
    double *h_lat;
    uint64_t size = sizeof(uint64_t);

    init_wrapper(&c, &v);

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();

    if (npes != 2) {
        fprintf(stderr, "This test requires exactly two processes \n");
        goto finalize;
    }

    alloc_tables(&h_tables, 2, 1);
    h_lat = (double *)h_tables[1];

    flag_d = (uint64_t *)nvshmem_malloc(sizeof(uint64_t));
    CUDA_CHECK(cudaMemset(flag_d, 0, sizeof(uint64_t)));

    int *hflag, *hflag_d;
    CUDA_CHECK(cudaHostAlloc((void **)&hflag, sizeof(uint64_t), 0));
    *hflag = 0;
    CUDA_CHECK(cudaHostGetDevicePointer(&hflag_d, hflag, 0));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    nvshmem_barrier_all();

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    if (mype == 0) {
        printf("Note: This test measures full round-trip latency\n");
    }
    
    {
        int status = 0;
        void *args[] = {&flag_d, &mype, &iter, &skip, &hflag_d, &h_lat};

        CUDA_CHECK(cudaMemset(flag_d, 0, sizeof(uint64_t)));
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

        CUDA_CHECK(cudaDeviceSynchronize());
    }

    if (mype == 0) {
        print_table("shmem_sig_ping_lat", "None", "size (Bytes)", "latency", "us", '-', &size, h_lat, 1);
    }
finalize:

    if (flag_d) nvshmem_free(flag_d);
    free_tables(h_tables, 2);
    finalize_wrapper();

    return 0;
}
