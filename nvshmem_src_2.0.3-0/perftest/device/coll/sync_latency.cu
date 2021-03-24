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

__device__ double *pWrk;

__global__ void test_sync_call_kern(nvshmem_team_t team, int mype, double *d_time_avg, double *h_thread_lat,
                                    double *h_warp_lat, double *h_block_lat) {
    int iter = MAX_ITERS;
    int skip = MAX_SKIP;
    long long int start = 0, stop = 0;
    double thread_usec, warp_usec, block_usec, time = 0;
    int i;
    double *dest_r, *source_r;
    int PE_size = nvshmem_team_n_pes(team);

    source_r = d_time_avg;
    dest_r = (double *)((double *)d_time_avg + 1);

    if (!blockIdx.x) nvshmemx_barrier_all_block();

    time = 0;

    if (!blockIdx.x && !threadIdx.x) {
        for (i = 0; i < (iter + skip); i++) {
            if (i > skip) start = clock64();
            nvshmem_team_sync(team);
            if (i > skip) stop = clock64();
            time += (stop - start);
        }
        nvshmem_barrier_all();
        if (!threadIdx.x) {
            *source_r = time;
            nvshmem_double_sum_reduce(team, dest_r, source_r, 1);
            time = *dest_r;

            if (mype == 0) {
                time = time / iter;
                time = time / PE_size;
                thread_usec = time * 1000 / clockrate;
            }
        }
    }

    __syncthreads();
    if (!blockIdx.x) nvshmemx_barrier_all_block();

    time = 0;

    if (!blockIdx.x && !(threadIdx.x / warpSize)) {
        for (i = 0; i < (iter + skip); i++) {
            if (i > skip) start = clock64();
            nvshmemx_team_sync_warp(team);
            if (i > skip) stop = clock64();
            time += (stop - start);
        }
        nvshmemx_barrier_all_warp();
        if (!threadIdx.x) {
            *source_r = time;
            nvshmem_double_sum_reduce(team, dest_r, source_r, 1);
            time = *dest_r;

            if (mype == 0) {
                time = time / iter;
                time = time / PE_size;
                warp_usec = time * 1000 / clockrate;
            }
        }
    }

    __syncthreads();
    if (!blockIdx.x) nvshmemx_barrier_all_block();

    time = 0;

    if (!blockIdx.x) {
        for (i = 0; i < (iter + skip); i++) {
            if (i > skip) start = clock64();
            nvshmemx_team_sync_block(team);
            if (i > skip) stop = clock64();
            time += (stop - start);
        }
        nvshmemx_barrier_all_block();
        if (!threadIdx.x) {
            *source_r = time;
            nvshmem_double_sum_reduce(team, dest_r, source_r, 1);
            time = *dest_r;

            if (mype == 0) {
                time = time / iter;
                time = time / PE_size;
                block_usec = time * 1000 / clockrate;
            }
        }
    }

    __syncthreads();
    if (!blockIdx.x && !threadIdx.x) nvshmem_barrier_all();

    if (!threadIdx.x && !blockIdx.x && !mype) {
        *h_thread_lat = thread_usec;
        *h_warp_lat = warp_usec;
        *h_block_lat = block_usec;
    }
}

__global__ void test_sync_all_call_kern(nvshmem_team_t team, int mype, double *d_time_avg, double *h_thread_lat,
                                        double *h_warp_lat, double *h_block_lat) {
    int iter = MAX_ITERS;
    int skip = MAX_SKIP;
    long long int start = 0, stop = 0;
    double thread_usec, warp_usec, block_usec, time = 0;
    int i;
    double *dest_r, *source_r;
    int PE_size = nvshmem_team_n_pes(team);

    source_r = d_time_avg;
    dest_r = (double *)((double *)d_time_avg + 1);

    if (!blockIdx.x) nvshmemx_barrier_all_block();

    time = 0;

    if (!blockIdx.x && !threadIdx.x) {
        for (i = 0; i < (iter + skip); i++) {
            if (i > skip) start = clock64();
            nvshmem_sync_all();
            if (i > skip) stop = clock64();
            time += (stop - start);
        }
        if (!threadIdx.x) {
            *source_r = time;
            nvshmem_double_sum_reduce(team, dest_r, source_r, 1);
            time = *dest_r;

            if (mype == 0) {
                time = time / iter;
                time = time / PE_size;
                thread_usec = time * 1000 / clockrate;
            }
        }
    }

    __syncthreads();
    if (!blockIdx.x) nvshmemx_barrier_all_block();

    time = 0;

    if (!blockIdx.x && !(threadIdx.x / warpSize)) {
        for (i = 0; i < (iter + skip); i++) {
            if (i > skip) start = clock64();
            nvshmemx_sync_all_warp();
            if (i > skip) stop = clock64();
            time += (stop - start);
        }
        if (!threadIdx.x) {
            *source_r = time;
            nvshmem_double_sum_reduce(team, dest_r, source_r, 1);
            time = *dest_r;

            if (mype == 0) {
                time = time / iter;
                time = time / PE_size;
                warp_usec = time * 1000 / clockrate;
            }
        }
    }

    __syncthreads();
    if (!blockIdx.x) nvshmemx_barrier_all_block();

    time = 0;

    if (!blockIdx.x) {
        for (i = 0; i < (iter + skip); i++) {
            if (i > skip) start = clock64();
            nvshmemx_sync_all_block();
            if (i > skip) stop = clock64();
            time += (stop - start);
        }
        if (!threadIdx.x) {
            *source_r = time;
            nvshmem_double_sum_reduce(team, dest_r, source_r, 1);
            time = *dest_r;

            if (mype == 0) {
                time = time / iter;
                time = time / PE_size;
                block_usec = time * 1000 / clockrate;
            }
        }
    }

    __syncthreads();
    if (!blockIdx.x && !threadIdx.x) nvshmem_barrier_all();

    if (!threadIdx.x && !blockIdx.x && !mype) {
        *h_thread_lat = thread_usec;
        *h_warp_lat = warp_usec;
        *h_block_lat = block_usec;
    }
}

int sync_calling_kernel(nvshmem_team_t team, cudaStream_t stream, int mype, double *d_time_avg, void **h_tables) {
    int status = 0;
    int nvshm_test_num_tpb = TEST_NUM_TPB_BLOCK;
    int num_blocks = 1;
    double *h_thread_lat = (double *)h_tables[0];
    double *h_warp_lat = (double *)h_tables[1];
    double *h_block_lat = (double *)h_tables[2];
    uint64_t num_tpb = TEST_NUM_TPB_BLOCK;

    nvshmem_barrier_all();
    test_sync_call_kern<<<num_blocks, nvshm_test_num_tpb, 0, stream>>>(team, mype, d_time_avg,
                                                                h_thread_lat, h_warp_lat, h_block_lat);
    cuda_check_error();
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (!mype) {
        print_table("sync_device", "thread", "threads per block", "latency", "us", '-', &num_tpb, h_thread_lat, 1);
        print_table("sync_device", "warp", "threads per block", "latency", "us", '-', &num_tpb, h_warp_lat, 1);
        print_table("sync_device", "block", "threads per block", "latency", "us", '-', &num_tpb, h_block_lat, 1);
    }

    nvshmem_barrier_all();
    test_sync_all_call_kern<<<num_blocks, nvshm_test_num_tpb, 0, stream>>>(team, mype, d_time_avg,
                                                                        h_thread_lat, h_warp_lat, h_block_lat);
    cuda_check_error();
    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (!mype) {
        print_table("sync_all_device", "thread", "threads per block", "latency", "us", '-', &num_tpb, h_thread_lat, 1);
        print_table("sync_all_device", "warp", "threads per block", "latency", "us", '-', &num_tpb, h_warp_lat, 1);
        print_table("sync_all_device", "block", "threads per block", "latency", "us", '-', &num_tpb, h_block_lat, 1);
    }

    return status;
}

int main(int argc, char **argv) {
    int mype;
    double *d_time_avg;
    cudaStream_t cstrm;
    void **h_tables;

    init_wrapper(&argc, &argv);
    alloc_tables(&h_tables, 3, 1);

    mype = nvshmem_my_pe();
    CUDA_CHECK(cudaStreamCreateWithFlags(&cstrm, cudaStreamNonBlocking));

    d_time_avg = (double *)nvshmem_malloc(sizeof(double) * 2);

    sync_calling_kernel(NVSHMEM_TEAM_WORLD, cstrm, mype, d_time_avg, h_tables);

    nvshmem_barrier_all();

    nvshmem_free(d_time_avg);

    CUDA_CHECK(cudaStreamDestroy(cstrm));
    free_tables(h_tables, 3);
    finalize_wrapper();

    return 0;
}
