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
#define DATATYPE int64_t

#define CALL_ALLTOALL(TYPENAME, TYPE)                                                                        \
    __global__ void test_##TYPENAME##_alltoall_call_kern(nvshmem_team_t team, TYPE *dest, const TYPE *source, \
                                                         size_t nelems, int mype, double *d_time_avg,   \
                                                         double *h_thread_lat, double *h_warp_lat,      \
                                                         double *h_block_lat) {                         \
        int iter = MAX_ITERS;                                                                      \
        int skip = MAX_SKIP;                                                                       \
        long long int start = 0, stop = 0;                                                         \
        double time = 0;                                                                           \
        double thread_usec = 0, warp_usec = 0, block_usec = 0;                                     \
        int i;                                                                                     \
        double *dest_r, *source_r;                                                                 \
        int PE_size = nvshmem_team_n_pes(team);                                                    \
                                                                                                   \
        source_r = d_time_avg;                                                                     \
        dest_r = (double *)((double *)d_time_avg + 1);                                             \
                                                                                                   \
        if (!blockIdx.x) nvshmemx_barrier_all_block();                                             \
                                                                                                   \
        time = 0;                                                                                  \
                                                                                                   \
        if (!blockIdx.x && !threadIdx.x && nelems < 512) {                                         \
            for (i = 0; i < (iter + skip); i++) {                                                  \
                nvshmem_barrier_all();                                                             \
                if (i > skip) start = clock64();                                                   \
                nvshmem_##TYPENAME##_alltoall(team, dest, source, nelems);                         \
                if (i > skip) stop = clock64();                                                    \
                time += (stop - start);                                                            \
            }                                                                                      \
            nvshmem_barrier_all();                                                                 \
            *source_r = time;                                                                      \
            nvshmem_double_sum_reduce(team, dest_r, source_r, 1);                                  \
            time = *dest_r;                                                                        \
                                                                                                   \
            if (mype == 0) {                                                                       \
                time = time / iter;                                                                \
                time = time / PE_size;                                                             \
                thread_usec = time * 1000 / clockrate;                                             \
            }                                                                                      \
        }                                                                                          \
                                                                                                   \
        __syncthreads();                                                                           \
        if (!blockIdx.x) nvshmemx_barrier_all_block();                                             \
                                                                                                   \
        time = 0;                                                                                  \
                                                                                                   \
        if (!blockIdx.x && !(threadIdx.x / warpSize) && nelems < 4096) {                           \
            for (i = 0; i < (iter + skip); i++) {                                                  \
                nvshmemx_barrier_all_warp();                                                       \
                if (i > skip) start = clock64();                                                   \
                nvshmemx_##TYPENAME##_alltoall_warp(team, dest, source, nelems);                   \
                if (i > skip) stop = clock64();                                                    \
                time += (stop - start);                                                            \
            }                                                                                      \
            nvshmemx_barrier_all_warp();                                                           \
            if (!threadIdx.x) {                                                                    \
                *source_r = time;                                                                  \
                nvshmem_double_sum_reduce(team, dest_r, source_r, 1);                              \
                time = *dest_r;                                                                    \
                                                                                                   \
                if (mype == 0) {                                                                   \
                    time = time / iter;                                                            \
                    time = time / PE_size;                                                         \
                    warp_usec = time * 1000 / clockrate;                                           \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
                                                                                                   \
        __syncthreads();                                                                           \
        if (!blockIdx.x) nvshmemx_barrier_all_block();                                             \
                                                                                                   \
        time = 0;                                                                                  \
                                                                                                   \
        if (!blockIdx.x) {                                                                         \
            for (i = 0; i < (iter + skip); i++) {                                                  \
                nvshmemx_barrier_all_block();                                                      \
                if (i > skip) start = clock64();                                                   \
                nvshmemx_##TYPENAME##_alltoall_block(team, dest, source, nelems);                  \
                if (i > skip) stop = clock64();                                                    \
                time += (stop - start);                                                            \
            }                                                                                      \
            nvshmemx_barrier_all_block();                                                          \
            if (!threadIdx.x) {                                                                    \
                *source_r = time;                                                                  \
                nvshmem_double_sum_reduce(team, dest_r, source_r, 1);                              \
                time = *dest_r;                                                                    \
                                                                                                   \
                if (mype == 0) {                                                                   \
                    time = time / iter;                                                            \
                    time = time / PE_size;                                                         \
                    block_usec = time * 1000 / clockrate;                                          \
                }                                                                                  \
            }                                                                                      \
        }                                                                                          \
                                                                                                   \
        if (!blockIdx.x) nvshmemx_barrier_all_block();                                             \
                                                                                                   \
        if (!threadIdx.x && !blockIdx.x && !mype) {                                                \
            *h_thread_lat = thread_usec;                                                           \
            *h_warp_lat = warp_usec;                                                               \
            *h_block_lat = block_usec;                                                             \
        }                                                                                          \
                                                                                                   \
    }

CALL_ALLTOALL(int32, int32_t);
CALL_ALLTOALL(int64, int64_t);

int alltoall_calling_kernel(nvshmem_team_t team, void *dest, const void *source, int mype, int max_elems,
                            cudaStream_t stream, double *d_time_avg, void **h_tables) {
    int status = 0;
    int nvshm_test_num_tpb = TEST_NUM_TPB_BLOCK;
    int num_blocks = 1;
    int num_elems = 1;
    int i;
    uint64_t *h_size_array = (uint64_t *)h_tables[0];
    double *h_thread_lat = (double *)h_tables[1];
    double *h_warp_lat = (double *)h_tables[2];
    double *h_block_lat = (double *)h_tables[3];

    nvshmem_barrier_all();
    i = 0;
    for (num_elems = 1; num_elems < max_elems; num_elems *= 2) {
        h_size_array[i] = num_elems * 4;
        test_int32_alltoall_call_kern<<<num_blocks, nvshm_test_num_tpb, 0, stream>>>(
            NVSHMEM_TEAM_WORLD, (int32_t *)dest, (const int32_t *)source, num_elems, mype, d_time_avg,
            &h_thread_lat[i], &h_warp_lat[i], &h_block_lat[i]);
        cuda_check_error();
        CUDA_CHECK(cudaStreamSynchronize(stream));
        i++;
    }

    if (!mype) {
        print_table("alltoall_device", "32-bit-thread", "size (Bytes)", "latency", "us", '-', h_size_array, h_thread_lat, i);
        print_table("alltoall_device", "32-bit-warp", "size (Bytes)", "latency", "us", '-', h_size_array, h_warp_lat, i);
        print_table("alltoall_device", "32-bit-block", "size (Bytes)", "latency", "us", '-', h_size_array, h_block_lat, i);
    }

    i = 0;
    for (num_elems = 1; num_elems < max_elems; num_elems *= 2) {
        h_size_array[i] = num_elems * 8;
        test_int64_alltoall_call_kern<<<num_blocks, nvshm_test_num_tpb, 0, stream>>>(
            NVSHMEM_TEAM_WORLD, (int64_t *)dest, (const int64_t *)source, num_elems, mype, d_time_avg,
            &h_thread_lat[i], &h_warp_lat[i], &h_block_lat[i]);
        cuda_check_error();
        CUDA_CHECK(cudaStreamSynchronize(stream));
        i++;
    }

    if (!mype) {
        print_table("alltoall_device", "64-bit-thread", "size (Bytes)", "latency", "us", '-', h_size_array, h_thread_lat, i);
        print_table("alltoall_device", "64-bit-warp", "size (Bytes)", "latency", "us", '-', h_size_array, h_warp_lat, i);
        print_table("alltoall_device", "64-bit-block", "size (Bytes)", "latency", "us", '-', h_size_array, h_block_lat, i);
    }

    return status;
}

int main(int argc, char **argv) {
    int status = 0;
    int mype, npes, array_size, max_elems;
    char *value = NULL;
    // size needs to hold psync array, source array (nelems) and dest array (nelems * npes)
    size_t size = (MAX_ELEMS * (MAX_NPES)*2) * sizeof(DATATYPE);
    size_t alloc_size;
    int num_elems;
    DATATYPE *h_buffer = NULL;
    DATATYPE *d_buffer = NULL;
    DATATYPE *d_source, *d_dest;
    DATATYPE *h_source, *h_dest;
    char size_string[100];
    double *d_time_avg;
    cudaStream_t cstrm;
    void **h_tables;

    max_elems = (MAX_ELEMS / 2);

    if (NULL != value) {
        max_elems = atoi(value);
        if (0 == max_elems) {
            fprintf(stderr, "Warning: min max elem size = 1\n");
            max_elems = 1;
        }
    }

    array_size = floor(log2((float)max_elems)) + 1;

    DEBUG_PRINT("symmetric size requested %lu\n", size);
    sprintf(size_string, "%lu", size);

    status = setenv("NVSHMEM_SYMMETRIC_SIZE", size_string, 1);
    if (status) {
        fprintf(stderr, "setenv failed \n");
        status = -1;
        goto out;
    }

    init_wrapper(&argc, &argv);
    alloc_tables(&h_tables, 4, array_size);

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();
    assert(npes <= MAX_NPES);

    DEBUG_PRINT("SHMEM: [%d of %d] hello shmem world! \n", mype, npes);
    CUDA_CHECK(cudaStreamCreateWithFlags(&cstrm, cudaStreamNonBlocking));

    d_time_avg = (double *)nvshmem_malloc(sizeof(double) * 2);

    num_elems = MAX_ELEMS / 2;
    alloc_size = (num_elems * (MAX_NPES)*2) * sizeof(DATATYPE);

    CUDA_CHECK(cudaHostAlloc(&h_buffer, alloc_size, cudaHostAllocDefault));
    h_source = (DATATYPE *)h_buffer;
    h_dest = (DATATYPE *)&h_source[num_elems * npes];

    d_buffer = (DATATYPE *)nvshmem_malloc(alloc_size);
    if (!d_buffer) {
        fprintf(stderr, "nvshmem_malloc failed \n");
        status = -1;
        goto out;
    }

    d_source = (DATATYPE *)d_buffer;
    d_dest = (DATATYPE *)&d_source[num_elems * npes];

    CUDA_CHECK(cudaMemcpyAsync(d_source, h_source, (sizeof(DATATYPE) * num_elems * npes),
                               cudaMemcpyHostToDevice, cstrm));
    CUDA_CHECK(cudaMemcpyAsync(d_dest, h_dest, (sizeof(DATATYPE) * num_elems * npes),
                               cudaMemcpyHostToDevice, cstrm));

    alltoall_calling_kernel(NVSHMEM_TEAM_WORLD, (void *)d_dest, (const void *)d_source, mype, max_elems, cstrm, d_time_avg, h_tables);

    CUDA_CHECK(cudaMemcpyAsync(h_source, d_source, (sizeof(DATATYPE) * num_elems * npes),
                               cudaMemcpyDeviceToHost, cstrm));
    CUDA_CHECK(cudaMemcpyAsync(h_dest, d_dest, (sizeof(DATATYPE) * num_elems * npes),
                               cudaMemcpyDeviceToHost, cstrm));

    nvshmem_barrier_all();

    CUDA_CHECK(cudaFreeHost(h_buffer));
    nvshmem_free(d_buffer);
    nvshmem_free(d_time_avg);

    CUDA_CHECK(cudaStreamDestroy(cstrm));
    free_tables(h_tables, 4);
    finalize_wrapper();

out:
    return 0;
}