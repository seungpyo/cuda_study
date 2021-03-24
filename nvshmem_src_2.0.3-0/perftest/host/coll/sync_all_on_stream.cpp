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

#include "coll_test.h"

int main(int c, char *v[]) {
    int status = 0;
    int mype, npes;
    size_t size = 128 * 1024;
    double latency_value;
    int iters = MAX_ITERS;
    int skip = MAX_SKIP;
    struct timeval t_start, t_stop;
    double latency = 0;
    cudaStream_t stream;

    init_wrapper(&c, &v);

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    DEBUG_PRINT("SHMEM: [%d of %d] hello shmem world! \n", mype, npes);

    latency = 0;
    for (iters = 0; iters < MAX_ITERS + skip; iters++) {
        if (iters >= skip) gettimeofday(&t_start, NULL);

        nvshmemx_sync_all_on_stream(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (iters >= skip) {
            gettimeofday(&t_stop, NULL);
            latency +=
                ((t_stop.tv_usec - t_start.tv_usec) + (1e+6 * (t_stop.tv_sec - t_start.tv_sec)));
        }
    }

    if (!mype) {
        latency_value = latency / MAX_ITERS;
        print_table("sync_all_on_stream", "None", "size (Bytes)", "latency", "us", '-', &size, &latency_value, 1);
    }

    nvshmem_barrier_all();

    CUDA_CHECK(cudaStreamDestroy(stream));

    finalize_wrapper();

out:
    return status;
}
