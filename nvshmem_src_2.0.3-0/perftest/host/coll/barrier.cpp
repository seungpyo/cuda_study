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
    char *buffer = NULL;
    int iters = 0;
    int skip = BARRIER_MAX_SKIP;
    struct timeval t_start, t_stop;
    double latency = 0;
    size_t size = 128 * 1024;
    double latency_value;
    char size_string[100];

    init_wrapper(&c, &v);

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();

    DEBUG_PRINT("SHMEM: [%d of %d] hello shmem world! \n", mype, npes);

    latency = 0;
    for (iters = 0; iters < BARRIER_MAX_ITERS + skip; iters++) {
        if (iters >= skip) gettimeofday(&t_start, NULL);

        nvshmem_barrier(NVSHMEM_TEAM_WORLD);

        if (iters >= skip) {
            gettimeofday(&t_stop, NULL);
            latency +=
                ((t_stop.tv_usec - t_start.tv_usec) + (1e+6 * (t_stop.tv_sec - t_start.tv_sec)));
        }
    }

    if (!mype) {
        latency_value = latency / BARRIER_MAX_ITERS;
        print_table("barrier", "None", "size (Bytes)", "latency", "us", '-', &size, &latency_value, 1);
    }

    nvshmem_barrier_all();

    finalize_wrapper();

out:
    return status;
}
