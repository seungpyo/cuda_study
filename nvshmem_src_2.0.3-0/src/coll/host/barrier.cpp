/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmemi_coll.h"
#include "cpu_coll.h"

/** The following two commented functions are not deleted.
They are kept around in case we plan to reinstate host based barrier
algorithms for performance reasons **/

/* void nvshmemi_sync_cpu(int start, int stride, int size, long *pSync) {
    int end = start + (stride * size);
    int root = start;
    int val = 1;
    volatile int *bcast_sync_arr = nvshm_cpu_coll_info.cpu_bcast_int_sync_arr;

    if (root == nvshmemi_state->mype) {
        for (int ii = start; ii < end; ii++) {
            if (root == ii) continue;
            while (0 != bcast_sync_arr[ii]) {
            }
        }

        for (int ii = start; ii < end; ii++) {
            if (root == ii) continue;
            bcast_sync_arr[ii] = val;
        }

        for (int ii = start; ii < end; ii++) {
            if (root == ii) continue;
            while (0 != bcast_sync_arr[ii]) {
            }
        }

    } else {
        while (val != bcast_sync_arr[nvshmemi_state->mype]) {
        }

        bcast_sync_arr[nvshmemi_state->mype] = 0;
    }
}

void nvshmemi_sync_p2p(int start, int stride, int size, long *pSync) {
    int end = start + (stride * size);
    int root = start;
    int val = 1;
    volatile int *gpu_bcast_sync_arr = nvshm_cpu_coll_info.gpu_bcast_int_sync_arr;

    if (root == nvshmemi_state->mype) {
        for (int ii = start; ii < end; ii++) {
            if (root == ii) continue;
            while (0 != gpu_bcast_sync_arr[ii]) {
            }
        }

        for (int ii = start; ii < end; ii++) {
            if (root == ii) continue;
            nvshmem_int_p((int *)&gpu_bcast_sync_arr[ii], val, ii);
        }

        for (int ii = start; ii < end; ii++) {
            if (root == ii) continue;
            while (0 != gpu_bcast_sync_arr[ii]) {
            }
        }

    } else {
        while (val != gpu_bcast_sync_arr[nvshmemi_state->mype]) {
        }

        nvshmem_int_p((int *)&gpu_bcast_sync_arr[nvshmemi_state->mype], 0, 0);
    }
}

} */

void nvshmemi_barrier(int start, int stride, int size, long *pSync, long *counter) {
    nvshmem_quiet();
    call_barrier_on_stream_kern(start, stride, size, pSync, counter, nvshmemi_state->my_stream);
    CUDA_CHECK(cuStreamSynchronize(nvshmemi_state->my_stream));
}

int nvshmem_barrier(nvshmem_team_t team) {
    NVSHMEM_CHECK_STATE_AND_INIT();
    
    nvshmemi_team_t *teami = nvshmemi_team_pool[team];
    nvshmemi_barrier(teami->start, teami->stride, teami->size,
                     nvshmemi_team_get_psync(teami, SYNC),
                     nvshmemi_team_get_sync_counter(teami));
    return 0;
}

void nvshmemi_barrier_all() {
    nvshmemi_team_t *teami = nvshmemi_team_pool[NVSHMEM_TEAM_WORLD];
    nvshmemi_barrier(teami->start, teami->stride, teami->size,
                     nvshmemi_team_get_psync(teami, SYNC),
                     nvshmemi_team_get_sync_counter(teami));

    return;
}

void nvshmem_barrier_all() {
    NVSHMEM_CHECK_STATE_AND_INIT();

    nvshmemi_barrier_all();
    return;
}

void nvshmemi_sync(int start, int stride, int size, long *pSync, long *counter) {
    call_sync_on_stream_kern(start, stride, size, pSync, counter, nvshmemi_state->my_stream);
    CUDA_CHECK(cuStreamSynchronize(nvshmemi_state->my_stream));
}

int nvshmem_team_sync(nvshmem_team_t team) {
    NVSHMEM_CHECK_STATE_AND_INIT();
    
    nvshmemi_team_t *teami = nvshmemi_team_pool[team];
    nvshmemi_sync(teami->start, teami->stride, teami->size,
                  nvshmemi_team_get_psync(teami, SYNC),
                  nvshmemi_team_get_sync_counter(teami));
    return 0;
}

void nvshmem_sync_all() {
    NVSHMEM_CHECK_STATE_AND_INIT();

    nvshmemi_team_t *teami = nvshmemi_team_pool[NVSHMEM_TEAM_WORLD];
    nvshmemi_sync(teami->start, teami->stride, teami->size,
                  nvshmemi_team_get_psync(teami, SYNC),
                  nvshmemi_team_get_sync_counter(teami));

    return;
}
