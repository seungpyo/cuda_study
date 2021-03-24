/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmemx.h"
#include "cpu_coll.h"
#include "nvshmemi_coll.h"

void nvshmemx_barrier_all_on_stream(cudaStream_t stream) {
    NVSHMEM_CHECK_STATE_AND_INIT();
    nvshmemx_barrier_on_stream(NVSHMEM_TEAM_WORLD, stream);
}

void nvshmemxi_barrier_on_stream(int PE_start, int PE_stride, int PE_size, long *pSync,
                                long *counter, cudaStream_t stream) {
    call_barrier_on_stream_kern(PE_start, PE_stride, PE_size, pSync, counter, stream);
}

int nvshmemx_barrier_on_stream(nvshmem_team_t team, cudaStream_t stream) {
    NVSHMEM_CHECK_STATE_AND_INIT();
    nvshmemi_team_t *teami = nvshmemi_team_pool[team];
    nvshmemxi_barrier_on_stream(teami->start, teami->stride, teami->size,
                                nvshmemi_team_get_psync(teami, SYNC),
                                nvshmemi_team_get_sync_counter(teami), stream);
    return 0;
}

void nvshmemx_sync_all_on_stream(cudaStream_t stream) {
    nvshmemx_team_sync_on_stream(NVSHMEM_TEAM_WORLD, stream);
}

int nvshmemx_team_sync_on_stream(nvshmem_team_t team, cudaStream_t stream) {
    NVSHMEM_CHECK_STATE_AND_INIT();
    
    nvshmemi_team_t *teami = nvshmemi_team_pool[team];
    call_sync_on_stream_kern(teami->start, teami->stride, teami->size,
                             nvshmemi_team_get_psync(teami, SYNC),
                             nvshmemi_team_get_sync_counter(teami), stream);
    return 0;
}
