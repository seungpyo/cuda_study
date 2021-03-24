/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmemx.h"
#include "nvshmem_internal.h"
#include "util.h"

#ifdef __cplusplus
extern "C" {
#endif
__device__ void nvshmem_quiet();
void nvshmemx_quiet_on_stream(cudaStream_t cstrm);
#ifdef __cplusplus
}
#endif

__global__ void nvshmemi_proxy_quiet_entrypoint() { nvshmem_quiet(); }

void nvshmemx_quiet_on_stream(cudaStream_t cstrm) {
    NVSHMEM_CHECK_STATE_AND_INIT();
    int status = 0;

    int tbitmap = nvshmemi_state->transport_bitmap;
    if (nvshmemi_state->npes_node > 1) {
        for (int s = 0; s < MAX_PEER_STREAMS; s++) {
            CUstream custrm = nvshmemi_state->custreams[s];
            CUevent cuev = nvshmemi_state->cuevents[s];
            status = cuEventRecord(cuev, custrm);
            NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                         "nvshmem_quiet_on_stream() failed \n");
            status = cuStreamWaitEvent((CUstream)cstrm, cuev, 0);
            NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                         "nvshmem_quiet_on_stream() failed \n");
        }
    }

    for (int j = 0; j < NVSHMEM_TRANSPORT_COUNT; j++) {
        if (tbitmap & 1) {
            if (j == NVSHMEM_TRANSPORT_ID_IBRC) {
                status = cudaLaunchKernel((const void *)nvshmemi_proxy_quiet_entrypoint, 1, 1, NULL,
                                          0, cstrm);
                if (status) {
                    ERROR_PRINT("cudaLaunchKernel() failed in nvshmem_quiet_on_stream \n");
                }
            }
            tbitmap >>= 1;
        }
    }
out:
    return;
}
