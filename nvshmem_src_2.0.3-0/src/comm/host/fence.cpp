/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmem_internal.h"
#include "nvshmemx_error.h"

void nvshmem_fence(void) {
    NVSHMEM_CHECK_STATE_AND_INIT();

    int status;
    int tbitmap = nvshmemi_state->transport_bitmap;
    for (int j = 0; j < NVSHMEM_TRANSPORT_COUNT; j++) {
        if (tbitmap & 1) {
            if (j == NVSHMEM_TRANSPORT_ID_P2P) {
                for (int s = 0; s < MAX_PEER_STREAMS; s++) {
                    CUstream custrm = nvshmemi_state->custreams[s];
                    status = cuStreamSynchronize(custrm);
                    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "nvshmem_fence() failed \n");
                }
            } else if (nvshmemi_state->fence[j]) {
                struct nvshmem_transport *tcurr =
                    ((nvshmem_transport_t *)nvshmemi_state->transports)[j];
                int ep_count = 1;
                if (tcurr->attr & NVSHMEM_TRANSPORT_ATTR_CONNECTED) {
                    ep_count = tcurr->ep_count * nvshmemi_state->npes;
                }
                for (int k = 0; k < ep_count; k++) {
                    if (!tcurr->ep[k]) continue;
                    status = nvshmemi_state->fence[j](tcurr->ep[k]);
                    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "nvshmem_fence() failed \n");
                }
            }
            tbitmap >>= 1;
        }
    }
out:
    return;
}
