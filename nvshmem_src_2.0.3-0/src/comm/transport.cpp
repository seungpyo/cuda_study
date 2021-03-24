/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmem_internal.h"

#include <assert.h>
#include "nvshmemx_error.h"
#include "util.h"
#include "topo.h"
#include "transport.h"
#include <sys/types.h>
#include <unistd.h>

int (*nvshmemi_transport_init_op[NVSHMEM_TRANSPORT_COUNT])(nvshmem_transport_t *transport);

int nvshmemi_transport_show_info(nvshmemi_state_t *state) {
    int status = 0;
    nvshmem_transport_t *transports = (nvshmem_transport_t *)state->transports;
    for (int i = 0; i < state->transport_count; ++i) {
        status = transports[i]->host_ops.show_info(state->handles, i, state->transport_count,
                                                   transports[i]->ep, transports[i]->ep_count,
                                                   state->npes, state->mype);
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "transport show info failed \n");
    }
out:
    return status;
}

void nvshmemi_add_transport(int id, int (*init_op)(nvshmem_transport_t *)) {
    nvshmemi_transport_init_op[id] = init_op;
}

void nvshmemi_transports_preinit() {
    nvshmemi_add_transport(NVSHMEM_TRANSPORT_ID_P2P, nvshmemt_p2p_init);
    nvshmemi_add_transport(NVSHMEM_TRANSPORT_ID_IBRC, nvshmemt_ibrc_init);
}

int nvshmemi_transport_init(nvshmemi_state_t *state) {
    int status = 0;
    nvshmem_transport_t *transports = NULL;

    nvshmemi_transports_preinit();

    if (!state->transports)
        state->transports =
            (nvshmem_transport_t *)calloc(NVSHMEM_TRANSPORT_COUNT, sizeof(nvshmem_transport_t));

    transports = (nvshmem_transport_t *)state->transports;

    for (int i = 0; i < NVSHMEM_TRANSPORT_COUNT; i++) {
        if (transports[i] == NULL) {
            if (nvshmemi_transport_init_op[i]) {
                status = nvshmemi_transport_init_op[i](transports + i);
                if (status) {
                    INFO(NVSHMEM_INIT, "init failed for transport: %d", i);
                    status = 0;
                    continue;
                }

                transports[i]->cap = (int *)calloc(state->npes, sizeof(int));
                INFO(NVSHMEM_INIT, "cap array for transport %d : %p", i,
                     transports[i]->cap);
            }
        }
    }

out:
    return status;
}

int nvshmemi_transport_finalize(nvshmemi_state_t *state) {
    int status = 0;
    nvshmem_transport_t *transports = NULL;
    ;

    if (!state->transports) return 0;

    transports = (nvshmem_transport_t *)state->transports;

    for (int i = 0; i < NVSHMEM_TRANSPORT_COUNT; i++) {
        if (transports[i] && nvshmemi_transport_init_op[i]) {
            status = transports[i]->host_ops.finalize(transports[i]);
            NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "transport finalize failed \n");
        }
    }
out:
    return status;
}

int nvshmemi_setup_connections(nvshmemi_state_t *state) {
    int status = 0;
    nvshmem_transport_t *transports = (nvshmem_transport_t *)state->transports;
    nvshmemt_ep_handle_t *local_ep_handles = NULL, *ep_handles = NULL;
    int tcount;

    // this can just be npes long if alltoall is used instead of allgather
    ep_handles =
        (nvshmemt_ep_handle_t *)calloc(state->npes, sizeof(nvshmemt_ep_handle_t));
    NULL_ERROR_JMP(ep_handles, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                   "failed allocating space for ep handles \n");

    tcount = 0;
    for (int i = 0; i < NVSHMEM_TRANSPORT_COUNT; i++) {
        // assumes symmetry of transport list at all PEs
        if ((state->transport_bitmap) & (1 << i)) {
            struct nvshmem_transport *tcurr = transports[i];

            tcount++;
            if (!(tcurr->attr & NVSHMEM_TRANSPORT_ATTR_CONNECTED)) continue;

            int ep_count = tcurr->ep_count = MAX_TRANSPORT_EP_COUNT;

            tcurr->ep = (nvshmemt_ep_t *)calloc(state->npes * ep_count, sizeof(nvshmemt_ep_t));
            NULL_ERROR_JMP(tcurr->ep, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                           "failed allocating space for endpoints \n");

            local_ep_handles = (nvshmemt_ep_handle_t *)calloc(state->npes * ep_count,
                                                              sizeof(nvshmemt_ep_handle_t));
            NULL_ERROR_JMP(local_ep_handles, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                           "failed allocating space for ep handles \n");

            /*if NVSHMEM_ENABLE_NIC_PE_MAPPING is not set, let transport manage the device binding
             * and do round robin*/
            if (nvshmemi_options.ENABLE_NIC_PE_MAPPING) {
                int ndev;

                status = tcurr->host_ops.get_device_count(&ndev, tcurr);
                NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "get_device_count failed \n");

                tcurr->dev_id = state->mype_node % ndev;
                INFO(NVSHMEM_INIT, "NVSHMEM_ENABLE_NIC_PE_MAPPING = 1, setting dev_id = %d", tcurr->dev_id);
            } else {
                status = get_device_by_distance(&tcurr->dev_id, state, tcurr);
                INFO(NVSHMEM_INIT, "Getting closest device by distance, device index = %d", tcurr->dev_id);
                NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                             "get_device_by_distance failed \n");
            }

            for (int j = 0; j < state->npes; j++) {
                for (int k = 0; k < ep_count; k++) {
                    status = tcurr->host_ops.ep_create((tcurr->ep + j * ep_count + k),
                                                       tcurr->dev_id, tcurr);
                    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                 "transport create ep failed \n");

                    int pid = getpid();
                    status = tcurr->host_ops.ep_get_handle(local_ep_handles + j * ep_count + k,
                                                           tcurr->ep[j * ep_count + k]);
                    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                 "transport get ep handle failed \n");
                }
            }

            // this could be more efficient with an alltoall
            status = state->boot_handle.alltoall(
                (void *)local_ep_handles, (void *)ep_handles,
                sizeof(nvshmemt_ep_handle_t) * ep_count, &state->boot_handle);
            NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "allgather of ep handles failed \n");

            for (int j = 0; j < state->npes; j++) {
                for (int k = 0; k < ep_count; k++) {
                    status = tcurr->host_ops.ep_connect(
                        tcurr->ep[j * ep_count + k],
                        ep_handles[j * ep_count + k]);
                    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                                 "transport create connect failed \n");
                }
            }

            status = state->boot_handle.barrier(&state->boot_handle);
            NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "barrier failed \n");
        }
    }

    assert(tcount == state->transport_count);

out:
    if (status) {
        for (int i = 0; i < NVSHMEM_TRANSPORT_COUNT; i++) {
            struct nvshmem_transport *tcurr = transports[i];
            // TODO: might have to destroy EP to clean up
            if (tcurr->ep) free(tcurr->ep);
        };
    }
    if (local_ep_handles) free(local_ep_handles);
    if (ep_handles) free(ep_handles);
    return status;
}
