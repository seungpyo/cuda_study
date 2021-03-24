/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmem_internal.h"

#include <string.h>
#include <assert.h>
#include "transport.h"
#include "nvshmemx_error.h"
#include "p2p.h"

int nvshmemt_p2p_init(nvshmem_transport_t *transport);

int nvshmemi_get_pcie_attrs(pcie_id_t *pcie_id, CUdevice cudev) {
    int status = 0;

    status = cuDeviceGetAttribute(&pcie_id->dev_id, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, cudev);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                 "cuDeviceGetAttribute failed \n");

    status = cuDeviceGetAttribute(&pcie_id->bus_id, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, cudev);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                 "cuDeviceGetAttribute failed \n");

    status = cuDeviceGetAttribute(&pcie_id->domain_id, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, cudev);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                 "cuDeviceGetAttribute failed \n");

out:
    return status;
}

int nvshmemt_p2p_show_info(nvshmem_mem_handle_t *mem_handles, int transport_id, int transport_count,
                           nvshmemt_ep_t *eps, int ep_count, int npes, int mype) {
    /*XXX : not implemented*/
    return 0;
}

int nvshmemt_p2p_can_reach_peer(int *access, struct nvshmem_transport_pe_info *peer_info,
                                nvshmem_transport_t transport) {
    int status = 0;
    int found = 0;
    int p2p_connected = 0;
    CUdevice peer_dev;
    transport_p2p_state_t *p2p_state = (transport_p2p_state_t *)transport->state;
    int can_access = 0;
    int atomics_supported = 0;

    INFO(NVSHMEM_TRANSPORT,
         "[%p] ndev %d pcie_devid %x cudevice %x peer host hash %x p2p host hash %x", p2p_state,
         p2p_state->ndev, peer_info->pcie_id.dev_id, p2p_state->cudevice, peer_info->hostHash,
         p2p_state->hostHash);
    if (peer_info->hostHash != p2p_state->hostHash) {
        *access = 0;
        goto out;
    }

    /*find device with the give pcie id*/
    for (int j = 0; j < p2p_state->ndev; j++) {
        if ((p2p_state->pcie_ids[j].dev_id == peer_info->pcie_id.dev_id) &&
            (p2p_state->pcie_ids[j].bus_id == peer_info->pcie_id.bus_id) &&
            (p2p_state->pcie_ids[j].domain_id == peer_info->pcie_id.domain_id)) {
            peer_dev = p2p_state->cudev[j];
            found = 1;
            break;
        }
    }

    if (!found) {
        //return access as true for a device that is not visible
        //cannot reliably detect P2P capability 
        //cannot determine if it is connected via NVLink, so no atomics
        *access = NVSHMEM_TRANSPORT_CAP_MAP_GPU_ST | NVSHMEM_TRANSPORT_CAP_MAP_GPU_LD;
        goto out;
    }

    if (peer_dev == p2p_state->cudevice) { 
       *access = NVSHMEM_TRANSPORT_CAP_MAP | NVSHMEM_TRANSPORT_CAP_MAP_GPU_ST | 
	       	 NVSHMEM_TRANSPORT_CAP_MAP_GPU_LD | NVSHMEM_TRANSPORT_CAP_MAP_GPU_ATOMICS;
       goto out;
    }

    //use CanAccessPeer if device is visible
    status = cuDeviceCanAccessPeer(&p2p_connected, p2p_state->cudevice, peer_dev);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                 "cuDeviceCanAccessPeer failed \n");

    if (p2p_connected) {
        *access = NVSHMEM_TRANSPORT_CAP_MAP | NVSHMEM_TRANSPORT_CAP_MAP_GPU_ST | 
		  NVSHMEM_TRANSPORT_CAP_MAP_GPU_LD;
        status = cuDeviceGetP2PAttribute(&atomics_supported, CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED,
                                         p2p_state->cudevice, peer_dev);
        NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                     "cuDeviceGetP2PAttribute failed \n");
        if (atomics_supported) {
            *access |= NVSHMEM_TRANSPORT_CAP_MAP_GPU_ATOMICS;
        }
    }

out:
    return status;
}

int nvshmemt_p2p_get_mem_handle(nvshmem_mem_handle_t *mem_handle, void *buf, size_t length, int dev,
                                nvshmem_transport_t transport) {
    int status = 0;
    CUipcMemHandle *ipc_handle = (CUipcMemHandle *)mem_handle;

    assert(sizeof(CUipcMemHandle) <= NVSHMEM_MEM_HANDLE_SIZE);

    INFO(NVSHMEM_TRANSPORT, "calling cuIpcGetMemHandle on buf: %p size: %d", buf, length);

    void *tobuf = NULL;
    status = cuIpcGetMemHandle(ipc_handle, (CUdeviceptr)buf);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INVALID_VALUE, out,
                 "cuIpcGetMemHandle failed \n");
out:
    return status;
}

int nvshmemt_p2p_release_mem_handle(nvshmem_mem_handle_t mem_handle) {
    // it is a noop
    return 0;
}

int nvshmemt_p2p_map(void **buf, nvshmem_mem_handle_t mem_handle) {
    int status = 0;
    CUipcMemHandle *ipc_handle = (CUipcMemHandle *)&mem_handle;

    status =
        cuIpcOpenMemHandle((CUdeviceptr *)buf, *ipc_handle, CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INVALID_VALUE, out,
                 "cuIpcOpenMemHandle failed with error %d \n", status);

out:
    return status;
}

int nvshmemt_p2p_unmap(void *buf) {
    int status = 0;

    status = cuIpcCloseMemHandle((CUdeviceptr)buf);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INVALID_VALUE, out,
                 "cuIpcCloseMemHandle failed with error %d \n", status);

out:
    return status;
}

int nvshmemt_p2p_finalize(nvshmem_transport_t transport) {
    int status = 0;

    if (!transport) return 0;

    if (transport->state) {
        transport_p2p_state_t *p2p_state = (transport_p2p_state_t *)transport->state;

        free(p2p_state->cudev);

        free(p2p_state->pcie_ids);

        free(p2p_state);
    }

    free(transport);

out:
    return status;
}

int nvshmemt_p2p_init(nvshmem_transport_t *t) {
    int status = 0;
    int leastPriority, greatestPriority;
    struct nvshmem_transport *transport;
    transport_p2p_state_t *p2p_state;

    transport = (struct nvshmem_transport *)malloc(sizeof(struct nvshmem_transport));
    memset(transport, 0, sizeof(struct nvshmem_transport));
    transport->is_successfully_initialized = false; /* set it to true after everything has been successfully initialized */

    p2p_state = (transport_p2p_state_t *)malloc(sizeof(transport_p2p_state_t));
    NULL_ERROR_JMP(p2p_state, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                   "p2p state allocation failed \n");

    status = cuCtxGetDevice(&p2p_state->cudevice);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "cuCtxGetDevice failed \n");

    p2p_state->hostHash = getHostHash();

    status = cuDeviceGetCount(&p2p_state->ndev);
    NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "cuDeviceGetCount failed \n");

    p2p_state->cudev = (CUdevice *)malloc(sizeof(CUdevice) * p2p_state->ndev);
    NULL_ERROR_JMP(p2p_state->cudev, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                   "p2p dev array allocation failed \n");

    p2p_state->pcie_ids = (pcie_id_t *)malloc(sizeof(pcie_id_t) * p2p_state->ndev);
    NULL_ERROR_JMP(p2p_state->pcie_ids, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                   "p2p pcie_ids array allocation failed \n");

    for (int i = 0; i < p2p_state->ndev; i++) {
        status = cuDeviceGet(&p2p_state->cudev[i], i);
        NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "cuDeviceGet failed \n");

        status = nvshmemi_get_pcie_attrs(&p2p_state->pcie_ids[i], p2p_state->cudev[i]);
        NE_ERROR_JMP(status, CUDA_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out,
                     "nvshmemi_get_pcie_attrs failed \n");
    }

    transport->host_ops.can_reach_peer = nvshmemt_p2p_can_reach_peer;
    transport->host_ops.get_mem_handle = nvshmemt_p2p_get_mem_handle;
    transport->host_ops.release_mem_handle = nvshmemt_p2p_release_mem_handle;
    transport->host_ops.map = nvshmemt_p2p_map;
    transport->host_ops.unmap = nvshmemt_p2p_unmap;
    transport->host_ops.finalize = nvshmemt_p2p_finalize;
    transport->host_ops.show_info = nvshmemt_p2p_show_info;

    transport->attr = NVSHMEM_TRANSPORT_ATTR_NO_ENDPOINTS;
    transport->state = p2p_state;
    transport->is_successfully_initialized = true;

    *t = transport;

out:
    return status;
}
