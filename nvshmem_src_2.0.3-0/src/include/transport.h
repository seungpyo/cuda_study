/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef __TRANSPORT_H
#define __TRANSPORT_H

#include <stdint.h>
#include "cuda.h"
#include "common.h"

#define NVSHMEM_TRANSPORT_DEVICE_SCORE_MAX 7

enum { NVSHMEM_TRANSPORT_WAIT_EQ = 0 };

enum {
    NVSHMEM_TRANSPORT_ID_P2P = 0,
    NVSHMEM_TRANSPORT_ID_IBRC,
    NVSHMEM_TRANSPORT_COUNT,
};

enum {
    NVSHMEM_TRANSPORT_MASK_P2P = 1 << NVSHMEM_TRANSPORT_ID_P2P,
    NVSHMEM_TRANSPORT_MASK_IBRC = 1 << NVSHMEM_TRANSPORT_ID_IBRC,
};

enum {
    NVSHMEM_TRANSPORT_CAP_MAP = 1,
    NVSHMEM_TRANSPORT_CAP_MAP_GPU_ST = 1 << 1,
    NVSHMEM_TRANSPORT_CAP_MAP_GPU_LD = 1 << 2,
    NVSHMEM_TRANSPORT_CAP_MAP_GPU_ATOMICS = 1 << 3,
    NVSHMEM_TRANSPORT_CAP_CPU_WRITE = 1 << 4,
    NVSHMEM_TRANSPORT_CAP_CPU_READ = 1 << 5,
    NVSHMEM_TRANSPORT_CAP_CPU_ATOMICS = 1 << 6,
};

enum {
    NVSHMEM_TRANSPORT_ATTR_NO_ENDPOINTS = 1,
    NVSHMEM_TRANSPORT_ATTR_CONNECTED = 1 << 1,
};

enum {
    NVSHMEM_TRANSPORT_DMA_FLUSH_ACTIVE = 0,
    NVSHMEM_TRANSPORT_DMA_FLUSH_COMPLETE,
    NVSHMEM_TRANSPORT_DMA_FLUSH_ERROR,
};

enum {
    NVSHMEM_TRANSPORT_MEMTYPE_HOST = 0,
    NVSHMEM_TRANSPORT_MEMTYPE_DEVICE,
    NVSHMEM_TRANSPORT_MEMTYPE_IOMEM
};

typedef struct pcie_identifier {
    int dev_id;
    int bus_id;
    int domain_id;
} pcie_id_t;

int nvshmemi_get_pcie_attrs(pcie_id_t *pcie_id, CUdevice cudev);

typedef struct nvshmem_transport_pe_info {
    pcie_id_t pcie_id;
    int pe;
    uint64_t hostHash;
} nvshmem_transport_pe_info_t;

/*inc*/
#define TRANSPORT_TYPE_INC(type, TYPE, opname) \
    void (*atomic_##type##_inc)(volatile TYPE *, nvshmem_mem_handle_t *, int);

/*finc, fetch*/
#define TRANSPORT_TYPE_FINC_FETCH(type, TYPE, opname) \
    TYPE (*atomic_##type##_##opname)(volatile TYPE *, nvshmem_mem_handle_t *, int);

/*and, or, xor, add, set*/
#define TRANSPORT_TYPE_AND_OR_XOR_ADD_SET(type, TYPE, opname) \
    void (*atomic_##type##_##opname)(volatile TYPE *, nvshmem_mem_handle_t *, TYPE, int);

/*fand, for, fxor, fadd, swap*/
#define TRANSPORT_TYPE_FAND_FOR_FXOR_FADD_SWAP(type, TYPE, opname) \
    TYPE (*atomic_##type##_##opname)(volatile TYPE *, nvshmem_mem_handle_t *, TYPE, int);

/*cswap*/
#define TRANSPORT_TYPE_CSWAP(type, TYPE, opname) \
    TYPE (*atomic_##type##_##opname)(volatile TYPE *, nvshmem_mem_handle_t *, TYPE, TYPE, int);

#define TRANSPORT_TYPE_COMMON_OPGROUP(OPNAME, opname)                   \
    TRANSPORT_TYPE_##OPNAME(uint, unsigned int, opname)                 \
    TRANSPORT_TYPE_##OPNAME(ulong, unsigned long, opname)               \
    TRANSPORT_TYPE_##OPNAME(ulonglong, unsigned long long, opname)      \
    TRANSPORT_TYPE_##OPNAME(int32, int32_t, opname)                     \
    TRANSPORT_TYPE_##OPNAME(int64, int64_t, opname)                     \
    TRANSPORT_TYPE_##OPNAME(uint32, uint32_t, opname)                   \
    TRANSPORT_TYPE_##OPNAME(uint64, uint64_t, opname)

#define TRANSPORT_TYPE_STANDARD_OPGROUP(OPNAME, opname)                 \
    TRANSPORT_TYPE_##OPNAME(int, int, opname)                           \
    TRANSPORT_TYPE_##OPNAME(long, long, opname)                         \
    TRANSPORT_TYPE_##OPNAME(longlong, long long, opname)                \
    TRANSPORT_TYPE_##OPNAME(size, size_t, opname)                       \
    TRANSPORT_TYPE_##OPNAME(ptrdiff, ptrdiff_t, opname)

#define TRANSPORT_TYPE_EXTENDED_OPGROUP(OPNAME, opname)                 \
    TRANSPORT_TYPE_##OPNAME(float, float, opname)                       \
    TRANSPORT_TYPE_##OPNAME(double, double, opname)

struct nvshmem_transport_host_ops {
    int (*get_device_count)(int *ndev, struct nvshmem_transport *transport);
    int (*get_pci_path)(int dev, char **pcipath, struct nvshmem_transport *transport);
    int (*can_reach_peer)(int *access, nvshmem_transport_pe_info_t *peer_info,
                          struct nvshmem_transport *transport);
    int (*ep_create)(nvshmemt_ep_t *ep, int devid, struct nvshmem_transport *transport);
    int (*ep_get_handle)(nvshmemt_ep_handle_t *ep_handle, nvshmemt_ep_t tep);
    int (*ep_connect)(nvshmemt_ep_t tep, nvshmemt_ep_handle_t remote_ep_handle);
    int (*get_mem_handle)(nvshmem_mem_handle_t *mem_handle, void *buf, size_t size, int dev_id,
                          struct nvshmem_transport *transport);
    int (*release_mem_handle)(nvshmem_mem_handle_t mem_handle);
    int (*map)(void **buf, nvshmem_mem_handle_t mem_handle);
    int (*unmap)(void *buf);
    int (*ep_destroy)(nvshmemt_ep_t ep);
    int (*finalize)(struct nvshmem_transport *transport);
    int (*show_info)(nvshmem_mem_handle_t *mem_handles, int transport_id, int transport_count,
                     nvshmemt_ep_t *eps, int ep_count, int npes, int mype);
    int (*progress)(struct nvshmem_transport *transport);

    rma_handle rma;
    amo_handle amo;
    fence_handle fence;
    quiet_handle quiet;
    // int (*enforce_consistency) (nvshmemt_ep_t tep);
    int (*enforce_cst)();
    int (*enforce_cst_at_target)();
};

struct nvshmem_transport {
    int attr;
    struct nvshmem_transport_host_ops host_ops;
    int *cap;
    void *state;
    int ep_idx;
    int ep_count;
    int dev_id;
    nvshmemt_ep_t *ep;
    bool is_successfully_initialized;
};

typedef struct nvshmem_transport *nvshmem_transport_t;

void nvshmemi_add_transport(int id, int (*init_op)(nvshmem_transport_t *));
int nvshmemi_transport_init(struct nvshmemi_state_dec *state);
int nvshmemi_transport_finalize(struct nvshmemi_state_dec *state);
int nvshmemi_transport_show_info(nvshmemi_state_dec *state);

/*Per transport struct*/

typedef struct {
    int ndev;
    CUdevice *cudev;
    CUdeviceptr *curetval;
    CUdevice cudevice;
    uint64_t hostHash;
    pcie_id_t *pcie_ids;
} transport_p2p_state_t;

int nvshmemt_p2p_init(nvshmem_transport_t *transport);
int nvshmemt_p2p_show_info(nvshmem_mem_handle_t *mem_handles, int transport_id, int transport_count,
                           nvshmemt_ep_t *eps, int ep_count, int npes, int mype);

int nvshmemt_ibrc_init(nvshmem_transport_t *transport);
int nvshmemt_ibrc_show_info(nvshmem_mem_handle_t *mem_handles, int transport_id,
                            int transport_count, nvshmemt_ep_t *eps, int ep_count, int npes,
                            int mype);
#endif
