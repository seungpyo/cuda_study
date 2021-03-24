/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"
#include "nvshmemx_error.h"
#include "cpu_coll.h"
#include "nvshmemi_coll.h"
#include <dlfcn.h>
#ifdef NVSHMEM_USE_NCCL
#include "nccl.h"
#endif /* NVSHMEM_USE_NCCL */

struct nccl_function_table nccl_ftable;
#define LOAD_SYM(handle, symbol, funcptr)  \
    do {                                   \
        void **cast = (void **)&funcptr;   \
        void *tmp = dlsym(handle, symbol); \
        *cast = tmp;                       \
    } while (0)

int nvshmemi_use_nccl = 0;


int nvshmemi_coll_common_cpu_read_env() {
    int status = 0;

fn_out:
    return status;
}

int nvshmemi_coll_common_cpu_init() {
    int status = 0;
#ifdef NVSHMEM_USE_NCCL
    void *nccl_handle = NULL;
#endif
    int nccl_build_version;

    status = nvshmemi_coll_common_cpu_read_env();
    if (status) NVSHMEMI_COLL_CPU_ERR_POP();

#ifdef NVSHMEM_USE_NCCL
    nvshmemi_use_nccl = 1;
    assert(NCCL_VERSION_CODE >= 2000);
    if (nvshmemi_options.DISABLE_NCCL) {
        nvshmemi_use_nccl = 0;
        goto fn_out;
    }

    nccl_handle = dlopen("libnccl.so", RTLD_LAZY);
    if (!nccl_handle) {
        WARN_PRINT("NCCL library not found...\n");
        nvshmemi_use_nccl = 0;
        goto fn_out;
    }
    
    nccl_build_version = NCCL_VERSION_CODE;
    LOAD_SYM(nccl_handle, "ncclGetVersion", nccl_ftable.GetVersion);
    int version;
    nccl_ftable.GetVersion(&version);
    if (version < nccl_build_version) {
        WARN_PRINT("NCCL library version (%d) is older than the"
                    " version (%d) with which NVSHMEM was built, skipping use...\n", version, nccl_build_version);
        nvshmemi_use_nccl = 0;
        goto fn_out;
    }
    LOAD_SYM(nccl_handle, "ncclGetUniqueId", nccl_ftable.GetUniqueId);
    LOAD_SYM(nccl_handle, "ncclCommInitRank", nccl_ftable.CommInitRank);
    LOAD_SYM(nccl_handle, "ncclCommDestroy", nccl_ftable.CommDestroy);
    LOAD_SYM(nccl_handle, "ncclAllReduce", nccl_ftable.AllReduce);
    LOAD_SYM(nccl_handle, "ncclBroadcast", nccl_ftable.Broadcast);
    LOAD_SYM(nccl_handle, "ncclAllGather", nccl_ftable.AllGather);
    LOAD_SYM(nccl_handle, "ncclGetErrorString", nccl_ftable.GetErrorString);

#endif /* NVSHMEM_USE_NCCL */
fn_out:
    return status;
fn_fail:
    return status;
}

int nvshmemi_coll_common_cpu_finalize() {
    int status = 0;

fn_out:
    return status;
fn_fail:
    return status;
}
