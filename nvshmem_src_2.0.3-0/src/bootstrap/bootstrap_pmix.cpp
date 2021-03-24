/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include "pmix.h"
#include "bootstrap.h"
#include "bootstrap_internal.h"

#define NVSHMEMX_ERROR_INTERNAL 7

#define NZ_ERROR_JMP(status, err, label, ...)                                           \
    do {                                                                                \
        if (__builtin_expect((status != 0), 0)) {                                       \
            fprintf(stderr, "%s:%d: non-zero status: %d ", __FILE__, __LINE__, status); \
            fprintf(stderr, __VA_ARGS__);                                               \
            status = err;                                                               \
            goto label;                                                                 \
        }                                                                               \
    } while (0)

//#define INFO(c, ...) printf(__VA_ARGS__)
#define INFO(...)

#define BOOTSTRAP_PMIX_KEYSIZE 64

static pmix_proc_t myproc;


static int bootstrap_pmix_barrier(bootstrap_handle_t *handle) {
    pmix_status_t status = PMIx_Fence(NULL, 0, NULL, 0);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "PMIx_Fence failed\n");

out:
    return status;
}


static pmix_status_t bootstrap_pmix_exchange(void) {
    pmix_status_t status;
    pmix_info_t info;
    bool flag = true;

    status = PMIx_Commit();
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "PMIx_Commit failed\n");

    PMIX_INFO_CONSTRUCT(&info);
    PMIX_INFO_LOAD(&info, PMIX_COLLECT_DATA, &flag, PMIX_BOOL);

    status = PMIx_Fence(NULL, 0, &info, 1);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "PMIx_Fence failed\n");

    PMIX_INFO_DESTRUCT(&info);

error:
out:
    return status;
}

static pmix_status_t bootstrap_pmix_put(char *key, void *value,
                                        size_t valuelen) {
    pmix_value_t val;
    pmix_status_t rc;

    PMIX_VALUE_CONSTRUCT(&val);
    val.type = PMIX_BYTE_OBJECT;
    val.data.bo.bytes = (char*) value;
    val.data.bo.size = valuelen;

    rc = PMIx_Put(PMIX_GLOBAL, key, &val);
    val.data.bo.bytes = NULL;  // protect the data
    val.data.bo.size = 0;
    PMIX_VALUE_DESTRUCT(&val);

    return rc;
}


static pmix_status_t bootstrap_pmix_get(int pe, char *key, void *value,
                                        size_t valuelen)
{
    pmix_proc_t proc;
    pmix_value_t *val;
    pmix_status_t rc;

    /* ensure the region is zero'd out */
    memset(value, 0, valuelen);

    /* setup the ID of the proc whose info we are getting */
    PMIX_LOAD_NSPACE(proc.nspace, myproc.nspace);

    proc.rank = (uint32_t) pe;

    rc = PMIx_Get(&proc, key, NULL, 0, &val);

    if (PMIX_SUCCESS == rc) {
        if (NULL != val) {
            /* see if the data fits into the given region */
            if (valuelen < val->data.bo.size) {
                PMIX_VALUE_RELEASE(val);
                return PMIX_ERROR;
            }
            /* copy the results across */
            memcpy(value, val->data.bo.bytes, val->data.bo.size);
            PMIX_VALUE_RELEASE(val);
        }
    }

    return rc;
}


static int bootstrap_pmix_allgather(const void *sendbuf, void *recvbuf, int length,
                                    bootstrap_handle_t *handle) {
    static int key_index = 1;

    pmix_status_t status = PMIX_SUCCESS;
    void *kvs_value;
    char kvs_key[BOOTSTRAP_PMIX_KEYSIZE]; // FIXME: assert( 64 < PMIX_MAX_KEYLEN);

    if (handle->pg_size == 1) {
        memcpy(recvbuf, sendbuf, length);
        return 0;
    }

    INFO(NVSHMEM_BOOTSTRAP, "PMIx allgather: transfer length: %d", length);

    snprintf(kvs_key, BOOTSTRAP_PMIX_KEYSIZE, "BOOTSTRAP-%04x", key_index);

    status = bootstrap_pmix_put(kvs_key, (void*) sendbuf, length);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "bootstrap_pmix_put failed\n");

    status = bootstrap_pmix_exchange();
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "bootstrap_pmix_exchange failed\n");

    for (int i = 0; i < handle->pg_size; i++) {
        snprintf(kvs_key, BOOTSTRAP_PMIX_KEYSIZE, "BOOTSTRAP-%04x", key_index);

        // assumes that same length is passed by all the processes
        status = bootstrap_pmix_get(i, kvs_key, (char *)recvbuf + length * i, length);
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "SPMI_KVS_Get failed\n");
    }

out:
    key_index++;
    return status;
}


static int bootstrap_pmix_alltoall(const void *sendbuf, void *recvbuf, int length,
                                   bootstrap_handle_t *handle) {
    static int key_index = 1;

    pmix_status_t status = 0;
    void *kvs_value;
    char kvs_key[BOOTSTRAP_PMIX_KEYSIZE];

    if (handle->pg_size == 1) {
        memcpy(recvbuf, sendbuf, length);
        return 0;
    }

    INFO(NVSHMEM_BOOTSTRAP, "PMIx alltoall: transfer length: %d", length);

    for (int i = 0; i < handle->pg_size; i++) {
        snprintf(kvs_key, BOOTSTRAP_PMIX_KEYSIZE, "BOOTSTRAP-%04x-%08x", key_index, i);

        status = bootstrap_pmix_put(kvs_key, (char *)sendbuf + i * length, length);
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "bootstrap_pmix_put failed\n");
    }

    status = bootstrap_pmix_exchange();
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "bootstrap_pmix_exchange failed\n");

    for (int i = 0; i < handle->pg_size; i++) {
        snprintf(kvs_key, BOOTSTRAP_PMIX_KEYSIZE, "BOOTSTRAP-%04x-%08x", key_index, handle->pg_rank);

        // assumes that same length is passed by all the processes
        status = bootstrap_pmix_get(i, kvs_key, (char *)recvbuf + length * i, length);
        NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "bootstrap_pmix_get failed\n");
    }

out:
    key_index++;
    return status;
}


extern "C" int nvshmemi_bootstrap_plugin_init(bootstrap_handle_t *handle) {
    pmix_status_t status = PMIX_SUCCESS;
    pmix_proc_t proc;
    proc.rank = PMIX_RANK_WILDCARD;
    pmix_value_t *val;

    PMIX_PROC_CONSTRUCT(&myproc);

    status = PMIx_Init(&myproc, NULL, 0);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "PMIx_Init failed\n");

    PMIX_LOAD_NSPACE(proc.nspace, myproc.nspace);
    proc.rank = PMIX_RANK_WILDCARD;

    status = PMIx_Get(&proc, PMIX_JOB_SIZE, NULL, 0, &val);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "PMIx_Get(PMIX_JOB_SIZE) failed\n");

    handle->pg_rank   = myproc.rank;
    handle->pg_size   = val->data.uint32;
    handle->allgather = bootstrap_pmix_allgather;
    handle->alltoall  = bootstrap_pmix_alltoall;
    handle->barrier   = bootstrap_pmix_barrier;
    /* handle->finalize is set by the loader */

    PMIX_VALUE_RELEASE(val);

out:
    return status != PMIX_SUCCESS;
}


extern "C" int nvshmemi_bootstrap_plugin_finalize(bootstrap_handle_t *handle) {
    pmix_status_t status;

    status = PMIx_Finalize(NULL, 0);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, error, "PMIx_Finalize failed\n");

error:
out:
    return status;
}
