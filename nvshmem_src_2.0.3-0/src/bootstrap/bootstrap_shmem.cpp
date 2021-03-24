/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmemx_error.h"
#include "util.h"
#include "shmem.h"
#include "bootstrap.h"
#include "bootstrap_internal.h"
#include <stdlib.h>
#include <dlfcn.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))

int (*shmem_wrapper_my_pe)(void);
int (*shmem_wrapper_n_pes)(void);
void *(*shmem_wrapper_malloc)(size_t size);
void (*shmem_wrapper_free)(void *ptr);
void (*shmem_wrapper_barrier)();
void (*shmem_wrapper_allgather)(void *recvbuf, const void *sendbuf, size_t length, int PE_sstart,
                                int logPE_stride, int PE_size, long *pSync);
void (*shmem_wrapper_alltoall)(void *recvbuf, const void *sendbuf, size_t length, int PE_sstart,
                                int logPE_stride, int PE_size, long *pSync);

int bootstrap_shmem_my_pe(int *my_pe) {
    int status = 0;

    *my_pe = shmem_wrapper_my_pe();
out:
    return status;
}

int bootstrap_shmem_n_pes(int *n_pes) {
    int status = 0;

    *n_pes = shmem_wrapper_n_pes();
out:
    return status;
}

int bootstrap_shmem_malloc(void **ptr, size_t bytes) {
    int status = 0;

    *ptr = shmem_wrapper_malloc(bytes);
    NULL_ERROR_JMP(*ptr, status, NVSHMEMX_ERROR_INTERNAL, out, "shmem_malloc failed \n");

    INFO(NVSHMEM_BOOTSTRAP, "bootstrap_shmem_malloc ptr %p bytes %ld", *ptr, bytes);
out:
    return status;
}

int bootstrap_shmem_free(void *ptr) {
    int status = 0;

    shmem_wrapper_free(ptr);
out:
    return status;
}

int bootstrap_shmem_barrier(struct bootstrap_handle *handle) {
    int status = 0;

    shmem_wrapper_barrier();
out:
    return status;
}

int bootstrap_shmem_allgather(const void *sendbuf, void *recvbuf, int length,
                              struct bootstrap_handle *handle) {
    int status = 0;
    void *sendbuf_i = NULL, *recvbuf_i = NULL;

    sendbuf_i = shmem_wrapper_malloc(length);
    NULL_ERROR_JMP(sendbuf_i, status, NVSHMEMX_ERROR_INTERNAL, out, "shmem_malloc failed \n");
    recvbuf_i = shmem_wrapper_malloc(length * handle->pg_size);
    NULL_ERROR_JMP(recvbuf_i, status, NVSHMEMX_ERROR_INTERNAL, out, "shmem_malloc failed \n");
    shmem_wrapper_barrier();

    INFO(NVSHMEM_BOOTSTRAP,
         "[%d] bootstrap_shmem_allgather recvbuf %p sendbuf %p length %d scratch %p",
         handle->pg_rank, recvbuf_i, sendbuf_i, length, handle->scratch);

    memcpy(sendbuf_i, sendbuf, length);
    INFO(NVSHMEM_BOOTSTRAP,
         "[%d] bootstrap_shmem_allgather recvbuf %p sendbuf %p *sendbuf %d length %d scratch %p",
         handle->pg_rank, recvbuf_i, sendbuf_i, *(int *)sendbuf, length, handle->scratch);

    shmem_wrapper_barrier();
    assert(handle->scratch_size >= SHMEM_COLLECT_SYNC_SIZE * sizeof(long));
    shmem_wrapper_allgather(recvbuf_i, sendbuf_i, length / 4, 0, 0, handle->pg_size,
                            (long *)handle->scratch);
    shmem_wrapper_barrier();

    memcpy(recvbuf, recvbuf_i, length * handle->pg_size);
    INFO(NVSHMEM_BOOTSTRAP, "[%d] bootstrap_shmem_allgather *recvbuf %d %d", handle->pg_rank,
         *(int *)recvbuf_i);

    shmem_wrapper_barrier();
    shmem_wrapper_free(sendbuf_i);
    shmem_wrapper_free(recvbuf_i);
    shmem_wrapper_barrier();
out:
    return status;
}

int bootstrap_shmem_alltoall(const void *sendbuf, void *recvbuf, int length,
                              struct bootstrap_handle *handle) {
    int status = 0;
    void *sendbuf_i = NULL, *recvbuf_i = NULL;

    sendbuf_i = shmem_wrapper_malloc(length * handle->pg_size);
    NULL_ERROR_JMP(sendbuf_i, status, NVSHMEMX_ERROR_INTERNAL, out, "shmem_malloc failed \n");
    recvbuf_i = shmem_wrapper_malloc(length * handle->pg_size);
    NULL_ERROR_JMP(recvbuf_i, status, NVSHMEMX_ERROR_INTERNAL, out, "shmem_malloc failed \n");
    shmem_wrapper_barrier();

    INFO(NVSHMEM_BOOTSTRAP,
         "[%d] bootstrap_shmem_alltoall recvbuf %p sendbuf %p length %d scratch %p",
         handle->pg_rank, recvbuf_i, sendbuf_i, length, handle->scratch);

    memcpy(sendbuf_i, sendbuf, length * handle->pg_size);
    INFO(NVSHMEM_BOOTSTRAP,
         "[%d] bootstrap_shmem_alltoall recvbuf %p sendbuf %p *sendbuf %d length %d scratch %p",
         handle->pg_rank, recvbuf_i, sendbuf_i, *(int *)sendbuf, length, handle->scratch);
    shmem_wrapper_barrier();
    assert(handle->scratch_size >= SHMEM_ALLTOALL_SYNC_SIZE * sizeof(long));
    shmem_wrapper_alltoall(recvbuf_i, sendbuf_i, length / 4, 0, 0, handle->pg_size,
                            (long *)handle->scratch);
    shmem_wrapper_barrier();

    memcpy(recvbuf, recvbuf_i, length * handle->pg_size);
    INFO(NVSHMEM_BOOTSTRAP, "[%d] bootstrap_shmem_alltoall *recvbuf %d %d", handle->pg_rank,
         *(int *)recvbuf_i);

    shmem_wrapper_barrier();
    shmem_wrapper_free(sendbuf_i);
    shmem_wrapper_free(recvbuf_i);
    shmem_wrapper_barrier();
out:
    return status;
}

#define get_symbol(lib_handle, name, var, status)                                                \
    do {                                                                                         \
        void **var_ptr = (void **)&var;                                                          \
        void *tmp = (void *)dlsym(lib_handle, name);                                             \
        NULL_ERROR_JMP(tmp, status, NVSHMEMX_ERROR_INTERNAL, out, "get shmem symbol failed \n"); \
        *var_ptr = tmp;                                                                          \
    } while (0)

int init_shmem_wrapper() {
    void *lshmem_handle = NULL;
    int status = 0;

    const char *oshmem_lib_name = nvshmemi_options.SHMEM_LIB_NAME;

    lshmem_handle = dlopen(oshmem_lib_name, RTLD_NOW | RTLD_GLOBAL | RTLD_DEEPBIND);
    NULL_ERROR_JMP(lshmem_handle, status, NVSHMEMX_ERROR_INTERNAL, out,
                   "could not find shmem library in environment \n");

    get_symbol(lshmem_handle, "shmem_my_pe", shmem_wrapper_my_pe, status);
    get_symbol(lshmem_handle, "shmem_n_pes", shmem_wrapper_n_pes, status);
    get_symbol(lshmem_handle, "shmem_malloc", shmem_wrapper_malloc, status);
    get_symbol(lshmem_handle, "shmem_free", shmem_wrapper_free, status);
    get_symbol(lshmem_handle, "shmem_barrier_all", shmem_wrapper_barrier, status);
    get_symbol(lshmem_handle, "shmem_collect32", shmem_wrapper_allgather, status);
    get_symbol(lshmem_handle, "shmem_alltoall32", shmem_wrapper_alltoall, status);
out:
    return status;
}

static int bootstrap_shmem_finalize(bootstrap_handle_t *handle) {
    int status = 0;

    status = bootstrap_shmem_free(handle->scratch);
    NE_ERROR_JMP(status, 0, NVSHMEMX_ERROR_INTERNAL, out, "shmem_free failed \n");

out:
    return status;
}

int bootstrap_shmem_init(bootstrap_handle_t *handle) {
    int status = 0;
    long *psync = 0;

    status = init_shmem_wrapper();
    NE_ERROR_JMP(status, 0, NVSHMEMX_ERROR_INTERNAL, out, "init shmem wrapper failed \n");

    status = bootstrap_shmem_my_pe(&handle->pg_rank);
    NE_ERROR_JMP(status, 0, NVSHMEMX_ERROR_INTERNAL, out, "shmem_my_pe failed \n");

    status = bootstrap_shmem_n_pes(&handle->pg_size);
    NE_ERROR_JMP(status, 0, NVSHMEMX_ERROR_INTERNAL, out, "shmem_n_pes failed \n");

    handle->scratch_size = MAX(SHMEM_COLLECT_SYNC_SIZE, SHMEM_ALLTOALL_SYNC_SIZE) * sizeof(long);
    status = bootstrap_shmem_malloc(&handle->scratch, handle->scratch_size);
    NE_ERROR_JMP(status, 0, NVSHMEMX_ERROR_INTERNAL, out, "shmem_malloc failed \n");

    psync = (long *)handle->scratch;
    for (int i = 0; i < handle->scratch_size / sizeof(long); ++i) {
        psync[i] = SHMEM_SYNC_VALUE;
    }

    handle->allgather = bootstrap_shmem_allgather;
    handle->alltoall = bootstrap_shmem_alltoall;
    handle->barrier = bootstrap_shmem_barrier;
    handle->finalize = bootstrap_shmem_finalize;

out:
    return status;
}
