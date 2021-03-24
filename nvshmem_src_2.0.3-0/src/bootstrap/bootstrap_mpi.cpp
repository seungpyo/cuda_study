/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"

#include "util.h"
#include "mpi.h"
#include "nvshmemx_error.h"
#include "bootstrap.h"
#include "bootstrap_internal.h"
#include <stdlib.h>
#include <dlfcn.h>

int (*mpi_wrapper_comm_rank)(MPI_Comm comm, int *rank);
int (*mpi_wrapper_comm_size)(MPI_Comm comm, int *size);
int (*mpi_wrapper_barrier)(MPI_Comm comm);
int (*mpi_wrapper_allgather)(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                             void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm);
int (*mpi_wrapper_alltoall)(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                             void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm);

#ifdef NVSHMEM_MPI_IS_OMPI
MPI_Datatype mpi_wrapper_byte;
#endif

typedef struct {
    MPI_Comm mpi_comm;
} mpi_info_t;

int bootstrap_mpi_barrier(struct bootstrap_handle *handle) {
    int status = 0;
    mpi_info_t *mpi_info = (mpi_info_t *)handle->internal;

    status = mpi_wrapper_barrier(mpi_info->mpi_comm);
    NE_ERROR_JMP(status, MPI_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "MPI_Barrier failed \n");

out:
    return status;
}

int bootstrap_mpi_allgather(const void *sendbuf, void *recvbuf, int length,
                            struct bootstrap_handle *handle) {
    int status = 0;
    mpi_info_t *mpi_info = (mpi_info_t *)handle->internal;

#ifdef NVSHMEM_MPI_IS_OMPI
    status = mpi_wrapper_allgather(sendbuf, length, mpi_wrapper_byte, recvbuf, length,
                                   mpi_wrapper_byte, mpi_info->mpi_comm);
    NE_ERROR_JMP(status, MPI_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "MPI_Allgather failed \n");
#else
    status = mpi_wrapper_allgather(sendbuf, length, MPI_BYTE, recvbuf, length, MPI_BYTE,
                                   mpi_info->mpi_comm);
    NE_ERROR_JMP(status, MPI_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "MPI_Allgather failed \n");
#endif

out:
    return status;
}

int bootstrap_mpi_alltoall(const void *sendbuf, void *recvbuf, int length,
                            struct bootstrap_handle *handle) {
    int status = 0;
    mpi_info_t *mpi_info = (mpi_info_t *)handle->internal;

#ifdef NVSHMEM_MPI_IS_OMPI
    status = mpi_wrapper_alltoall(sendbuf, length, mpi_wrapper_byte, recvbuf, length,
                                   mpi_wrapper_byte, mpi_info->mpi_comm);
    NE_ERROR_JMP(status, MPI_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "MPI_Alltoall failed \n");
#else
    status = mpi_wrapper_alltoall(sendbuf, length, MPI_BYTE, recvbuf, length, MPI_BYTE,
                                   mpi_info->mpi_comm);
    NE_ERROR_JMP(status, MPI_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "MPI_Alltoall failed \n");
#endif

out:
    return status;
}


#define get_symbol(lib_handle, name, var, status)                                              \
    do {                                                                                       \
        void **var_ptr = (void **)&var;                                                        \
        void *tmp = (void *)dlsym(lib_handle, name);                                           \
        NULL_ERROR_JMP(tmp, status, NVSHMEMX_ERROR_INTERNAL, out, "get mpi symbol failed \n"); \
        *var_ptr = tmp;                                                                        \
    } while (0)

int init_mpi_wrapper() {
    void *lmpi_handle = NULL;
    int status = 0;

    const char *mpi_lib_name = nvshmemi_options.MPI_LIB_NAME;

    lmpi_handle = dlopen(mpi_lib_name, RTLD_NOW | RTLD_GLOBAL | RTLD_DEEPBIND);

    if (lmpi_handle == NULL) {
        // Spectrum MPI names its library differently; try again for that case.
        mpi_lib_name = (char *)"libmpi_ibm.so";
        lmpi_handle = dlopen((const char *)mpi_lib_name, RTLD_NOW | RTLD_GLOBAL | RTLD_DEEPBIND);
    }

    NULL_ERROR_JMP(lmpi_handle, status, NVSHMEMX_ERROR_INTERNAL, out,
                   "could not find mpi library in environment \n");

    get_symbol(lmpi_handle, "MPI_Comm_rank", mpi_wrapper_comm_rank, status);
    get_symbol(lmpi_handle, "MPI_Comm_size", mpi_wrapper_comm_size, status);
    get_symbol(lmpi_handle, "MPI_Barrier", mpi_wrapper_barrier, status);
    get_symbol(lmpi_handle, "MPI_Allgather", mpi_wrapper_allgather, status);
    get_symbol(lmpi_handle, "MPI_Alltoall", mpi_wrapper_alltoall, status);
#ifdef NVSHMEM_MPI_IS_OMPI
    get_symbol(lmpi_handle, "ompi_mpi_byte", mpi_wrapper_byte, status);
#endif
out:
    return status;
}

static int bootstrap_mpi_finalize(bootstrap_handle_t *handle) {
    mpi_info_t *mpi_info = (mpi_info_t *)handle->internal;

    if (mpi_info) {
        free(mpi_info);
        handle->internal = NULL;
    }

    return 0;
}

int bootstrap_mpi_init(void *mpi_comm, bootstrap_handle_t *handle) {
    int status = 0;
    mpi_info_t *mpi_info;

    status = init_mpi_wrapper();
    NE_ERROR_JMP(status, 0, NVSHMEMX_ERROR_INTERNAL, out, "init mpi wrapper failed \n");

    status = mpi_wrapper_comm_rank(*((MPI_Comm *)mpi_comm), &handle->pg_rank);
    NE_ERROR_JMP(status, MPI_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "MPI_Comm_rank failed \n");

    status = mpi_wrapper_comm_size(*((MPI_Comm *)mpi_comm), &handle->pg_size);
    NE_ERROR_JMP(status, MPI_SUCCESS, NVSHMEMX_ERROR_INTERNAL, out, "MPI_Comm_size failed \n");

    mpi_info = (mpi_info_t *)malloc(sizeof(mpi_info_t));
    NULL_ERROR_JMP(mpi_info, status, NVSHMEMX_ERROR_OUT_OF_MEMORY, out,
                   "memory allocation for mpi_info failed \n");

    mpi_info->mpi_comm = *((MPI_Comm *)mpi_comm);
    handle->internal = (void *)mpi_info;
    handle->allgather = bootstrap_mpi_allgather;
    handle->alltoall = bootstrap_mpi_alltoall;
    handle->barrier = bootstrap_mpi_barrier;
    handle->finalize = bootstrap_mpi_finalize;

out:
    return status;
}
