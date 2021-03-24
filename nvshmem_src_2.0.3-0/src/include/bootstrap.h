/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef __NVSHMEM_BOOTSTRAP_H
#define __NVSHMEM_BOOTSTRAP_H

#include <cstddef>

#define PREINIT_HANDLE_SIZE 128
#define MAX_LENGTH_ERROR_STRING 128

enum {
    BOOTSTRAP_MPI,
    BOOTSTRAP_SHMEM,
    BOOTSTRAP_PMI
};

typedef struct {
    char internal[PREINIT_HANDLE_SIZE];
} bootstrap_preinit_handle_t;

typedef struct bootstrap_attr {
    bootstrap_attr() : npes(1), mpi_comm(NULL) {}
    int npes;
    void *mpi_comm;
    bootstrap_preinit_handle_t *preinit_handle;
} bootstrap_attr_t;

typedef struct bootstrap_handle {
    bootstrap_handle() : internal(NULL), allgather(NULL), alltoall(NULL), barrier(NULL) {}
    int pg_rank;
    int pg_size;
    void *internal;
    void *scratch;
    uint64_t scratch_size;
    int (*allgather)(const void *sendbuf, void *recvbuf, int bytes,
                     struct bootstrap_handle *handle);
    int (*alltoall)(const void *sendbuf, void *recvbuf, int bytes,
                     struct bootstrap_handle *handle);
    int (*barrier)(struct bootstrap_handle *handle);
    int (*finalize)(struct bootstrap_handle *handle);
} bootstrap_handle_t;

int bootstrap_init(int mode, bootstrap_attr_t *attr, bootstrap_handle_t *handle);
int bootstrap_finalize(bootstrap_handle_t *handle);

#endif
