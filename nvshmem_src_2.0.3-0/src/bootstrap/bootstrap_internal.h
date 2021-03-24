/*
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */
#ifndef _BOOTSTRAP_INTERNAL_H_
#define _BOOTSTRAP_INTERNAL_H_

int bootstrap_mpi_init(void *mpi_comm, bootstrap_handle_t *handle);
int bootstrap_shmem_init(bootstrap_handle_t *handle);
int bootstrap_pmi_init(bootstrap_handle_t *handle);
int bootstrap_pmi2_init(bootstrap_handle_t *handle);
int bootstrap_loader_init(const char *plugin, bootstrap_handle_t *handle);

#endif
