/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include "nvshmem.h"

#include "util.h"
#include "nvshmemx_error.h"
#include "bootstrap.h"
#include "bootstrap_internal.h"

int bootstrap_preinit(bootstrap_preinit_handle_t *handle) {
    ERROR_PRINT("not implemented");
    return NVSHMEMX_ERROR_INTERNAL;
}

int bootstrap_init(int mode, bootstrap_attr_t *attr, bootstrap_handle_t *handle) {
    int status = NVSHMEMX_SUCCESS;

    switch (mode) {
        case BOOTSTRAP_MPI:
#ifdef NVSHMEM_MPI_SUPPORT
            status = bootstrap_mpi_init(attr->mpi_comm, handle);
            NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "bootstrap_mpi_init returned error\n");
#else
            ERROR_PRINT("MPI-based initialization selected but NVSHMEM not built with MPI Support\n");
#endif
            break;
        case BOOTSTRAP_SHMEM:
#ifdef NVSHMEM_SHMEM_SUPPORT
            status = bootstrap_shmem_init(handle);
            NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                    "bootstrap_shmem_init returned error\n");
#else
            ERROR_PRINT("OpenSHMEM-based bootstrap selected but NVSHMEM not built with OpenSHMEM Support\n");
#endif
            break;
        case BOOTSTRAP_PMI:
            if (strcmp(nvshmemi_options.BOOTSTRAP_PMI, "PMIX") == 0) {
#ifdef NVSHMEM_PMIX_SUPPORT
                status = bootstrap_loader_init("nvshmem_pmix.so", handle);
                NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "bootstrap_loader_init returned error\n");
#else
                ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
                          "PMIx bootstrap selected but NVSHMEM not built with PMIx support\n");
#endif
            } else if (strcmp(nvshmemi_options.BOOTSTRAP_PMI, "PMI-2") == 0 ||
                       strcmp(nvshmemi_options.BOOTSTRAP_PMI, "PMI2") == 0) {
                status = bootstrap_pmi2_init(handle);
                NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "bootstrap_pmi_init returned error\n");
            } else if (strcmp(nvshmemi_options.BOOTSTRAP_PMI, "PMI") == 0) {
                status = bootstrap_pmi_init(handle);
                NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "bootstrap_pmi_init returned error\n");
            } else {
                ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out, "bootstrap_pmi_init invalid bootstrap '%s'\n",
                        nvshmemi_options.BOOTSTRAP_PMI);
            }
            break;
        default:
            ERROR_PRINT("Invalid bootstrap mode selected\n");
    }

out:
    return status;
}

int bootstrap_finalize(bootstrap_handle_t *handle) {
    int status = NVSHMEMX_SUCCESS;

    status = handle->finalize(handle);
    NZ_ERROR_JMP(status, NVSHMEMX_ERROR_INTERNAL, out,
            "bootstrap finalization returned error\n");

out:
    return status;
}
