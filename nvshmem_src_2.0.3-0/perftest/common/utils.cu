/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * See COPYRIGHT.txt for license information
 */

#include "utils.h"

double *d_latency = NULL;
double *d_avg_time = NULL;
double *latency = NULL;
double *avg_time = NULL;
int mype = 0;
int npes = 0;
int use_mpi = 0;
int use_shmem = 0;
__device__ int clockrate;

void select_device() {
    cudaDeviceProp prop;
    int dev_count;
    int mype_node;

    mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);

    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    CUDA_CHECK(cudaSetDevice(mype_node % dev_count));

    CUDA_CHECK(cudaGetDeviceProperties(&prop, mype_node % dev_count));
    fprintf(stderr, "mype: %d device name: %s bus id: %d \n", mype_node, prop.name, prop.pciBusID);
    CUDA_CHECK(cudaMemcpyToSymbol(clockrate, (void *)&prop.clockRate, sizeof(int), 0,
                                  cudaMemcpyHostToDevice));
}

void init_wrapper(int *c, char ***v) {
    char *value;

#ifdef NVSHMEM_MPI_SUPPORT
    value = getenv("NVSHMEMTEST_USE_MPI_LAUNCHER");
    if (value) use_mpi = atoi(value);
#endif

#ifdef NVSHMEM_SHMEM_SUPPORT
    value = getenv("NVSHMEMTEST_USE_SHMEM_LAUNCHER");
    if (value) use_shmem = atoi(value);
#endif

#ifdef NVSHMEM_MPI_SUPPORT
    if (use_mpi) {
        MPI_Init(c, v);
        int rank, nranks;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &nranks);
        DEBUG_PRINT("MPI: [%d of %d] hello MPI world! \n", rank, nranks);
        MPI_Comm mpi_comm = MPI_COMM_WORLD;

        nvshmemx_init_attr_t attr;
        attr.mpi_comm = &mpi_comm;
        nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

        select_device();

        return;
    }
#endif

#ifdef NVSHMEM_SHMEM_SUPPORT
    if (use_shmem) {
        shmem_init();
        mype = shmem_my_pe();
        npes = shmem_n_pes();
        DEBUG_PRINT("SHMEM: [%d of %d] hello SHMEM world! \n", my_pe, n_pes);

        latency = (double *)shmem_malloc(sizeof(double));
        if (!latency) ERROR_EXIT("(shmem_malloc) failed \n");

        avg_time = (double *)shmem_malloc(sizeof(double));
        if (!avg_time) ERROR_EXIT("(shmem_malloc) failed \n");

        nvshmemx_init_attr_t attr;
        nvshmemx_init_attr(NVSHMEMX_INIT_WITH_SHMEM, &attr);

        select_device();

        return;
    }
#endif

    nvshmem_init();

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();
    select_device();

    d_latency = (double *)nvshmem_malloc(sizeof(double));
    if (!d_latency) ERROR_EXIT("nvshmem_malloc failed \n");

    d_avg_time = (double *)nvshmem_malloc(sizeof(double));
    if (!d_avg_time) ERROR_EXIT("nvshmem_malloc failed \n");

    DEBUG_PRINT("end of init \n");
    return;
}

void finalize_wrapper() {
#ifdef NVSHMEM_SHMEM_SUPPORT
    if (use_shmem) {
        shmem_free(latency);
        shmem_free(avg_time);
    }
#endif

#if !defined(NVSHMEM_SHMEM_SUPPORT) && !defined(NVSHMEM_MPI_SUPPORT)
    if (!use_mpi && !use_shmem) {
        nvshmem_free(d_latency);
        nvshmem_free(d_avg_time);
    }
#endif
    nvshmem_finalize();

#ifdef NVSHMEM_MPI_SUPPORT
    if (use_mpi) MPI_Finalize();
#endif
#ifdef NVSHMEM_SHMEM_SUPPORT
    if (use_shmem) shmem_finalize();
#endif
}

void
alloc_tables(void ***table_mem, int num_tables, int num_entries_per_table)
{
    void **tables;
    int i, dev_property;

    CUDA_CHECK(cudaDeviceGetAttribute(&dev_property, cudaDevAttrUnifiedAddressing, nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE)));
    assert(dev_property == 1);

    assert(num_tables >= 1);
    assert(num_entries_per_table >= 1);
    CUDA_CHECK(cudaHostAlloc(table_mem, num_tables * sizeof(void *), cudaHostAllocMapped));
    tables = *table_mem;

    /* Just allocate an array of 8 byte values. The user can decide if they want to use double or uint64_t */
    for (i = 0; i < num_tables; i++) {
        CUDA_CHECK(cudaHostAlloc(&tables[i], num_entries_per_table * sizeof(double), cudaHostAllocMapped));
    }
}

void
free_tables(void **tables, int num_tables)
{
    int i;
    for (i = 0; i < num_tables; i++) {
        CUDA_CHECK(cudaFreeHost(tables[i]));
    }
    CUDA_CHECK(cudaFreeHost(tables));
}

void
print_table(const char *job_name, const char *subjob_name, const char *var_name,
            const char *output_var, const char *units, const char plus_minus,
            uint64_t *size, double *value, int num_entries)
{
	int i;

/* Used for automated test output. It outputs the data in a non human-friendly format. */
#ifdef NVSHMEM_MACHINE_READABLE_OUTPUT

    printf("%s\n", job_name);
    for (i = 0; i < num_entries; i++) {
        if (size[i] != 0 && value[i] != 0.00) {
            printf("&&&& PERF %s___%s___size__%lu___%s %lf %c%s\n", job_name, subjob_name, size[i], output_var, value[i], plus_minus, units);
        }
    }
#else
	printf("+------------------------+----------------------+\n");
	printf("| %-22s | %-20s |\n", job_name, subjob_name);
	printf("+------------------------+----------------------+\n");
	printf("| %-22s | %10s %-9s |\n", var_name, output_var, units);
	printf("+------------------------+----------------------+\n");
	for (i = 0; i < num_entries; i++) {
		printf("| %-22.0lu | %-20.6lf |\n", size[i], value[i]);
		printf("+------------------------+----------------------+\n");
	}
#endif
	printf("\n\n");
}
