/****
 * Copyright (c) 2016-2020, NVIDIA Corporation.  All rights reserved.
 *
 * Copyright 2011 Sandia Corporation. Under the terms of Contract
 * DE-AC04-94AL85000 with Sandia Corporation, the U.S.  Government
 * retains certain rights in this software.
 *
 * Copyright (c) 2017 Intel Corporation. All rights reserved.
 * This software is available to you under the BSD license.
 *
 * Portions of this file are derived from Sandia OpenSHMEM.
 *
 * See COPYRIGHT for license information
 ****/

/* NVSHMEMI_ENV_DEF( name, kind, default, category, short description )
 *
 * Kinds: long, size, bool, string
 * Categories: NVSHMEMI_ENV_CAT_OPENSHMEM, NVSHMEMI_ENV_CAT_OTHER,
 *             NVSHMEMI_ENV_CAT_COLLECTIVES, NVSHMEMI_ENV_CAT_TRANSPORT,
 *             NVSHMEMI_ENV_CAT_HIDDEN
 */

/*
 * Preincluded header requirements: nvshmem_internal.h
 */

NVSHMEMI_ENV_DEF(VERSION, bool, false, NVSHMEMI_ENV_CAT_OPENSHMEM,
                 "Print library version at startup")
NVSHMEMI_ENV_DEF(INFO, bool, false, NVSHMEMI_ENV_CAT_OPENSHMEM,
                 "Print environment variable options at startup")
NVSHMEMI_ENV_DEF(INFO_HIDDEN, bool, false, NVSHMEMI_ENV_CAT_HIDDEN,
                 "Print hidden environment variable options at startup")
NVSHMEMI_ENV_DEF(SYMMETRIC_SIZE, size, (size_t)(SYMMETRIC_SIZE_DEFAULT), NVSHMEMI_ENV_CAT_OPENSHMEM,
                 "Symmetric heap size")
NVSHMEMI_ENV_DEF(DEBUG, string, "", NVSHMEMI_ENV_CAT_OPENSHMEM,
                 "Set to enable debugging messages.\n"
                 "\tOptional values: VERSION, WARN, INFO, ABORT, TRACE")

/** Debugging **/

NVSHMEMI_ENV_DEF(DEBUG_SUBSYS, string, "", NVSHMEMI_ENV_CAT_HIDDEN,
                 "Comma separated list of debugging message sources. Prefix with '^' to exclude.\n"
                 "\tValues: INIT, COLL, P2P, PROXY, TRANSPORT, MEM, BOOTSTRAP, TOPO, UTIL, ALL")
NVSHMEMI_ENV_DEF(DEBUG_FILE, string, "", NVSHMEMI_ENV_CAT_OTHER,
                 "Debugging output filename, may contain %h for hostname and %p for pid")
NVSHMEMI_ENV_DEF(ENABLE_ERROR_CHECKS, bool, false, NVSHMEMI_ENV_CAT_HIDDEN,
                 "Enable error checks")

/** Bootstrap **/

#if   defined(NVSHMEM_DEFAULT_PMIX)
#define NVSHMEMI_ENV_BOOTSTRAP_DEFAULT "PMIX"
#elif defined(NVSHMEM_DEFAULT_PMI2)
#define NVSHMEMI_ENV_BOOTSTRAP_DEFAULT "PMI-2"
#else
#define NVSHMEMI_ENV_BOOTSTRAP_DEFAULT "PMI"
#endif

NVSHMEMI_ENV_DEF(BOOTSTRAP_PMI, string, NVSHMEMI_ENV_BOOTSTRAP_DEFAULT, NVSHMEMI_ENV_CAT_OTHER,
                 "Name of the bootstrap that should be used to initialize\n"
                 "\tNVSHMEM. Allowed values: PMI, PMI-2"
#ifdef NVSHMEM_PMIX_SUPPORT
                 ", PMIX"
#endif
                 )

#undef NVSHMEMI_ENV_BOOTSTRAP_DEFAULT

NVSHMEMI_ENV_DEF(MPI_LIB_NAME, string, "libmpi.so", NVSHMEMI_ENV_CAT_OTHER,
                 "Name of the MPI shared library to be used for bootstrap")
NVSHMEMI_ENV_DEF(SHMEM_LIB_NAME, string, "liboshmem.so", NVSHMEMI_ENV_CAT_OTHER,
                 "Name of the OpenSHMEM shared library to be used for bootstrap")

NVSHMEMI_ENV_DEF(BYPASS_ACCESSIBILITY_CHECK, bool, false, NVSHMEMI_ENV_CAT_HIDDEN,
                 "Bypass peer GPU accessbility checks")

#if defined(NVSHMEM_PPC64LE)
NVSHMEMI_ENV_DEF(CUDA_LIMIT_STACK_SIZE, size, 0, NVSHMEMI_ENV_CAT_OTHER,
                 "Specify limit on stack size of each GPU thread")
#endif

/** General Collectives **/

NVSHMEMI_ENV_DEF(DISABLE_NCCL, bool, false, NVSHMEMI_ENV_CAT_COLLECTIVES,
                 "Disable use of NCCL for collective operations")
NVSHMEMI_ENV_DEF(BARRIER_DISSEM_KVAL, int, 2, NVSHMEMI_ENV_CAT_COLLECTIVES,
                 "Radix of the dissemination algorithm used for barriers")
NVSHMEMI_ENV_DEF(BARRIER_TG_DISSEM_KVAL, int, 2, NVSHMEMI_ENV_CAT_COLLECTIVES,
                 "Radix of the dissemination algorithm used for thread group barriers")
NVSHMEMI_ENV_DEF(REDUCE_RECEXCH_KVAL, int, 2, NVSHMEMI_ENV_CAT_HIDDEN,
                 "Radix of the recursive exchange reduction algorithm")

/** CPU Collectives **/

NVSHMEMI_ENV_DEF(RDX_NUM_TPB, int, 32, NVSHMEMI_ENV_CAT_HIDDEN,
                 "Number of threads per block used for reduction purposes")

/** Transport **/

NVSHMEMI_ENV_DEF(ASSERT_ATOMICS_SYNC, bool, false, NVSHMEMI_ENV_CAT_HIDDEN,
                 "Bypass flush on wait_until at target")
NVSHMEMI_ENV_DEF(DISABLE_IB_NATIVE_ATOMICS, bool, false, NVSHMEMI_ENV_CAT_TRANSPORT,
                 "disable use of InfiniBand native atomics")
NVSHMEMI_ENV_DEF(DISABLE_GDRCOPY, bool, false, NVSHMEMI_ENV_CAT_TRANSPORT,
                 "disable use of GDRCopy in IB RC Transport")
NVSHMEMI_ENV_DEF(BYPASS_FLUSH, bool, false, NVSHMEMI_ENV_CAT_HIDDEN,
                 "Bypass flush in proxy when enforcing consistency")
NVSHMEMI_ENV_DEF(ENABLE_NIC_PE_MAPPING, bool, false, NVSHMEMI_ENV_CAT_TRANSPORT,
                 "When not set or set to 0, a PE is assigned the NIC on the node that is\n"
                 "\tclosest to it by distance. When set to 1, NVSHMEM either assigns NICs to\n"
                 "\tPEs on a round-robin basis or uses NVSHMEM_HCA_PE_MAPPING or\n"
                 "\tNVSHMEM_HCA_LIST when they are specified.")
NVSHMEMI_ENV_DEF(IB_GID_INDEX, int, 0, NVSHMEMI_ENV_CAT_TRANSPORT,
                 "Source GID Index for ROCE")
NVSHMEMI_ENV_DEF(IB_TRAFFIC_CLASS, int, 0, NVSHMEMI_ENV_CAT_TRANSPORT,
                 "Traffic calss for ROCE")
NVSHMEMI_ENV_DEF(IB_SL, int, 0, NVSHMEMI_ENV_CAT_TRANSPORT,
                 "Service level to use over IB/ROCE")

NVSHMEMI_ENV_DEF(HCA_LIST, string, "", NVSHMEMI_ENV_CAT_TRANSPORT,
                 "Comma-separated list of HCAs to use in the NVSHMEM application. Entries\n"
                 "\tare of the form hca_name:port, e.g. mlx5_1:1,mlx5_2:2 and entries\n"
                 "\tprefixed by ^ are excluded. NVSHMEM_ENABLE_NIC_PE_MAPPING must be set to\n"
                 "\t1 for this variable to be effective.")

NVSHMEMI_ENV_DEF(HCA_PE_MAPPING, string, "", NVSHMEMI_ENV_CAT_TRANSPORT,
                 "Specifies mapping of HCAs to PEs as a comma-separated list. Each entry\n"
                 "\tin the comma separated list is of the form hca_name:port:count.  For\n"
                 "\texample, mlx5_0:1:2,mlx5_0:2:2 indicates that PE0, PE1 are mapped to\n"
                 "\tport 1 of mlx5_0, and PE2, PE3 are mapped to port 2 of mlx5_0.\n"
                 "\tNVSHMEM_ENABLE_NIC_PE_MAPPING must be set to 1 for this variable to be\n"
                 "\teffective.")
NVSHMEMI_ENV_DEF(QP_DEPTH, int, 1024, NVSHMEMI_ENV_CAT_HIDDEN,
                 "Number of WRs in QP")
NVSHMEMI_ENV_DEF(SRQ_DEPTH, int, 16384, NVSHMEMI_ENV_CAT_HIDDEN,
                 "Number of WRs in SRQ")

/** Runtime optimimzations **/

NVSHMEMI_ENV_DEF(IS_P2P_RUN, bool, false, NVSHMEMI_ENV_CAT_HIDDEN,
                 "Tells whether all GPUs are p2p connected. NVSHMEM can use this information\n"
                 "\tfor runtime optimizations. Results can be erroneous if this env variable is\n"
                 "\tset to 1 but all the GPUs are not p2p connected.")
