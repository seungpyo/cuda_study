/****
 * Copyright (c) 2016-2018, NVIDIA Corporation.  All rights reserved.
 *
 * See COPYRIGHT for license information
 ****/

#define __STDC_FORMAT_MACROS 1

#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include "util.h"
#include <execinfo.h>
#include <signal.h>

void sig_handler(int sig) {
    void *array[10];
    size_t size;

    // get void*'s for all entries on the stack
    size = backtrace(array, 10);

    // print out all the frames to stderr
    backtrace_symbols_fd(array, size, STDERR_FILENO);

    exit(1);
}

void setup_sig_handler() { signal(SIGSEGV, sig_handler); }

/* based on DJB2, result = result * 33 + char */
uint64_t getHostHash() {
    char hostname[1024];
    uint64_t result = 5381;
    int status = 0;

    status = gethostname(hostname, 1024);
    if (status) ERROR_EXIT("gethostname failed \n");

    for (int c = 0; hostname[c] != '\0'; c++) {
        result = ((result << 5) + result) + hostname[c];
    }

    INFO(NVSHMEM_UTIL, "host name: %s hash %" PRIu64, hostname, result);

    return result;
}

double parsesize(char *value) {
    long long int units;
    double size;

    if (value == NULL) return -1;

    if (strchr(value, 'G') != NULL) {
        units = 1e9;
    } else if (strchr(value, 'M') != NULL) {
        units = 1e6;
    } else if (strchr(value, 'K') != NULL) {
        units = 1e3;
    } else {
        units = 1;
    }

    size = atof(value) * units;

    return size;
}

// TODO: force to single node
int nvshmemu_get_num_gpus_per_node() { return 128; }

int cuCheck(CUresult res) {
    cuGetErrorString(res, (const char **)&p_err_str);
    if (CUDA_SUCCESS != res) {
        fprintf(stderr, "[%s:%d] cuda failed with %s\n", __FILE__, __LINE__, p_err_str);
        return static_cast<int>(res);
    }
    return static_cast<int>(res);
}

int cudaCheck(cudaError_t res) {
    char *errstr = (char *)cudaGetErrorString(res);
    if (cudaSuccess != res) {
        fprintf(stderr, "[%s:%d] cuda failed with %s\n", __FILE__, __LINE__, errstr);
        return static_cast<int>(res);
    }
    return static_cast<int>(res);
}

/* Convert data to a hexadecimal string */
char * nvshmemu_hexdump(void *ptr, size_t len) {
    const char *hex = "0123456789abcdef";

    char *str = (char *) malloc(len*2 + 1);
    if (str == NULL) return NULL;

    char *ptr_c = (char *) ptr;

    for (size_t i = 0; i < len; i++) {
        str[i*2]   = hex[(ptr_c[i] >> 4) & 0xF];
        str[i*2+1] = hex[ptr_c[i] & 0xF];
    }

    str[len*2] = '\0';

    return str;
}
