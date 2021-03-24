/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#include <pthread.h>
#include "util.h"

pthread_mutex_t global_mutex;

void nvshmemu_thread_cs_init() {
    int status = 0;

    status = pthread_mutex_init(&global_mutex, NULL);
    NZ_EXIT(status, "mutex initialization failed \n");
}

void nvshmemu_thread_cs_finalize() {
    int status = pthread_mutex_destroy(&global_mutex);
    NZ_EXIT(status, "mutex destroy failed \n");
}

void nvshmemu_thread_cs_enter() {
    int status = pthread_mutex_lock(&global_mutex);
    NZ_EXIT(status, "mutex lock failed \n");
}

void nvshmemu_thread_cs_exit() {
    int status = pthread_mutex_unlock(&global_mutex);
    NZ_EXIT(status, "mutex unlock failed \n");
}
