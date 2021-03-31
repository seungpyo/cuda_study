#pragma once

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string>
#include <vector>
#include <algorithm>
#include <thread>
#include <mutex>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <sys/mman.h>
#include <atomic>
#include <semaphore.h>

static const char * barrier_name = "MemMapManagerBarrier";
static sem_t sem;

typedef struct sharedMemoryInfo_st {
    void *addr;
    size_t size;
    int shmFd;
} sharedMemoryInfo;

static sharedMemoryInfo shmInfo;

int sharedMemoryCreate(const char *name, size_t sz, sharedMemoryInfo *info);

int sharedMemoryOpen(const char *name, size_t sz, sharedMemoryInfo *info);

void sharedMemoryClose(sharedMemoryInfo *info);


typedef struct shmStruct_st {
  size_t nprocesses;
  int counter;
  int sense;
} shmStruct;

#define cpu_atomic_add32(a, x) __sync_add_and_fetch(a, x)

// void barrierWait(volatile int *barrier, volatile int *sense, unsigned int n);
void waitServerInit(volatile int *sense, volatile int *counter, bool isServer);
