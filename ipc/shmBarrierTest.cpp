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
#include "cuda.h"

const char * barrier_name = "simplebarrier";

typedef struct sharedMemoryInfo_st {
    void *addr;
    size_t size;
    int shmFd;
} sharedMemoryInfo;

int sharedMemoryCreate(const char *name, size_t sz, sharedMemoryInfo *info) {
  int status = 0;

  info->size = sz;

  info->shmFd = shm_open(name, O_RDWR | O_CREAT, 0777);
  if (info->shmFd < 0) {
    return errno;
  }

  status = ftruncate(info->shmFd, sz);
  if (status != 0) {
    return status;
  }

  info->addr = mmap(0, sz, PROT_READ | PROT_WRITE, MAP_SHARED, info->shmFd, 0);
  if (info->addr == NULL) {
    return errno;
  }

  return 0;
}

int sharedMemoryOpen(const char *name, size_t sz, sharedMemoryInfo *info) {
  info->size = sz;

  info->shmFd = shm_open(name, O_RDWR, 0777);
  if (info->shmFd < 0) {
    return errno;
  }

  info->addr = mmap(0, sz, PROT_READ | PROT_WRITE, MAP_SHARED, info->shmFd, 0);
  if (info->addr == NULL) {
    return errno;
  }

  return 0;
}

void sharedMemoryClose(sharedMemoryInfo *info) {
  if (info->addr) {
    munmap(info->addr, info->size);
  }
  if (info->shmFd) {
    close(info->shmFd);
  }
}

typedef struct shmStruct_st {
  size_t nprocesses;
  int barrier;
  int sense;
} shmStruct;
#define cpu_atomic_add32(a, x) __sync_add_and_fetch(a, x)
static void barrierWait(volatile int *barrier, volatile int *sense,
                        unsigned int n) {
  int count;
  // std::cout << getpid() << " is entering barrier!" << std::endl;
  // Check-in
  count = cpu_atomic_add32(barrier, 1);
  // std::cout << "checked in: barrier = " << count << std::endl;
  if (count == n) {  // Last one in
    *sense = 1;
  }
  while (!*sense)
    ;
    // std::cout << "middle : barrier = " << count << std::endl;

  // Check-out
  count = cpu_atomic_add32(barrier, -1);
  // std::cout << "checked out: barrier = " << count << std::endl;
  if (count == 0) {  // Last one out
    *sense = 0;
  }
  // std::cout << "spin lock at the end, count = " << count << std::endl;
  while (*sense)
    ;
    // std::cout << getpid() << " is exiting barrier!" << std::endl;
}
#define SHM_SIZE 128
int main() {
    // Ensure that MemMapManager instance exists before forking.
    
    pid_t pid;
    if ((pid = fork()) == 0) {
        // And then, launch client process.
        
        sharedMemoryInfo shmInfo;
        if (sharedMemoryOpen(barrier_name, SHM_SIZE,  &shmInfo) < 0) {
            perror("main, sharedMemoryCreate");
        }
        volatile char * shm = (volatile char *)shmInfo.addr;
        std::cout << "child: " << shm[0] << std::endl;
        sleep(3);
        std::cout << "child read " << shm[0] << std::endl;
        sharedMemoryClose(&shmInfo);
    } else {
        // parent process as a demo server.
        sharedMemoryInfo shmInfo;
        if (sharedMemoryCreate(barrier_name, SHM_SIZE,  &shmInfo) < 0) {
            perror("main, sharedMemoryCreate");
        }
        
        volatile char * shm = (volatile char *)shmInfo.addr;
        std::cout << "parent " << shm[0] << std::endl;
        shm[0] = 'X';
        std::cout << "parent wrote " << shm[0] << std::endl;
        int status;
        wait(&status);
        // barrierWait(&shm->barrier, &shm->sense, (unsigned int)(shm->nprocesses));
        sharedMemoryClose(&shmInfo);
    }
    
    return 0;
}