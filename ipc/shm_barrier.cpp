#include "shm_barrier.h"

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

void waitServerInit(volatile int *sense, volatile int * counter, bool isServer) {
  if (isServer) {
    std::cout << "Server in" << std::endl;
    *sense = 1;
  } else {
    std::cout << "client in" << std::endl;
    cpu_atomic_add32(counter, 1);
  }
  while(!*sense);
  if (!isServer) {
    cpu_atomic_add32(counter, -1);
    while(*counter);
    *sense = 0;
  }
}
