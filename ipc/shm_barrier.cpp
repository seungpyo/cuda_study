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

/*
void barrierWait(volatile int *barrier, volatile int *sense, unsigned int n) {
  int count;
  // Check-in
  
  count = cpu_atomic_add32(barrier, 1);
  if (count == n) {  // Last one in
    *sense = 1;
  }
  while (!*sense);

  // Check-out

  count = cpu_atomic_add32(barrier, -1);
  std::cout << getpid() << " checks out" << std::endl;
  if (count == 0) {
    *sense = 0;
  } else {
    std::cout << getpid() << " yields barrier = " << *barrier << std::endl;
  }
  while (*sense);
  std::cout << getpid() << " exits barrierWait" << std::endl;
}
*/
void waitServerInit(volatile int *sense, volatile int * counter, bool isServer) {
  if (isServer) {
    *sense = 1;
  } else {
    cpu_atomic_add32(counter, 1);
  }
  while(!*sense);
  if (!isServer) {
    cpu_atomic_add32(counter, -1);
    while(*counter);
    *sense = 0;
  }
}
