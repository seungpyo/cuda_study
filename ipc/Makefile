NVCC=/usr/local/cuda-11.2/bin/nvcc
all:
	$(NVCC) -g -lcuda -o memMapManager_test memMapManager.cpp memMapManager_test.cpp shm_barrier.cpp
clean:
	rm memMapManager_test
	rm *_ipc
