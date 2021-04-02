NVCC=/usr/local/cuda-11.2/bin/nvcc
all:
	$(NVCC) -g -lcuda -lrt -o memMapManager_test memMapManager.cpp memMapManager_test.cpp
clean:
	rm memMapManager_test
	rm pid_*
	rm /dev/shm/sem.MemMapManager_Server_Barrier
	rm MemMapManager_Server_EndPoint
