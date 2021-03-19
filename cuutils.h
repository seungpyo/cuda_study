#ifdef CUUTIL_DEBUG
#define CUDA_ERRCHK(x) do { \
	(x); \
	cudaError_t e = cudaGetLastError(); \
	if(e != cudaSuccess) { \
		printf("CUDA failure at %s %d: %s\n", \
			__FILE__, __LINE__, cudaGetErrorString(e)); \
		exit(-1);  \
	} \
} while(0)
#endif
#ifndef CUUTIL_DEBUG
#define CUDA_ERRCHK(x) (x)
#endif
