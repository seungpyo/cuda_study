#include <iostream>
#include "cuutils.h"

#define N 8
#define CUUTIL_DEBUG
using namespace std;

__global__ void vect_add(int *c, int *a, int *b) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	for(; tid < N; tid += blockDim.x * gridDim.x) {

		c[tid] = a[tid] + b[tid];
	}
}

__global__ void vect_fill(int *v, int val) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	for(; tid < N; tid += blockDim.x * gridDim.x) {
		v[tid] = val;
	}
}	


	
int main() {
	int a[N], b[N], c[N];
	int *dev_a, *dev_b, *dev_c;

	for(int i = 0; i < N; ++i) {
		a[i] = -1;
		b[i] = -2;
		c[i] = -10;
	}
	CUDA_ERRCHK( cudaMalloc( (void**)&dev_a, N*sizeof(int) ) );
	CUDA_ERRCHK( cudaMalloc( (void**)&dev_b, N*sizeof(int) ) );
	CUDA_ERRCHK( cudaMalloc( (void**)&dev_c, N*sizeof(int) ) );
	
	CUDA_ERRCHK( cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice) );
	CUDA_ERRCHK( cudaMemcpy(dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice) );

	vect_fill<<<N, 1>>>(dev_a, 1);
	vect_fill<<<N, 1>>>(dev_b, 2);
	vect_add<<<N, 1>>>(dev_c, dev_a, dev_b);
	CUDA_ERRCHK( cudaMemcpy(a, dev_a, N*sizeof(int), cudaMemcpyDeviceToHost) );
	CUDA_ERRCHK( cudaMemcpy(b, dev_b, N*sizeof(int), cudaMemcpyDeviceToHost) );
	CUDA_ERRCHK( cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost) );
	cout << "a: "; for(int i = 0; i < N; ++i) cout << a[i] << ", "; cout << endl;
	cout << "b: "; for(int i = 0; i < N; ++i) cout << b[i] << ", "; cout << endl;
	cout << "c: "; for(int i = 0; i < N; ++i) cout << c[i] << ", "; cout << endl;
	return 0;
}
