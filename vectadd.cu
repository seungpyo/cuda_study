#include <iostream>
#include <cunistd>
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


int main() {
	int a[N], b[N], c[N];
	int *dev_a, *dev_b, *dev_c;

	for(int i = 0; i < N; ++i) {
		a[i] = -1;
		b[i] = -2;
		c[i] = -10;
	}
	CUUTIL_ERRCHK( cudaMalloc( (void**)&dev_a, N*sizeof(int) ) );
	CUUTIL_ERRCHK( cudaMalloc( (void**)&dev_b, N*sizeof(int) ) );
	CUUTIL_ERRCHK( cudaMalloc( (void**)&dev_c, N*sizeof(int) ) );
	CUUTIL_ERRCHK( cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice) );
	CUUTIL_ERRCHK( cudaMemcpy(dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice) );
	CUUTIL_ERRCHK( cudaMemcpy(dev_c, c, N*sizeof(int), cudaMemcpyHostToDevice) );
	
	vect_add<<<1, N>>>(dev_c, dev_a, dev_b);
	CUUTIL_ERRCHK( cudaMemcpy(a, dev_a, N*sizeof(int), cudaMemcpyDeviceToHost) );
	CUUTIL_ERRCHK( cudaMemcpy(b, dev_b, N*sizeof(int), cudaMemcpyDeviceToHost) );
	CUUTIL_ERRCHK( cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost) );
	cout << "a: "; for(int i = 0; i < N; ++i) cout << a[i] << ", "; cout << endl;
	cout << "b: "; for(int i = 0; i < N; ++i) cout << b[i] << ", "; cout << endl;
	cout << "c: "; for(int i = 0; i < N; ++i) cout << c[i] << ", "; cout << endl;

	

	return 0;
}
