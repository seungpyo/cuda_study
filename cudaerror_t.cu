#include <iostream>
using namespace std;

int main() {
	cout << cudaGetErrorString(cudaErrorMemoryAllocation) << endl;
	cout << cudaGetErrorString(cudaErrorInvalidValue) << endl;
	cout << cudaGetErrorString(cudaSuccess) << endl;
	return 0;
}

