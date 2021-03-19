#include <iostream>
#include <cstdlib>
#include <cassert>
#include <chrono>

#include "cuutils.h"
#define CUUTIL_DEBUG

using namespace std;

enum Device {
    DEVICE_GPU,
    DEVICE_CPU
};

__global__ void matmul_gpu(float *ret, float *lhs, float *rhs, const unsigned int lhs_height, const unsigned int rhs_width, const unsigned int K) {
    for(int y = threadIdx.y + blockIdx.y * blockDim.y; y < lhs_height; y += blockDim.y * gridDim.y) {
        for(int x = threadIdx.x + blockIdx.x * blockDim.x; x < rhs_width; x += blockDim.x * gridDim.x) {
            float dot_product = 0.0f;
            for(register int k = 0; k < K; ++k) {
                dot_product += lhs[y * K + k] * rhs[k * rhs_width + x];
            }
            ret[y * rhs_width + x] = dot_product;
        }
    }

}

void matmul_cpu(float *ret, float *lhs, float *rhs, const unsigned int lhs_height, const unsigned int rhs_width, const unsigned int K) {
    for(int i = 0; i < lhs_height; ++i) {
        for(int j = 0; j < rhs_width; ++j) {
            ret[i * rhs_width + j] = 0.0f;
            for(int k = 0; k < K; ++k) {
                ret[i * rhs_width + j] += lhs[i * K + k] * rhs[k * rhs_width + j];
            }
        }
    }
}


__global__ void matfill_gpu(float *m, const unsigned int height, const unsigned int width, float value) {
    for(int y = threadIdx.y + blockIdx.y * blockDim.y; y < height; y += blockDim.y * gridDim.y) {
        for(int x = threadIdx.x + blockIdx.x * blockDim.x; x < width; x += blockDim.x * gridDim.x) {
            m[y * width + x] = value;
        }
    }
}

void matfill_cpu(float *m, const unsigned int height, const unsigned int width, float value) {
    for(int i = 0; i < height; ++i) {
        for(int j = 0; j < width; ++j) {
            m[i * width + j] = value;
        }
    }
}


void matmul(float *ret, float *lhs, float *rhs, const unsigned int lhs_height, const unsigned int rhs_width, const unsigned int K, Device device) {
    if(device == DEVICE_GPU) {
        dim3 dimGrid_matmul(2, 3);
        dim3 dimBlock_matmul(32, 32);
        matmul_gpu<<<dimGrid_matmul, dimBlock_matmul>>>(ret, lhs, rhs, lhs_height, rhs_width, K);
    }
    else matmul_cpu(ret, lhs, rhs, lhs_height, rhs_width, K);
}

void matfill(float *m, const unsigned int height, const unsigned int width, float value, Device device) {
    if(device == DEVICE_GPU) {
        dim3 dimGrid_matfill(2, 2);
        dim3 dimBlock_matfill(3, 3);
        matfill_gpu<<<dimGrid_matfill, dimBlock_matfill>>>(m, height, width, value);
    }
    else matfill_cpu(m, height, width, value);
}

int matcmp_cpu(float *lhs, float *rhs, const unsigned int height, const unsigned int width) {
    for(int i = 0; i < height; ++i) {
        for(int j = 0; j < width; ++j) {
            if(lhs[i * width + j] != rhs[i * width + j]) {
                return 1;
            }
        }
    }
    return 0;
}

float * matmalloc_cpu(const unsigned int height, const unsigned int width) {
    return (float *)malloc(height * width * sizeof(float));
}

void matprint(const char * name, float *m, const unsigned int height, const unsigned int width) {
    cout << name << endl;
    for(int i = 0; i < height; ++i) {
        for(int j = 0; j < width; ++j) {
            cout << m[i * width + j] << " ";
        }
        cout << endl;
    }
}

#define A_WIDTH 1024
#define A_HEIGHT 1024
#define B_WIDTH 1024
#define B_HEIGHT 1024
int main() {
    assert (A_WIDTH == B_HEIGHT);
    printf("Matmul Test: (%d x %d) * (%d x %d) --> (%d x %d)\n", A_HEIGHT, A_WIDTH, B_HEIGHT, B_WIDTH, A_HEIGHT, B_WIDTH);
    std::chrono::steady_clock::time_point begin, end;

    float * A = matmalloc_cpu(A_HEIGHT, A_WIDTH);
    float * B = matmalloc_cpu(B_HEIGHT, B_WIDTH);
    float * C = matmalloc_cpu(A_HEIGHT, B_WIDTH);
    float * C_cpu = matmalloc_cpu(A_HEIGHT, B_WIDTH);

    float *dev_a, *dev_b, *dev_c;
    CUDA_ERRCHK( cudaMalloc((void **)&dev_a, A_HEIGHT * A_WIDTH * sizeof(float)) );
    CUDA_ERRCHK( cudaMalloc((void **)&dev_b, B_HEIGHT * B_WIDTH * sizeof(float)) );
    CUDA_ERRCHK( cudaMalloc((void **)&dev_c, A_HEIGHT * B_WIDTH * sizeof(float)) );

    begin = std::chrono::steady_clock::now();
    matfill(A, A_HEIGHT, A_WIDTH, 1.0f, DEVICE_CPU);
    matfill(B, B_HEIGHT, B_WIDTH, 2.0f, DEVICE_CPU);
    matfill(C, A_HEIGHT, B_WIDTH, 1.0f, DEVICE_CPU);
    end = std::chrono::steady_clock::now();
    std::cout << "Elapsed time for matfill_cpu(A, B, C) = " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() << "[ns]" << std::endl;

    begin = std::chrono::steady_clock::now();
    matfill(dev_a, A_HEIGHT, A_WIDTH, 1.0f, DEVICE_GPU);
    matfill(dev_b, B_HEIGHT, B_WIDTH, 2.0f, DEVICE_GPU);
    matfill(dev_c, A_HEIGHT, B_WIDTH, 3.0f, DEVICE_GPU);
    end = std::chrono::steady_clock::now();
    std::cout << "Elapsed time for matfill_gpu(A, B, C) = " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() << "[ns]" << std::endl;
    


    
    begin = std::chrono::steady_clock::now();
    matmul(C_cpu, A, B, A_HEIGHT, B_WIDTH, A_WIDTH, DEVICE_CPU);
    end = std::chrono::steady_clock::now();
    std::cout << "Elapsed time for matmul_cpu = " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() << "[ns]" << std::endl;

    begin = std::chrono::steady_clock::now();
    matmul(dev_c, dev_a, dev_b, A_HEIGHT, B_WIDTH, A_WIDTH, DEVICE_GPU);
    end = std::chrono::steady_clock::now();
    std::cout << "Elapsed time for matmul_gpu = " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() << "[ns]" << std::endl;
    
    CUDA_ERRCHK( cudaMemcpy(A, dev_a, A_HEIGHT * A_WIDTH * sizeof(float), cudaMemcpyDeviceToHost) );
    CUDA_ERRCHK( cudaMemcpy(B, dev_b, B_HEIGHT * B_WIDTH * sizeof(float), cudaMemcpyDeviceToHost) );
    CUDA_ERRCHK( cudaMemcpy(C, dev_c, A_HEIGHT * B_WIDTH * sizeof(float), cudaMemcpyDeviceToHost) );
    
    /*
    cout << "After matmul process" << endl;
    matprint("C", C, A_HEIGHT, B_WIDTH);
    matprint("C_cpu", C_cpu, A_HEIGHT, B_WIDTH);
    */

    if(matcmp_cpu(C, C_cpu, A_HEIGHT, B_WIDTH)) {
        cout << "Test failed" << endl;
    } else {
        cout << "Test passed" << endl;
    }

    CUDA_ERRCHK( cudaFree((void *)dev_a) );
    CUDA_ERRCHK( cudaFree((void *)dev_b) );
    CUDA_ERRCHK( cudaFree((void *)dev_c) );
    free(A);
    free(B);
    free(C);

    return 0;
}