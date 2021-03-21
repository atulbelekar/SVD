#include <device_functions.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include<iostream>
#include "Matrix.h"

using namespace std;


__global__ void matmul(Matrix a, Matrix b, Matrix c) {
    
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < a.row; i+= (blockDim.x * gridDim.x)) {
        for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < b.col; j+=(blockDim.x * gridDim.x)) {
            double sum = 0;
            for (int k = 0; k < a.row; k++) {
                sum += (a.dp[i * a.row + k] * b.dp[k * b.col + j]);
            }
            c.dp[i * a.row + j] = sum;
        }
    }
}

int main() {
    
    Matrix a(2,2);
    Matrix b(2,2);
    Matrix c(2,2);
    a.cuda_malloc();
    b.cuda_malloc();
    c.cuda_malloc();
    dim3 block_dim(30, 30);
    matmul << <1, block_dim >> > (a,b,c);
    c.cuda_copy_to_host();
    c.print();
    
}

