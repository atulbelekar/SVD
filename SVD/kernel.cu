#include <device_functions.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include<iostream>
#include "Matrix.h"
#include "device_functions.h"
#include<random>
using namespace std;


__global__ void matmul(Matrix a, Matrix b, Matrix c) {
    
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < a.row; i+= (blockDim.x * gridDim.x)) {
        for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < b.col; j+=(blockDim.x * gridDim.x)) {
            double sum = 0;
            for (int k = 0; k < a.row; k++) {
                sum += (a.dp[i * a.col + k] * b.dp[k * b.col + j]);
            }
            c.dp[i * c.col + j] = sum;
        }
    }
}

__global__ void Gram_schmidt(Matrix a, Matrix ans) {
    for (int i = 0; i < a.col; i++) {
        for (int j = i - 1; j >= 0; j--) {
            double dot = dot_prod(a, ans, i, j);
            for (int k = 0; k < a.row; k++) {
                ans.dp[k*(ans.col)+i] += dot* (ans.dp[k*(ans.col)+j]);
            }
        }
        for (int k = 0; k < a.row; k++) {
            ans.dp[k * (ans.col) + i] = (a.dp[k * (ans.col) + i] - ans.dp[k * (ans.col) + i]);
        }
        double norm_temp = norm(ans,i);
        for (int k = 0; k < a.row; k++) {
            ans.dp[k * (ans.col) + i] = ans.dp[k * (ans.col) +i] / norm_temp;
        }
    }
    
}

int main() {
    Matrix a(3,3);
    a.p[0] = 1;
    a.p[1] = 8;
    a.p[2] = 4;
    a.p[3]= 7;
    a.p[4] = 3;
    a.p[5]= 1;
    a.p[6]= 5;
    a.p[7]= 2;
    a.p[8]= 9;
    a.cuda_malloc();
    Matrix ans(3,3);
    ans.set_zero();
    a.cuda_malloc();
    ans.cuda_malloc();
    Gram_schmidt << <1, 1 >> > (a,ans);
    ans.cuda_copy_to_host();
    ans.print();
    
    
}

