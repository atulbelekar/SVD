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
            ans.dp[k * (ans.col) + i] = (ans.dp[k * (ans.col) + i] - a.dp[k * (ans.col) + i]);
        }
        double norm_temp = norm(ans,i);
        for (int k = 0; k < a.row; k++) {
            ans.dp[k * (ans.col) + i] = ans.dp[k * (ans.col) +i] / norm_temp;
        }
    }
    
}

__global__ void R_matrix(Matrix a, Matrix Q, Matrix R) {

    for (int i = 0; i < a.row; i++) {
        for (int j = i; j < a.col; j++) {
            R.dp[i*(R.col)+j] = dot_prod(a, Q, j, i);
        }
    }
}


int main() {
   
    Matrix A(2, 2);
    A.p[0] = 7;
    A.p[1] = 3;
    A.p[2] = 3;
    A.p[3] = -1;
    Matrix B(2,2);    
    A.cuda_malloc();
    Matrix res(2, 2);
    B = A;
    res.cuda_malloc();
    dim3 size(10, 10);
    int n = 3;
    Matrix Q(2, 2);
    Q.cuda_malloc();
    Matrix R(2, 2);
    R.cuda_malloc();
    while (n--) {
        cout << "Q : \n";
        Q.cuda_copy_to_host();
        Q.print();
        cout << "R: \n";
        R.cuda_copy_to_host();
        R.print();
        cout << "A: \n";
        A.cuda_copy_to_host();
        A.print();
        cout << "****\n";
        Gram_schmidt << <1, size >> > (B, Q);
        R_matrix << <1, size >> > (B, Q, R);
        matmul << <1, size >> > (R, Q, B);
        
    }
    


}

