#include <device_functions.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include<iostream>
#include "Matrix.h"
#include "helper_functions.h"
#include<random>
using namespace std;
__global__ void matmul(Matrix a, Matrix b, Matrix c) {

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < a.row; i += (blockDim.x * gridDim.x)) {
        for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < b.col; j += (blockDim.y * gridDim.x)) {
            double sum = 0;
            for (int k = 0; k < b.row; k++) {
                sum += (a.dp[i * a.col + k] * b.dp[k * b.col + j]);
            }
            c.dp[i * c.col + j] = sum;
        }
    }
}

__global__ void set_zero(Matrix c) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < c.row; i += (blockDim.x * gridDim.x)) {
        for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < c.col; j += (blockDim.y * gridDim.x)) {
           
            c.dp[i * c.col + j] = 0;
        }
    }
}
__global__ void Gram_schmidt(Matrix a, Matrix ans) 
{
    for (int j = 0; j < a.col;j++) 
    {
        for (int l = 0; l < a.row; l++) 
        {
            ans.dp[l * ans.col + j] = a.dp[l * ans.col + j];
        }
        for (int k = 0; k < j; k++)
        {
            double dot = dot_prod(ans, a, k, j);
            double norm_ = dot_prod(ans,ans, k,k);
            for (int l = 0; l < ans.row; l++)
            {
                ans.dp[l * ans.col + j] = ans.dp[l * ans.col + j] - (dot / norm_) * ans.dp[l * ans.col + k];
            }
        }

    }
    for (int j = 0; j < a.col; j++) 
    {
        double norm_ = norm(ans, j);
        for (int k=0;k<a.row;k++)
        {
            ans.dp[k * ans.col + j] = ans.dp[k * ans.col + j] / norm_;
        }
        
    }
      
}

__global__ void R_matrix(Matrix a, Matrix Q, Matrix R) {

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < a.row; i += (blockDim.x * gridDim.x)) {
        for (int j =i+blockIdx.y * blockDim.y + threadIdx.y; j < a.col; j += (blockDim.y * gridDim.x)) {
            R.dp[i * (R.col) + j] = dot_prod(a, Q, j, i);
        }
    }
}

__global__ void sqrt_d(Matrix a) {
    for (int i = 0; i < a.row; i++)
    {
        a.dp[i * a.col + i] = sqrt(a.dp[i * a.col + i]);
    }
}

void Find_eigen_values(Matrix &A,Matrix &Eig, int n) {
    Matrix Q(A.row, A.col);
    Q.cuda_malloc();
    Matrix R(A.row, A.col);
    R.cuda_malloc();
    dim3 size(31, 31);
    Gram_schmidt << <1, 1 >> > (A, Q);
    Q.cuda_copy_to_host();
   
    R_matrix << <1, size >> > (A, Q, R);
    R.cuda_copy_to_host();
    
    Matrix temp(Eig.row,Eig.col);
    temp.cuda_malloc();
    matmul << <1, size >> > (Q, Eig, temp);
    Eig = temp;
    while (n--) {
        matmul << <1, size >> > (R, Q, A);
        set_zero << <1, size >> > (Q);
        Gram_schmidt << <1, 1 >> > (A, Q);
        
        R_matrix << <1, size >> > (A, Q, R);
        R.cuda_copy_to_host();
        
        matmul << <1, size >> > (Eig, Q, temp);
        Eig = temp;          
    }
    
}

Matrix InverseOfMatrix(Matrix A)
{
    Matrix I(A.row,A.col);
    I.make_identity();

    for (int i = 0; i < A.row-1; i++) {
        for (int j = i + 1; j < A.row; j++) {
            double temp=A.p[j*A.col+i]/A.p[i*A.col+i];
            for (int k = 0; k < A.row; k++) {
                A.p[j * A.row + k] = A.p[j * A.row + k] - A.p[i * A.row + k] * temp;
                I.p[j * A.row + k] = I.p[j * A.row + k] - I.p[i * A.row + k] * temp;
            }
        }
    }
    for (int i = A.row - 1; i >0; i--) {
        for (int j = i - 1; j >= 0; j--) {
            double temp = A.p[j * A.col + i] / A.p[i * A.col + i];
            for (int k = 0; k < A.row; k++) {
                A.p[j * A.row + k] = A.p[j * A.row + k] - A.p[i * A.row + k] * temp;
                I.p[j * A.row + k] = I.p[j * A.row + k] - I.p[i * A.row + k] * temp;
            }
        }
    }
    for (int i = 0; i < A.row; i++) {
        for (int j = 0; j < A.row; j++) {
            I.p[i * A.row + j] = I.p[i * A.row + j] / A.p[i * A.row + i];
        }
    }
    return I;
}

__global__ void transpose(Matrix a, Matrix b) {
    
    double* p = new double[a.row * a.col];
    
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < a.row; i += (blockDim.x * gridDim.x)) {
        for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < a.col; j += (blockDim.y * gridDim.x)) {
            b.dp[j * b.col + i] = a.dp[i * a.col + j];
        }  
    }   
}

void SVD(Matrix& A) {
    A.print();
    cout << endl;
    dim3 size(31, 31);
    Matrix At(A.col, A.row);
    At.cuda_malloc();
    transpose << <1, size >> > (A, At);
    Matrix AAt(A.row, At.col);
    AAt.cuda_malloc();
    matmul << <1, size >> > (A, At, AAt);
    AAt.cuda_copy_to_host();
    
    Matrix AtA(At.row, A.col);
    AtA.cuda_malloc();
    matmul << <1, size >> > (At, A, AtA);
    AtA.cuda_copy_to_host();
    
    Matrix U(AAt.row, AAt.col);
    U.make_identity();
    U.cuda_malloc();
    Find_eigen_values(AAt, U, 1000);
    U.cuda_copy_to_host();
    U.print();
    cout << endl;
    sqrt_d << <1, 1 >> > (AAt);
    AAt.cuda_copy_to_host();
    AAt.print();
    cout << endl;
    Matrix V(AtA.row, AtA.col);
    V.make_identity();
    V.cuda_malloc();
    Find_eigen_values(AtA, V,1000);
    V.cuda_copy_to_host();
    V.print();
}
int main() {
   
   /* Matrix A(3, 2);
    A.p[0] = 1;
    A.p[1] = 2;
    A.p[2] = 8;
    A.p[3] = 2;
    A.p[4] = 1;
    A.p[5] = 5;
    A.cuda_malloc();
    SVD(A);*/

    Matrix A(10, 5);
    //A.p[0] = 1;  // 5;
    //A.p[1] = 2;
    //A.p[2] = 3;
    //A.p[3] = 10;
    //A.p[4] = 2;
    //A.p[5] = 4;
    //A.p[6] = 8;
    //A.p[7] = 7;
    //A.p[8] = 3;
    //A.p[9] = 8;
    //A.p[10] = 5;
    //A.p[11] = 6;
    A.random_init(1, 10);
    A.cuda_malloc();
    SVD(A); 
   
}

