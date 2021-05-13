#include <device_functions.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include<iostream>
#include "Matrix.h"
#include "helper_functions.h"
#include<random>
#include<chrono>
using namespace std;
using namespace std::chrono;

__global__ void matmul(double* a, double* b, double* c,int arow, int acol, int brow,int bcol) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < arow && j < bcol) {
            double sum = 0;
            for (int k = 0; k < brow; k++) {
                sum += (a[i * acol + k] * b[k * bcol + j]);
            }
            c[i * bcol + j] = sum;
        
    }
}

__global__ void set_zero(double* c, int crow, int ccol) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i<crow && j<ccol){
            c[i * ccol + j] = 0;
    }
}

//__global__ void Gram_schmidt(Matrix a, Matrix ans) 
//{
//    for (int i = 0; i < a.row; i++) {
//        for (int j = i - 1; j >= 0; j--) {
//            double dot = dot_prod(a, ans, i, j);
//            for (int k = 0; k < a.row; k++) {
//                ans.dp[k * (ans.col) + i] += dot * (ans.dp[k * (ans.col) + j]);
//            }
//        }
//        for (int k = 0; k < a.row; k++) {
//            ans.dp[k * (ans.col) + i] = (ans.dp[k * (ans.col) + i] - a.dp[k * (ans.col) + i]);
//        }
//        double norm_temp = norm(ans, i);
//        for (int k = 0; k < a.row; k++) {
//            ans.dp[k * (ans.col) + i] = ans.dp[k * (ans.col) + i] / norm_temp;
//        }
//    }
//      
//}
void gram_schmidt_host(Matrix &a, Matrix &ans) {
    
    for (int i = 0; i < a.row; i++) {
        for (int j = i - 1; j >= 0; j--) {
            double dot = dot_prod_host(a, ans, i, j);
            for (int k = 0; k < a.row; k++) {
                ans.p[k * (ans.col) + i] += dot * (ans.p[k * (ans.col) + j]);
            }
        }
        for (int k = 0; k < a.row; k++) {
            ans.p[k * (ans.col) + i] = (ans.p[k * (ans.col) + i] - a.p[k * (ans.col) + i]);
        }
        double norm_temp = norm_host(ans, i);
        for (int k = 0; k < a.row; k++) {
            ans.p[k * (ans.col) + i] = ans.p[k * (ans.col) + i] / norm_temp;
        }
    }

    /*Matrix v(a.row, a.col);
    for (int j = 0; j<a.col; j++) {
        for (int i = 0; i < a.row; i++) {
            v.p[i * a.col + j] = a.p[i * a.col + j];
        }
    }
    for (int j = 0; j<a.col; j++) {
        double norm_temp = norm_host(v, j);
        for (int i = 0; i < a.row; i++) {
            q.p[i * a.col + j] = v.p[i * a.col + j]/norm_temp;
        }
        for (int k = j + 1; k < a.col; k++) {
            double dot = dot_prod_host(q, v, j, k);
            for (int i = 0; i < a.row; i++) {
                v.p[i * a.col + k] = v.p[i * a.col + k]- q.p[i * a.col + j]*dot;
            }
        }
    }*/
}

__global__ void R_matrix(double* a, double* Q, double* R, int arow, int acol,int Qrow, int Qcol, int Rrow, int Rcol) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < arow && j < acol) {
            R[i * (Rcol) + j] = dot_prod(a, Q, j, i,arow, acol,Qcol);
        
    }
}

__global__ void copy(double* a, double* b, int arow, int acol, int brow, int bcol)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < arow && j < arow) {
            b[i * acol + j] = a[i * acol + j];
        
    }
}

__global__ void transpose(Matrix a, Matrix b) {

    double* p = new double[a.row * a.col];

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < a.row; i += (blockDim.x * gridDim.x)) {
        for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < a.col; j += (blockDim.y * gridDim.x)) {
            b.dp[j * b.col + i] = a.dp[i * a.col + j];
        }
    }
}

__global__ void nodiag_normalize(double* A, double* I, int n, int i) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < n && y < n)
        if (x == i && x != y) {
            I[x * n + y] /= A[i * n + i];
            A[x * n + y] /= A[i * n + i];
        }

}

__global__ void diag_normalize(double* A, double* I, int n, int i) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < n && y < n)
        if (x == y && x == i) {
            I[x * n + y] /= A[i * n + i];
            A[x * n + y] /= A[i * n + i];
        }

}

__global__ void gaussjordan(double* A, double* I, int n, int i)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < n && y < n) {
        if (x != i) {
            I[x * n + y] -= I[i * n + y] * A[x * n + i];
            if (y != i) {
                A[x * n + y] -= A[i * n + y] * A[x * n + i];
            }
        }
    }

}
__global__ void set_zero_(double* A, double* I, int n, int i) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < n && y < n) {
        if (x != i) {
            if (y == i) {
                A[x * n + y] = 0;
            }
        }
    }
}


void sqrt_d(Matrix &a) {
    for (int i = 0; i < a.row; i++)
    {
        for (int j = 0; j < a.col; j++) {
            if (i == j) {
                a.p[i * a.col + i] = sqrt(abs(a.p[i * a.col + i]));
            }
            else {
                a.p[i * a.col + j] = 0;
            }
        }
        
    }
}

void Inversehost(Matrix& s) {
    for (int i = 0; i < s.row; i++) {
        for (int j = 0; j < s.col; j++) {
            if (i == j) {
                if (s.p[i * s.col + j] != 0) {
                    s.p[i * s.col + j] = 1 / s.p[i * s.col + j];
                }
            }
        }
    }
}

void Find_eigen_values(Matrix &A,Matrix &Eig, int n) {
    
    dim3 size(32, 32);
    Matrix Q(A.row, A.col);
    Q.cuda_malloc();
    Matrix R(A.row, A.col);
    R.cuda_malloc();
    Matrix temp(A.row, A.col);
    temp.make_identity();
    temp.cuda_malloc();

    int Qrows = ceil(Q.row / 32) + 1;
    int Qcols = ceil(Q.col / 32) + 1;
    dim3 Qsize(Qrows, Qcols);

    dim3 tempSize((ceil(temp.row / 32) + 1), (ceil(temp.col / 32) + 1));

    while (n--)
    {
        
        dim3 blockSize(Qrows, Qcols);

        set_zero << <blockSize, size >> > (Q.dp,Q.row,Q.col);
        Q.cuda_copy_to_host();
        gram_schmidt_host(A, Q);
        Q.cuda_free();
        Q.cuda_malloc();
        matmul << <tempSize, size >> > (Eig.dp, Q.dp, temp.dp,Eig.row,Eig.col,Q.row,Q.col);
        copy << <tempSize, size >> > (temp.dp, Eig.dp,temp.row,temp.col,Eig.row,Eig.col);
        R_matrix << <Qsize, size >> > (A.dp, Q.dp, R.dp,A.row,A.col,Q.row,Q.col,R.row,R.col);
        matmul << <Qsize, size >> > (R.dp, Q.dp, A.dp, R.row,R.col,Q.row,Q.col);
        A.cuda_copy_to_host();
        
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

void left_singular_value(Matrix& A,string str)
{
    dim3 size(32, 32);
    dim3 ASize((ceil(A.row / 32) + 1), (ceil(A.col / 32) + 1));

    Matrix At(A.col, A.row);
    At.cuda_malloc();
    transpose << <ASize, size >> > (A, At);
    
    Matrix AAt(A.row, At.col);

    dim3 AAtSize((ceil(A.row / 32) + 1), (ceil(A.row / 32) + 1));
    AAt.cuda_malloc();
    matmul << <AAtSize, size >> > (A.dp, At.dp, AAt.dp, A.row,A.col,At.row,At.col);
    AAt.cuda_copy_to_host();
    
    Matrix Eigvec(AAt.row, AAt.col);
    Eigvec.make_identity();
    Eigvec.cuda_malloc();
    Find_eigen_values(AAt, Eigvec, 10);
    if (str == "True") {
        AAt.cuda_copy_to_host();
        sqrt_d(AAt);
        
       AAt.write_csv("S");
    }
    Eigvec.cuda_copy_to_host();
    Eigvec.write_csv("U");
    Eigvec.cuda_free();
    At.cuda_free();
    AAt.cuda_free();
}

void right_singular_value(Matrix& A,string str)
{
    dim3 size(32, 32);
    dim3 ASize((ceil(A.row / 32) + 1), (ceil(A.col / 32) + 1));
    dim3 AtASize((ceil(A.col / 32) + 1), (ceil(A.col / 32) + 1));

    Matrix At(A.col, A.row);
    At.cuda_malloc();
    transpose << <ASize, size >> > (A, At);
    Matrix AtA(At.row, A.col);
    AtA.cuda_malloc();
    matmul << <AtASize, size >> > (At.dp, A.dp, AtA.dp, At.row,At.col, A.row,A.col);
    AtA.cuda_copy_to_host();
    Matrix Eigvec(AtA.row, AtA.col);
    Eigvec.make_identity();
    Eigvec.cuda_malloc();
    Find_eigen_values(AtA, Eigvec, 10);
    if (str == "True") {
        AtA.cuda_copy_to_host();
        sqrt_d(AtA);
 
        AtA.write_csv("S");
    }
    Eigvec.cuda_copy_to_host();
    At.cuda_free();
    Eigvec.write_csv("V");
    Eigvec.cuda_free();
    AtA.cuda_free();
}

void SVD(Matrix& A) {
    if (A.row <= A.col) {
        left_singular_value(A,"True");
        right_singular_value(A,"False");
    }
    else
    {
        left_singular_value(A, "False");
        right_singular_value(A, "True");
    }
   
}

void InverseGPU(Matrix& A) {
    Matrix I(A.row, A.col);
    I.make_identity();
    I.cuda_malloc();
    int blocksize = 32;
    dim3 threadsPerBlock(blocksize, blocksize);

    dim3 numBlocks((A.row + blocksize - 1) / blocksize, (A.row + blocksize - 1) / blocksize);

    for (int i = 0; i < A.row; i++) {
        nodiag_normalize << <numBlocks, threadsPerBlock >> > (A.dp, I.dp, A.row, i);
        diag_normalize << <numBlocks, threadsPerBlock >> > (A.dp, I.dp, A.row, i);
        gaussjordan << <numBlocks, threadsPerBlock >> > (A.dp, I.dp, A.row, i);
        set_zero_ << <numBlocks, threadsPerBlock >> > (A.dp, I.dp, A.row, i);
    }
    I.cuda_copy_to_host();
}

void matmul_3(Matrix &Y, Matrix &V, Matrix &Si, Matrix &ans) {
    Matrix ans2(Y.row, V.col);
    dim3 size(32, 32);
    ans2.cuda_malloc();

    dim3 yv_size(ceil(Y.row / 32) + 1, ceil(V.col / 32) + 1);
    matmul<<< yv_size, size >>> (Y.dp, V.dp ,ans2.dp, Y.row, Y.col, V.row, V.col);
    dim3 ansSi_size(ceil(ans2.row / 32) + 1, ceil(Si.col / 32) + 1);
    matmul<<< ansSi_size, size >>> (ans2.dp, Si.dp, ans.dp, ans2.row,ans2.col,Si.row, Si.col);
    ans.cuda_copy_to_host();
    ans2.cuda_free();
    

}

void DMD(Matrix &X, Matrix &Y) {
    dim3 size(32, 32);
    
    X.cuda_malloc();
    Y.cuda_malloc();
    SVD(X);
    Matrix U(X.row, X.row);
    U.read_csv("U");
    Matrix S(X.row,X.col);
    S.read_csv("S");
    Matrix V(X.col, X.col);
    V.read_csv("V");
    Inversehost(S);
    Matrix Si(S.col, S.row);
    Si.cuda_malloc();
    S.cuda_malloc();

    dim3 sSize(ceil(S.row / 32) + 1, ceil(S.row / 32) + 1);
    transpose << <sSize, size >> > (S, Si);
    Si.cuda_copy_to_host();
    
    U.cuda_malloc();
    Matrix Ut(U.col, U.row);
    Ut.cuda_malloc();

    dim3 uSize(ceil(U.row / 32) + 1, ceil(U.row / 32) + 1);
    transpose << <uSize, size >> > (U, Ut);
    Ut.cuda_copy_to_host();
    
    Matrix ans(U.col, U.row);
    ans.cuda_malloc();
    V.cuda_malloc();
    matmul_3(Y, V, Si, ans);
    ans.cuda_copy_to_host();
    //ans.print();
    Matrix ansf(U.col, U.row);
    ansf.cuda_malloc();
    //dim3 size(32, 32);

    dim3 size_mul(ceil(Ut.row / 32) + 1, ceil(ans.col / 32) + 1);
    matmul<<< size_mul, size >>> (Ut.dp,ans.dp,ansf.dp, Ut.row, Ut.col, ans.row, ans.col);
    ansf.cuda_copy_to_host();
    ansf.write_csv("Atilde");
    ansf.cuda_free();
    //////////////////

}



int main() {
   

    Matrix a(1000, 30);
    a.random_init(-10, 10);

    Matrix b(1000, 30);
    b.random_init(1, 5);

    auto start = high_resolution_clock::now();
    DMD(a, b);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    cout << "Matrix Multiplication time is : " << duration.count() << " milliseconds" << endl;
}

