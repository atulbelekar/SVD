#pragma once
#include<stdio.h>
#include <device_functions.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<iostream>
#include<math.h>
#include"Matrix.h"

__device__ double norm( double* mat, int col,int matrow, int matcol) {
    double ans = 0;
    for (int i = 0; i < matrow; i++) {
        ans += (mat[i*matcol+col] * mat[i*matcol+col]); 
    }
    return sqrt(ans);
}
__device__ double dot_prod(double* a, double* e, int x, int y,int arow, int acol, int ecol) {
    double ans = 0;
    for (int i = 0; i < arow; i++) {
        ans =ans+ (a[(i*acol)+x] * e[(i*ecol)+y]);
    }
    return ans;

}
double norm_host(Matrix &mat, int col) {
    double ans = 0;
    for (int i = 0; i < mat.row; i++) {
        ans += (mat.p[i * mat.col + col] * mat.p[i * mat.col + col]);
    }
    return sqrt(ans);
}
double dot_prod_host(Matrix &a, Matrix &e, int x, int y) {
    double ans = 0;
    for (int i = 0; i < a.row; i++) {
        ans = ans + (a.p[(i * a.col) + x] * e.p[(i * e.col) + y]);
    }
    return ans;

}