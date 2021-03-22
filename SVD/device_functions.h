#pragma once
#include<stdio.h>
#include <device_functions.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<iostream>
#include<math.h>
#include"Matrix.h"

__device__ double norm( Matrix mat, int col) {
    double ans = 0;
    for (int i = 0; i < mat.row; i++) {
        ans += (mat.dp[i*mat.col+col] * mat.dp[i*mat.col+col]); 
    }
    return sqrt(ans);
}
__device__ double dot_prod(Matrix a, Matrix e, int x, int y) {
    double ans = 0;
    for (int i = 0; i < a.row; i++) {
        ans += (a.dp[i*a.col+x] * e.dp[i*e.col+y]);
    }
    return ans;
}