#pragma once
#include<iostream>
#include<stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
class Matrix
{
public:
	double* p;
	int row, col;
	double* dp;
	Matrix(int row, int col) {
		this->row = row;
		this->col = col;
		double* p = new double [(long)row*(long)col];
		for (int i = 0; i < row*col; i++) {
			p[i] = 5;
		}
		this->p = p;
	}
	double operator ()(int i,int j) {
		return p[i*row+j];
	}
	void print() {
		double* p = this->p;
		for (int i = 0; i < this->row; i++) {
			for (int j = 0; j < this->col; j++) {
				std::cout<<p[i * row + j]<<" ";
			}
			printf("\n");
		}
	}
	void cuda_malloc() {
		
		int status;
		cudaMalloc((void**)&(this->dp), sizeof(double) * row*col);
		cudaMemcpy(this->dp, this->p, sizeof(double)* col*row, cudaMemcpyHostToDevice);
		
	}
	void cuda_free() {
		cudaFree(this->dp);
	}
	void cuda_copy_to_host() {
		cudaMemcpy(this->p, this->dp, sizeof(double)*(this->row) * this->col, cudaMemcpyDeviceToHost);
	}
};

