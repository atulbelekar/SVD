#pragma once
#include<iostream>
#include<stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<random>
#include<fstream>
#include<string>
#include<sstream>
using namespace std;
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
			p[i] = 0;
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
				printf("%.1f\t", p[i * col + j]);
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
	void transpose() {
		int row = this->col;
		int col = this->row;
		double* p;
		p = new double[row * col];
		for (int i = 0; i < row; i++) {
			for (int j = 0; j< col; j++) {
				p[i * col + j] = this->p[j * this->col + i];
			}
		}
		this->p = p;
		this->row = row;
		this->col = col;
	
	}
	void random_init(double low, double high) 
	{
		std::random_device rand;
		std::mt19937_64 generator(rand());
		std::uniform_real_distribution<double> distribution(low, high);
		for (int i = 0; i < this->row; i++) {
			for (int j = 0; j < this->col; j++) {
				this->p[i*(this->col) + j] = distribution(generator);
			}
		}
	}
	void set_zero()
	{
		for (int i = 0; i < this->row; i++) {
			for (int j = 0; j < this->col; j++) {
				this->p[i * (this->col) + j] = 0;
			}
		}
	}
	void make_identity() {
		for (int i = 0; i < this->row; i++) {
			for (int j = 0; j < this->col; j++) {
				if (i == j) {
					this->p[i * this->col + j] = 1;
				}
				else {
					this->p[i * this->col + j] = 0;
				}
			}

		}
	}
	void read_csv(string filename) {
		int i{ 0 };
		string line, word;
		ifstream file(filename, ios::in);
		//if (file.is_open()) { cout << "open" << endl; }

		while (!file.eof()) {
			getline(file, line);
			stringstream s(line);
			int j{ 0 };
			while (getline(s, word, ',')) {
				p[i * this->col + j] = stod(word);
				j++;
			}
			i++;
		}
	}

};

