#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <stdio.h>
#include "functions.h"
#include "common/cuda/EventManagement.cuh"
#include "common/cuda/Util.cuh"

__global__ void addKernel(int *dst, int val, int width, int height){
	int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);

	if (ix >= width || iy >= height)
		return;

	int idx = iy * width + ix;
	dst[idx] += val;
}

void test(){
	int width = 1024;
	int height = 1024;
	int *data = (int*)malloc(width*height*sizeof(int));
	memset(data, 0, width*height*sizeof(int));

	int *d_data;
	cudaMalloc(&d_data, width*height*sizeof(int));
	cudaMemset(d_data, 0, width*height*sizeof(int));

	EventRecord time;
	time.addRecord("Start");

	dim3 blocks(16, 8);
	dim3 grids(DIVUP(width, blocks.x), DIVUP(height, blocks.y));
	addKernel << <grids, blocks >> >(d_data, 2, width, height);

	time.addRecord("Stop");
	//time.print();
	std::cout << "time costs: " << time.getCurrentTime() << std::endl;

	cudaMemcpy(data, d_data, width*height*sizeof(int), cudaMemcpyDeviceToHost);

}
