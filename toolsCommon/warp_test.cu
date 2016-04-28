#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include "functions.h"

__global__ void bcast(int arg) {
	int laneId = threadIdx.x & 0x1f; 
	int value; 
	if (laneId == 0) // Note unused variable for 
		value = arg; // all threads except lane 0 
	value = __shfl(value, 0); // Get "value" from lane 0 
	if (value != arg) 
		printf("Thread %d failed.\n", threadIdx.x); 
} 

__global__ void scan4() {
	int laneId = threadIdx.x & 0x1f; // Seed sample starting value (inverse of lane ID) 
	int value = laneId;// 31 - laneId; // Loop to accumulate scan within my partition. // Scan requires log2(n) == 3 steps for 8 threads // It works by an accumulated sum up the warp // by 1, 2, 4, 8 etc. steps. 
	for (int i=1; i<=4; i*=2) { 
		// Note: shfl requires all threads being 
		// accessed to be active. Therefore we do 
		// the __shfl unconditionally so that we 
		// can read even from threads which won't do a 
		// sum, and then conditionally assign the result. 
		int n = __shfl_up(value, i, 8); //value: laneId 持有的值; n: 获取(laneId-i)对应的value； i: 偏移
		printf("laneID %d value = %d n = %d\n", laneId, value, n);
		if (laneId >= i) 
			value += n; 
	} 
	printf("Thread %d final value = %d n = %d\n", threadIdx.x, value);
}

int warp_test() { 
	//bcast<<< 1, 32 >>>(1234); 
	scan4 << < 1, 32 >> >();
	cudaDeviceSynchronize(); 
	return 0; 
}