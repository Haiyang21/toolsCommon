#ifndef _MEMORY_H_
#define _MEMORY_H_

#include <cuda_runtime.h>

template<typename T>
__device__ __inline__ T read2D(cudaTextureObject_t ref, int x, int y){
	return tex2D<T>(ref, x, y);
}

template<typename T>
__device__ __inline__ T read2D(cudaTextureObject_t ref, float x, float y){
	return tex2D<T>(ref, x, y);
}

template<typename T>
__device__ __inline__ void write2D(cudaSurfaceObject_t ref,
	const T& val, int x, int y) {
	surf2Dwrite(val, ref, x * sizeof(T), y);
}

template<typename T>
__device__ __inline__ void write2D(void* out, const T& val, int x, int y, int width_stride) {
	int idx = y * width_stride + sizeof(T)* x;
	reinterpret_cast<T*>(reinterpret_cast<char*>(out)+idx)[0] = val;
}

#endif//#ifndef _MEMORY_H_