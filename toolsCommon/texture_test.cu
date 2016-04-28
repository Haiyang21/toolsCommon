#include <cuda_runtime.h>
#include <cv.h>
#include <highgui.h>
#include "functions.h"
#include "common/cuda/BufferManager.h"
#include "common/cuda/Types.h"
#include "common/cuda/Util.cuh"
#include "common/cuda/Memory.h"

#define PITCH2D_TEST

/**
 * \brief: 通过pitch2D 写数据
*/
__global__ void transparentKernel(void* out, cudaTextureObject_t in, int width, int height){
	int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);

	if (ix >= width || iy >= height)
		return;
	uchar val = read2D<uchar>(in, ix, iy);
	val = (uchar)(val * 0.5);
	write2D<uchar>(out, val, ix, iy, width);
}

/**
 * \brief: 通过cudaSurfaceObject_t 写数据
*/
__global__ void transparentKernel(cudaSurfaceObject_t out, cudaTextureObject_t in, int width, int height){
	int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
	int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);

	if (ix >= width || iy >= height)
		return;
	uchar val = read2D<uchar>(in, ix, iy);
	val = (uchar)(val * 0.5);
	write2D<uchar>(out, val, ix, iy);
}

void transparent(BufferManager &out, BufferManager &in){
	int width = in.width();
	int height = in.height();
	dim3 blocks(16, 8);
	dim3 grids(DIVUP(width, blocks.x), DIVUP(height, blocks.y));
#ifdef PITCH2D_TEST
	transparentKernel << <grids, blocks >> >(out.ptr(), in.cu_tex_obj(), width, height);
#else
	transparentKernel << <grids, blocks >> >(out.cu_surf_obj(), in.cu_tex_obj(), width, height);
#endif
}

void texture_test(){
	std::string filename = "image.bmp";
	cv::Mat image = cv::imread(filename, 0);
	cv::Mat out(image.rows, image.cols, CV_8UC1);

	BufferManager d_in, d_out;
#ifdef PITCH2D_TEST
	d_in.create(image.cols, image.rows, UCHAR, PITCH2D, RD_ELEMENT_TYPE);
	d_out.create(image.cols, image.rows, UCHAR, PITCH2D, RD_ELEMENT_TYPE);
#else 
	d_in.create(image.cols, image.rows, UCHAR, BLOCK_LINEAR, RD_ELEMENT_TYPE);
	d_out.create(image.cols, image.rows, UCHAR, BLOCK_LINEAR, RD_ELEMENT_TYPE);
#endif
	d_in.upload(image.data, image.step, 0, 0, image.cols, image.rows);
	transparent(d_out, d_in);

	d_out.download(out.data, image.step, 0, 0, image.cols, image.rows);

	cv::imshow("image", image);
	cv::imshow("out", out);
	cv::waitKey(0);
}