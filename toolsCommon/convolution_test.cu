#include <cuda_runtime.h>
#include <cv.h>
#include <highgui.h>
#include "functions.h"
#include "common/cuda/BufferManager.h"
#include "common/cuda/Types.h"
#include "common/cuda/Util.cuh"
#include "common/cuda/Memory.h"
#include "common/cuda/EventManagement.cuh"

template<typename T>
__device__ __inline__ float Convolution3x3(cudaTextureObject_t in,
	int ctr_t_x, int ctr_t_y) {
	/** Since Gaussian filter is symmetric -
	1 2 1
	2 4 2
	1 2 1
	l c r
	We can spilt it into 3 columns: left, center, right, respectively.
	As the convolution goes, we noticed that the result of r can be reused as l in the next convolution.
	Why? Because of the downsampling game we play:

	O O O O
	O X O X
	O O O O
	O X O X

	The above is 4x4 tile and X are the pixels we picked in the down sampling.
	Only the convoluted result of 'X' is important, we skipped all the 'O' pixels. Note that between 'X' they shares the
	same column of the pixels, just different side (right vs. left). */

	// now we have avoid expensive SHL, SHR
	int top_t_y = ctr_t_y - 1;
	int btm_t_y = ctr_t_y + 1;
	int lft_t_x = ctr_t_x - 1;
	int rgt_t_x = ctr_t_x + 1;

	float lft, ctr, rgt; // left center right (columns)
	lft = read2D<T>(in, lft_t_x, top_t_y)
		+ (read2D<T>(in, lft_t_x, ctr_t_y) * 2.0)
		+ read2D<T>(in, lft_t_x, btm_t_y);
	ctr = read2D<T>(in, ctr_t_x, top_t_y)
		+ (read2D<T>(in, ctr_t_x, ctr_t_y) * 2.0)
		+ read2D<T>(in, ctr_t_x, btm_t_y);
	rgt = read2D<T>(in, rgt_t_x, top_t_y)
		+ (read2D<T>(in, rgt_t_x, ctr_t_y) * 2.0)
		+ read2D<T>(in, rgt_t_x, btm_t_y);

	return (lft + (ctr * 2.0) + rgt) / 16.0;
}

template<typename T>
__global__ void DownSampleHalfScale3x3(cudaSurfaceObject_t out, cudaTextureObject_t in,
	int x_offset, int y_offset, int width, int height) {
	int x = x_offset + blockIdx.x * blockDim.x + threadIdx.x;
	int y = y_offset + blockIdx.y * blockDim.y + threadIdx.y;

	if (x < (width >> 1) && y < (height >> 1)) {
		float val = Convolution3x3<T>(in, (x << 1) + 1, (y << 1) + 1);
		T vout = (T)val;
		write2D<T>(out, val, x, y);
	}
}

void convolution_test(){
	std::string filename = "image.bmp";
	cv::Mat image = cv::imread(filename, 0);
	cv::Mat out(image.rows/2, image.cols/2, CV_8UC1);

	BufferManager d_in, d_out;
#ifdef PITCH2D_TEST
	d_in.create(image.cols, image.rows, UCHAR, PITCH2D, RD_ELEMENT_TYPE);
	d_out.create(image.cols, image.rows, UCHAR, PITCH2D, RD_ELEMENT_TYPE);
#else 
	d_in.create(image.cols, image.rows, UCHAR, BLOCK_LINEAR, RD_ELEMENT_TYPE);
	d_out.create(image.cols/2, image.rows/2, UCHAR, BLOCK_LINEAR, RD_ELEMENT_TYPE);
#endif
	d_in.upload(image.data, image.step, 0, 0, image.cols, image.rows);

	/*EventRecord time;
	time.addRecord("Start");*/

	//使用event计算时间
	float time_elapsed = 0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);    //创建Event
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);    //记录当前时间

	dim3 blocks(16, 8);
	dim3 grids(DIVUP(image.cols / 2, blocks.x), DIVUP(image.rows/2, blocks.y));
	DownSampleHalfScale3x3<uchar> << <grids, blocks >> >(d_out.cu_surf_obj(), d_in.cu_tex_obj(), 0, 0, image.cols, image.rows);

	cudaEventRecord(stop, 0);    //记录当前时间

	cudaEventSynchronize(start);    //Waits for an event to complete.
	cudaEventSynchronize(stop);    //Waits for an event to complete.Record之前的任务
	cudaEventElapsedTime(&time_elapsed, start, stop);    //计算时间差
	printf("执行时间：%f(ms)\n", time_elapsed);
	//time.addRecord("Stop");
	////time.print();
	//std::cout << "time costs: " << time.getCurrentTime() << std::endl;

	d_out.download(out.data, image.cols/2, 0, 0, image.cols/2, image.rows/2);

	cv::imshow("image", image);
	cv::imshow("out", out);
	cv::waitKey(0);
}