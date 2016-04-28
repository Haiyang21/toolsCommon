#include <assert.h>
#include "BufferManager.h"
#include "Log.h"

inline cudaChannelFormatDesc createChannelDesc(PixelType type) {
	cudaChannelFormatDesc channel_desc;
	switch (type) {
	case UCHAR: {
		channel_desc = cudaCreateChannelDesc<uchar1>();
		break;
	}	
	case INT: {
		channel_desc = cudaCreateChannelDesc<int1>();
		break;
	}
	case FLOAT: {
		channel_desc = cudaCreateChannelDesc<float1>();
		break;
	}
	case UCHAR4: {
		channel_desc = cudaCreateChannelDesc<uchar4>();
		break;
	}	
	case FLOAT4: {
		channel_desc = cudaCreateChannelDesc<float4>();
		break;
	}
	}
	return channel_desc;
}

inline size_t getSizeOf(PixelType type) {
	switch (type) {
	case UCHAR: {
		return sizeof(uchar1);
	}
	case SHORT: {
		return sizeof(short1);
	}
	case INT: {
		return sizeof(int1);
	}
	case FLOAT: {
		return sizeof(float1);
	}
	case UCHAR4: {
		return sizeof(uchar4);
	}
	case FLOAT4: {
		return sizeof(float4);
	}
	}

	return 0;
}

/*
 * \brief: Allocates a CUDA array according to the ::cudaChannelFormatDesc structure
 */
inline CU_STATUS createCudaArray(cudaArray_t* array,
	const cudaChannelFormatDesc* channel_desc,
	int width, int height) {
	cudaError_t error;
	error = cudaMallocArray(array, channel_desc, width, height,
		cudaArraySurfaceLoadStore);
	//cudaArraySurfaceLoadStore: Allocates an array that can be read from or written to using a surface reference

	if (error != cudaSuccess) {
		CULOG(CULOG_ERROR) << __FUNCTION__
			<< ": unable to allocate a cuda array with width=" << width
			<< ", height=" << height;
		return CU_FAILED;
	}
	return CU_SUCCESS;
}

inline CU_STATUS bindTexture(cudaTextureObject_t* tex_obj,
	const cudaArray_t array, TexReadMode read_mode,
	TexFilterMode filter_mode) {
	cudaError_t error;

	//! 资源描述符，用于获取纹理数据
	struct cudaResourceDesc res_desc;
	memset(&res_desc, 0, sizeof(res_desc));
	res_desc.resType = cudaResourceTypeArray;
	res_desc.res.array.array = array;

	//! 纹理描述符，用于描述纹理参数
	struct cudaTextureDesc tex_desc;
	memset(&tex_desc, 0, sizeof(tex_desc));
	tex_desc.addressMode[0] = cudaAddressModeMirror;
	tex_desc.addressMode[1] = cudaAddressModeMirror;
	if (filter_mode == FL_POINT)
		tex_desc.filterMode = cudaFilterModePoint;
	else
		tex_desc.filterMode = cudaFilterModeLinear;
	if (read_mode == RD_NORMALIZED_FLOAT)
		tex_desc.readMode = cudaReadModeNormalizedFloat;
	else
		tex_desc.readMode = cudaReadModeElementType;
	tex_desc.normalizedCoords = 0;

	//! 创建纹理对象
	error = cudaCreateTextureObject(tex_obj, &res_desc, &tex_desc, NULL);

	if (error != cudaSuccess) {
		CULOG(CULOG_ERROR) << __FUNCTION__
			<< ": unable to bind texture, error = " << error;
		return CU_FAILED;
	}

	return CU_SUCCESS;
}

inline CU_STATUS bindTexture(cudaTextureObject_t* tex_obj, void* ptr,
	size_t pitch, int width, int height, PixelType type,
	const cudaChannelFormatDesc& channel_desc,
	TexReadMode read_mode, TexFilterMode filter_mode) {
	cudaError_t error;

	struct cudaResourceDesc res_desc;
	memset(&res_desc, 0, sizeof(res_desc));
	res_desc.resType = cudaResourceTypePitch2D;
	res_desc.res.pitch2D.devPtr = ptr;
	res_desc.res.pitch2D.desc = channel_desc;
	res_desc.res.pitch2D.height = height;
	res_desc.res.pitch2D.width = width;// * GetSizeOf(type);
	res_desc.res.pitch2D.pitchInBytes = pitch;

	struct cudaTextureDesc tex_desc;
	memset(&tex_desc, 0, sizeof(tex_desc));
	tex_desc.addressMode[0] = cudaAddressModeClamp;
	tex_desc.addressMode[1] = cudaAddressModeClamp;
	if (filter_mode == FL_LINEAR)
		tex_desc.filterMode = cudaFilterModeLinear;
	else
		tex_desc.filterMode = cudaFilterModePoint;
	if (read_mode == RD_NORMALIZED_FLOAT)
		tex_desc.readMode = cudaReadModeNormalizedFloat;
	else
		tex_desc.readMode = cudaReadModeElementType;
	tex_desc.normalizedCoords = 0;

	error = cudaCreateTextureObject(tex_obj, &res_desc, &tex_desc, NULL);

	if (error != cudaSuccess) {
		CULOG(CULOG_ERROR) << __FUNCTION__
			<< ": unable to bind texture, error = " << cudaGetErrorString(error);
		return CU_FAILED;
	}

	return CU_SUCCESS;
}

inline CU_STATUS bindSurface(cudaSurfaceObject_t* surf_obj,
	const cudaArray_t array) {
	cudaError_t error;

	struct cudaResourceDesc res_desc;
	memset(&res_desc, 0, sizeof(res_desc));
	res_desc.resType = cudaResourceTypeArray;
	res_desc.res.array.array = array;

	error = cudaCreateSurfaceObject(surf_obj, &res_desc);

	if (error != cudaSuccess) {
		CULOG(CULOG_ERROR) << __FUNCTION__ << ": unable to bind surface";
		return CU_FAILED;
	}

	return CU_SUCCESS;
}

/*
 * \brief: 行多字节对齐，处理速度快，行与行之间一般都有间距
*/
inline CU_STATUS createPitch2D(void** ptr, size_t* pitch, int width, int height,
	PixelType type) {
	cudaError_t error;
	error = cudaMallocPitch(ptr, pitch, width * getSizeOf(type), height);

	if (error != cudaSuccess) {
		CULOG(CULOG_ERROR) << __FUNCTION__
			<< ": unable to allocate a pitch2D with width=" << width
			<< ", height=" << height;
		return CU_FAILED;
	}
	return CU_SUCCESS;
}

BufferManager::~BufferManager(){
	cudaError_t error;

	// 1. destroy texture object
	error = cudaDestroyTextureObject(cu_tex_obj_);
	assert(error == cudaSuccess);

	if (format_ == BLOCK_LINEAR) {
		// 2. destroy surface object
		error = cudaDestroySurfaceObject(cu_surf_obj_);
		assert(error == cudaSuccess);

		// 3. destroy cuda array
		error = cudaFreeArray(cu_array_);
		assert(error == cudaSuccess);
	}
	else {
		error = cudaFree(ptr_);
		assert(error == cudaSuccess);
	}
}

bool BufferManager::create(int width, int height, PixelType type, 
							MemoryFormat format, TexReadMode read_mode, 
							TexFilterMode filter_mode){
	width_ = width; height_ = height;
	type_ = type; format_ = format;
	cu_array_ = 0; ptr_ = 0;

	CU_STATUS status;
	// 1. create cuda channel desc
	cu_channel_desc_ = createChannelDesc(type);

	if (format == BLOCK_LINEAR) {
		// 2. create cuda array
		status = createCudaArray(&cu_array_, &cu_channel_desc_, width_, height_);
		assert(status == CU_SUCCESS);
		// 3. map to texture
		status = bindTexture(&cu_tex_obj_, cu_array_, read_mode, filter_mode);
		assert(status == CU_SUCCESS);
		// 4. map to surface
		status = bindSurface(&cu_surf_obj_, cu_array_);
		assert(status == CU_SUCCESS);
	}
	else {
		// 2. create pitch linear
		status = createPitch2D(reinterpret_cast<void**>(&ptr_), &pitch_, width_,
			height_, type);
		assert(status == CU_SUCCESS);
		// 3. map to texture
		status = bindTexture(&cu_tex_obj_, ptr_, pitch_, width_, height_, type_,
			cu_channel_desc_, read_mode, filter_mode);
		assert(status == CU_SUCCESS);
	}

	return true;
}

bool BufferManager::upload(const void* buf, size_t pitch, int x_offset,
	int y_offset, unsigned int width, unsigned int height) {

	cudaError_t error;
	if (format_ == BLOCK_LINEAR) {
		error = cudaMemcpy2DToArray(cu_array_, x_offset, y_offset, buf, pitch,
			width, height, cudaMemcpyHostToDevice);
	}
	else {
		error = cudaMemcpy2D(ptr_, pitch_, buf, pitch, width, height,
			cudaMemcpyHostToDevice);
	}

	if (error != cudaSuccess) {
		CULOG(CULOG_ERROR) << __FUNCTION__ << ": unable to upload data.";
		return CU_FAILED;
	}

	return true;
}

bool BufferManager::download(void* buf, size_t pitch, int x_offset,
	int y_offset, unsigned int width, unsigned int height) const {

	cudaError_t error;
	if (format_ == BLOCK_LINEAR) {
		error = cudaMemcpy2DFromArray(buf, pitch, cu_array_, x_offset, y_offset,
			width, height, cudaMemcpyDeviceToHost);
	}
	else {
		error = cudaMemcpy2D(buf, pitch, ptr_, pitch_, width, height,
			cudaMemcpyDeviceToHost);
	}

	if (error != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(error), __FILE__, __LINE__);
		CULOG(CULOG_ERROR) << __FUNCTION__ << ": unable to download data.";
		return CU_FAILED;
	}

	return true;
}