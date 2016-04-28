#ifndef BUFFERMANAGER_H_
#define BUFFERMANAGER_H_

#include <cuda_runtime.h>
#include "Types.h"

class BufferManager{
public:
	BufferManager(){};
	~BufferManager();

	// data transfer
	bool upload(const void* buf, size_t pitch, int x_offset, int y_offset,
		unsigned int width, unsigned int height);
	bool download(void* buf, size_t pitch, int x_offset, int y_offset,
		unsigned int width, unsigned int height) const;

	bool create(int width, int height, PixelType type, MemoryFormat format, 
		TexReadMode read_mode, TexFilterMode filter_mode = FL_POINT);

	unsigned int width() const { return width_; }
	unsigned int height() const { return height_; }
	size_t pitch() const { return pitch_; }

	void* ptr() { return reinterpret_cast<void*>(ptr_); }

	cudaArray* cu_array() { return cu_array_; }
	cudaTextureObject_t cu_tex_obj() const { return cu_tex_obj_; }
	cudaSurfaceObject_t cu_surf_obj() const { return cu_surf_obj_; }

private:
	unsigned int width_;
	unsigned int height_;
	unsigned char* ptr_;//pitch2D pointer
	size_t pitch_;
	PixelType type_;
	MemoryFormat format_;
	cudaArray* cu_array_;
	cudaChannelFormatDesc cu_channel_desc_;
	cudaTextureObject_t cu_tex_obj_;//便于作为参数传递纹理内存
	cudaSurfaceObject_t cu_surf_obj_;

};

#endif//#ifndef BUFFERMANAGER_H_