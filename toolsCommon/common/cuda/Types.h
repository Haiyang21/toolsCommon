#ifndef TYPES_H_
#define TYPES_H_

enum CU_STATUS {
	CU_SUCCESS, CU_FAILED
};

enum PixelType {
	UCHAR,
	SHORT,
	INT,
	FLOAT,
	UCHAR4,
	FLOAT4
};

enum TexReadMode {
	RD_NORMALIZED_FLOAT,
	RD_ELEMENT_TYPE
};

enum TexFilterMode {
	FL_POINT,
	FL_LINEAR
};

enum MemoryFormat {
	BLOCK_LINEAR,
	PITCH2D
};

#endif//#ifndef TYPES_H_