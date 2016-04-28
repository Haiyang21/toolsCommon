#ifndef UTIL_CUH_
#define UTIL_CUH_

#include <cuda_runtime.h>

//Maps to a single instruction on G8x / G9x / G10x
#define IMAD(a, b, c) ( __mul24((a), (b)) + (c) )

#define DIVUP(X, Y) ((X) + (Y) - 1) / (Y)

#define ROUNDED_HALF(x) (((x)>>1)+((x)&0x01)) //x & 0x01 ÅÐ¶ÏÆæÅ¼Êý


#endif //#ifndef UTIL_CUH_