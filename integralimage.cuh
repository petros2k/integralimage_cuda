#pragma once


#define __CUDA(F) do { cudaError_t e = (F); \
	if (e != cudaSuccess) { \
		fprintf( stderr, "CUDA Error: %d, %s\n", e, cudaGetErrorString(e) ); \
		fprintf( stderr, "in %s at %s, line %d\n",__FUNCTION__,__FILE__,__LINE__); \
		abort(); \
	} \
} while(0)


enum Format{ PGM, PPM, DEFAULT };

struct ImageStructure
{
	enum Format fmt;
	size_t w, h;
	size_t pix_maxval;
	unsigned char **img      = NULL;
	//unsigned char **d_img    = NULL;
	//unsigned int  **intimg   = NULL;
	//unsigned int  **d_intimg = NULL;
};
typedef struct ImageStructure Image;

// functions for manage PPM/PGM picture
bool loadImage( const char *filename, Image *img );
bool writeImage( const char *filename, Image img, int fmt );
bool ppmtopgm( Image *img );
void freeImage( Image *img );

// pixel value reduction
unsigned int reductionCPU( const Image img );
unsigned int reductionGPU( const Image img, int method=4 );

// integral image
bool integralImageCPU( const Image img, unsigned int ***intimg=NULL );
bool integralImageGPU( const Image img, unsigned int ***intimg=NULL, int method=0 );
void freeIntImg( unsigned int **intimg );

// GPU kernels
__global__ void reduction_naive( const unsigned char* img, size_t w, size_t h, size_t pitch, unsigned int *result );
__global__ void reduction_smem ( const unsigned char* img, size_t w, size_t h, size_t pitch, unsigned int *result );
__global__ void reduction_shfl1( const unsigned char* img, size_t w, size_t h, size_t pitch, unsigned int *result );
__global__ void reduction_shfl2( const unsigned char* img, size_t w, size_t h, size_t pitch, unsigned int *result );
__global__ void reduction_shfl3( const unsigned char* __restrict__ img, size_t w, size_t h, size_t pitch, unsigned int *result );
__global__ void reduction_shfl4( cudaTextureObject_t tex, unsigned int *result );
__global__ void reduction_shfl5( cudaTextureObject_t tex, unsigned int *result );

__global__ void integral_row_naive( const unsigned char* img, size_t w, size_t h, size_t ipitch, size_t opitch, unsigned int *intimg=NULL );
__global__ void integral_col_naive( const unsigned char* img, size_t w, size_t h, size_t ipitch, size_t opitch, unsigned int *intimg=NULL );

__global__ void integral_row_shfl ( const Image img, unsigned int **intimg=NULL, unsigned int **block_sum=NULL );
__global__ void integral_col_shfl ( const Image img, unsigned int **intimg=NULL, unsigned int **block_sum=NULL );
__global__ void integral_row_shfl_uniform( const Image img, unsigned int **block_sum=NULL );
__global__ void integral_row_shfl_apply( const Image img, unsigned int **intimg=NULL, unsigned int **block_sum=NULL );
__global__ void integral_col_shfl_uniform( const Image img, unsigned int **block_sum=NULL );
__global__ void integral_col_shfl_apply( const Image img, unsigned int **intimg=NULL, unsigned int **block_sum=NULL );

__global__ void integral_row_shfl2( const Image img, unsigned int **intimg=NULL, unsigned int **block_sum=NULL );
__global__ void integral_col_shfl_trans( const Image img, unsigned int **intimg=NULL, unsigned int **block_sum=NULL );
__global__ void integral_row_shfl_uniform2( const Image img, unsigned int **intimg=NULL, unsigned int **block_sum=NULL );
__global__ void integral_col_shfl_trans_uniform( const Image img, unsigned int **intimg=NULL, unsigned int **block_sum=NULL );

// kernel description
const char kernel_desc[][128] =
{
	"naive reduction",
	"reduction with smem",
	"reduction with simple warp-shuffle",
	"reduction with warp-shuffle and vec4",
	"reduction with warp-shuffle and vec16",
	"reduction with warp-shuffle and texture obj 4vec",
	"reduction with warp-shuffle and texture obj 16vec",
};

// GPU device functions
__device__ __forceinline__ unsigned int warp_shuffle_up( unsigned int val, int lane_id )
{
#pragma unroll
	for (int i=1; i<=warpSize; i*=2)
	{
		int num = __shfl_up( val, i, warpSize );
		if (lane_id >= i) val += num;
	}
	return val;
}

__device__ __forceinline__ unsigned int warp_shuffle_xor( unsigned int val, int lane_id )
{
	unsigned int _val0 = val;
	#pragma unroll
	for (int mask=warpSize/2; mask>0; mask >>= 1)
	{
			unsigned int _val1 = __shfl_xor( _val0, mask, warpSize );
			if (lane_id < mask) _val0 += _val1;
	}
	return _val0;
}

__device__ __forceinline__ unsigned int sum_uint4_uchar( uint4 data )
{
	unsigned char sum = 0;
	sum += (unsigned char)( (data.x >>  0) & 0xff );
	sum += (unsigned char)( (data.x >>  8) & 0xff );
	sum += (unsigned char)( (data.x >> 16) & 0xff );
	sum += (unsigned char)( (data.x >> 24) & 0xff );

	sum += (unsigned char)( (data.y >>  0) & 0xff );
	sum += (unsigned char)( (data.y >>  8) & 0xff );
	sum += (unsigned char)( (data.y >> 16) & 0xff );
	sum += (unsigned char)( (data.y >> 24) & 0xff );

	sum += (unsigned char)( (data.z >>  0) & 0xff );
	sum += (unsigned char)( (data.z >>  8) & 0xff );
	sum += (unsigned char)( (data.z >> 16) & 0xff );
	sum += (unsigned char)( (data.z >> 24) & 0xff );

	sum += (unsigned char)( (data.w >>  0) & 0xff );
	sum += (unsigned char)( (data.w >>  8) & 0xff );
	sum += (unsigned char)( (data.w >> 16) & 0xff );
	sum += (unsigned char)( (data.w >> 24) & 0xff );

	return sum;
}
