#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <windows.h>

#define _REDUCTION
//#define _REDUCTION_TEST
#define _INTEGRAL
//#define _INTEGRAL_TEST

#define N_THREADS 64
#define TILE_DIM 32

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
	size_t x,y;
	size_t maxval;
	unsigned char *img = NULL;
	unsigned char *d_img = NULL;
};
typedef struct ImageStructure Image;

struct IntegralImageStructure
{
	size_t x,y;
	unsigned int *img = NULL;
	unsigned int *d_img = NULL;
};
typedef struct IntegralImageStructure IntegralImage;

int loadImage(const char *filename, Image *img)
{
	FILE *fp = NULL;
	char buf[16];
	cudaError_t err;

	// initialize
	img->fmt = Format::DEFAULT; img->x = 0; img->y = 0; img->maxval = 0; img->img = NULL;

	if (!(fp = fopen(filename, "rb")))
	{
		fprintf( stderr, "cannot open file \"%s\".\n", filename );
		return -1;
	}

	if (!fgets(buf, sizeof(buf), fp))
	{
		fprintf( stderr, "failed to load file.\n" );
		return -1;
	}

	if (buf[0] != 'P')
	{
		fprintf( stderr, "invalid image format.\n" );
		return -1;
	}
	switch (buf[1])
	{
		case '5':
			img->fmt = Format::PGM;
			break;

		case '6':
			img->fmt = Format::PPM;
			break;

		default:
			fprintf( stderr, "invalid image format.\n" );
			return -1;
	}

	fscanf(fp, "%lu %lu", &(img->x), &(img->y));
	fscanf(fp, "%d", &(img->maxval));

	while (fgetc(fp) != '\n') ;

	switch (img->fmt)
	{
		case Format::PGM:
			if ((img->img = (unsigned char*)malloc( img->x * img->y )) == NULL)
			{
				fprintf( stderr, "fail to allocate memory while loading image\n" );
				return -1;
			}
			fread( img->img, img->x, img->y, fp);
			break;

		case Format::PPM:
			if ((img->img = (unsigned char*)malloc(3 * img->x * img->y)) == NULL)
			{
				fprintf( stderr, "fail to allocate memory while loading image:\n" );
				return -1;
			}
			fread(img->img, 3 * img->x, img->y, fp);
			break;

		default:
			fprintf( stderr, "invalid image format.\n" );
			return -1;
	}

	fclose(fp);

	return 0;
}


int writeImage(const Image img, const char *filename, int fmt)
{
	FILE *fp = NULL;
	
	if (!(fp = fopen( filename, "wb" )))
	{
		fprintf( stderr, "cannot open file %s.\n", filename );
		return -1;
	}

	switch (fmt)
	{
		case Format::PGM:
			fprintf( fp, "P5\n%d %d\n%d\n", img.x, img.y, img.maxval );
			fwrite( img.img, img.x, img.y, fp );
			break;

		case Format::PPM:
			fprintf( fp, "P6\n%d %d\n%d\n", img.x, img.y, img.maxval );
			fwrite( img.img, 3 * img.x, img.y, fp );
			break;

		default:
			break;
	}

	fclose(fp);

	return 0;
}


int ppmtopgm(Image *img)
{
	unsigned char *pgm;

	if (img->fmt == Format::PGM)  return 0;

	if ((pgm = (unsigned char*)malloc(img->x * img->y)) == NULL)
	{
		fprintf( stderr, "fail to allocate memory\n" );
		return -1;
	}

	for (int i = 0; i < img->y; ++i)
	{
		for (int j = 0, jo = 0; j < img->x; ++j, jo+=3)
		{
			double px;
			px = img->img[ i * (img->x * 3) + jo + 0] * .299f
			   + img->img[ i * (img->x * 3) + jo + 1] * .587f
			   + img->img[ i * (img->x * 3) + jo + 2] * .114f;
			pgm[ i * img->x + j ] = (unsigned char)fmod(px, img->maxval + 1.);
		}
	}

	img->fmt = Format::PGM;

	free(img->img);
	img->img = pgm;

	return 0;
}


cudaError_t freeImage(Image *img)
{
	cudaError_t e = cudaSuccess;
	if (img->img != NULL)
	{
		free(img->img);
		img->img = NULL;
	}
	if (img->d_img != NULL)
	{
		e = cudaFree(img->d_img);
		img->d_img = NULL;
	}
	return e;
}

cudaError_t freeIntegralImage(IntegralImage *img)
{
	cudaError_t e = cudaSuccess;
	if (img->img != NULL)
	{
		free(img->img);
		img->img = NULL;
	}
	if (img->d_img != NULL)
	{
		e = cudaFree(img->d_img);
		img->d_img = NULL;
	}
	return e;
}


unsigned int reduction_cpu(const Image img)
{
	unsigned int result = 0;
	for (int i = 0; i < img.y; ++i)
	{
		for (int j = 0; j < img.x; ++j)
		{
			result += img.img[i * img.x + j];
		}
	}

	return result;
}


int integralImageCPU(const Image img, IntegralImage *integral)
{
	if (integral == NULL) return -1;

	integral->x = img.x; integral->y = img.y;
	if (integral->img != NULL)
	{
		freeIntegralImage(integral);
		integral->img = NULL;
	}
	size_t size = integral->x * integral->y * sizeof(unsigned int);
	unsigned int *p_intg = NULL;
	if ((p_intg = (unsigned int*)malloc(size)) == NULL)
	{
		fprintf( stderr, "fail to allocate memory\n" );
		return -1;
	}
	integral->img = p_intg;

	for (int i=0; i < integral->y; ++i)
	{
		for (int j=0; j < integral->x; ++j)
		{
			int u  = (i - 1) * integral->x + j;
			int ul = u - 1;
			int c  = i * integral->x + j;
			int l  = c - 1;
			unsigned int uv, ulv, lv, cv;
			(u < 0)  ? (uv = 0) : (uv = integral->img[u]);
			(j == 0) ? (lv = 0) : (lv = integral->img[l]);
			((ul < 0) || (j == 0)) ? (ulv = 0) : (ulv = integral->img[ul]);
			cv = img.img[c];

			integral->img[c] = uv + lv + cv - ulv;
		}
	}

	return 0;
}


__global__ void reductionGPUsmem(const Image img, unsigned int *result)
{
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockIdx.x * blockDim.x + tid;
	unsigned int size = img.x * img.y;
	extern __shared__ unsigned int smem[];

	if (idx == 0)  *result = 0;

	smem[tid] = 0;
	if (idx < size)
	{
		smem[tid] = img.d_img[idx];
	}
	__syncthreads();

	for (int mask = blockDim.x/2; mask > 0; mask >>= 1)
	{
		if (tid < mask && tid < size)  smem[tid] += smem[tid ^ mask];
		__syncthreads();
	}
	if (tid == 0)  atomicAdd( result, smem[0] );
}


__global__ void reductionGPUshfl(const Image img, unsigned int *result)
{
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockIdx.x * blockDim.x + tid;
	unsigned int num = 0;
	unsigned int size = img.x * img.y;

	if (idx == 0) *result = 0;

	if (idx < size) num = img.d_img[idx];

	#pragma unroll
	for (int mask = warpSize/2; mask > 0; mask >>= 1)
	{
		num += __shfl_xor( num, mask, warpSize );
	}
	if (!(tid & (warpSize-1))) atomicAdd( result, num );
}


__global__ void reductionGPUshfl2(const Image img, unsigned int *result)
{
	unsigned int tid = threadIdx.x;
	unsigned int bid = blockIdx.x;
	unsigned int bdm = blockDim.x;
	unsigned int gdm = gridDim.x;
	unsigned int idx = bid * (bdm * 4) + tid;
	unsigned int num = 0;
	unsigned int sum = 0;
	unsigned int size = img.x * img.y;
	unsigned int total_blocks = (size + N_THREADS - 1) / N_THREADS;

	if (idx == 0) *result = 0;

	for (int i=0; (i*gdm*4+bid) < total_blocks; ++i)
	{
		num = 0;
		if (idx + N_THREADS * 3 < size)
		{
			num = img.d_img[idx]
				+ img.d_img[idx + N_THREADS];
				+ img.d_img[idx + N_THREADS * 2];
				+ img.d_img[idx + N_THREADS * 3];
		}
		else if (idx + N_THREADS * 2 < size)
		{
			num = img.d_img[idx]
				+ img.d_img[idx + N_THREADS];
				+ img.d_img[idx + N_THREADS * 2];
		}
		else if (idx + N_THREADS < size )
		{
			num = img.d_img[idx]
				+ img.d_img[idx + N_THREADS];
		}
		else if (idx < size)
		{
			num = img.d_img[idx];
		}

		#pragma unroll
		for (int mask = warpSize/2; mask > 0; mask >>= 1)
		{
			num += __shfl_xor( num, mask, warpSize );
		}
		if (!(tid & (warpSize-1)))
		{
			sum += num;
		}
		idx = idx + gdm * bdm * 4;
	}
	if (!(tid & (warpSize - 1))) atomicAdd(result, sum);
}


texture< unsigned char, 1, cudaReadModeElementType > teximg;
__global__ void integralGPUhorz(IntegralImage integral)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < integral.y)
	{
		unsigned int num = 0;
		for (int i=0; i < integral.x; ++i)
		{
			num += tex1Dfetch( teximg, idx * integral.x + i);
			integral.d_img[idx * integral.x + i] = num;
		}
		num = 0;
	}
}

__global__ void integralGPUvert(IntegralImage integral)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < integral.x)
	{
		unsigned int num = integral.d_img[idx];
		for (int i=1; i < integral.y; ++i)
		{
			num += integral.d_img[i * integral.x + idx];
			integral.d_img[i * integral.x + idx] = num;
		}
	}
}

int integralImageGPU(const Image img, IntegralImage *integral)
{
	if (integral == NULL) return -1;

	if (integral->img != NULL)
	{
		freeIntegralImage(integral);
		integral->img = NULL;
	}
	integral->x = img.x; integral->y = img.y;
	size_t size = integral->x * integral->y * sizeof(unsigned int);
	unsigned int *p_intg = NULL;
	unsigned int *dp_intg = NULL;
	if ((p_intg = (unsigned int*)malloc(size)) == NULL)
	{
		fprintf( stderr, "fail to allocate memory\n" );
		return -1;
	}
	if (cudaError_t err = cudaMalloc( &dp_intg, size ))
	{
		fprintf( stderr, "cuda memory allocation error: %s\n", cudaGetErrorString(err) );
		return err;
	}
	__CUDA( cudaMemset( dp_intg, 0, size ) );
	integral->img = p_intg;
	integral->d_img = dp_intg;

	size_t img_size = img.x * img.y * sizeof(unsigned char);
	__CUDA( cudaBindTexture( 0, teximg, img.d_img, img_size ) );

	int gd = (integral->y + N_THREADS - 1) / N_THREADS;
	integralGPUhorz<<< gd, N_THREADS >>>( *integral );
#ifdef _DEBUG	
	cudaDeviceSynchronize();
#endif
	gd = (integral->x + N_THREADS - 1) / N_THREADS;
	integralGPUvert<<< gd, N_THREADS >>>( *integral );
#ifdef _DEBUG	
	cudaDeviceSynchronize();
#endif

	__CUDA( cudaUnbindTexture( teximg ) );
	__CUDA( cudaMemcpy( integral->img, integral->d_img, size, cudaMemcpyDeviceToHost ) );

	return cudaSuccess;
}


__global__ void integralGPUhorz_shfl(const Image img, IntegralImage integral, unsigned int *block_sum=NULL)
{
	__shared__ unsigned int sum[N_THREADS];

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int x_bid = blockIdx.x % ((integral.x + blockDim.x - 1) / blockDim.x);
	int y_idx = blockIdx.x / ((integral.x + blockDim.x - 1) / blockDim.x);
	int x_idx = x_bid * blockDim.x + threadIdx.x;
//	int y_idx = y_bid;
	int img_idx = y_idx * integral.x + x_idx;
	int lane_id = (blockIdx.x * blockDim.x + threadIdx.x) % warpSize;
	int warp_id = threadIdx.x / warpSize;
	int intx = integral.x, inty = integral.y;

	unsigned int val = 0;
	
	if (idx < img.x*img.y)
	val = img.d_img[idx];
	if ((x_idx < intx) && (y_idx < inty))
	{
		if (block_sum != NULL)
			val = (unsigned int)img.d_img[img_idx];
		else
			val = integral.d_img[img_idx];
	}
	sum[threadIdx.x] = 0;
	__syncthreads();

#pragma unroll
	for (int i=1; i <= warpSize; i*=2)
	{
		int num = __shfl_up( val, i, warpSize );
		if (lane_id >= i) val += num;
	}

	if (threadIdx.x % warpSize == warpSize - 1)
	{
		sum[warp_id] = val;
	}
	__syncthreads();

	if (warp_id == 0)
	{
		int warp_val = sum[lane_id];

#pragma unroll
		for (int i=1; i <= warpSize; i*=2)
		{
			int num = __shfl_up(warp_val, i, warpSize);
			if (lane_id >= i) warp_val += num;
		}
		sum[lane_id] = warp_val;
	}
	__syncthreads();

	int thread_val = 0;
	if (warp_id > 0)
	{
		thread_val = sum[warp_id - 1];
	}
	
	val += thread_val;

	if ((x_idx < intx) && (y_idx < inty))
	{
 		integral.d_img[img_idx] = val;
	}

	if (block_sum != NULL && threadIdx.x == blockDim.x - 1)
	{
 		block_sum[blockIdx.x] = val;
	}
}

__global__ void horizontalBlockAdd(IntegralImage integral, unsigned int *block_sum)
{
	__shared__ unsigned int sum;
	__shared__ unsigned int smem[N_THREADS];
	int x_bdim = (integral.x + blockDim.x - 1) / blockDim.x;
	int x_bid = blockIdx.x % x_bdim;
	int y_bid = blockIdx.x / x_bdim;
	int x_idx = x_bid * blockDim.x + threadIdx.x;
	int y_idx = y_bid;
	int img_idx = y_idx * integral.x + x_idx;
	int head = x_bdim * y_bid;
	int tail = x_bdim * (y_bid + 1) - 1;

	smem[threadIdx.x] = 0;
	if (threadIdx.x < x_bdim)
	{
		smem[threadIdx.x] = block_sum[head + threadIdx.x];
	}
	__syncthreads();

	

	if ((x_bid == 0) || (x_idx >= integral.x) || (y_idx >= integral.y)) return;

	if (threadIdx.x == 0)
	{
		sum = block_sum[blockIdx.x - 1];
	}
	__syncthreads();

	integral.d_img[img_idx] += sum;
}


__global__ void integralGPUhorz_shfl3(const Image img, IntegralImage integral, unsigned int *block_sum = NULL)
{
	__shared__ unsigned int sums[N_THREADS];

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int x_bid = blockIdx.x % (((integral.x / 16) + blockDim.x - 1) / blockDim.x);  // block index in x-axis
	int y_idx = blockIdx.x / (((integral.x / 16) + blockDim.x - 1) / blockDim.x);  // and y
	int x_idx = x_bid * blockDim.x + threadIdx.x;
	int x_idx4 = x_bid * (blockDim.x * 4) + (threadIdx.x * 4);
//	int y_idx = y_bid;
	int intx = integral.x / 16; int inty = integral.y;
	int intx4 = integral.x / 4;
	int img_idx = y_idx * intx + x_idx;
	int img_idx4 = y_idx * intx4 + x_idx4;
	int lane_id = idx % warpSize;
	int warp_id = threadIdx.x / warpSize;

	unsigned int val[16];
	val[0]  = val[1]  = val[2]  = val[3]  = 0;
	val[4]  = val[5]  = val[6]  = val[7]  = 0;
	val[8]  = val[9]  = val[10] = val[11] = 0;
	val[12] = val[13] = val[14] = val[15] = 0;

	if ((x_idx < intx) && (y_idx < inty))
	{
		uint4 data = ((uint4*)img.d_img)[img_idx];
		uint4 mask;	
		mask.x = 0xff; mask.y = 0xff00; mask.z = 0xff0000; mask.w = 0xff000000;
		val[0]  = (data.x & mask.x) >> 0;
		val[1]  = (data.x & mask.y) >> 8;
		val[2]  = (data.x & mask.z) >> 16;
		val[3]  = (data.x & mask.w) >> 24;
		val[4]  = (data.y & mask.x) >> 0;
		val[5]  = (data.y & mask.y) >> 8;
		val[6]  = (data.y & mask.z) >> 16;
		val[7]  = (data.y & mask.w) >> 24;
		val[8]  = (data.z & mask.x) >> 0;
		val[9]  = (data.z & mask.y) >> 8;
		val[10] = (data.z & mask.z) >> 16;
		val[11] = (data.z & mask.w) >> 24;
		val[12] = (data.w & mask.x) >> 0;
		val[13] = (data.w & mask.y) >> 8;
		val[14] = (data.w & mask.z) >> 16;
		val[15] = (data.w & mask.w) >> 24;
	}

#pragma unroll
	for (int i=0; i<15; ++i)
	{
		val[i+1] += val[i];
	}

	unsigned int sum = val[15];

	sums[threadIdx.x] = 0;
	__syncthreads();

#pragma unroll
	for (int i=1; i <= warpSize; i *= 2)
	{
		int num = __shfl_up(sum, i, warpSize);

		if (lane_id >= i)
		{
#pragma unroll
			for (int i=0; i < 16; ++i)
			{
				val[i] += num;
			}
			sum += num;
		}
	}

	if (threadIdx.x % warpSize == warpSize - 1)
	{
		sums[warp_id] = val[15];
	}
	__syncthreads();

	if (warp_id == 0)
	{
		int warp_val = sums[lane_id];

#pragma unroll
		for (int i = 1; i <= warpSize; i *= 2)
		{
			int num = __shfl_up(warp_val, i, warpSize);

			if (lane_id >= i) warp_val += num;
		}
		sums[lane_id] = warp_val;
	}
	__syncthreads();

	int thread_val = 0;
	if (warp_id > 0)
	{
		thread_val = sums[warp_id - 1];

	#pragma unroll
		for (int i=0; i < 16; ++i)
		{
			val[i] += thread_val;
		}
	}

	if ((x_idx4 < intx4) && (y_idx < inty))
	{
		int id = 0;
#pragma unroll
		for ( int i=0; i<4; ++i)
		{
			uint4 data;
			data.x = val[id++];
			data.y = val[id++];
			data.z = val[id++];
			data.w = val[id++];
			((uint4*)integral.d_img)[img_idx4 + i] = data;
		}
		//memcpy( (integral.d_img + img_idx4), val, 16 * sizeof(unsigned int) );
	}

	if (block_sum != NULL && threadIdx.x == blockDim.x - 1)
	{
		block_sum[blockIdx.x] = val[15];
	}
}


__global__ void integralGPUhorz_shfl2(const Image img, IntegralImage integral, unsigned int *block_sum = NULL)
{
	__shared__ unsigned int sums[N_THREADS];

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int x_bid = blockIdx.x % (((integral.x / 4) + blockDim.x - 1) / blockDim.x);  // block index in x-axis
	int y_idx = blockIdx.x / (((integral.x / 4) + blockDim.x - 1) / blockDim.x);  // and y
	int x_idx = x_bid * blockDim.x + threadIdx.x;
//	int y_idx = y_bid;
	int intx = integral.x / 4; int inty = integral.y;
	int img_idx = y_idx * intx + x_idx;  // image index for uint4 or uchar4 access
	int lane_id = idx % warpSize;
	int warp_id = threadIdx.x / warpSize;

	unsigned int val[4];
	val[0] = val[1] = val[2] = val[3] = 0;
	uint4 mask;
	mask.x = 0xff; mask.y = 0xff00; mask.z = 0xff0000; mask.w = 0xff000000;

	if ((x_idx < intx) && (y_idx < inty))
	{
		//if (block_sum != NULL)
		//{
		//uchar4 data = ((uchar4*)img.d_img)[img_idx];
		//val[0] = data.x;
		//val[1] = data.y;
		//val[2] = data.z;
		//val[3] = data.w;
		unsigned int data = ((unsigned int*)img.d_img)[img_idx];
		val[0] = (data & mask.x) >> 0;
		val[1] = (data & mask.y) >> 8;
		val[2] = (data & mask.z) >> 16;
		val[3] = (data & mask.w) >> 24;
		//}
		//else
		//{
		//	uint4 data = ((uint4*)integral.d_img)[img_idx];
		//	val[0] = data.x;
		//	val[1] = data.y;
		//	val[2] = data.z;
		//	val[3] = data.w;
		//}
	}

	val[1] += val[0];
	val[2] += val[1];
	val[3] += val[2];

	unsigned int sum = val[3];

	sums[threadIdx.x] = 0;
	__syncthreads();

#pragma unroll
	for (int i=1; i <= warpSize; i *= 2)
	{
		int num = __shfl_up(sum, i, warpSize);

		if (lane_id >= i)
		{
#pragma unroll
			for (int i=0; i < 4; ++i)
			{
				val[i] += num;
			}
			sum += num;
		}
	}

	if (threadIdx.x % warpSize == warpSize - 1)
	{
		sums[warp_id] = val[3];
	}
	__syncthreads();

	if (warp_id == 0)
	{
		int warp_val = sums[lane_id];

#pragma unroll
		for (int i = 1; i <= warpSize; i *= 2)
		{
			int num = __shfl_up(warp_val, i, warpSize);

			if (lane_id >= i) warp_val += num;
		}
		sums[lane_id] = warp_val;
	}
	__syncthreads();

	int thread_val = 0;
	if (warp_id > 0)
	{
		thread_val = sums[warp_id - 1];

	#pragma unroll
		for (int i=0; i < 4; ++i)
		{
			val[i] += thread_val;
		}
	}

	if ((x_idx < intx) && (y_idx < inty))
	{
		uint4 data;
		data.x = val[0];
		data.y = val[1];
		data.z = val[2];
		data.w = val[3];
		((uint4*)integral.d_img)[img_idx] = data;
	}

	if (block_sum != NULL && threadIdx.x == blockDim.x - 1)
	{
		block_sum[blockIdx.x] = val[3];
	}
}

__global__ void horizontalBlockAdd2(IntegralImage integral, unsigned int *block_sum=NULL)
{
//	__shared__ unsigned int sum;
	__shared__ unsigned int sums[N_THREADS];
	int x_gdim = ((integral.x / 4) + blockDim.x - 1) / blockDim.x;
	int x_bid = blockIdx.x % x_gdim;  // block index in x-axis
	int y_idx = blockIdx.x / x_gdim;  // and y
	int x_idx = x_bid * blockDim.x + threadIdx.x;
	int intx = integral.x / 4, inty = integral.y;
	int img_idx = y_idx * intx + x_idx;  // image index for uint4 or uchar4 access
	int lane_id = (blockIdx.x * blockDim.x + threadIdx.x) % warpSize;
	int warp_id = threadIdx.x / warpSize;

//	// scan block_sum[] and add
	unsigned int val = 0;
//
//	if ((x_idx >= integral.x) || (y_idx >= integral.y)) return;
//
	sums[threadIdx.x] = 0;
	if ((block_sum != NULL) && (threadIdx.x < x_gdim) && (y_idx < integral.y))
	{
		sums[threadIdx.x] = block_sum[y_idx * x_gdim + threadIdx.x];
	}
	__syncthreads();

	if (threadIdx.x == 0)
	{
		for (int i=0; i < x_gdim-1; ++i)
		{
			sums[i+1] += sums[i];
		}
	}
	__syncthreads();
//
//	for (int i=1; i < warpSize; i*=2)
//	{
//		int num = __shfl_up( val, i, warpSize );
//		if (lane_id >= i) val += num;
//	}
//
//	if (threadIdx.x % warpSize == warpSize - 1)
//	{
//		sums[warp_id] = val;
//	}
//	__syncthreads();
//
//	if (warp_id == 0)
//	{
//		int warp_val = sums[lane_id];
//#pragma unroll
//		for (int i=1; i <= warpSize; i*=2)
//		{
//			int num = __shfl_up( warp_val, i, warpSize );
//			if (lane_id >= i) warp_val += num;
//		}
//		sums[lane_id] = warp_val;
//	}
//	__syncthreads();
//
//	int thread_val = 0;
//	if (warp_id > 0)
//	{
//		thread_val = sums[warp_id - 1];
//	}
//	val += thread_val;

	if (x_bid > 0)
	{
		val = sums[x_bid - 1];
	}
	if ((x_idx < intx) && (y_idx < inty))
	{
		uint4 data = ((uint4*)integral.d_img)[img_idx];
		data.x += val;
		data.y += val;
		data.z += val;
		data.w += val;
		((uint4*)integral.d_img)[img_idx] = data;
	}
}


__global__ void horizontalBlockAdd3(IntegralImage integral, unsigned int *block_sum=NULL)
{
//	__shared__ unsigned int sum;
	__shared__ unsigned int sums[N_THREADS];
	int x_gdim = ((integral.x / 16) + blockDim.x - 1) / blockDim.x;
	int x_bid = blockIdx.x % x_gdim;  // block index in x-axis
	int y_idx = blockIdx.x / x_gdim;  // and y
	int x_idx = x_bid * (blockDim.x * 4) + (threadIdx.x * 4);
	int intx = integral.x / 4, inty = integral.y;
	int img_idx = y_idx * intx + x_idx;
	int lane_id = (blockIdx.x * blockDim.x + threadIdx.x) % warpSize;
	int warp_id = threadIdx.x / warpSize;

//	// scan block_sum[] and add
	unsigned int val = 0;
	sums[threadIdx.x] = 0;
	if ((block_sum != NULL) && (threadIdx.x < x_gdim) && (y_idx < integral.y))
	{
		sums[threadIdx.x] = block_sum[y_idx * x_gdim + threadIdx.x];
	}
	__syncthreads();

	if (threadIdx.x == 0)
	{
		for (int i=0; i < x_gdim-1; ++i)
		{
			sums[i+1] += sums[i];
		}
	}
	__syncthreads();

	if (x_bid > 0)
	{
		val = sums[x_bid - 1];
	}
	if ((x_idx < intx) && (y_idx < inty))
	{
		uint4 data0 = ((uint4*)integral.d_img)[img_idx + 0];
		uint4 data1 = ((uint4*)integral.d_img)[img_idx + 1];
		uint4 data2 = ((uint4*)integral.d_img)[img_idx + 2];
		uint4 data3 = ((uint4*)integral.d_img)[img_idx + 3];
		data0.x += val; data0.y += val; data0.z += val; data0.w += val;
		data1.x += val; data1.y += val; data1.z += val; data1.w += val;
		data2.x += val; data2.y += val; data2.z += val; data2.w += val;
		data3.x += val; data3.y += val; data3.z += val; data3.w += val;
		((uint4*)integral.d_img)[img_idx + 0] = data0;
		((uint4*)integral.d_img)[img_idx + 1] = data1;
		((uint4*)integral.d_img)[img_idx + 2] = data2;
		((uint4*)integral.d_img)[img_idx + 3] = data3;
	}
}


__global__ void integralGPUvert_shfl(IntegralImage integral, unsigned int *block_sum = NULL)
{
	__shared__ unsigned int sum[N_THREADS];
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int y_bid = blockIdx.x % ((integral.y + blockDim.x - 1) / blockDim.x);
	int x_bid = blockIdx.x / ((integral.y + blockDim.x - 1) / blockDim.x);
	int y_idx = y_bid * blockDim.x + threadIdx.x;
	int x_idx = x_bid;
	int img_idx = y_idx * integral.x + x_idx;
	int lane_id = idx % warpSize;
	int warp_id = threadIdx.x / warpSize;
	int val = 0;
	if ((x_idx < integral.x) && (y_idx < integral.y))
	{
		val = integral.d_img[img_idx];
	}
	sum[threadIdx.x] = 0;
	__syncthreads();

#pragma unroll
	for (int i = 1; i <= warpSize; i *= 2)
	{
		int num = __shfl_up(val, i, warpSize);
		if (lane_id >= i) val += num;
	}

	if (threadIdx.x % warpSize == warpSize - 1)
	{
		sum[warp_id] = val;
	}
	__syncthreads();

	if (warp_id == 0)
	{
		int warp_val = sum[lane_id];

#pragma unroll
		for (int i = 1; i <= warpSize; i *= 2)
		{
			int num = __shfl_up(warp_val, i, warpSize);
			if (lane_id >= i) warp_val += num;
		}
		sum[lane_id] = warp_val;
	}
	__syncthreads();

	int thread_val = 0;
	if (warp_id > 0)
	{
		thread_val = sum[warp_id - 1];
	}

	val += thread_val;

	if ((x_idx < integral.x) && (y_idx < integral.y))
	{
		integral.d_img[img_idx] = val;
	}

	if (threadIdx.x == blockDim.x - 1)
	{
		block_sum[blockIdx.x] = val;
	}
}

__global__ void verticalBlockAdd(IntegralImage integral, unsigned int *block_sum)
{
	__shared__ unsigned int sum;
	int y_bid = blockIdx.x % ((integral.y + blockDim.x - 1) / blockDim.x);
	int x_bid = blockIdx.x / ((integral.y + blockDim.x - 1) / blockDim.x);
	int y_idx = y_bid * blockDim.x + threadIdx.x;
	int x_idx = x_bid;
	int img_idx = y_idx * integral.x + x_idx;

	if ((y_bid == 0) || (x_idx >= integral.x) || (y_idx >= integral.y)) return;

	if (threadIdx.x == 0)
	{
		sum = block_sum[blockIdx.x - 1];
	}
	__syncthreads();

	integral.d_img[img_idx] += sum;
}


int integralImageGPUshfl(const Image img, IntegralImage *integral)
{
	if (integral == NULL) return -1;

	integral->x = img.x; integral->y = img.y;
	if (integral->img != NULL)
	{
		freeIntegralImage(integral);
		integral->img = NULL;
	}
	size_t size = integral->x * integral->y * sizeof(unsigned int);
	unsigned int *p_intg = NULL;
	unsigned int *dp_intg = NULL;
	if ((p_intg = (unsigned int*)malloc(size)) == NULL)
	{
		fprintf(stderr, "fail to allocate memory\n");
		return -1;
	}
	if (cudaError_t err = cudaMalloc(&dp_intg, size))
	{
		fprintf(stderr, "cuda memory allocation error: %s\n", cudaGetErrorString(err));
		return err;
	}
	__CUDA(cudaMemset(dp_intg, 0, size));
	integral->img = p_intg;
	integral->d_img = dp_intg;

	int xblocksum_size = ((integral->x + N_THREADS - 1) / N_THREADS * integral->y) * sizeof(int);
	unsigned int *xblock_sum = NULL;
	if (cudaError_t err = cudaMalloc(&xblock_sum, xblocksum_size))
	{
		fprintf(stderr, "cuda memory allocation error: %s\n", cudaGetErrorString(err));
		return err;
	}
	__CUDA(cudaMemset(xblock_sum, 0, xblocksum_size));

	int yblocksum_size = ((integral->y + N_THREADS - 1) / N_THREADS * integral->x) * sizeof(int);
	//int vert_x_gdim = (integral->x + TILE_DIM - 1) / TILE_DIM;
	//int vert_y_gdim = (integral->y + TILE_DIM - 1) / TILE_DIM;
	//int yblocksum_size = TILE_DIM * vert_x_gdim * vert_y_gdim * sizeof(int);
	unsigned int *yblock_sum = NULL;
	if (cudaError_t err = cudaMalloc(&yblock_sum, yblocksum_size))
	{
		fprintf(stderr, "cuda memory allocation error: %s\n", cudaGetErrorString(err));
		return err;
	}
	__CUDA(cudaMemset(yblock_sum, 0, yblocksum_size));

#ifdef _DEBUG	
	__CUDA(cudaDeviceSynchronize());
	__CUDA(cudaGetLastError());
#endif

	// horizontal scan
	int x_thrs = integral->x / 4;  // 4 pixels proceeded by one thread. the num of threads in x-axis
	int x_gdim = ((x_thrs + N_THREADS - 1) / N_THREADS);  // the num of blocks in x-axis
	int gridDim = x_gdim * integral->y;
	integralGPUhorz_shfl2<<< gridDim, N_THREADS >>>(img, *integral, xblock_sum);
#ifdef _DEBUG	
	__CUDA(cudaDeviceSynchronize());
	__CUDA(cudaGetLastError());
#endif
#if 0
	if (x_gdim > 1)
	{
		Image empty_img;  empty_img.x = empty_img.y = 0;
		IntegralImage block_intg;
		block_intg.x = x_gdim;   block_intg.y = integral->y;   block_intg.d_img = xblock_sum;
		integralGPUhorz_shfl << < gridDim, N_THREADS >> >(empty_img, block_intg);
	}
#endif
	horizontalBlockAdd2 <<< gridDim, N_THREADS >>>(*integral, xblock_sum);
#ifdef _DEBUG	
	__CUDA(cudaDeviceSynchronize());
	__CUDA(cudaGetLastError());
#endif

	// vertical scan
	int y_gdim = (integral->y + N_THREADS - 1) / N_THREADS;
	gridDim = y_gdim * integral->x;
	integralGPUvert_shfl<<< gridDim, N_THREADS >>>(*integral, yblock_sum);
#ifdef _DEBUG	
	__CUDA(cudaDeviceSynchronize());
	__CUDA(cudaGetLastError());
#endif
	if (y_gdim > 1)
	{
		//size_t shsize = TILE_DIM * vert_y_gdim * sizeof(unsigned int);
		//verticalBlockAdd<<< gridDim, TILE_DIM * TILE_DIM, shsize >> >(*integral, yblock_sum);

				Image empty_img;  empty_img.x = empty_img.y = 0;
				IntegralImage block_intg;
				block_intg.x = y_gdim;   block_intg.y = integral->x;   block_intg.d_img = yblock_sum;
				integralGPUhorz_shfl<<< gridDim, N_THREADS >>>( empty_img, block_intg );
	}
	verticalBlockAdd<<< gridDim, N_THREADS >>>( *integral, yblock_sum );
#ifdef _DEBUG	
	__CUDA(cudaDeviceSynchronize());
	__CUDA(cudaGetLastError());
#endif

	__CUDA(cudaMemcpy(integral->img, integral->d_img, size, cudaMemcpyDeviceToHost));

	__CUDA(cudaFree(xblock_sum));
	__CUDA(cudaFree(yblock_sum));

#ifdef _DEBUG	
	__CUDA(cudaGetLastError());
#endif

	return cudaSuccess;
}


__global__ void integralGPUvert_shfltrans(IntegralImage integral, unsigned int *block_sum = NULL)
{
	__shared__ unsigned int sum[TILE_DIM];
	__shared__ unsigned int tile[TILE_DIM * TILE_DIM];
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int x_gdim = (integral.x + TILE_DIM - 1) / TILE_DIM;
	int y_gdim = (integral.y + TILE_DIM - 1) / TILE_DIM;
	int x_tid = threadIdx.x % TILE_DIM;
	int y_tid = threadIdx.x / TILE_DIM;
	int x_bid = blockIdx.x % x_gdim;
	int y_bid = blockIdx.x / x_gdim;
	int x_idx = x_bid * TILE_DIM + x_tid;
	int y_idx = y_bid * TILE_DIM + y_tid;
	int intx = integral.x, inty = integral.y;
	int img_idx = y_idx * integral.x + x_idx;
	int intile_idx = (threadIdx.x % TILE_DIM) * TILE_DIM + (threadIdx.x / TILE_DIM);
	int blocksum_idx = (x_bid * y_gdim + y_bid) * TILE_DIM + (threadIdx.x / warpSize);
	int lane_id = idx % warpSize;
	int warp_id = threadIdx.x / warpSize;
	int val = 0;
	tile[threadIdx.x] = 0;
	if ((x_idx < intx) && (y_idx < inty))
	{
		tile[threadIdx.x] = integral.d_img[img_idx];
	}
	__syncthreads();

	val = tile[intile_idx];
	__syncthreads();

#pragma unroll
	for (int i = 1; i <= warpSize; i *= 2)
	{
		int num = __shfl_up(val, i, warpSize);
		if (lane_id >= i) val += num;
	}

	// write to global memory from tile base line
	if (threadIdx.x % warpSize == warpSize - 1)
	{
//		sum[warp_id] = val;
//		block_sum[blockIdx.x + y_tid] = val;
		block_sum[blocksum_idx] = val;
	}
	tile[intile_idx] = val;
	__syncthreads();

//	if (warp_id == 0)
//	{
//		int warp_val = sum[lane_id];
//
//#pragma unroll
//		for (int i = 1; i <= warpSize; i *= 2)
//		{
//			int num = __shfl_up(warp_val, i, warpSize);
//			if (lane_id >= i) warp_val += num;
//		}
//		sum[lane_id] = warp_val;
//	}
//	__syncthreads();
//
//	int thread_val = 0;
//	if (warp_id > 0)
//	{
//		thread_val = sum[warp_id - 1];
//	}
//
//	val += thread_val;

	if ((x_idx < intx) && (y_idx < inty))
	{
		 integral.d_img[img_idx] = tile[threadIdx.x];
	}
//	if ((x_idx < integral.x) && (y_idx < integral.y))
//	{
//		integral.d_img[img_idx] = val;
//	}

//	if (threadIdx.x == blockDim.x - 1)
//	{
//		block_sum[blockIdx.x] = val;
//	}
}

__global__ void integralGPUvert_shfltrans2(IntegralImage integral, unsigned int *block_sum = NULL)
{
	__shared__ unsigned int sum[TILE_DIM];
	__shared__ unsigned int tile[(TILE_DIM+1) * TILE_DIM];
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int x_gdim = (integral.x + TILE_DIM - 1) / TILE_DIM;
	int y_gdim = (integral.y + TILE_DIM - 1) / TILE_DIM;
	int x_tid = threadIdx.x % TILE_DIM;
	int y_tid = threadIdx.x / TILE_DIM;
	int x_bid = blockIdx.x % x_gdim;
	int y_bid = blockIdx.x / x_gdim;
	int x_idx = x_bid * TILE_DIM + x_tid;
	int y_idx = y_bid * TILE_DIM + y_tid;
	int intx = integral.x, inty = integral.y;
	int img_idx = y_idx * intx + x_idx;
	int intile_idx = (threadIdx.x % TILE_DIM) * (TILE_DIM+1) + (threadIdx.x / TILE_DIM);
	int blocksum_idx = (x_bid * y_gdim + y_bid) * TILE_DIM + threadIdx.x;
	int lane_id = idx % warpSize;
	int warp_id = threadIdx.x / warpSize;
	int val = 0;
	__shared__ unsigned int sblksum[TILE_DIM];

	// compute 1 block (32x32pix) with 256 threads.
	// so 1 thread processes 4 pixels.
#pragma unroll
	for (int k=0; k < 4; ++k)
	{
		int id = (threadIdx.x / TILE_DIM) * (TILE_DIM+1) + (threadIdx.x % TILE_DIM) + (blockDim.x + 8) * k;
		tile[id] = 0;
		if ((x_idx < intx) && ((y_idx + 8*k) < inty))
		{
			tile[id] = integral.d_img[img_idx + intx * 8*k];
		}
	}
	__syncthreads();

#pragma unroll
	for (int k=0; k < 4; ++k)
	{
		val = tile[intile_idx + 8*k];
//		__syncthreads();

#pragma unroll
		for (int i=1; i <= warpSize; i*=2)
		{
			int num = __shfl_up(val, i, warpSize);
			if (lane_id >= i) val += num;
		}

		// write to shared memory from tile base line
		if (threadIdx.x % warpSize == warpSize - 1)
		{
			int id = (threadIdx.x + blockDim.x * k) / warpSize;
			sblksum[id] = val;
		}
		tile[intile_idx + 8*k] = val;
		__syncthreads();
	}

//	__syncthreads();
	if (threadIdx.x < TILE_DIM)
	{
		block_sum[blocksum_idx] = sblksum[threadIdx.x];
	}
//	__syncthreads();

#pragma unroll
	for (int k=0; k < 4; ++k)
	{
		int id = (threadIdx.x / TILE_DIM) * (TILE_DIM+1) + (threadIdx.x % TILE_DIM) + (blockDim.x + 8) * k;
		if ((x_idx < intx) && ((y_idx + 8*k) < inty))
		{
			 integral.d_img[img_idx + intx * 8*k] = tile[id];
		}
	}
}

__global__ void verticalBlockAddTrans(IntegralImage integral, unsigned int *block_sum)
{
//	__shared__ unsigned int sum;
	extern __shared__ unsigned int s_bsum[];
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int x_gdim = (integral.x + TILE_DIM - 1) / TILE_DIM;
	int y_gdim = (integral.y + TILE_DIM - 1) / TILE_DIM;
	int x_tid = threadIdx.x % TILE_DIM;
	int y_tid = threadIdx.x / TILE_DIM;
	int x_bid = blockIdx.x % x_gdim;
	int y_bid = blockIdx.x / x_gdim;
	int x_idx = x_bid * TILE_DIM + x_tid;
	int y_idx = y_bid * TILE_DIM + y_tid;
	int intx = integral.x, inty = integral.y;
	int img_idx = y_idx * integral.x + x_idx;
	int blocksum_head = (x_bid * y_gdim) * TILE_DIM;
	int intile_idx = (threadIdx.x % TILE_DIM) * TILE_DIM + (threadIdx.x / TILE_DIM);
	int lane_id = idx % warpSize;
	
	unsigned int val = 0;
	unsigned int prev_sum = 0;

//	if ((y_bid == 0) || (x_idx >= integral.x) || (y_idx >= integral.y)) return;

	for (int k=0; (y_tid + TILE_DIM * k) < y_gdim; ++k)
	{
//		val = block_sum[blocksum_head + k * blockDim.x + threadIdx.x];
		s_bsum[k * blockDim.x + threadIdx.x] = block_sum[blocksum_head + k * blockDim.x + threadIdx.x];
	}
	__syncthreads();


	for (int k=0; (TILE_DIM * k) < y_gdim; ++k)
	{
		int id = intile_idx + blockDim.x * k;
		val = (id < (TILE_DIM * y_gdim)) ? s_bsum[id] : 0;

#pragma unroll
		for (int i=1; i <= warpSize; i *= 2)
		{
			int num = __shfl_up( val, i, warpSize );

			if (lane_id >= i) val += num;
		}
		//__syncthreads();

		val += prev_sum;
		prev_sum = __shfl( val, warpSize - 1, warpSize );

//		if (threadIdx.x % warpSize == warpSize - 1)
//		{
//			prev_sum = val;
//		}

		(id < (TILE_DIM * y_gdim)) ? (s_bsum[id] = val) : val;
		__syncthreads();
	}
//	if (threadIdx.x < TILE_DIM)
//	{
//		for (int i=1; i < y_gdim; ++i)
//		{
//			s_bsum[i * TILE_DIM + threadIdx.x] += s_bsum[(i-1) * TILE_DIM + threadIdx.x];
//			__syncthreads();
//		}
//	}
	__syncthreads();

	if ((y_bid > 0) && (x_idx < intx) && (y_idx < inty)) {
		integral.d_img[img_idx] += s_bsum[(y_bid - 1) * TILE_DIM + x_tid];
	}
//	integral.d_img[img_idx] += sum;
}


__global__ void verticalBlockAddTrans2(IntegralImage integral, unsigned int *block_sum)
{
	extern __shared__ unsigned int s_bsum[];
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int x_gdim = (integral.x + TILE_DIM - 1) / TILE_DIM;
	int y_gdim = (integral.y + TILE_DIM - 1) / TILE_DIM;
	int x_tid = threadIdx.x % TILE_DIM;
	int y_tid = threadIdx.x / TILE_DIM;
	int x_bid = blockIdx.x % x_gdim;
	int y_bid = blockIdx.x / x_gdim;
	int x_idx = x_bid * TILE_DIM + x_tid;
	int y_idx = y_bid * TILE_DIM + y_tid;
	int intx = integral.x, inty = integral.y;
	int img_idx = y_idx * integral.x + x_idx;
	int blocksum_head = (x_bid * y_gdim) * TILE_DIM;
	int intile_idx = (threadIdx.x % TILE_DIM) * (TILE_DIM+1) + (threadIdx.x / TILE_DIM);
	int lane_id = idx % warpSize;
	
	unsigned int val = 0;
	unsigned int prev_sum[4];
	prev_sum[0] = prev_sum[1] = prev_sum[2] = prev_sum[3] = 0;


	for (int l=0; (y_tid + (TILE_DIM/4) * l) < y_gdim; ++l)
	{
		int id = (threadIdx.x / TILE_DIM) * (TILE_DIM+1) + (threadIdx.x % TILE_DIM) + (blockDim.x+8) * l;
		s_bsum[id] = block_sum[blocksum_head + l * blockDim.x + threadIdx.x];
	}
	__syncthreads();


	for (int k=0; (TILE_DIM * k) < y_gdim; ++k)
	{
#pragma unroll
		for (int l=0; l < 4; ++l)
		{
			int id = intile_idx + 8*l + (TILE_DIM+1)*TILE_DIM * k;
			val = (id < ((TILE_DIM+1) * y_gdim)) ? s_bsum[id] : 0;

#pragma unroll
			for (int i=1; i <= warpSize; i *= 2)
			{
				int num = __shfl_up( val, i, warpSize );

				if (lane_id >= i) val += num;
			}
			//__syncthreads();

			val += prev_sum[l];
			prev_sum[l] = __shfl( val, warpSize - 1, warpSize );


			//__syncthreads();
			(id < ((TILE_DIM+1) * y_gdim)) ? (s_bsum[id] = val) : val;
		}
	}
	__syncthreads();

#pragma unroll
	for (int l=0; l < 4; ++l)
	{
		if ((y_bid > 0) && (x_idx < intx) && ((y_idx + 8*l) < inty)) {
			integral.d_img[img_idx + intx * 8*l] += s_bsum[(y_bid - 1) * (TILE_DIM+1) + x_tid];
		}
	}
}


int integralImageGPUshfl_trans(const Image img, IntegralImage *integral)
{
	if (integral == NULL) return -1;

	integral->x = img.x; integral->y = img.y;
	if (integral->img != NULL)
	{
		freeIntegralImage(integral);
		integral->img = NULL;
	}
	size_t size = integral->x * integral->y * sizeof(unsigned int);
	unsigned int *p_intg = NULL;
	unsigned int *dp_intg = NULL;
	if ((p_intg = (unsigned int*)malloc(size)) == NULL)
	{
		fprintf( stderr, "fail to allocate memory\n" );
		return -1;
	}
	if (cudaError_t err = cudaMalloc( &dp_intg, size ))
	{
		fprintf( stderr, "cuda memory allocation error: %s\n", cudaGetErrorString(err) );
		return err;
	}
	__CUDA( cudaMemset( dp_intg, 0, size ) );
	integral->img = p_intg;
	integral->d_img = dp_intg;

	int xblocksum_size = ((integral->x + N_THREADS - 1) / N_THREADS * integral->y) * sizeof(int);
	unsigned int *xblock_sum = NULL;
	if (cudaError_t err = cudaMalloc( &xblock_sum, xblocksum_size ))
	{
		fprintf( stderr, "cuda memory allocation error: %s\n", cudaGetErrorString(err) );
		return err;
	}
	__CUDA( cudaMemset( xblock_sum, 0, xblocksum_size ) );

//	int yblocksum_size = ((integral->y + N_THREADS - 1) / N_THREADS * integral->x) * sizeof(int);
	int vert_x_gdim = (integral->x + TILE_DIM - 1) / TILE_DIM;
	int vert_y_gdim = (integral->y + TILE_DIM - 1) / TILE_DIM;
	int yblocksum_size = TILE_DIM * vert_x_gdim * vert_y_gdim * sizeof(int);
	unsigned int *yblock_sum = NULL;
	if (cudaError_t err = cudaMalloc( &yblock_sum, yblocksum_size ))
	{
		fprintf( stderr, "cuda memory allocation error: %s\n", cudaGetErrorString(err) );
		return err;
	}
	__CUDA( cudaMemset( yblock_sum, 0, yblocksum_size ) );

#ifdef _DEBUG	
	__CUDA( cudaDeviceSynchronize() );
	__CUDA( cudaGetLastError() );
#endif

	// horizontal scan
	int x_thrs = integral->x / 4;  // 4 pixels proceeded by one thread. the num of threads in x-axis
	int x_gdim = ((x_thrs + N_THREADS - 1) / N_THREADS);  // the num of blocks in x-axis
	int gridDim = x_gdim * integral->y;
	integralGPUhorz_shfl2<<< gridDim, N_THREADS >>>( img, *integral, xblock_sum );
#ifdef _DEBUG	
	__CUDA( cudaDeviceSynchronize() );
	__CUDA( cudaGetLastError() );
#endif
#if 0
	if (x_gdim > 1)
	{
		Image empty_img;  empty_img.x = empty_img.y = 0;
		IntegralImage block_intg;
		block_intg.x = x_gdim;   block_intg.y = integral->y;   block_intg.d_img = xblock_sum;
		integralGPUhorz_shfl<<< gridDim, N_THREADS >>>( empty_img, block_intg );
	}
#endif
	horizontalBlockAdd2<<< gridDim, N_THREADS >>>(*integral, xblock_sum);
#ifdef _DEBUG	
	__CUDA( cudaDeviceSynchronize() );
	__CUDA( cudaGetLastError() );
#endif

	// vertical scan
//	int y_gdim = (integral->y + N_THREADS - 1) / N_THREADS;
	gridDim = vert_x_gdim * vert_y_gdim;
	integralGPUvert_shfltrans<<< gridDim, TILE_DIM * TILE_DIM >>>( *integral, yblock_sum );
#ifdef _DEBUG	
	__CUDA(cudaDeviceSynchronize());
	__CUDA( cudaGetLastError() );
#endif
	if (vert_y_gdim > 1)
	{
		size_t shsize = TILE_DIM * vert_y_gdim * sizeof(unsigned int);
		verticalBlockAddTrans<<< gridDim, TILE_DIM * TILE_DIM, shsize >>>( *integral, yblock_sum );

//		Image empty_img;  empty_img.x = empty_img.y = 0;
//		IntegralImage block_intg;
//		block_intg.x = y_gdim;   block_intg.y = integral->x;   block_intg.d_img = yblock_sum;
//		integralGPUhorz_shfl<<< gridDim, N_THREADS >>>( empty_img, block_intg );
	}
//	verticalBlockAdd<<< gridDim, N_THREADS >>>( *integral, yblock_sum );
#ifdef _DEBUG	
	__CUDA( cudaDeviceSynchronize() );
	__CUDA( cudaGetLastError() );
#endif

	__CUDA( cudaMemcpy( integral->img, integral->d_img, size, cudaMemcpyDeviceToHost ) );

	__CUDA( cudaFree( xblock_sum ) );
	__CUDA( cudaFree( yblock_sum ) );

#ifdef _DEBUG	
	__CUDA( cudaGetLastError() );
#endif

	return cudaSuccess;
}


int integralImageGPUshfl_trans2(const Image img, IntegralImage *integral)
{
	if (integral == NULL) return -1;

	integral->x = img.x; integral->y = img.y;
	if (integral->img != NULL)
	{
		freeIntegralImage(integral);
		integral->img = NULL;
	}
	size_t size = integral->x * integral->y * sizeof(unsigned int);
	unsigned int *p_intg = NULL;
	unsigned int *dp_intg = NULL;
	if ((p_intg = (unsigned int*)malloc(size)) == NULL)
	{
		fprintf( stderr, "fail to allocate memory\n" );
		return -1;
	}
	if (cudaError_t err = cudaMalloc( &dp_intg, size ))
	{
		fprintf( stderr, "cuda memory allocation error: %s\n", cudaGetErrorString(err) );
		return err;
	}
	__CUDA( cudaMemset( dp_intg, 0, size ) );
	integral->img = p_intg;
	integral->d_img = dp_intg;

	int xblocksum_size = ((integral->x + N_THREADS - 1) / N_THREADS * integral->y) * sizeof(int);
	unsigned int *xblock_sum = NULL;
	if (cudaError_t err = cudaMalloc( &xblock_sum, xblocksum_size ))
	{
		fprintf( stderr, "cuda memory allocation error: %s\n", cudaGetErrorString(err) );
		return err;
	}
	__CUDA( cudaMemset( xblock_sum, 0, xblocksum_size ) );

//	int yblocksum_size = ((integral->y + N_THREADS - 1) / N_THREADS * integral->x) * sizeof(int);
	int vert_x_gdim = (integral->x + TILE_DIM - 1) / TILE_DIM;
	int vert_y_gdim = (integral->y + TILE_DIM - 1) / TILE_DIM;
	int yblocksum_size = TILE_DIM * vert_x_gdim * vert_y_gdim * sizeof(int);
	unsigned int *yblock_sum = NULL;
	if (cudaError_t err = cudaMalloc( &yblock_sum, yblocksum_size ))
	{
		fprintf( stderr, "cuda memory allocation error: %s\n", cudaGetErrorString(err) );
		return err;
	}
	__CUDA( cudaMemset( yblock_sum, 0, yblocksum_size ) );

#ifdef _DEBUG	
	__CUDA( cudaDeviceSynchronize() );
	__CUDA( cudaGetLastError() );
#endif

	// horizontal scan
	int x_thrs = integral->x / 16;  // 16 pixels proceeded by one thread. the num of threads in x-axis
	int x_gdim = ((x_thrs + N_THREADS - 1) / N_THREADS);  // the num of blocks in x-axis
	int gridDim = x_gdim * integral->y;
	integralGPUhorz_shfl3<<< gridDim, N_THREADS >>>( img, *integral, xblock_sum );
#ifdef _DEBUG	
	__CUDA( cudaDeviceSynchronize() );
	__CUDA( cudaGetLastError() );
#endif
#if 0
	if (x_gdim > 1)
	{
		Image empty_img;  empty_img.x = empty_img.y = 0;
		IntegralImage block_intg;
		block_intg.x = x_gdim;   block_intg.y = integral->y;   block_intg.d_img = xblock_sum;
		integralGPUhorz_shfl<<< gridDim, N_THREADS >>>( empty_img, block_intg );
	}
#endif
	horizontalBlockAdd3<<< gridDim, N_THREADS >>>(*integral, xblock_sum);
#ifdef _DEBUG	
	__CUDA( cudaDeviceSynchronize() );
	__CUDA( cudaGetLastError() );
#endif

	// vertical scan
//	int y_gdim = (integral->y + N_THREADS - 1) / N_THREADS;
	gridDim = vert_x_gdim * vert_y_gdim;
	integralGPUvert_shfltrans2<<< gridDim, TILE_DIM * TILE_DIM / 4 >>>( *integral, yblock_sum );
#ifdef _DEBUG	
	__CUDA( cudaDeviceSynchronize() );
	__CUDA( cudaGetLastError() );
#endif
	if (vert_y_gdim > 1)
	{
		size_t shsize = (TILE_DIM+1) * vert_y_gdim * sizeof(unsigned int);
		verticalBlockAddTrans2<<< gridDim, TILE_DIM * TILE_DIM / 4, shsize >>>( *integral, yblock_sum );

//		Image empty_img;  empty_img.x = empty_img.y = 0;
//		IntegralImage block_intg;
//		block_intg.x = y_gdim;   block_intg.y = integral->x;   block_intg.d_img = yblock_sum;
//		integralGPUhorz_shfl<<< gridDim, N_THREADS >>>( empty_img, block_intg );
	}
//	verticalBlockAdd<<< gridDim, N_THREADS >>>( *integral, yblock_sum );
#ifdef _DEBUG	
	__CUDA( cudaDeviceSynchronize() );
	__CUDA( cudaGetLastError() );
#endif

	__CUDA( cudaMemcpy( integral->img, integral->d_img, size, cudaMemcpyDeviceToHost ) );

	__CUDA( cudaFree( xblock_sum ) );
	__CUDA( cudaFree( yblock_sum ) );

#ifdef _DEBUG	
	__CUDA( cudaGetLastError() );
#endif

	return cudaSuccess;
}


// array for reduction test
__managed__ unsigned char arr[] = 
{ 1 ,  2 ,  3 ,  4 ,  5 ,
  6 ,  7 ,  8 ,  9 , 10 ,
 11 , 12 , 13 , 14 , 15 ,
 16 , 17 , 18 , 19 , 20 ,
 21 , 22 , 23 , 24 , 25 };


int main(int argc, void **argv)
{
	Image img, test;
	IntegralImage intg, test_intg, d_intg, d_test_intg, cpu_intg;
	unsigned int result, *d_result;
	dim3 gridDim, blockDim;
	float elapsed_time = 0.f;
	double t = 0.;
	cudaDeviceProp prop;
	cudaEvent_t start, stop;
	LARGE_INTEGER cpu_start, cpu_stop, freq;
	test.img = test.d_img = arr;
	test.x = 5; test.y = 5; test.maxval = 255; test.fmt = Format::PGM;

	__CUDA( cudaGetDeviceProperties( &prop, 0 ) );

	__CUDA( cudaMalloc( &d_result, sizeof(unsigned int)) );
	__CUDA( cudaMemset( d_result, 0, sizeof(unsigned int) ) );
	result = 0;

	__CUDA( cudaEventCreate( &start ) );
	__CUDA( cudaEventCreate( &stop ) );

	// load image and change format to pgm
	if (loadImage( ((const char **)argv)[argc - 1], &img ))  exit(-1);
	if (img.fmt == Format::PPM)  ppmtopgm(&img);
	//if (writeImage( &img, "result.pgm", Format::PGM ))  exit(-1);
	__CUDA( cudaMalloc( &img.d_img, img.x * img.y * sizeof(char) ) );
	__CUDA( cudaMemcpy( img.d_img, img.img, img.x * img.y * sizeof(char), cudaMemcpyHostToDevice ) );

	QueryPerformanceFrequency(&freq);

#ifdef _REDUCTION_TEST
	// CPU reduction test
	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&cpu_start);
	printf( "reduction CPU [test]: %ld\n", reduction_cpu(test) );
	QueryPerformanceCounter(&cpu_stop);
	t = (double)(cpu_stop.QuadPart - cpu_start.QuadPart) / freq.QuadPart * 1000;
	printf( "time: %f ms\n", t );
	printf( "\t-> %f MB/s\n", (test.x * test.y * sizeof(unsigned char) / 1024 / 1024) / (elapsed_time / 1000) );

	printf("\n");

	// GPU reduction test with shared memory
	gridDim = dim3( (test.x * test.y + N_THREADS - 1) / N_THREADS );
	blockDim = dim3( N_THREADS );
	__CUDA( cudaEventRecord( start, 0 ) );
	reductionGPUsmem<<<gridDim, blockDim, N_THREADS * sizeof(int)>>>(test, d_result);
	__CUDA( cudaEventRecord( stop, 0 ) );
	__CUDA( cudaEventSynchronize( stop ) );
	__CUDA( cudaEventElapsedTime( &elapsed_time, start, stop ) );
	__CUDA( cudaMemcpy( &result, d_result, sizeof(int) ) );
	printf( "reduction GPU smem [test]: %ld\n", result );
	printf( "time: %f ms\n", elapsed_time );
	printf( "\t-> %f MB/s\n", (test.x * test.y * sizeof(unsigned char) / 1024 / 1024) / (elapsed_time / 1000) );
	__CUDA( cudaGetLastError() );

	printf("\n");

	// GPU reduction test with shuffle
	__CUDA( cudaEventRecord( start, 0 ) );
	reductionGPUshfl<<<gridDim, blockDim, N_THREADS * sizeof(int)>>>(test, d_result);
	__CUDA( cudaEventRecord( stop, 0 ) );
	__CUDA( cudaEventSynchronize( stop ) );
	__CUDA( cudaEventElapsedTime( &elapsed_time, start, stop ) );
	__CUDA( cudaMemcpy( &result, d_result, sizeof(int) ) );
	printf( "reduction GPU shfl [test]: %ld\n", result );
	printf( "time: %f ms\n", elapsed_time );
	printf( "\t-> %f MB/s\n", (test.x * test.y * sizeof(unsigned char) / 1024 / 1024) / (elapsed_time / 1000) );
	__CUDA( cudaGetLastError() );

	printf("\n");

	// GPU reduction with shuffle2 method
	__CUDA( cudaEventRecord( start, 0 ) );
	reductionGPUshfl2<<<2, blockDim, N_THREADS * sizeof(int)>>>(test, d_result);
	__CUDA( cudaEventRecord( stop, 0 ) );
	__CUDA( cudaEventSynchronize( stop ) );
	__CUDA( cudaEventElapsedTime( &elapsed_time, start, stop ) );
	__CUDA( cudaMemcpy( &result, d_result, sizeof(int) ) );
	printf( "reduction GPU shfl2 [test]: %ld\n", result );
	printf( "time: %f ms\n", elapsed_time );
	printf( "\t-> %f MB/s\n", (test.x * test.y * sizeof(unsigned char) / 1024 / 1024) / (elapsed_time / 1000) );
	__CUDA( cudaGetLastError() );

	printf("\n");
#endif // _REDUCTION_TEST

#ifdef _INTEGRAL_TEST
	// CPU integral image test
	QueryPerformanceCounter(&cpu_start);
	if (integralImageCPU( test, &test_intg )) exit(-1);
	QueryPerformanceCounter(&cpu_stop);
	printf( "Integral Image CPU [test]: %ld\n", test_intg.img[test_intg.x * test_intg.y - 1] );
	t = (double)(cpu_stop.QuadPart - cpu_start.QuadPart) / freq.QuadPart * 1000;
	printf( "time: %f ms\n", t );
	printf( "\t-> %f MB/s\n", (test.x * test.y * sizeof(unsigned int) / 1024 / 1024) / (t / 1000) );

	printf("\n");
	
	// GPU integral image test
	__CUDA(cudaEventRecord(start, 0));
	integralImageGPU(test, &d_test_intg);
	__CUDA(cudaEventRecord(stop, 0));
	__CUDA(cudaEventSynchronize(stop));
	__CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));
	printf("Integral Image GPU [test]: %ld\n", d_test_intg.img[d_test_intg.x * d_test_intg.y - 1]);
	printf("time: %f ms\n", elapsed_time);
	printf( "\t-> %f MB/s\n", (test.x * test.y * sizeof(unsigned int) / 1024 / 1024) / (elapsed_time / 1000) );
	__CUDA(cudaGetLastError());

	printf("\n");

	// GPU integral image test with __shfl
	__CUDA(cudaEventRecord(start, 0));
	integralImageGPUshfl(test, &d_test_intg);
	__CUDA(cudaEventRecord(stop, 0));
	__CUDA(cudaEventSynchronize(stop));
	__CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));
	printf("Integral Image GPU with shfl [test]: %ld\n", d_test_intg.img[d_test_intg.x * d_test_intg.y - 1]);
	printf("time: %f ms\n", elapsed_time);
	printf("\t-> %f MB/s\n", (test.x * test.y * sizeof(unsigned int) / 1024 / 1024) / (elapsed_time / 1000) );
	__CUDA(cudaGetLastError());

	printf("\n");
#endif //_INTEGRAL_TEST

#ifdef _REDUCTION
	// CPU reduction
	QueryPerformanceCounter(&cpu_start);
	printf( "reduction CPU: %ld\n", reduction_cpu(img) );
	QueryPerformanceCounter(&cpu_stop);
	t = (double)(cpu_stop.QuadPart - cpu_start.QuadPart) / freq.QuadPart * 1000;
	printf( "time: %f ms\n", t );
	printf( "\t-> %f MB/s\n", (img.x * img.y * sizeof(unsigned char) / 1024 / 1024) / (t / 1000) );

	printf("\n");

	// GPU reduction with shared memory
	gridDim = dim3( (img.x * img.y + N_THREADS - 1) / N_THREADS );
	blockDim = dim3( N_THREADS );
	__CUDA( cudaEventRecord( start, 0 ) );
	reductionGPUsmem<<<gridDim, blockDim, N_THREADS * sizeof(int)>>>(img, d_result);
	__CUDA( cudaEventRecord( stop, 0 ) );
	__CUDA( cudaEventSynchronize( stop ) );
	__CUDA( cudaEventElapsedTime( &elapsed_time, start, stop ) );
	__CUDA( cudaMemcpy( &result, d_result, sizeof(int), cudaMemcpyDeviceToHost ) );
	printf( "reduction GPU smem: %ld\n", result );
	printf( "time: %f ms\n", elapsed_time );
	printf("\t-> %f MB/s\n", (img.x * img.y * sizeof(unsigned char) / 1024 / 1024) / (elapsed_time / 1000));
	__CUDA( cudaGetLastError() );

	printf("\n");

	// GPU reduction with shuffle
	__CUDA( cudaEventRecord( start, 0 ) );
	reductionGPUshfl<<<gridDim, blockDim, N_THREADS * sizeof(int)>>>(img, d_result);
	__CUDA( cudaEventRecord( stop, 0 ) );
	__CUDA( cudaEventSynchronize( stop ) );
	__CUDA( cudaEventElapsedTime( &elapsed_time, start, stop ) );
	__CUDA( cudaMemcpy( &result, d_result, sizeof(int), cudaMemcpyDeviceToHost ) );
	printf( "reduction GPU shfl: %ld\n", result );
	printf( "time: %f ms\n", elapsed_time );
	printf("\t-> %f MB/s\n", (img.x * img.y * sizeof(unsigned char) / 1024 / 1024) / (elapsed_time / 1000));
	__CUDA( cudaGetLastError() );

	printf("\n");

	// GPU reduction with shuffle2 method
	__CUDA( cudaEventRecord( start, 0 ) );
	reductionGPUshfl2<<<2048, blockDim, N_THREADS * sizeof(int)>>>(img, d_result);
	__CUDA( cudaEventRecord( stop, 0 ) );
	__CUDA( cudaEventSynchronize( stop ) );
	__CUDA( cudaEventElapsedTime( &elapsed_time, start, stop ) );
	__CUDA( cudaMemcpy( &result, d_result, sizeof(int), cudaMemcpyDeviceToHost ) );
	printf( "reduction GPU shfl2: %ld\n", result );
	printf( "time: %f ms\n", elapsed_time );
	printf( "\t-> %f MB/s\n", (img.x * img.y * sizeof(unsigned char) / 1024 / 1024) / (elapsed_time / 1000) );
	__CUDA( cudaGetLastError() );

	printf("\n");
#endif // _REDUCTION

#ifdef _INTEGRAL
	// CPU integral image
	QueryPerformanceCounter(&cpu_start);
	if (integralImageCPU( img, &cpu_intg )) exit(-1);
	QueryPerformanceCounter(&cpu_stop);
	printf( "Integral Image CPU: %ld\n", cpu_intg.img[cpu_intg.x * cpu_intg.y - 1] );
	t = (double)(cpu_stop.QuadPart - cpu_start.QuadPart) / freq.QuadPart * 1000;
	printf( "time: %f ms\n", t );
	printf("\t-> %f MB/s\n", (img.x * img.y * sizeof(unsigned int) / 1024 / 1024) / (t / 1000));

	printf("\n");

	// GPU integral image
	__CUDA(cudaEventRecord(start, 0));
	integralImageGPU(img, &intg);
	__CUDA(cudaEventRecord(stop, 0));
	__CUDA(cudaEventSynchronize(stop));
	__CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));
	printf("Integral Image GPU: %ld\n", intg.img[intg.x * intg.y - 1]);
	printf("time: %f ms\n", elapsed_time);
	printf("\t-> %f MB/s\n", (img.x * img.y * sizeof(unsigned int) / 1024 / 1024) / (elapsed_time / 1000));
	__CUDA(cudaGetLastError());

	printf("\n");

	// GPU integral image with __shfl
	__CUDA(cudaEventRecord(start, 0));
	integralImageGPUshfl_trans2(img, &d_intg);
	__CUDA(cudaEventRecord(stop, 0));
	__CUDA(cudaEventSynchronize(stop));
	__CUDA(cudaEventElapsedTime(&elapsed_time, start, stop));
	printf("Integral Image GPU with shfl2: %ld\n", d_intg.img[d_intg.x * d_intg.y - 1]);
	printf("time: %f ms\n", elapsed_time);
	printf("\t-> %f MB/s\n", (img.x * img.y * sizeof(unsigned int) / 1024 / 1024) / (elapsed_time / 1000));
	__CUDA(cudaGetLastError());

	printf("\n");

	FILE *err;
	err = fopen("error.log", "w");
	bool correct = true;
	for (int i=0; i < d_intg.y; ++i)
	{
		for (int j=0; j < d_intg.x; ++j)
		{
			if (cpu_intg.img[i * cpu_intg.x + j] != d_intg.img[i * d_intg.x + j])
			{
				correct = false;
				fprintf( err, "(x,y) : (%d,%d), answer:%d, result:%d\n", j, i, cpu_intg.img[i*cpu_intg.x + j], d_intg.img[i*d_intg.x + j]);
			}
		}
	}
	printf( "integral image : %s\n", (correct ? "success" : "fail") );
	printf("\n");
	fclose(err);
#endif // _INTEGRAL


	__CUDA( freeImage(&img) );
	__CUDA( freeIntegralImage(&test_intg) );
	__CUDA( freeIntegralImage(&intg) );
	__CUDA( freeIntegralImage(&d_test_intg) );
	__CUDA( freeIntegralImage(&d_intg) );
	__CUDA( cudaEventDestroy(start) );
	__CUDA( cudaEventDestroy(stop) );
	__CUDA( cudaDeviceReset() );

	return 0;
}

