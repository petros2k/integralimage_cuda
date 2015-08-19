#include <stdio.h>
#include <stdlib.h>
#include <memory.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "integralimage.cuh"

#define N_THREADS 128


int main(void)
{
	// setting up test image data
	Image img0;
	img0.w = 2688; img0.h = 1520; img0.pix_maxval = 255; img0.fmt = Format::PGM;

	// allocate memory in 1D map
	img0.img = (unsigned char**)malloc( img0.h * sizeof(unsigned char*) );
	img0.img[0] = (unsigned char*)malloc( img0.w * img0.h * sizeof(unsigned char) );
	memset( img0.img[0], 1, img0.w * img0.h * sizeof(unsigned char) );
	// set 2D array address
	for (int i=0; i<img0.h; ++i)
	{
		img0.img[i] = img0.img[0] + i * img0.w * sizeof(unsigned char);
	}

	unsigned int res_cpu = 0, res_gpu = 0;
	int gpu_method = 6;
	res_cpu = reductionCPU( img0 );
	res_gpu = reductionGPU( img0, gpu_method );

	printf( "CPU result: %ld\n", res_cpu );
	printf( "GPU result: %ld (method %d : %s)\n", res_gpu, gpu_method, reduce_kernel_desc[gpu_method] );
	
	gpu_method = 2;
	unsigned int **intimg = NULL;
	unsigned int **d_intimg = NULL;
	bool eq = true;

	printf( "\n" );
	printf( "[Integral Image]\n(GPU method %d : %s)\n", gpu_method, integral_kernel_desc[gpu_method] );
	if (!integralImageCPU( img0, &intimg ))
	{
		printf( "Integral Image CPU failed.\n" );
		eq = false;
	}
	if (!integralImageGPU( img0, &d_intimg, gpu_method ))
	{
		printf( "Integral Image GPU failed.\n" );
		eq = false;
	}

	int diff = 0;
	for (int i=0; i<img0.h; ++i)
	{
		for (int j=0; j<img0.w; ++j)
		{
			if (intimg[i][j] != d_intimg[i][j])
			{
				eq = false;
				++diff;
			}
		}
	}
	printf( "result %s, diff=%d\n", eq==true?"success":"failure", diff );

	freeIntImg( intimg );
	freeIntImg( d_intimg );
	free( img0.img[0] );
	free( img0.img );

	cudaDeviceReset();

	return 0;
}


bool loadImage(const char *filename, Image *img)
{
	FILE *fp = NULL;
	char buf[16];
	cudaError_t err;

	// initialize
	img->fmt = Format::DEFAULT; img->w = 0; img->h = 0; img->pix_maxval = 0; img->img = NULL;

	if (!(fp = fopen(filename, "rb")))
	{
		fprintf( stderr, "cannot open file \"%s\".\n", filename );
		return -1;
	}

	if (!fgets(buf, sizeof(buf), fp))
	{
		fprintf( stderr, "failed to load file.\n" );
		return false;
	}

	if (buf[0] != 'P')
	{
		fprintf( stderr, "invalid image format.\n" );
		return false;
	}
	switch (buf[1])
	{
		case '5':
		{
			img->fmt = Format::PGM;
			break;
		}
		case '6':
		{
			img->fmt = Format::PPM;
			break;
		}
		default:
		{
			fprintf( stderr, "invalid image format.\n" );
			return false;
		}
	}

	// read image size
	while(1)
	{
		if (!fgets(buf, sizeof(buf), fp))
		{
			fprintf( stderr, "failed to load file.\n" );
			return false;
		}
		if (buf[0] != '#') break;
	}
	sscanf(buf, "%lu %lu", &(img->w), &(img->h));

	// read max value
	while(1)
	{
		if (!fgets(buf, sizeof(buf), fp))
		{
			fprintf( stderr, "failed to load file.\n" );
			return false;
		}
		if (buf[0] != '#') break;
	}
	sscanf(buf, "%d", &(img->pix_maxval));

	char ch;
	while ((ch=fgetc(fp)) != '\n')
	{
		if (ch == EOF)
		{
			fprintf( stderr, "failed to load file.\n" );
			return false;
		}
	}

	if ((img->img = (unsigned char**)malloc( sizeof(char*) * img->h )) == NULL)
	{
		fprintf( stderr, "fail to allocate memory while loading image\n" );
		return false;
	}

	switch (img->fmt)
	{
		case Format::PGM:
		{
			for (int i=0; i<img->h; ++i)
			{
				if ((img->img[i] = (unsigned char*)malloc( img->w )) == NULL)
				{
					fprintf( stderr, "fail to allocate memory while loading image\n" );
					return false;
				}
				fread( img->img[i], sizeof(unsigned char), img->w, fp);
			}
			break;
		}
		case Format::PPM:
		{
			for (int i=0; i<img->h; ++i)
			{
				if ((img->img[i] = (unsigned char*)malloc(3 * img->w)) == NULL)
				{
					fprintf( stderr, "fail to allocate memory while loading image:\n" );
					return false;
				}
				fread(img->img[i], sizeof(unsigned char) * 3, img->w, fp);
			}
			break;
		}
		default:
		{
			fprintf( stderr, "invalid image format.\n" );
			return false;
		}
	}

	fclose(fp);

	return true;
}


bool writeImage( const char *filename, const Image img, int fmt )
{
	FILE *fp = NULL;

	if (!(fp = fopen( filename, "wb" )))
	{
		fprintf( stderr, "cannot open file %s.\n", filename );
		return false;
	}

	switch (fmt)
	{
		case Format::PGM:
		{
			fprintf( fp, "P5\n%d %d\n%d\n", img.w, img.h, img.pix_maxval );
			for (int i=0; i<img.h; ++i)
			{
				fwrite( img.img[i], sizeof(unsigned char), img.w, fp );
				break;
			}
		}
		case Format::PPM:
		{
			fprintf( fp, "P6\n%d %d\n%d\n", img.w, img.h, img.pix_maxval );
			for (int i=0; i<img.h; ++i)
			{
				fwrite( img.img[i], sizeof(unsigned char) * 3, img.w, fp );
				break;
			}
		}
		default:
			break;
	}

	fclose(fp);

	return true;
}


bool ppmtopgm( Image *img )
{
	unsigned char **pgm;

	if (img->fmt == Format::PGM) return true;

	if ((pgm = (unsigned char**)malloc( img->h * sizeof(char*) )) == NULL)
	{
		fprintf( stderr, "fail to allocate memory.\n" );
		return false;
	}

	for (int i=0; i<img->h; ++i)
	{
		if ((pgm[i] = (unsigned char*)malloc( img->w * sizeof(unsigned char) )) == NULL)
		{
			fprintf( stderr, "fail to allocate memory.\n" );
			return false;
		}

		for (int j=0; j<img->w; ++j)
		{
			double px;
			px = img->img[i][j*3 + 0] * .299f
			   + img->img[i][j*3 + 1] * .587f
			   + img->img[i][j*3 + 2] * .114f;
			pgm[i][j] = (unsigned char)fmod( px, img->pix_maxval + 1. );
		}

		free( img->img[i] );
	}

	img->fmt = Format::PGM;
	img->img = pgm;

	return true;
}


void freeImage( Image *img )
{
	for (int i=0; i<img->h; ++i)
	{
		free( img->img[i] );
		img->img[i] = NULL;
	}
	free(img);
}


unsigned int reductionCPU( const Image img )
{
	unsigned int result = 0;
	for (int i=0; i<img.h; ++i)
	{
		for (int j=0; j<img.w; ++j)
		{
			result += img.img[i][j];
		}
	}
	return result;
}


unsigned int reductionGPU( const Image img, int method )
{
	unsigned char *d_img = NULL;
	unsigned int result = 0;
	unsigned int *dp_result;
	dim3 gridDim = dim3( (img.w + N_THREADS - 1) / N_THREADS, img.h );
	dim3 blockDim = dim3( N_THREADS );

	__CUDA( cudaMalloc( &dp_result, sizeof(int) ) );
	__CUDA( cudaMemset( dp_result, 0, sizeof(int) ) );

	size_t pitch = 0;
	switch (method)
	{
		case 0:
		{
			__CUDA( cudaMallocPitch( &d_img, &pitch, img.w, img.h ) );
			__CUDA( cudaMemset2D( d_img, pitch, 0, img.w, img.h ) );
			__CUDA( cudaMemcpy2D( d_img, pitch, img.img[0], img.w,
						img.w, img.h, cudaMemcpyHostToDevice ) );
			reduction_naive<<< 1, 1 >>>( d_img, img.w, img.h, pitch, dp_result );
			__CUDA( cudaGetLastError() );
			__CUDA( cudaFree( d_img ) );
			break;
		}
		case 1:
		{
			int smems = N_THREADS * sizeof(int);
			__CUDA( cudaMallocPitch( &d_img, &pitch, img.w, img.h ) );
			__CUDA( cudaMemset2D( d_img, pitch, 0, img.w, img.h ) );
			__CUDA( cudaMemcpy2D( d_img, pitch, img.img[0], img.w,
						img.w, img.h, cudaMemcpyHostToDevice ) );
			reduction_smem<<< gridDim, blockDim, smems >>>( d_img, img.w, img.h, pitch, dp_result );
			__CUDA( cudaGetLastError() );
			__CUDA( cudaFree( d_img ) );
			break;
		}
		case 2:
		{
			__CUDA( cudaMallocPitch( &d_img, &pitch, img.w, img.h ) );
			__CUDA( cudaMemset2D( d_img, pitch, 0, img.w, img.h ) );
			__CUDA( cudaMemcpy2D( d_img, pitch, img.img[0], img.w,
						img.w, img.h, cudaMemcpyHostToDevice ) );
			reduction_shfl1<<< gridDim, blockDim >>>( d_img, img.w, img.h, pitch, dp_result );
			__CUDA( cudaGetLastError() );
			__CUDA( cudaFree( d_img ) );
			break;
		}
		case 3:
		{
			gridDim = dim3( (img.w + N_THREADS*4 - 1) / (N_THREADS*4), img.h );
			int w = gridDim.x * N_THREADS * sizeof(uchar4);
			__CUDA( cudaMallocPitch( &d_img, &pitch, w, img.h ) );
			__CUDA( cudaMemset2D( d_img, pitch, 0, w, img.h ) );
			__CUDA( cudaMemcpy2D( d_img, pitch, img.img[0], img.w,
						img.w, img.h, cudaMemcpyHostToDevice ) );
			reduction_shfl2<<< gridDim, blockDim >>>( d_img, img.w, img.h, pitch, dp_result );
			__CUDA( cudaGetLastError() );
			__CUDA( cudaFree( d_img ) );
			break;
		}
		case 4:
		{
			gridDim = dim3( (img.w + N_THREADS*16 - 1) / (N_THREADS*16), img.h );
			int w = gridDim.x * N_THREADS * sizeof(uint4);
			__CUDA( cudaMallocPitch( &d_img, &pitch, w, img.h ) );
			__CUDA( cudaMemset2D( d_img, pitch, 0, w, img.h ) );
			__CUDA( cudaMemcpy2D( d_img, pitch, img.img[0], img.w,
						img.w, img.h, cudaMemcpyHostToDevice ) );
			reduction_shfl3<<< gridDim, blockDim >>>( d_img, img.w, img.h, pitch, dp_result );
			__CUDA( cudaGetLastError() );
			__CUDA( cudaFree( d_img ) );
			break;
		}
		case 5:
		{
			gridDim = dim3( (img.w + N_THREADS*4 - 1) / (N_THREADS*4), img.h );
			__CUDA( cudaMallocPitch( &d_img, &pitch, img.w, img.h ) );
			__CUDA( cudaMemset2D( d_img, pitch, 0, img.w, img.h ) );
			__CUDA( cudaMemcpy2D( d_img, pitch, img.img[0], img.w,
						img.w, img.h, cudaMemcpyHostToDevice ) );
			cudaResourceDesc resDesc;
			memset( &resDesc, 0, sizeof(resDesc) );
			resDesc.resType = cudaResourceTypePitch2D;
			resDesc.res.pitch2D.devPtr = d_img;
			resDesc.res.pitch2D.desc.f = cudaChannelFormatKindUnsigned;
			resDesc.res.pitch2D.desc.x = 8;
			resDesc.res.pitch2D.desc.y = 8;
			resDesc.res.pitch2D.desc.z = 8;
			resDesc.res.pitch2D.desc.w = 8;
			resDesc.res.pitch2D.pitchInBytes = pitch;
			resDesc.res.pitch2D.width  = (img.w + 4 - 1) / 4;
			resDesc.res.pitch2D.height = img.h;

			cudaTextureDesc texDesc;
			memset( &texDesc, 0, sizeof(texDesc) );
			texDesc.addressMode[0] = cudaAddressModeBorder;
			texDesc.addressMode[1] = cudaAddressModeBorder;
			texDesc.readMode = cudaReadModeElementType;

			cudaTextureObject_t texObj = 0;
			__CUDA( cudaCreateTextureObject( &texObj, &resDesc, &texDesc, NULL ) );

			reduction_shfl4<<< gridDim, blockDim >>>( texObj, dp_result );

			__CUDA( cudaGetLastError() );
			__CUDA( cudaDestroyTextureObject( texObj ) );
			__CUDA( cudaFree( d_img ) );
			break;
		}
		case 6:
		default:
		{
			gridDim = dim3( (img.w + N_THREADS*16 - 1) / (N_THREADS*16), img.h );
			__CUDA( cudaMallocPitch( &d_img, &pitch, img.w, img.h ) );
			__CUDA( cudaMemset2D( d_img, pitch, 0, img.w, img.h ) );
			__CUDA( cudaMemcpy2D( d_img, pitch, img.img[0], img.w,
						img.w, img.h, cudaMemcpyHostToDevice ) );
			cudaResourceDesc resDesc;
			memset( &resDesc, 0, sizeof(resDesc) );
			resDesc.resType = cudaResourceTypePitch2D;
			resDesc.res.pitch2D.devPtr = d_img;
			resDesc.res.pitch2D.desc.f = cudaChannelFormatKindUnsigned;
			resDesc.res.pitch2D.desc.x = 32;
			resDesc.res.pitch2D.desc.y = 32;
			resDesc.res.pitch2D.desc.z = 32;
			resDesc.res.pitch2D.desc.w = 32;
			resDesc.res.pitch2D.pitchInBytes = pitch;
			resDesc.res.pitch2D.width  = (img.w + 16 - 1) / 16;
			resDesc.res.pitch2D.height = img.h;

			cudaTextureDesc texDesc;
			memset( &texDesc, 0, sizeof(texDesc) );
			texDesc.addressMode[0] = cudaAddressModeBorder;
			texDesc.addressMode[1] = cudaAddressModeBorder;
			texDesc.readMode = cudaReadModeElementType;

			cudaTextureObject_t texObj = 0;
			__CUDA( cudaCreateTextureObject( &texObj, &resDesc, &texDesc, NULL ) );

			reduction_shfl5<<< gridDim, blockDim >>>( texObj, dp_result );

			__CUDA( cudaGetLastError() );
			__CUDA( cudaDestroyTextureObject( texObj ) );
			__CUDA( cudaFree( d_img ) );
			break;
		}
	}

	__CUDA( cudaMemcpy( &result, dp_result, sizeof(int), cudaMemcpyDeviceToHost ) );
	__CUDA( cudaFree( dp_result ) );

	return result;
}


void freeIntImg( unsigned int **intimg )
{
	free( intimg[0] );
	free( intimg );
}


bool integralImageCPU( const Image img, unsigned int ***intimg )
{
	unsigned int **p_intg = NULL;
	if (!(p_intg = (unsigned int**)malloc( img.h * sizeof(unsigned int*) )))
	{
		fprintf( stderr, "fail to allocate memory.\n" );
		return false;
	}
	if (!(p_intg[0] = (unsigned int*)malloc( img.w * img.h * sizeof(unsigned int) )))
	{
		fprintf( stderr, "fail to allocate memory.\n" );
		return false;
	}
	for (int i=1; i<img.h; ++i)
	{
		p_intg[i] = &(p_intg[0][i * img.w]);
	}

	for (int i=0; i<img.h; ++i)
	{
		for (int j=0; j<img.w; ++j)
		{
			unsigned int u_val, l_val, ul_val, c_val;
			(i > 0) ? (u_val = p_intg[i-1][j]) : (u_val = 0);
			(j > 0) ? (l_val = p_intg[i][j-1]) : (l_val = 0);
			((i>0) && (j>0)) ? (ul_val = p_intg[i-1][j-1]) : (ul_val = 0);
			c_val = img.img[i][j];

			p_intg[i][j] = u_val + l_val + c_val - ul_val;
		}
	}

	*intimg = p_intg;

	return true;
}


bool integralImageGPU( const Image img, unsigned int ***intimg, int method )
{
	unsigned int **p_intg = NULL;
	unsigned int *dp_intg = NULL;
	unsigned char *d_img  = NULL;
	dim3 gridDim = dim3( (img.w + N_THREADS - 1) / N_THREADS, img.h );
	dim3 blockDim = dim3( N_THREADS );

	// allocate host memory for integral image output
	if (!(p_intg = (unsigned int**)malloc( img.h * sizeof(unsigned int*) )))
	{
		fprintf( stderr, "fail to allocate memory.\n" );
		return false;
	}
	if (!(p_intg[0] = (unsigned int*)malloc( img.w * img.h * sizeof(unsigned int) )))
	{
		fprintf( stderr, "fail to allocate memory.\n" );
		return false;
	}
	memset( p_intg[0], 0, img.w * img.h * sizeof(unsigned int) );
	for (int i=1; i<img.h; ++i)
	{
		p_intg[i] = &(p_intg[0][i * img.w]);
	}

	size_t ipitch = 0;
	size_t opitch = 0;
	switch (method)
	{
		case 0: // naive
		{
			gridDim = dim3( (img.h + N_THREADS - 1) / N_THREADS, 1 );
			// allocate device memory for input data
			__CUDA( cudaMallocPitch( &d_img, &ipitch, img.w, img.h ) );
			__CUDA( cudaMemset2D( d_img, ipitch, 0, img.w, img.h ) );
			__CUDA( cudaMemcpy2D( d_img, ipitch, img.img[0], img.w,
						img.w, img.h, cudaMemcpyHostToDevice ) );
			// allocate device memory for output data
			__CUDA( cudaMallocPitch( &dp_intg, &opitch, img.w * sizeof(int), img.h ) );
			__CUDA( cudaMemset2D( dp_intg, opitch, 0, img.w, img.h ) );
			integral_row_naive<<< gridDim, blockDim >>>( d_img, img.w, img.h,
						ipitch, opitch, dp_intg );
			#ifdef _DEBUG
			__CUDA( cudaDeviceSynchronize() );
			#endif
			gridDim = dim3( (img.w + N_THREADS - 1) / N_THREADS, 1 );
			integral_col_naive <<< gridDim, blockDim >>>(d_img, img.w, img.h,
						ipitch, opitch, dp_intg );
			#ifdef _DEBUG
			__CUDA( cudaDeviceSynchronize() );
			#endif
			__CUDA( cudaMemcpy2D( p_intg[0], img.w * sizeof(int), dp_intg, opitch,
						img.w * sizeof(int), img.h, cudaMemcpyDeviceToHost ) );
			__CUDA( cudaFree( d_img ) );
			__CUDA( cudaFree( dp_intg ) );
			break;
		}
		case 1: // shared memory
		{
			//break;
		}
		case 2: // shuffle instruction
		{
			// allocate device memory for input data
			__CUDA( cudaMallocPitch( &d_img, &ipitch, img.w, img.h ) );
			__CUDA( cudaMemset2D( d_img, ipitch, 0, img.w, img.h ) );
			__CUDA( cudaMemcpy2D( d_img, ipitch, img.img[0], img.w,
						img.w, img.h, cudaMemcpyHostToDevice ) );
			// allocate device memory for output data
			__CUDA( cudaMallocPitch( &dp_intg, &opitch, img.w * sizeof(int), img.h ) );
			__CUDA( cudaMemset2D( dp_intg, opitch, 0, img.w, img.h ) );
			unsigned int *row_block_sum = NULL;
			__CUDA( cudaMalloc( &row_block_sum, gridDim.x * sizeof(int) * img.h ) );
			__CUDA( cudaMemset( row_block_sum, 0, gridDim.x * sizeof(int) * img.h ) );
			// call kernel to integral in row
			integral_row_shfl<<< gridDim, blockDim >>>( d_img, img.w, img.h,
						ipitch, opitch, dp_intg, row_block_sum );
			integral_row_shfl_uniform<<< dim3(1, img.h), gridDim.x >>>( row_block_sum );
			integral_row_shfl_apply<<< gridDim, blockDim >>>( img.w, img.h, opitch, dp_intg, row_block_sum );
			#ifdef _DEBUG
			__CUDA( cudaDeviceSynchronize() );
			#endif

			// integral in col
			// allocate device memory for inter-block integral
			gridDim = dim3( (img.h + N_THREADS - 1) / N_THREADS, img.w );
			unsigned int *col_block_sum = NULL;
			__CUDA( cudaMalloc( &col_block_sum, gridDim.x * sizeof(int) * img.w ) );
			__CUDA( cudaMemset( col_block_sum, 0, gridDim.x * sizeof(int) * img.w ) );
			integral_col_shfl<<< gridDim, blockDim >>>( d_img, img.w, img.h,
						ipitch, opitch, dp_intg, col_block_sum );
			integral_col_shfl_uniform<<< dim3(1, img.w), gridDim.x >>>( col_block_sum );
			integral_col_shfl_apply<<< gridDim, blockDim >>>( img.w, img.h, opitch, dp_intg, col_block_sum );
			#ifdef _DEBUG
			__CUDA( cudaDeviceSynchronize() );
			#endif
			__CUDA( cudaMemcpy2D( p_intg[0], img.w * sizeof(int), dp_intg, opitch,
						img.w * sizeof(int), img.h, cudaMemcpyDeviceToHost ) );
			__CUDA( cudaFree( d_img ) );
			__CUDA( cudaFree( dp_intg ) );
			__CUDA( cudaFree( row_block_sum ) );
			__CUDA( cudaFree( col_block_sum ) );
			break;
		}
	}

	*intimg = p_intg;

	return true;
}


__global__ void reduction_naive( const unsigned char* d_img, size_t w, size_t h, size_t pitch, unsigned int *result )
{
	unsigned int sum = 0;
	for (int i=0; i<h; ++i)
	{
		for (int j=0; j<w; ++j)
		{
			sum += d_img[i * pitch + j];
		}
	}
	*result = sum;
}


__global__ void reduction_smem( const unsigned char* d_img, size_t w, size_t h, size_t pitch, unsigned int *result )
{
	unsigned int tid = threadIdx.x;
	unsigned int lane_id = tid % warpSize;
	unsigned int x_idx = blockIdx.x * blockDim.x + tid;
	unsigned int y_idx = blockIdx.y;
	extern __shared__ unsigned int smem[];
	//__shared__ unsigned int sum;

	//if (tid == 0) sum = 0;
	smem[tid] = 0;
	if (x_idx < w)
	{
		smem[tid] = d_img[y_idx * pitch + x_idx];
	}
	__syncthreads();

#pragma unroll
	for (int mask = warpSize/2; mask>0; mask >>= 1)
	{
		if (lane_id < mask)
		{
			smem[tid] += smem[tid ^ mask];
		}
		__syncthreads();
	}

	//if (lane_id == 0) sum += smem[tid];
	//__syncthreads();

	if (lane_id == 0) atomicAdd( result, smem[tid] );
}


__global__ void reduction_shfl1( const unsigned char* d_img, size_t w, size_t h, size_t pitch, unsigned int *result )
{
	unsigned int tid = threadIdx.x;
	unsigned int lane_id = tid % warpSize;
	unsigned int x_idx = blockIdx.x * blockDim.x + tid;
	unsigned int y_idx = blockIdx.y;
	unsigned int val = 0;
	unsigned int sum = 0;

	if (x_idx < w) val = d_img[y_idx * pitch + x_idx];

	sum = warp_shuffle_xor( val, lane_id );
	if (lane_id == 0) atomicAdd( result, sum );
}


__global__ void reduction_shfl2( const unsigned char* d_img, size_t w, size_t h, size_t pitch, unsigned int *result )
{
	unsigned int tid = threadIdx.x;
	unsigned int lane_id = tid % warpSize;
	unsigned int x_idx = blockIdx.x * blockDim.x + tid;
	unsigned int y_idx = blockIdx.y;
	unsigned int val = 0;
	unsigned int sum = 0;
	unsigned int _pitch = pitch / sizeof(uchar4);

	uchar4 data = ((uchar4*)d_img)[y_idx * _pitch + x_idx];
	val = data.x + data.y + data.z + data.w;

	sum = warp_shuffle_xor( val, lane_id );
	if (lane_id == 0) atomicAdd( result, sum );
}


__global__ void reduction_shfl3( const unsigned char* __restrict__ d_img, size_t w, size_t h, size_t pitch, unsigned int *result )
{
	unsigned int tid = threadIdx.x;
	unsigned int lane_id = tid % warpSize;
	unsigned int x_idx = blockIdx.x * blockDim.x + tid;
	unsigned int y_idx = blockIdx.y;
	unsigned int val = 0;
	unsigned int sum = 0;
	unsigned int _pitch = pitch / sizeof(uint4);

	uint4 data = ((uint4*)d_img)[y_idx * _pitch + x_idx];
	val = sum_uint4_uchar( data );

	sum = warp_shuffle_xor( val, lane_id );
	if (lane_id == 0) atomicAdd( result, sum );
}


__global__ void reduction_shfl4( cudaTextureObject_t tex, unsigned int *result )
{
	unsigned int tid = threadIdx.x;
	unsigned int lane_id = tid % warpSize;
	unsigned int x_idx = blockIdx.x * blockDim.x + tid;
	unsigned int y_idx = blockIdx.y;
	unsigned int val = 0;
	unsigned int sum = 0;

	uchar4 data = tex2D<uchar4>( tex, x_idx, y_idx );
	val = data.x + data.y + data.z + data.w;

	sum = warp_shuffle_xor( val, lane_id );
	if (lane_id == 0) atomicAdd( result, sum );
}


__global__ void reduction_shfl5( cudaTextureObject_t tex, unsigned int *result )
{
	unsigned int tid = threadIdx.x;
	unsigned int lane_id = tid % warpSize;
	unsigned int x_idx = blockIdx.x * blockDim.x + tid;
	unsigned int y_idx = blockIdx.y;
	unsigned int val = 0;
	unsigned int sum = 0;

	uint4 data = tex2D<uint4>( tex, x_idx, y_idx );
	val = sum_uint4_uchar( data );

	sum = warp_shuffle_xor( val, lane_id );
	if (lane_id == 0) atomicAdd( result, sum );
}


__global__ void integral_row_naive( const unsigned char* img, size_t w,
					size_t h, size_t ipitch, size_t opitch, unsigned int *intimg )
{
	unsigned int y_idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int val = 0;
	unsigned int _opitch = opitch / sizeof(int);
	if (y_idx >= h) return;
	for (int i=0; i<w; ++i)
	{
		val += img[y_idx * ipitch + i];
		intimg[y_idx * _opitch + i] = val;
	}
}


__global__ void integral_col_naive( const unsigned char* img, size_t w,
					size_t h, size_t ipitch, size_t opitch, unsigned int *intimg )
{
	unsigned int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int val = 0;
	unsigned int _opitch = opitch / sizeof(int);
	if (x_idx >= w) return;
	for (int i=0; i<h; ++i)
	{
		val += intimg[i * _opitch + x_idx];
		intimg[i * _opitch + x_idx] = val;
	}
}


__global__ void integral_row_shfl( const unsigned char* img, size_t w, size_t h,
	 				size_t ipitch, size_t opitch, unsigned int *intimg, unsigned int *block_sum )
{
	__shared__ unsigned int sum[N_THREADS];

	int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int h_idx = blockIdx.y;
	int lane_id = (blockIdx.x * blockDim.x + threadIdx.x) % warpSize;
	int warp_id = threadIdx.x / warpSize;
	int _opitch = opitch / sizeof(int);

	unsigned int val = img[h_idx * ipitch + w_idx];
	sum[threadIdx.x] = 0;
	__syncthreads();

	val = warp_shuffle_up( val, lane_id );
	if (threadIdx.x % warpSize == warpSize-1)
	{
		sum[warp_id] = val;
	}
	__syncthreads();

	if (warp_id == 0)
	{
		int warp_val = sum[lane_id];
		sum[lane_id] = warp_shuffle_up( warp_val, lane_id );
	}
	__syncthreads();

	int thread_val = 0;
	if (warp_id > 0)
	{
		thread_val = sum[warp_id - 1];
	}

	val += thread_val;

	intimg[h_idx * _opitch + w_idx] = val;

	if (block_sum != NULL && threadIdx.x == blockDim.x - 1)
	{
		block_sum[h_idx * gridDim.x + blockIdx.x] = val;
	}
}


__global__ void integral_col_shfl( const unsigned char* img, size_t w, size_t h,
	 				size_t ipitch, size_t opitch, unsigned int *intimg, unsigned int *block_sum )
{
	__shared__ unsigned int sum[N_THREADS];
	int w_idx = blockIdx.y;
	int h_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int lane_id = h_idx % warpSize;
	int warp_id = threadIdx.x / warpSize;
	int _opitch = opitch / sizeof(int);

	if (h_idx >= h) return;
	unsigned int val = intimg[h_idx * _opitch + w_idx];
	sum[threadIdx.x] = 0;
	__syncthreads();

	val = warp_shuffle_up( val, lane_id );

	if (threadIdx.x % warpSize == warpSize - 1)
	{
		sum[warp_id] = val;
	}
	__syncthreads();

	if (warp_id == 0)
	{
		int warp_val = sum[lane_id];
		sum[lane_id] = warp_shuffle_up( warp_val, lane_id );
	}
	__syncthreads();

	int thread_val = 0;
	if (warp_id > 0)
	{
		thread_val = sum[warp_id - 1];
	}

	val += thread_val;

	intimg[h_idx * _opitch + w_idx] = val;

	if (threadIdx.x == blockDim.x - 1)
	{
		block_sum[blockIdx.y * gridDim.x + blockIdx.x] = val;
	}
}


__global__ void integral_row_shfl_uniform( unsigned int *block_sum )
{
	__shared__ unsigned int sum[N_THREADS];

	int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int h_idx = blockIdx.y;
	int lane_id = (blockIdx.x * blockDim.x + threadIdx.x) % warpSize;
	int warp_id = threadIdx.x / warpSize;

	unsigned int val = block_sum[h_idx * blockDim.x + threadIdx.x];
	sum[threadIdx.x] = 0;
	__syncthreads();

	val = warp_shuffle_up( val, lane_id );
	if (threadIdx.x % warpSize == warpSize-1)
	{
		sum[warp_id] = val;
	}
	__syncthreads();

	if (warp_id == 0)
	{
		int warp_val = sum[lane_id];
		sum[lane_id] = warp_shuffle_up( warp_val, lane_id );
	}
	__syncthreads();

	int thread_val = 0;
	if (warp_id > 0)
	{
		thread_val = sum[warp_id - 1];
	}

	val += thread_val;

	block_sum[blockIdx.y * blockDim.x + threadIdx.x] = val;

}


__global__ void integral_col_shfl_uniform( unsigned int *block_sum )
{
	__shared__ unsigned int sum[N_THREADS];

	int w_idx = blockIdx.y;
	int h_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int lane_id = h_idx % warpSize;
	int warp_id = threadIdx.x / warpSize;

	unsigned int val = block_sum[blockIdx.y * blockDim.x + threadIdx.x];
	sum[threadIdx.x] = 0;
	__syncthreads();

	val = warp_shuffle_up( val, lane_id );

	if (threadIdx.x % warpSize == warpSize - 1)
	{
		sum[warp_id] = val;
	}
	__syncthreads();

	if (warp_id == 0)
	{
		int warp_val = sum[lane_id];
		sum[lane_id] = warp_shuffle_up( warp_val, lane_id );
	}
	__syncthreads();

	int thread_val = 0;
	if (warp_id > 0)
	{
		thread_val = sum[warp_id - 1];
	}

	val += thread_val;

	block_sum[blockIdx.y * blockDim.x + threadIdx.x] = val;

}

__global__ void integral_row_shfl_apply( size_t w, size_t h, size_t pitch,
	 													unsigned int *intimg, unsigned int *block_sum )
{
	int w_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int h_idx = blockIdx.y;
	int _pitch = pitch / sizeof(int);

	if (blockIdx.x > 0)
	{
		intimg[h_idx * _pitch + w_idx] += block_sum[blockIdx.y * gridDim.x + blockIdx.x-1];
	}
}


__global__ void integral_col_shfl_apply( size_t w, size_t h, size_t pitch,
	 													unsigned int *intimg, unsigned int *block_sum )
{
	int w_idx = blockIdx.y;
	int h_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int _pitch = pitch / sizeof(int);

	if (blockIdx.x > 0 && h_idx < h)
	{
		intimg[h_idx * _pitch + w_idx] += block_sum[blockIdx.y * gridDim.x + blockIdx.x-1];
	}
}
