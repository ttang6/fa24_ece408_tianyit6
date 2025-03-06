#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <cuda_fp16.h>

#define TILE_WIDTH 16
#define BLOCK_SIZE 256

__global__ void matrix_unrolling_kernel(const half *input, half *output,
                                      const int Batch, const int Channel,
                                      const int Height, const int Width,
                                      const int K) {
    /*
    Modify this function to implement the input matrix unrolling kernel.

    Function paramter definitions:
    input - input
    output - output
    Batch - batch_size (number of images in x)
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)

    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    // TODO: Insert your input matrix unrolling kernel code here
    int t = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const size_t W_unroll = Batch * Height_out * Width_out;

    if(t < Channel * W_unroll)
    {
        int b = t / (Channel * Height_out * Width_out);
        int s = t % (Channel * Height_out * Width_out);
        int c = s / (Height_out * Width_out);
        int s2 = s % (Height_out * Width_out);
        int h_out = s2 / Width_out;
        int w_out = s2 % Width_out;

        for(int p = 0; p < K; p++)
        {
            for(int q = 0; q < K; q++)
            {
                int h_unroll = c * K * K + p * K + q;
                int w_unroll = b * (Height_out * Width_out) + h_out * Width_out + w_out;
                output[h_unroll * W_unroll + w_unroll] = in_4d(b, c, h_out + p, w_out + q);
            }
        }
    }

    #undef in_4d
}

__global__ void matrixMultiplyShared(const half *A, const half *B, half *C,
                                   int numARows, int numAColumns,
                                   int numBRows, int numBColumns,
                                   int numCRows, int numCColumns)
{
    __shared__ half tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ half tileB[TILE_WIDTH][TILE_WIDTH];

    int by = blockIdx.y, bx = blockIdx.x, ty = threadIdx.y, tx = threadIdx.x;

    int row = by * TILE_WIDTH + ty, col = bx * TILE_WIDTH + tx;
    half val = __float2half(0.0f);

    for (int tileId = 0; tileId < (numAColumns - 1) / TILE_WIDTH + 1; tileId++) {
        if (row < numARows && tileId * TILE_WIDTH + tx < numAColumns) {
            tileA[ty][tx] = A[(size_t)row * numAColumns + tileId * TILE_WIDTH + tx];
        } else {
            tileA[ty][tx] = __float2half(0.0f);
        }
        if (col < numBColumns && tileId * TILE_WIDTH + ty < numBRows) {
            tileB[ty][tx] = B[((size_t)tileId * TILE_WIDTH + ty) * numBColumns + col];
        } else {
            tileB[ty][tx] = __float2half(0.0f);
        }
        __syncthreads();

        if (row < numCRows && col < numCColumns) {
            for (int i = 0; i < TILE_WIDTH; i++) {
                val = __hadd(val, __hmul(tileA[ty][i], tileB[i][tx]));
            }
        }
        __syncthreads();
    }

    if (row < numCRows && col < numCColumns) {
        C[row * numCColumns + col] = val;
    }
}

__global__ void matrix_permute_kernel(const half *input, half *output, int Map_out,
                                    int Batch, int image_size) {
    int b = blockIdx.y;
    int x = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (x < image_size) {
        for (int m = 0; m < Map_out; m++) {
            output[b * Map_out * image_size + m * image_size + x] =
                input[m * Batch * image_size + b * image_size + x];
        }
    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    int outputSize = Batch * Map_out * Height_out * Width_out;
    int inputSize = Batch * Channel * Width * Height;
    int maskSize = Channel * Map_out * K * K;

    half *device_input_half, *device_mask_half, *device_output_half;
    cudaMalloc((void **)&device_input_half, inputSize * sizeof(half));
    cudaMalloc((void **)&device_mask_half, maskSize * sizeof(half));
    cudaMalloc((void **)&device_output_half, outputSize * sizeof(half));

    half *host_input_half = (half*)malloc(inputSize * sizeof(half));
    half *host_mask_half = (half*)malloc(maskSize * sizeof(half));

    for(int i = 0; i < inputSize; i++) {
        host_input_half[i] = __float2half(host_input[i]);
    }
    for(int i = 0; i < maskSize; i++) {
        host_mask_half[i] = __float2half(host_mask[i]);
    }

    cudaMemcpy(device_input_half, host_input_half, inputSize * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(device_mask_half, host_mask_half, maskSize * sizeof(half), cudaMemcpyHostToDevice);

    *device_input_ptr = (float*)device_input_half;
    *device_mask_ptr = (float*)device_mask_half;
    *device_output_ptr = (float*)device_output_half;

    free(host_input_half);
    free(host_mask_half);
}

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const size_t Height_unrolled = Channel * K * K;
    const size_t Width_unrolled = Batch * Height_out * Width_out;

    half *device_input_half = (half*)device_input;
    half *device_mask_half = (half*)device_mask;
    half *device_output_half = (half*)device_output;

    half *unrolled_matrix;
    half *matmul_output;
    cudaMalloc((void**)&unrolled_matrix, (size_t)Batch * Channel * K * K * Height_out * Width_out * sizeof(half));
    cudaMalloc((void**)&matmul_output, (Batch * Map_out * Height_out * Width_out) * sizeof(half));

    dim3 DimGrid;
    dim3 DimBlock;

    DimBlock = dim3(BLOCK_SIZE, 1, 1);
    DimGrid = dim3(ceil(Batch * Channel * Width_out * Height_out / (1.0 * BLOCK_SIZE)), 1, 1);

    matrix_unrolling_kernel<<<DimGrid, DimBlock>>>(device_input_half, unrolled_matrix, Batch, Channel, Height, Width, K);

    DimBlock = dim3(TILE_WIDTH, TILE_WIDTH, 1);
    DimGrid = dim3(ceil(Width_unrolled / (1.0 * TILE_WIDTH)), ceil(Map_out / (1.0 * TILE_WIDTH)), 1);
    
    matrixMultiplyShared<<<DimGrid, DimBlock>>>(device_mask_half, unrolled_matrix, matmul_output, Map_out, Height_unrolled, Height_unrolled, Width_unrolled, Map_out, Width_unrolled);

    const int out_image_size = Height_out * Width_out;
    dim3 permute_grid_dim((out_image_size - 1) / BLOCK_SIZE + 1, Batch, 1);
    matrix_permute_kernel<<<permute_grid_dim, BLOCK_SIZE>>>(matmul_output, device_output_half, Map_out, Batch, out_image_size);

    cudaFree(matmul_output);
    cudaFree(unrolled_matrix);
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Copy the output back to host
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    int outputSize = Batch * Map_out * Width_out * Height_out;

    half *device_output_half = (half*)device_output;
    half *host_output_half = new half[outputSize];

    cudaMemcpy(host_output_half, device_output_half, outputSize * sizeof(half), cudaMemcpyDeviceToHost);

    for(int i = 0; i < outputSize; i++) {
        host_output[i] = __half2float(host_output_half[i]);
    }

    delete[] host_output_half;
    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);
}