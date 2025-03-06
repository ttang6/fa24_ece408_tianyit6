#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <cuda_fp16.h>

#define TILE_WIDTH 16
#define BLOCK_SIZE 256

__global__ void conv_forward_kernel(half *output, const half *input, const half *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int batch_idx = blockIdx.y;
    const int map_idx = blockIdx.z;
    const int h_out = blockIdx.x / Width_out * TILE_WIDTH + ty;
    const int w_out = blockIdx.x % Width_out * TILE_WIDTH + tx;
    
    if (h_out < Height_out && w_out < Width_out) {
        half acc = __float2half(0.0f);
        
        for (int c = 0; c < Channel; c++) {
            for (int p = 0; p < K; p++) {
                for (int q = 0; q < K; q++) {
                    int h_in = h_out + p;
                    int w_in = w_out + q;
                    if (h_in < Height && w_in < Width) {
                        half input_val = input[(batch_idx * Channel * Height * Width) + 
                                             (c * Height * Width) + 
                                             (h_in * Width) + 
                                             w_in];
                        half mask_val = mask[(map_idx * Channel * K * K) + 
                                           (c * K * K) + 
                                           (p * K) + 
                                           q];
                        acc = __hfma(input_val, mask_val, acc);
                    }
                }
            }
        }
        
        output[(batch_idx * Map_out * Height_out * Width_out) + 
               (map_idx * Height_out * Width_out) + 
               (h_out * Width_out) + 
               w_out] = acc;
    }
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    int outputSize = Batch * Map_out * Height_out * Width_out;
    int inputSize = Batch * Channel * Width * Height;
    int maskSize = Channel * Map_out * K * K;

    half *device_input_half, *device_mask_half, *device_output_half;
    cudaMalloc((void **)&device_input_half, inputSize * sizeof(half));
    cudaMalloc((void **)&device_mask_half, maskSize * sizeof(half));
    cudaMalloc((void **)&device_output_half, outputSize * sizeof(half));

    half *host_input_half = new half[inputSize];
    half *host_mask_half = new half[maskSize];

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

    delete[] host_input_half;
    delete[] host_mask_half;

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    dim3 DimGrid;
    dim3 DimBlock;

    DimBlock = dim3(TILE_WIDTH, TILE_WIDTH, 1);
    DimGrid = dim3(ceil(Height_out * Width_out / (1.0 * TILE_WIDTH)), Batch, Map_out);
    
    half *device_output_half = (half*)device_output;
    half *device_input_half = (half*)device_input;
    half *device_mask_half = (half*)device_mask;
    
    conv_forward_kernel<<<DimGrid, DimBlock>>>(device_output_half, device_input_half, device_mask_half, Batch, Map_out, Channel, Height, Width, K);
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    int outputSize = Batch * Map_out * Width_out * Height_out;

    half *host_output_half = new half[outputSize];
    
    cudaMemcpy(host_output_half, (half*)device_output, outputSize * sizeof(half), cudaMemcpyDeviceToHost);

    for(int i = 0; i < outputSize; i++) {
        host_output[i] = __half2float(host_output_half[i]);
    }

    delete[] host_output_half;
    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}