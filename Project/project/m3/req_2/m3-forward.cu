#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16
#define BLOCK_SIZE 256

__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
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
        float acc = 0.0f;
        
        for (int c = 0; c < Channel; c++) {
            for (int p = 0; p < K; p++) {
                for (int q = 0; q < K; q++) {
                    int h_in = h_out + p;
                    int w_in = w_out + q;
                    if (h_in < Height && w_in < Width) {
                        float input_val = input[(batch_idx * Channel * Height * Width) + 
                                              (c * Height * Width) + 
                                              (h_in * Width) + 
                                              w_in];
                        float mask_val = mask[(map_idx * Channel * K * K) + 
                                            (c * K * K) + 
                                            (p * K) + 
                                            q];
                        acc += input_val * mask_val;
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

    cudaMalloc((void **) device_input_ptr, inputSize * sizeof(float));
    cudaMalloc((void **) device_output_ptr, outputSize * sizeof(float));
    cudaMalloc((void **) device_mask_ptr, maskSize * sizeof(float));

    cudaMemcpy(*device_input_ptr, host_input, inputSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, maskSize * sizeof(float), cudaMemcpyHostToDevice);

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

    dim3 gridDim((Height_out * Width_out + TILE_WIDTH - 1) / TILE_WIDTH, Batch, Map_out);
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    
    conv_forward_kernel<<<gridDim, blockDim>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Copy the output back to host
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    int outputSize = Batch * Map_out * Width_out * Height_out;

    cudaMemcpy(host_output, device_output, outputSize * sizeof(float), cudaMemcpyDeviceToHost);

    // TODO: Free device memory
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