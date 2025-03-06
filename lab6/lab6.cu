// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>
#include <iostream>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float *output, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float T[2*BLOCK_SIZE];
  int i = 2 * (blockDim.x * blockIdx.x + threadIdx.x);
  int j = i + 1;

  if(i < len)
    T[2 * threadIdx.x] = input[i];
  else
    T[2 * threadIdx.x] = 0.0f;
  if(j < len)
    T[2 * threadIdx.x + 1] = input[j];
  else
    T[2 * threadIdx.x + 1] = 0.0f;

  __syncthreads();
  
  int stride = 1;
  while(stride < 2*BLOCK_SIZE)
  {
    __syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if(index < 2*BLOCK_SIZE && (index-stride) >= 0)
      T[index] += T[index-stride];
    stride *= 2;
  }

  stride = BLOCK_SIZE / 2;
  while(stride > 0)
  {
    __syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if((index+stride) < 2*BLOCK_SIZE)
      T[index+stride] += T[index];
    stride /= 2;
  }

  __syncthreads();

  if(i < len)
    output[i] = T[2 * threadIdx.x];
  if(j < len)
    output[j] = T[2 * threadIdx.x + 1];
}

__global__ void last_block_num(float *output, float *d_nums, int num_blocks) {
    int i = 2 * (blockDim.x * blockIdx.x + threadIdx.x) + 1;
    
    if ((threadIdx.x == blockDim.x - 1) && (blockIdx.x != (num_blocks - 1)))
        d_nums[blockIdx.x] = output[i];
}

__global__ void block_add(float *output, float *d_nums, int num_blocks, int len) {
    int i = 2 * (blockDim.x * blockIdx.x + threadIdx.x);
    int j = i + 1;

    float val1 = output[i];
    float val2 = output[j];

    float block_sum = 0.0f;
    int k = 0;
    while(k < num_blocks - 1){
      if(blockIdx.x > k)
          block_sum += d_nums[k];
        k++;
    }

    val1 += block_sum;
    val2 += block_sum;

    if (i < len)
        output[i] = val1;
    if (j < len)
        output[j] = val2;
}


int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  float *d_nums;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  // Import data and create memory on host
  // The number of input elements in the input is numElements
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));

  // Allocate GPU memory.
  int num_blocks = ceil(numElements / (2.0*BLOCK_SIZE));
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&d_nums, num_blocks * sizeof(float)));

  // Clear output memory.
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));

  // Copy input memory to the GPU.
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));

  //@@ Initialize the grid and block dimensions here
  // int num_blocks = ceil(numElements / (2*BLOCK_SIZE));
  // printf("%d\n", num_blocks);
  dim3 DimBlock(BLOCK_SIZE, 1, 1);
  dim3 DimGrid(num_blocks, 1, 1);

  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, numElements);
  last_block_num<<<DimGrid, DimBlock>>>(deviceOutput, d_nums, num_blocks);
  block_add<<<DimGrid, DimBlock>>>(deviceOutput, d_nums, num_blocks, numElements);
  cudaDeviceSynchronize();

  // Copying output memory to the CPU
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));

  //@@  Free GPU Memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}

