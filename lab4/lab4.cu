#include <wb.h>
#include <iostream>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define MASK_WIDTH 3
#define TILE_WIDTH 8


//@@ Define constant memory for device kernel here
__constant__ float Mc[MASK_WIDTH][MASK_WIDTH][MASK_WIDTH];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  const int N_width = TILE_WIDTH + MASK_WIDTH - 1;
  __shared__ float N_ds[N_width][N_width][N_width];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
  // printf("tidx= %d %d %d\n", tx, ty, tz);
  // printf("value= %d\n", threadIdx.x);

  int col_o = blockIdx.x * TILE_WIDTH + tx;
  int row_o = blockIdx.y * TILE_WIDTH + ty;
  int hei_o = blockIdx.z * TILE_WIDTH + tz;
  // printf("out= %d %d %d\n", row, col, hei);

  int col_i = col_o - MASK_WIDTH / 2;
  int row_i = row_o - MASK_WIDTH / 2;
  int hei_i = hei_o - MASK_WIDTH / 2;
  // printf("in= %d %d %d\n", row_i, col_i, hei_i);
  
  float Pvalue = 0.0f;

  if((col_i >= 0) && (col_i < x_size) &&
    (row_i >= 0) && (row_i < y_size) &&
    (hei_i >= 0) && (hei_i < z_size))
  {
    int i = hei_i * y_size * x_size + row_i * x_size + col_i;
    N_ds[tz][ty][tx] = input[i];
  }
  else
  {
    N_ds[tz][ty][tx] = 0.0f;
  }

  __syncthreads();


  if(tz < TILE_WIDTH && ty < TILE_WIDTH && tx < TILE_WIDTH)
  {
    for(int i = 0; i < MASK_WIDTH; i++)
    {
      for(int j = 0; j < MASK_WIDTH; j++)
      {
        for(int k = 0; k < MASK_WIDTH; k++)
        {
          Pvalue += Mc[i][j][k] * N_ds[i+tz][j+ty][k+tx];
        }
      }
    }
    if(col_o < x_size && row_o < y_size && hei_o < z_size)
    {
      // printf("pval: %.2f\n", Pvalue);
      output[hei_o*y_size*x_size + row_o*x_size + col_o] = Pvalue;
    }
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  //@@ Initial deviceInput and deviceOutput here.
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  int size = (inputLength - 3) * sizeof(float);
  int sizeK = kernelLength * sizeof(float);

  cudaMalloc((void **) &deviceInput, size);
  cudaMalloc((void **) &deviceOutput, size);

  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpy(deviceInput, &hostInput[3], size, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(Mc, hostKernel, sizeK);

  //@@ Initialize grid and block dimensions here
  dim3 DimBlock(TILE_WIDTH+MASK_WIDTH-1, TILE_WIDTH+MASK_WIDTH-1, TILE_WIDTH+MASK_WIDTH-1);
  dim3 DimGrid(ceil(x_size / (1.0*TILE_WIDTH)), ceil(y_size / (1.0*TILE_WIDTH)), ceil(z_size / (1.0*TILE_WIDTH)));

  //@@ Launch the GPU kernel here
  conv3d<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  cudaDeviceSynchronize();

  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(&hostOutput[3], deviceOutput, size, cudaMemcpyDeviceToHost);

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  //@@ Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);

  return 0;
}

