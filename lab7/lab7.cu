// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 256
#define TILE_WIDTH 16
//@@ insert code here
__global__ void float_to_char(unsigned char *ucharImage, float *inputImage, int width, int height, int channel)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int size = width * height * channel;

  if(i < size)
    ucharImage[i] = (unsigned char) (255 * inputImage[i]);
}

__global__ void color_to_grey(unsigned char *grayImage, unsigned char *rgbImage, int width, int height)
{
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;

  if(col < width && row < height)
  {
    int grayOffset = row * width + col;
    int rgbOffset = 3 * grayOffset;
    unsigned char r = rgbImage[rgbOffset];
    unsigned char g = rgbImage[rgbOffset + 1];
    unsigned char b = rgbImage[rgbOffset + 2];
    grayImage[grayOffset] = 0.21f * r + 0.71f * g + 0.07f * b;
  }
}

__global__ void compute_histogram(unsigned char *grayImage, unsigned int *histogram, int width, int height)
{
    __shared__ unsigned int histo_private[HISTOGRAM_LENGTH];

    if(threadIdx.x < HISTOGRAM_LENGTH)
        histo_private[threadIdx.x] = 0;

    __syncthreads();

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i < width * height)
        atomicAdd(&histo_private[grayImage[i]], 1); 

    __syncthreads();

    if(threadIdx.x < HISTOGRAM_LENGTH)
        atomicAdd(&(histogram[threadIdx.x]), histo_private[threadIdx.x]);
}

__global__ void scan(unsigned int *input, float *output, int width, int height) 
{
    __shared__ float T[HISTOGRAM_LENGTH];

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i < HISTOGRAM_LENGTH)
        T[i] = input[i];
    else
        T[i] = 0.0f;

    __syncthreads();

    int stride = 1;
    while(stride < HISTOGRAM_LENGTH) 
    {
        int index = (i + 1) * stride * 2 - 1;
        if (index < HISTOGRAM_LENGTH && (index - stride) >= 0)
            T[index] += T[index - stride];
        stride *= 2;
        __syncthreads();
    }

    stride = HISTOGRAM_LENGTH / 2;
    while(stride > 0) 
    {
        int index = (i + 1) * stride * 2 - 1;
        if ((index + stride) < HISTOGRAM_LENGTH)
            T[index + stride] += T[index];
        stride /= 2;
        __syncthreads();
    }

    if(i < HISTOGRAM_LENGTH)
        output[i] = T[i] / (width * height);
}

__global__ void equalize(unsigned char *ucharImage, float *cdf, int width, int height, int channels)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int size = width * height * channels;

    if(i < size)
        ucharImage[i] = min(max(255.0f * (cdf[ucharImage[i]] - cdf[0]) / (1.0f - cdf[0]), 0.0f), 255.0f);
}

__global__ void char_to_float(unsigned char *ucharImage, float *outputImage, int width, int height, int channel)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int size = width * height * channel;

  if(i < size)
    outputImage[i] = (ucharImage[i] / 255.0f);
}


int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float *d_inputImage;
  float *d_outputImage;
  unsigned char *d_ucharImage;
  unsigned char *d_grayImage;
  float *d_cdf;
  unsigned int *d_histo;


  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  //Import data and create memory on host
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  int size = imageHeight * imageWidth;

  cudaMalloc((void **) &d_inputImage, size * imageChannels * sizeof(float));
  cudaMalloc((void **) &d_outputImage, size * imageChannels * sizeof(float));
  cudaMalloc((void **) &d_ucharImage, size * imageChannels * sizeof(unsigned char));
  cudaMalloc((void **) &d_grayImage, size * sizeof(unsigned char));
  cudaMalloc((void **) &d_histo, HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMalloc((void **) &d_cdf, HISTOGRAM_LENGTH * sizeof(float));
 
  //@@ insert code here
  cudaMemset((void *) d_histo, 0, HISTOGRAM_LENGTH * sizeof(int));
  cudaMemcpy(d_inputImage, hostInputImageData, size * imageChannels * sizeof(float), cudaMemcpyHostToDevice);

  dim3 DimGrid;
  dim3 DimBlock;

  DimBlock = dim3(BLOCK_SIZE, 1, 1);
  DimGrid = dim3(ceil(size * imageChannels / (1.0 * BLOCK_SIZE)), 1, 1);
  float_to_char<<<DimGrid, DimBlock>>>(d_ucharImage, d_inputImage, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();

  DimBlock = dim3(TILE_WIDTH, TILE_WIDTH, 1);
  DimGrid = dim3(ceil(imageWidth / (1.0 * TILE_WIDTH)), ceil(imageHeight / (1.0 * TILE_WIDTH)), 1);
  color_to_grey<<<DimGrid, DimBlock>>>(d_grayImage, d_ucharImage, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  DimBlock = dim3(BLOCK_SIZE, 1, 1);
  DimGrid = dim3(ceil(size / (1.0 * BLOCK_SIZE)), 1, 1);
  compute_histogram<<<DimGrid, DimBlock>>>(d_grayImage, d_histo, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  DimBlock = dim3(BLOCK_SIZE, 1, 1);
  DimGrid = dim3(1, 1, 1);
  scan<<<DimGrid, DimBlock>>>(d_histo, d_cdf, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  DimBlock = dim3(BLOCK_SIZE, 1, 1);
  DimGrid = dim3(ceil(size * imageChannels / (1.0 * BLOCK_SIZE)), 1, 1);
  equalize<<<DimGrid, DimBlock>>>(d_ucharImage, d_cdf, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();

  DimBlock = dim3(BLOCK_SIZE, 1, 1);
  DimGrid = dim3(ceil(size * imageChannels / (1.0 * BLOCK_SIZE)), 1, 1);
  char_to_float<<<DimGrid, DimBlock>>>(d_ucharImage, d_outputImage, imageWidth, imageHeight, imageChannels);
  cudaDeviceSynchronize();

  cudaMemcpy(hostOutputImageData, d_outputImage, size * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);

  // wbExport("out.ppm", outputImage);

  wbSolution(args, outputImage);

  //@@ insert code here
  cudaFree(d_inputImage);
  cudaFree(d_outputImage);
  cudaFree(d_grayImage);
  cudaFree(d_ucharImage);
  cudaFree(d_cdf);
  cudaFree(d_histo);

  return 0;
}