#include <iostream>
#include <cmath>
#include <stdio.h>

#include <FreeImage.h>

using namespace std;

struct RGB_24 {
    unsigned char r;
    unsigned char g;
    unsigned char b;
};

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void convertToGrayscale(RGB_24* d_in, float* d_out, int numRows, int numCols) {

    if (blockIdx.x == (int) numCols/blockDim.x && threadIdx.x + blockIdx.x * blockDim.x >= numCols) return;
    else if (blockIdx.y == (int) numRows/blockDim.y && threadIdx.y + blockIdx.y * blockDim.y >= numRows) return;
    
    unsigned long toffset = threadIdx.x + threadIdx.y * numCols;
    unsigned long boffset = blockIdx.y * blockDim.x * numCols + blockDim.y * blockIdx.x;

    unsigned long id = toffset + boffset;
    d_out[id] = float(d_in[id].r) * 0.2989f + float(d_in[id].g) * 0.587f + float(d_in[id].b) * 0.114f;
}

int main(int argc, char** argv) {
    
    if (argc < 2 || argc > 2) return -1;
    FreeImage_Initialise(); 
    FREE_IMAGE_FORMAT format = FreeImage_GetFileType(argv[1]);
    FIBITMAP* immap = FreeImage_Load(format, argv[1]);

    int numRows = FreeImage_GetHeight(immap);
    int numCols = FreeImage_GetWidth(immap);
    int pitch = FreeImage_GetPitch(immap);

    
    RGB_24* h_in = new RGB_24[numRows * numCols];
    float* h_out = new float[numRows * numCols];
    RGB_24* d_in;
    float* d_out;
    
    gpuErrchk(cudaMalloc((void **) &d_in, sizeof(RGB_24) * numRows * numCols));
    gpuErrchk(cudaMalloc((void **) &d_out, sizeof(float) * numRows * numCols));

    FREE_IMAGE_TYPE type = FreeImage_GetImageType(immap);
    int i = 0;
    if(type == FIT_BITMAP) {
        BYTE* bits = (BYTE*)FreeImage_GetBits(immap);
        for(int y = 0; y < numRows; y++) {
            BYTE* pixel = (BYTE *) bits;
            for(int x = 0; x < numCols; x++) {
                h_in[i].r = pixel[FI_RGBA_RED];
                h_in[i].g = pixel[FI_RGBA_GREEN];
                h_in[i++].b = pixel[FI_RGBA_BLUE];
                pixel += 3;
            }
            bits += pitch;
        }
    }
    gpuErrchk(cudaMemcpy(d_in, h_in, numRows * numCols * sizeof(RGB_24), cudaMemcpyHostToDevice));
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    
    cout<<ceil(numCols/32.0)<<" --- "<<ceil(numRows/32.0)<<endl;
    convertToGrayscale<<<dim3(ceil(numCols/32.0), ceil(numRows/32.0), 1),dim3(32, 32, 1)>>>(d_in, d_out, numRows, numCols);
    gpuErrchk(cudaPeekAtLastError());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);
    cout<<"Time taken "<<time<<endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    gpuErrchk(cudaMemcpy(h_out, d_out, numRows * numCols * sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(d_out));
    gpuErrchk(cudaFree(d_in));
    cudaDeviceSynchronize();

    BYTE* bits = (BYTE*)FreeImage_GetBits(immap);
    i = 0;
    if(type == FIT_BITMAP) {
        for(int y = 0; y < numRows; y++) {
            BYTE* pixel = (BYTE *) bits;
            for(int x = 0; x < numCols; x++) {
                pixel[FI_RGBA_RED] = pixel[FI_RGBA_GREEN] = pixel[FI_RGBA_BLUE] = h_out[i++];
                pixel += 3;
            }
            bits += pitch;
        }
    }

    FreeImage_Save(FIF_JPEG, immap, "gray.jpeg", JPEG_DEFAULT);
    FreeImage_DeInitialise();

    return 0; 
}
