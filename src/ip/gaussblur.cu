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

__global__ void separateChannels(RGB_24* d_in, unsigned char* d_r, unsigned char* d_g, unsigned char* d_b, int numRows, int numCols) {
    
    if (blockIdx.x == (int) numCols/blockDim.x && threadIdx.x + blockIdx.x * blockDim.x >= numCols) return;
    else if (blockIdx.y == (int) numRows/blockDim.y && threadIdx.y + blockIdx.y * blockDim.y >= numRows) return;

    unsigned long toffset = threadIdx.x + threadIdx.y * numCols;
    unsigned long boffset = blockIdx.y * blockDim.x * numCols + blockDim.y * blockIdx.x;

    unsigned long id = toffset + boffset;

    d_r[id] = d_in[id].r;
    d_g[id] = d_in[id].g;
    d_b[id] = d_in[id].b;
}

__global__ void gaussBlur(RGB_24* d_out, unsigned char* d_r, unsigned char* d_g, unsigned char* d_b, int numRows, int numCols) {
    
    if (blockIdx.x == (int) numCols/blockDim.x && threadIdx.x + blockIdx.x * blockDim.x >= numCols) return;
    else if (blockIdx.y == (int) numRows/blockDim.y && threadIdx.y + blockIdx.y * blockDim.y >= numRows) return;
    
    unsigned long toffset = threadIdx.x + threadIdx.y * numCols;
    unsigned long boffset = blockIdx.y * blockDim.y * numCols + blockDim.x * blockIdx.x;

    unsigned long id = toffset + boffset;
    __shared__ RGB_24 pixels[34*34];
    unsigned long poffset = (blockDim.x + 2) * (threadIdx.y + 1) + threadIdx.x + 1;
    pixels[poffset].r = d_r[id];
    pixels[poffset].g = d_g[id];
    pixels[poffset].b = d_b[id];

    unsigned int t_poffset = poffset;
    unsigned long tid = id;
    
    if (id == 0) {
        t_poffset--;
        pixels[t_poffset].r = pixels[t_poffset].g = pixels[t_poffset].b = 0;
        t_poffset -= blockDim.x + 2;
        pixels[t_poffset].r = pixels[t_poffset].g = pixels[t_poffset].b = 0;
        t_poffset++;
        pixels[t_poffset].r = pixels[t_poffset].g = pixels[t_poffset].b = 0;
    } else if (id == numCols - 1) {
        t_poffset++;
        pixels[t_poffset].r = pixels[t_poffset].g = pixels[t_poffset].b = 0;
        t_poffset -= blockDim.x + 2;
        pixels[t_poffset].r = pixels[t_poffset].g = pixels[t_poffset].b = 0;
        t_poffset--;
        pixels[t_poffset].r = pixels[t_poffset].g = pixels[t_poffset].b = 0;
    } else if (id == numCols * (numRows - 1)) {
        t_poffset--;
        pixels[t_poffset].r = pixels[t_poffset].g = pixels[t_poffset].b = 0;
        t_poffset += blockDim.x + 2;
        pixels[t_poffset].r = pixels[t_poffset].g = pixels[t_poffset].b = 0;
        t_poffset++;
        pixels[t_poffset].r = pixels[t_poffset].g = pixels[t_poffset].b = 0;
    } else if (id == numCols * numRows - 1) {
        t_poffset++;
        pixels[t_poffset].r = pixels[t_poffset].g = pixels[t_poffset].b = 0;
        t_poffset += blockDim.x + 2;
        pixels[t_poffset].r = pixels[t_poffset].g = pixels[t_poffset].b = 0;
        t_poffset--;
        pixels[t_poffset].r = pixels[t_poffset].g = pixels[t_poffset].b = 0;
    } else if (id < numCols) {
        t_poffset -= blockDim.x + 2;
        pixels[t_poffset].r = pixels[t_poffset].g = pixels[t_poffset].b = 0;
        if (threadIdx.x == 0) {
            t_poffset--;
            pixels[t_poffset].r = pixels[t_poffset].g = pixels[t_poffset].b = 0;
            t_poffset += blockDim.x + 2;
            tid--;
            pixels[t_poffset].r = d_r[tid];
            pixels[t_poffset].g = d_g[tid];
            pixels[t_poffset].b = d_b[tid];
        } else if (threadIdx.x == blockDim.x - 1) {
            t_poffset++;
            pixels[t_poffset].r = pixels[t_poffset].g = pixels[t_poffset].b = 0;
            t_poffset += blockDim.x + 2;
            tid++;
            pixels[t_poffset].r = d_r[tid];
            pixels[t_poffset].g = d_g[tid];
            pixels[t_poffset].b = d_b[tid];
        }
    } else if (id % numCols == 0) {
        t_poffset--;
        pixels[t_poffset].r = pixels[t_poffset].g = pixels[t_poffset].b = 0;
        if (threadIdx.y == blockDim.y - 1) {
            t_poffset += blockDim.x + 2;
            pixels[t_poffset].r = pixels[t_poffset].g = pixels[t_poffset].b = 0;
            t_poffset++;
            tid += numCols;
            pixels[t_poffset].r = d_r[tid];
            pixels[t_poffset].g = d_g[tid];
            pixels[t_poffset].b = d_b[tid];
        } else if (threadIdx.y == 0) {
            t_poffset -= blockDim.x + 2;
            pixels[t_poffset].r = pixels[t_poffset].g = pixels[t_poffset].b = 0;
            t_poffset++;
            tid -= numCols;
            pixels[t_poffset].r = d_r[tid];
            pixels[t_poffset].g = d_g[tid];
            pixels[t_poffset].b = d_b[tid];
        }

    } else if (id % numCols == numCols - 1) {
        t_poffset++;
        pixels[t_poffset].r = pixels[t_poffset].g = pixels[t_poffset].b = 0;
        if (threadIdx.y == blockDim.y - 1) {
            t_poffset += blockDim.x + 2;
            pixels[t_poffset].r = pixels[t_poffset].g = pixels[t_poffset].b = 0;
            t_poffset--;
            tid -= numCols;
            pixels[t_poffset].r = d_r[tid];
            pixels[t_poffset].g = d_g[tid];
            pixels[t_poffset].b = d_b[tid];
        } else if (threadIdx.y == 0) {
            t_poffset -= blockDim.x + 2;
            pixels[t_poffset].r = pixels[t_poffset].g = pixels[t_poffset].b = 0;
            t_poffset--;
            tid -= numCols;
            pixels[t_poffset].r = d_r[tid];
            pixels[t_poffset].g = d_g[tid];
            pixels[t_poffset].b = d_b[tid];
        }
    } else if (id > numCols * (numRows - 1)) {
        t_poffset += blockDim.x + 2;
        pixels[t_poffset].r = pixels[t_poffset].g = pixels[t_poffset].b = 0;
        if (threadIdx.x == 0) {
            t_poffset--;
            pixels[t_poffset].r = pixels[t_poffset].g = pixels[t_poffset].b = 0;
            t_poffset -= blockDim.x + 2;
            tid--;
            pixels[t_poffset].r = d_r[tid];
            pixels[t_poffset].g = d_g[tid];
            pixels[t_poffset].b = d_b[tid];
        } else if (threadIdx.x == blockDim.x - 1) {
            t_poffset++;
            pixels[t_poffset].r = pixels[t_poffset].g = pixels[t_poffset].b = 0;
            t_poffset -= blockDim.x + 2;
            tid++;
            pixels[t_poffset].r = d_r[tid];
            pixels[t_poffset].g = d_g[tid];
            pixels[t_poffset].b = d_b[tid];
        }
    } else if (threadIdx.x == 0 && threadIdx.y == 0) {
        t_poffset--;
        tid--;
        pixels[t_poffset].r = d_r[tid];
        pixels[t_poffset].g = d_g[tid];
        pixels[t_poffset].b = d_b[tid];
        
        t_poffset -= blockDim.x + 2;
        tid -= numCols;
        pixels[t_poffset].r = d_r[tid];
        pixels[t_poffset].g = d_g[tid];
        pixels[t_poffset].b = d_b[tid];

        t_poffset++;
        tid++;
        pixels[t_poffset].r = d_r[tid];
        pixels[t_poffset].g = d_g[tid];
        pixels[t_poffset].b = d_b[tid];
    } else if (threadIdx.x == blockDim.x - 1 && threadIdx.y == 0) {
        t_poffset++;
        tid++;
        pixels[t_poffset].r = d_r[tid];
        pixels[t_poffset].g = d_g[tid];
        pixels[t_poffset].b = d_b[tid];
        
        t_poffset -= blockDim.x + 2;
        tid -= numCols;
        pixels[t_poffset].r = d_r[tid];
        pixels[t_poffset].g = d_g[tid];
        pixels[t_poffset].b = d_b[tid];

        t_poffset--;
        tid--;
        pixels[t_poffset].r = d_r[tid];
        pixels[t_poffset].g = d_g[tid];
        pixels[t_poffset].b = d_b[tid];
    } else if (threadIdx.x == 0 && threadIdx.y == blockDim.y - 1) {
        t_poffset--;
        tid--;
        pixels[t_poffset].r = d_r[tid];
        pixels[t_poffset].g = d_g[tid];
        pixels[t_poffset].b = d_b[tid];
        
        t_poffset += blockDim.x + 2;
        tid += numCols;
        pixels[t_poffset].r = d_r[tid];
        pixels[t_poffset].g = d_g[tid];
        pixels[t_poffset].b = d_b[tid];

        t_poffset++;
        tid++;
        pixels[t_poffset].r = d_r[tid];
        pixels[t_poffset].g = d_g[tid];
        pixels[t_poffset].b = d_b[tid];
    } else if (threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1) {
        t_poffset++;
        tid++;
        pixels[t_poffset].r = d_r[tid];
        pixels[t_poffset].g = d_g[tid];
        pixels[t_poffset].b = d_b[tid];
        
        t_poffset += (blockDim.x + 2);
        tid += numCols;
        pixels[t_poffset].r = d_r[tid];
        pixels[t_poffset].g = d_g[tid];
        pixels[t_poffset].b = d_b[tid];

        t_poffset--;
        tid--;
        pixels[t_poffset].r = d_r[tid];
        pixels[t_poffset].g = d_g[tid];
        pixels[t_poffset].b = d_b[tid];
    } else if (threadIdx.y == 0) {
        t_poffset -= blockDim.x + 2;
        tid -= numCols;
        pixels[t_poffset].r = d_r[tid];
        pixels[t_poffset].g = d_g[tid];
        pixels[t_poffset].b = d_b[tid];
    } else if (threadIdx.x == 0) {
        t_poffset--;
        tid--;
        pixels[t_poffset].r = d_r[tid];
        pixels[t_poffset].g = d_g[tid];
        pixels[t_poffset].b = d_b[tid];
    } else if (threadIdx.y == blockDim.y - 1) {
        t_poffset += blockDim.x + 2;
        tid += numCols;    
        pixels[t_poffset].r = d_r[tid];
        pixels[t_poffset].g = d_g[tid];
        pixels[t_poffset].b = d_b[tid];
    } else if (threadIdx.x == blockDim.x - 1) {
        t_poffset++;
        tid++;    
        pixels[t_poffset].r = d_r[tid];
        pixels[t_poffset].g = d_g[tid];
        pixels[t_poffset].b = d_b[tid];
    }

    __syncthreads();

    float r, g, b;
    r = 0.147761f * float(pixels[poffset].r) +
        0.118318f * (float(pixels[poffset+1].r) 
                + float(pixels[poffset-1].r) 
                + float(pixels[poffset+blockDim.x+2].r) 
                + float(pixels[poffset-blockDim.x-2].r)) +
        0.0947416f * (float(pixels[poffset+blockDim.x+3].r) 
                + float(pixels[poffset+blockDim.x+1].r) 
                + float(pixels[poffset-blockDim.x-3].r) 
                + float(pixels[poffset-blockDim.x-1].r));
    g = 0.147761f * float(pixels[poffset].g) +
        0.118318f * (float(pixels[poffset+1].g) 
                + float(pixels[poffset-1].g) 
                + float(pixels[poffset+blockDim.x+2].g) 
                + float(pixels[poffset-blockDim.x-2].g)) +
        0.0947416f * (float(pixels[poffset+blockDim.x+3].g) 
                + float(pixels[poffset+blockDim.x+1].g) 
                + float(pixels[poffset-blockDim.x-3].g) 
                + float(pixels[poffset-blockDim.x-1].g));
    b = 0.147761f * float(pixels[poffset].b) +
        0.118318f * (float(pixels[poffset+1].b) 
                + float(pixels[poffset-1].b) 
                + float(pixels[poffset+blockDim.x+2].b) 
                + float(pixels[poffset-blockDim.x-2].b)) +
        0.0947416f * (float(pixels[poffset+blockDim.x+3].b) 
                + float(pixels[poffset+blockDim.x+1].b) 
                + float(pixels[poffset-blockDim.x-3].b) 
                + float(pixels[poffset-blockDim.x-1].b));
    
    /*if (id == 0) {
        r = 0.147761f * (float) d_r[id] + 0.118318f * ((float) d_r[id+1] + (float) d_r[id+numCols]) + 0.0947416f * (float) d_r[id+numCols+1];
        g = 0.147761f * (float) d_g[id] + 0.118318f * ((float) d_g[id+1] + (float) d_g[id+numCols]) + 0.0947416f * (float) d_g[id+numCols+1];
        b = 0.147761f * (float) d_b[id] + 0.118318f * ((float) d_b[id+1] + (float) d_b[id+numCols]) + 0.0947416f * (float) d_b[id+numCols+1];
    } else if (id == numCols - 1) {
        r = 0.147761f * (float) d_r[id] + 0.118318f * ((float) d_r[id-1] + (float) d_r[id+numCols]) + 0.0947416f * (float) d_r[id+numCols-1];
        g = 0.147761f * (float) d_g[id] + 0.118318f * ((float) d_g[id-1] + (float) d_g[id+numCols]) + 0.0947416f * (float) d_g[id+numCols-1];
        b = 0.147761f * (float) d_b[id] + 0.118318f * ((float) d_b[id-1] + (float) d_b[id+numCols]) + 0.0947416f * (float) d_b[id+numCols-1];
    } else if (id == numCols * (numRows - 1)) {
        r = 0.147761f * (float) d_r[id] + 0.118318f * ((float) d_r[id+1] + (float) d_r[id-numCols]) + 0.0947416f * (float) d_r[id-numCols+1];
        g = 0.147761f * (float) d_g[id] + 0.118318f * ((float) d_g[id+1] + (float) d_g[id-numCols]) + 0.0947416f * (float) d_g[id-numCols+1];
        b = 0.147761f * (float) d_b[id] + 0.118318f * ((float) d_b[id+1] + (float) d_b[id-numCols]) + 0.0947416f * (float) d_b[id-numCols+1];
    } else if (id == numCols * numRows - 1) {
        r = 0.147761f * (float) d_r[id] + 0.118318f * ((float) d_r[id-1] + (float) d_r[id-numCols]) + 0.0947416f * (float) d_r[id-numCols-1];
        g = 0.147761f * (float) d_g[id] + 0.118318f * ((float) d_g[id-1] + (float) d_g[id-numCols]) + 0.0947416f * (float) d_g[id-numCols-1];
        b = 0.147761f * (float) d_b[id] + 0.118318f * ((float) d_b[id-1] + (float) d_b[id-numCols]) + 0.0947416f * (float) d_b[id-numCols-1];
    } else if (id < numCols) {
        r = 0.147761f * (float) d_r[id] + 0.118318f * ((float) d_r[id-1] + (float) d_r[id+1] + 
            (float) d_r[id+numCols]) + 0.0947416f * ((float) d_r[id+numCols-1] + (float) d_r[id+numCols+1]);
        g = 0.147761f * (float) d_g[id] + 0.118318f * ((float) d_g[id-1] + (float) d_g[id+1] +
            (float) d_g[id+numCols]) + 0.0947416f * ((float) d_g[id+numCols-1] + (float) d_g[id+numCols+1]);
        b = 0.147761f * (float) d_b[id] + 0.118318f * ((float) d_b[id-1] + (float) d_b[id+1] + 
            (float) d_b[id+numCols]) + 0.0947416f * ((float) d_b[id+numCols-1] + (float) d_b[id+numCols+1]);
    } else if (id % numCols == 0) {
        r = 0.147761f * (float) d_r[id] + 0.118318f * ((float) d_r[id+1] + (float) d_r[id+numCols] +
            (float) d_r[id-numCols]) + 0.0947416f * ((float) d_r[id+numCols+1] + (float) d_r[id-numCols+1]);
        g = 0.147761f * (float) d_g[id] + 0.118318f * ((float) d_g[id+1] + (float) d_g[id+numCols] +
            (float) d_g[id-numCols]) + 0.0947416f * ((float) d_g[id+numCols+1] + (float) d_g[id-numCols+1]);
        b = 0.147761f * (float) d_b[id] + 0.118318f * ((float) d_b[id+1] + (float) d_b[id+numCols] +
            (float) d_b[id-numCols]) + 0.0947416f * ((float) d_b[id+numCols+1] + (float) d_b[id-numCols+1]);
    } else if (id % numCols == numCols - 1) {
        r = 0.147761f * (float) d_r[id] + 0.118318f * ((float) d_r[id-1] + (float) d_r[id+numCols] + 
            (float) d_r[id-numCols]) + 0.0947416f * ((float) d_r[id+numCols-1] + (float) d_r[id-numCols-1]);
        g = 0.147761f * (float) d_g[id] + 0.118318f * ((float) d_g[id-1] + (float) d_g[id+numCols] + 
            (float) d_g[id-numCols]) + 0.0947416f * ((float) d_g[id+numCols-1] + (float) d_g[id-numCols-1]);
        b = 0.147761f * (float) d_b[id] + 0.118318f * ((float) d_b[id-1] + (float) d_b[id+numCols] + 
            (float) d_b[id-numCols]) + 0.0947416f * ((float) d_b[id+numCols-1] + (float) d_b[id-numCols-1]);
    } else if (id > numCols * (numRows - 1)) {
        r = 0.147761f * (float) d_r[id] + 0.118318f * ((float) d_r[id-1] + (float) d_r[id+1] +
            (float) d_r[id-numCols]) + 0.0947416f * ((float) d_r[id-numCols-1] + (float) d_r[id-numCols+1]);
        g = 0.147761f * (float) d_g[id] + 0.118318f * ((float) d_g[id-1] + (float) d_g[id+1] +
            (float) d_g[id-numCols]) + 0.0947416f * ((float) d_g[id-numCols-1] + (float) d_g[id-numCols+1]);
        b = 0.147761f * (float) d_b[id] + 0.118318f * ((float) d_b[id-1] + (float) d_b[id+1] + 
            (float) d_b[id-numCols]) + 0.0947416f * ((float) d_b[id-numCols-1] + (float) d_b[id-numCols+1]);
    } else {
        r = 0.147761f * (float) d_r[id] + 0.118318f * ((float) d_r[id-1] + (float) d_r[id+1] + (float) d_r[id-numCols] + (float) d_r[id+numCols]) +
            0.0947416f * ((float) d_r[id-numCols-1] + (float) d_r[id-numCols+1] + (float) d_r[id+numCols-1] + (float) d_r[id+numCols+1]);
        g = 0.147761f * (float) d_g[id] + 0.118318f * ((float) d_g[id-1] + (float) d_g[id+1] + (float) d_g[id-numCols] + (float) d_g[id+numCols]) +
            0.0947416f * ((float) d_g[id-numCols-1] + (float) d_g[id-numCols+1] + (float) d_g[id+numCols-1] + (float) d_g[id+numCols+1]);
        b = 0.147761f * (float) d_b[id] + 0.118318f * ((float) d_b[id-1] + (float) d_b[id+1] + (float) d_b[id-numCols] + (float) d_b[id+numCols]) +
            0.0947416f * ((float) d_b[id-numCols-1] + (float) d_b[id-numCols+1] + (float) d_b[id+numCols-1] + (float) d_b[id+numCols+1]);
    }*/

    d_out[id].r = r ; d_out[id].g = g ; d_out[id].b = b;
    /*d_out[id].r = pixels[poffset].r; 
    d_out[id].g = pixels[poffset].g;
    d_out[id].b = pixels[poffset].b;*/
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
    RGB_24* h_out = new RGB_24[numRows * numCols];
    RGB_24* d_in;
    unsigned char* d_r;
    unsigned char* d_g;
    unsigned char* d_b;
    RGB_24* d_out;
    
    gpuErrchk(cudaMalloc((void **) &d_in, sizeof(RGB_24) * numRows * numCols));

    gpuErrchk(cudaMalloc((void **) &d_r, sizeof(unsigned char) * numRows * numCols));
    gpuErrchk(cudaMalloc((void **) &d_g, sizeof(unsigned char) * numRows * numCols));
    gpuErrchk(cudaMalloc((void **) &d_b, sizeof(unsigned char) * numRows * numCols));
    
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
    
    cout<<sizeof(float)<<endl;
    separateChannels<<<dim3(ceil(numCols/32.0), ceil(numRows/32.0), 1), dim3(32, 32, 1)>>>(d_in, d_r, d_g, d_b, numRows, numCols);
    
    gpuErrchk(cudaMalloc((void **) &d_out, sizeof(RGB_24) * numRows * numCols));
   
    gpuErrchk(cudaFree(d_in));
    
    gaussBlur<<<dim3(ceil(numCols/32.0), ceil(numRows/32.0), 1), dim3(32, 32, 1)>>>(d_out, d_r, d_g, d_b, numRows, numCols);
    gpuErrchk(cudaPeekAtLastError());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);
    cout<<"Time taken "<<time<<endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    gpuErrchk(cudaMemcpy(h_out, d_out, numRows * numCols * sizeof(RGB_24), cudaMemcpyDeviceToHost));
   
    gpuErrchk(cudaFree(d_out));
    gpuErrchk(cudaFree(d_r));
    gpuErrchk(cudaFree(d_g));
    gpuErrchk(cudaFree(d_b));
    
    //cudaDeviceSynchronize();

    BYTE* bits = (BYTE*)FreeImage_GetBits(immap);
    i = 0;
    if(type == FIT_BITMAP) {
        for(int y = 0; y < numRows; y++) {
            BYTE* pixel = (BYTE *) bits;
            for(int x = 0; x < numCols; x++) {
                pixel[FI_RGBA_RED] = h_out[i].r;
                pixel[FI_RGBA_GREEN] = h_out[i].g;
                pixel[FI_RGBA_BLUE] = h_out[i++].b;
                pixel += 3;
            }
            bits += pitch;
        }
    }

    FreeImage_Save(FIF_PNG, immap, "blur.png", JPEG_DEFAULT);
    FreeImage_DeInitialise();

    return 0; 
}
