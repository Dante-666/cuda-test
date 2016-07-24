#include <iostream>
#include <cmath>
#include <chrono>

#include <FreeImage.h>

using namespace std;

struct RGB_24 {
    unsigned char r;
    unsigned char g;
    unsigned char b;
};

void separateChannels(RGB_24* h_in, float* h_r, float* h_g, float* h_b, int numRows, int numCols) {
    for(int id = 0; id < numRows*numCols; id++) {
        h_r[id] = float(h_in[id].r);
        h_g[id] = float(h_in[id].g);
        h_b[id] = float(h_in[id].b);
    }
}

void gaussBlur(RGB_24* h_out, float* h_r, float* h_g, float* h_b, int numRows, int numCols) {
    float r, g, b;
    for(int id = 0; id < numRows*numCols; id++) {
        if (id == 0) {
            r = 0.147761f * (float) h_r[id] + 0.118318f * ((float) h_r[id+1] + (float) h_r[id+numCols]) + 0.0947416f * (float) h_r[id+numCols+1];
            g = 0.147761f * (float) h_g[id] + 0.118318f * ((float) h_g[id+1] + (float) h_g[id+numCols]) + 0.0947416f * (float) h_g[id+numCols+1];
            b = 0.147761f * (float) h_b[id] + 0.118318f * ((float) h_b[id+1] + (float) h_b[id+numCols]) + 0.0947416f * (float) h_b[id+numCols+1];
        } else if (id == numCols - 1) {
            r = 0.147761f * (float) h_r[id] + 0.118318f * ((float) h_r[id-1] + (float) h_r[id+numCols]) + 0.0947416f * (float) h_r[id+numCols-1];
            g = 0.147761f * (float) h_g[id] + 0.118318f * ((float) h_g[id-1] + (float) h_g[id+numCols]) + 0.0947416f * (float) h_g[id+numCols-1];
            b = 0.147761f * (float) h_b[id] + 0.118318f * ((float) h_b[id-1] + (float) h_b[id+numCols]) + 0.0947416f * (float) h_b[id+numCols-1];
        } else if (id == numCols * (numRows - 1)) {
            r = 0.147761f * (float) h_r[id] + 0.118318f * ((float) h_r[id+1] + (float) h_r[id-numCols]) + 0.0947416f * (float) h_r[id-numCols+1];
            g = 0.147761f * (float) h_g[id] + 0.118318f * ((float) h_g[id+1] + (float) h_g[id-numCols]) + 0.0947416f * (float) h_g[id-numCols+1];
            b = 0.147761f * (float) h_b[id] + 0.118318f * ((float) h_b[id+1] + (float) h_b[id-numCols]) + 0.0947416f * (float) h_b[id-numCols+1];
        } else if (id == numCols * numRows - 1) {
            r = 0.147761f * (float) h_r[id] + 0.118318f * ((float) h_r[id-1] + (float) h_r[id-numCols]) + 0.0947416f * (float) h_r[id-numCols-1];
            g = 0.147761f * (float) h_g[id] + 0.118318f * ((float) h_g[id-1] + (float) h_g[id-numCols]) + 0.0947416f * (float) h_g[id-numCols-1];
            b = 0.147761f * (float) h_b[id] + 0.118318f * ((float) h_b[id-1] + (float) h_b[id-numCols]) + 0.0947416f * (float) h_b[id-numCols-1];
        } else if (id < numCols) {
            r = 0.147761f * (float) h_r[id] + 0.118318f * ((float) h_r[id-1] + (float) h_r[id+1] + 
                (float) h_r[id+numCols]) + 0.0947416f * ((float) h_r[id+numCols-1] + (float) h_r[id+numCols+1]);
            g = 0.147761f * (float) h_g[id] + 0.118318f * ((float) h_g[id-1] + (float) h_g[id+1] +
                (float) h_g[id+numCols]) + 0.0947416f * ((float) h_g[id+numCols-1] + (float) h_g[id+numCols+1]);
            b = 0.147761f * (float) h_b[id] + 0.118318f * ((float) h_b[id-1] + (float) h_b[id+1] + 
                (float) h_b[id+numCols]) + 0.0947416f * ((float) h_b[id+numCols-1] + (float) h_b[id+numCols+1]);
        } else if (id % numCols == 0) {
            r = 0.147761f * (float) h_r[id] + 0.118318f * ((float) h_r[id+1] + (float) h_r[id+numCols] +
                (float) h_r[id-numCols]) + 0.0947416f * ((float) h_r[id+numCols+1] + (float) h_r[id-numCols+1]);
            g = 0.147761f * (float) h_g[id] + 0.118318f * ((float) h_g[id+1] + (float) h_g[id+numCols] +
                (float) h_g[id-numCols]) + 0.0947416f * ((float) h_g[id+numCols+1] + (float) h_g[id-numCols+1]);
            b = 0.147761f * (float) h_b[id] + 0.118318f * ((float) h_b[id+1] + (float) h_b[id+numCols] +
                (float) h_b[id-numCols]) + 0.0947416f * ((float) h_b[id+numCols+1] + (float) h_b[id-numCols+1]);
        } else if (id % numCols == numCols - 1) {
            r = 0.147761f * (float) h_r[id] + 0.118318f * ((float) h_r[id-1] + (float) h_r[id+numCols] + 
                (float) h_r[id-numCols]) + 0.0947416f * ((float) h_r[id+numCols-1] + (float) h_r[id-numCols-1]);
            g = 0.147761f * (float) h_g[id] + 0.118318f * ((float) h_g[id-1] + (float) h_g[id+numCols] + 
                (float) h_g[id-numCols]) + 0.0947416f * ((float) h_g[id+numCols-1] + (float) h_g[id-numCols-1]);
            b = 0.147761f * (float) h_b[id] + 0.118318f * ((float) h_b[id-1] + (float) h_b[id+numCols] + 
                (float) h_b[id-numCols]) + 0.0947416f * ((float) h_b[id+numCols-1] + (float) h_b[id-numCols-1]);
        } else if (id > numCols * (numRows - 1)) {
            r = 0.147761f * (float) h_r[id] + 0.118318f * ((float) h_r[id-1] + (float) h_r[id+1] +
                (float) h_r[id-numCols]) + 0.0947416f * ((float) h_r[id-numCols-1] + (float) h_r[id-numCols+1]);
            g = 0.147761f * (float) h_g[id] + 0.118318f * ((float) h_g[id-1] + (float) h_g[id+1] +
                (float) h_g[id-numCols]) + 0.0947416f * ((float) h_g[id-numCols-1] + (float) h_g[id-numCols+1]);
            b = 0.147761f * (float) h_b[id] + 0.118318f * ((float) h_b[id-1] + (float) h_b[id+1] + 
                (float) h_b[id-numCols]) + 0.0947416f * ((float) h_b[id-numCols-1] + (float) h_b[id-numCols+1]);
        } else {
            r = 0.147761f * (float) h_r[id] + 0.118318f * ((float) h_r[id-1] + (float) h_r[id+1] + (float) h_r[id-numCols] + (float) h_r[id+numCols]) +
                0.0947416f * ((float) h_r[id-numCols-1] + (float) h_r[id-numCols+1] + (float) h_r[id+numCols-1] + (float) h_r[id+numCols+1]);
            g = 0.147761f * (float) h_g[id] + 0.118318f * ((float) h_g[id-1] + (float) h_g[id+1] + (float) h_g[id-numCols] + (float) h_g[id+numCols]) +
                0.0947416f * ((float) h_g[id-numCols-1] + (float) h_g[id-numCols+1] + (float) h_g[id+numCols-1] + (float) h_g[id+numCols+1]);
            b = 0.147761f * (float) h_b[id] + 0.118318f * ((float) h_b[id-1] + (float) h_b[id+1] + (float) h_b[id-numCols] + (float) h_b[id+numCols]) +
                0.0947416f * ((float) h_b[id-numCols-1] + (float) h_b[id-numCols+1] + (float) h_b[id+numCols-1] + (float) h_b[id+numCols+1]);
        }
        h_out[id].r = r ; h_out[id].g = g ; h_out[id].b = b;
    }
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
    float *h_r = new float[numRows * numCols];
    float *h_g = new float[numRows * numCols];
    float *h_b = new float[numRows * numCols];
    RGB_24* h_out = new RGB_24[numRows * numCols];

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

    auto start = chrono::high_resolution_clock::now();    

    cout<<ceil(numCols/32.0)<<" --- "<<ceil(numRows/32.0)<<endl;

    separateChannels(h_in, h_r, h_g, h_b, numRows, numCols);
    delete h_in;
    gaussBlur(h_out, h_r, h_g, h_b, numRows, numCols);

    auto stop = chrono::high_resolution_clock::now() - start;
       
    cout<<"Time taken "<<chrono::duration_cast<chrono::microseconds>(stop).count()<<endl;

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

    FreeImage_Save(FIF_JPEG, immap, "blur.jpeg", JPEG_DEFAULT);
    FreeImage_DeInitialise();

    return 0; 
}
