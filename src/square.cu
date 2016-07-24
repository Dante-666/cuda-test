//#include <cuda.h>
#include <stdio.h>

__global__ void  square(const float *d_in, float *d_out){
    d_out[threadIdx.x] = d_in[threadIdx.x]*d_in[threadIdx.x];
}

int main() {
    const int ARRAY_SIZE = 64;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);
    float *h_in, *h_out, *d_in, *d_out;
   
    h_in = (float *) malloc(ARRAY_BYTES);
    h_out = (float *) malloc(ARRAY_BYTES);

    for (int i = 0; i < ARRAY_SIZE;) {
        h_in[i] = (float) ++i;
        h_out[i-1] = h_in[i-1];
    }

    cudaMalloc((void **) &d_in, ARRAY_BYTES);
    cudaMalloc((void **) &d_out, ARRAY_BYTES);

    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

    square<<<1, ARRAY_SIZE>>>(d_in, d_out);
    
    cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
   
    for (int i=0; i<ARRAY_SIZE; i++){
        printf("%f\t", h_out[i]);
    }
    cudaFree(d_in);
    cudaFree(d_out);
}
