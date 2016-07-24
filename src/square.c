#include <stdio.h>
#include <stdlib.h>

int main(char* argv, char** argc) {
	const int ARRAY_SIZE = 64;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);
	float * d_in = (float *) malloc(ARRAY_BYTES);
	float * d_out = (float *) malloc(ARRAY_BYTES);

	for (int i=0; i<ARRAY_SIZE; i++){
		d_in[i] = (float) i;
	}

	for (int i=0; i<ARRAY_SIZE; i++){
		d_out[i] = d_in[i]*d_in[i];
	}

	for (int i=0; i<ARRAY_SIZE; i++){
		printf("%f\t", d_out[i]);
	}
	free(d_in);
	free(d_out);
}
