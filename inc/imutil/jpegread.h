#ifndef __IITK_CUDA_JPEGREAD_H
#include <jpeglib.h>
#include <fstream>

/**
struct uchar4 {
    unsigned char
}**/

class JpegRead {
    public:
        uchar4 * FetchRGBA(ifstream image);
};
