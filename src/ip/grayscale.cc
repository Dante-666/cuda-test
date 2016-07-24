#include <iostream>
#include <FreeImage.h>

using namespace std;

int main(int argc, char** argv) {
    
    if (argc < 2 || argc > 2) return -1;
    FreeImage_Initialise() ;
    FREE_IMAGE_FORMAT format = FreeImage_GetFileType(argv[1]);
    FIBITMAP* immap = FreeImage_Load(format, argv[1]);

    int numRows = FreeImage_GetHeight(immap);
    int numCols = FreeImage_GetWidth(immap);
    int pitch = FreeImage_GetPitch(immap);
    cout<<numRows<<" "<<numCols<<endl;

    FREE_IMAGE_TYPE type = FreeImage_GetImageType(immap);

    if(type == FIT_BITMAP) {
        BYTE* bits = (BYTE*)FreeImage_GetBits(immap);
        for(int y = 0; y < numRows; y++) {
            BYTE* pixel = (BYTE *) bits;
            for(int x = 0; x < numCols; x++) {
                float grey = float(pixel[FI_RGBA_RED]) * 0.2989f + float(pixel[FI_RGBA_GREEN]) * 0.587f + float(pixel[FI_RGBA_BLUE]) * 0.114f;
                pixel[FI_RGBA_RED] = grey;
                pixel[FI_RGBA_GREEN] = grey;
                pixel[FI_RGBA_BLUE] = grey;
                pixel += 3;
            }
            bits += pitch;
        }
    }
    FreeImage_Save(FIF_JPEG, immap, "gray.jpeg", JPEG_DEFAULT);    
    FreeImage_DeInitialise();

    return 0; 
}
