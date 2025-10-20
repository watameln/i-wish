#include "uiblur.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

static inline float _gauss(float x, float s){ return expf(-(x*x)/(2.0f*s*s)); }

void uiblur_init(UIBlur* b, float sigma){
    if(!b) return;
    if (sigma < 0.5f) sigma = 0.5f;
    b->sigma = sigma;
}

static void premultiply(uint8_t* p, int w, int h, int stride){
    for(int y=0; y<h; ++y){
        uint8_t* row = p + y*stride;
        for(int x=0; x<w; ++x){
            uint8_t* px = row + 4*x;
            float a = px[3] / 255.0f;
            px[0] = (uint8_t)lrintf(px[0]*a);
            px[1] = (uint8_t)lrintf(px[1]*a);
            px[2] = (uint8_t)lrintf(px[2]*a);
        }
    }
}

static void unpremultiply(uint8_t* p, int w, int h, int stride){
    for(int y=0; y<h; ++y){
        uint8_t* row = p + y*stride;
        for(int x=0; x<w; ++x){
            uint8_t* px = row + 4*x;
            uint8_t a8 = px[3];
            if (a8 == 0) { px[0]=px[1]=px[2]=0; continue; }
            float invA = 255.0f / (float)a8;
            int r = (int)lrintf(px[0]*invA);
            int g = (int)lrintf(px[1]*invA);
            int b = (int)lrintf(px[2]*invA);
            px[0] = (uint8_t)(r<0?0:(r>255?255:r));
            px[1] = (uint8_t)(g<0?0:(g>255?255:g));
            px[2] = (uint8_t)(b<0?0:(b>255?255:b));
        }
    }
}

static void make_kernel(float sigma, float** outK, int* outR){
    int r = (int)ceilf(3.0f * sigma);
    if (r < 1) r = 1;
    int size = 2*r + 1;

    float* k = (float*)malloc(sizeof(float)*size);
    float sum = 0.0f;
    for(int i=-r;i<=r;++i){ float v = _gauss((float)i, sigma); k[i+r]=v; sum+=v; }
    float inv = 1.0f / sum;
    for(int i=0;i<size;++i) k[i]*=inv;

    *outK = k; *outR = r;
}

static void convolve_1d_rgba8(
    const uint8_t* src, uint8_t* dst,
    int w, int h, int stride,
    const float* k, int r, int dir)
{
    if (dir==0) {
        for(int y=0;y<h;++y){
            const uint8_t* srow = src + y*stride;
            uint8_t* drow = dst + y*stride;
            for(int x=0;x<w;++x){
                float accR=0,accG=0,accB=0,accA=0;
                for(int i=-r;i<=r;++i){
                    int xx = x+i; if(xx<0) xx=0; else if(xx>=w) xx=w-1;
                    const uint8_t* p = srow + 4*xx;
                    float wgt = k[i+r];
                    accR += wgt * p[0];
                    accG += wgt * p[1];
                    accB += wgt * p[2];
                    accA += wgt * p[3];
                }
                uint8_t* q = drow + 4*x;
                q[0] = (uint8_t)lrintf(accR);
                q[1] = (uint8_t)lrintf(accG);
                q[2] = (uint8_t)lrintf(accB);
                q[3] = (uint8_t)lrintf(accA);
            }
        }
    } else {
        for(int y=0;y<h;++y){
            for(int x=0;x<w;++x){
                float accR=0,accG=0,accB=0,accA=0;
                for(int i=-r;i<=r;++i){
                    int yy = y+i; if(yy<0) yy=0; else if(yy>=h) yy=h-1;
                    const uint8_t* p = src + yy*stride + 4*x;
                    float wgt = k[i+r];
                    accR += wgt * p[0];
                    accG += wgt * p[1];
                    accB += wgt * p[2];
                    accA += wgt * p[3];
                }
                uint8_t* q = dst + y*stride + 4*x;
                q[0] = (uint8_t)lrintf(accR);
                q[1] = (uint8_t)lrintf(accG);
                q[2] = (uint8_t)lrintf(accB);
                q[3] = (uint8_t)lrintf(accA);
            }
        }
    }
}

int uiblur_apply_rgba8(
    const UIBlur* b,
    const uint8_t* src, uint8_t* dst,
    int w, int h, int stride)
{
    if(!b || !src || !dst || w<=0 || h<=0 || stride < w*4) return -1;

    uint8_t* workA = (uint8_t*)malloc((size_t)stride*h);
    uint8_t* workB = (uint8_t*)malloc((size_t)stride*h);
    if(!workA || !workB){ free(workA); free(workB); return -2; }

    for(int y=0;y<h;++y) memcpy(workA + y*stride, src + y*stride, (size_t)stride);

    premultiply(workA, w, h, stride);

    float* k = NULL; int r = 0;
    make_kernel(b->sigma, &k, &r);
    if(!k){ free(workA); free(workB); return -3; }

    convolve_1d_rgba8(workA, workB, w, h, stride, k, r, 0);
    convolve_1d_rgba8(workB, workA, w, h, stride, k, r, 1);

    unpremultiply(workA, w, h, stride);
    for(int y=0;y<h;++y) memcpy(dst + y*stride, workA + y*stride, (size_t)stride);

    free(k); free(workA); free(workB);
    return 0;
}
