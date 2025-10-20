#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float sigma;
} UIBlur;

void uiblur_init(UIBlur* b, float sigma);

int uiblur_apply_rgba8(
    const UIBlur* b,
    const uint8_t* src, uint8_t* dst,
    int width, int height, int stride);

#ifdef __cplusplus
}
#endif
