#include "preprocesamiento/alineacion_oreja.h"
#include <cmath>
#include <algorithm>
#include <vector>

namespace {

inline uint8_t clamp_u8(int v) {
    return (uint8_t)std::min(255, std::max(0, v));
}

inline bool is_fg(uint8_t v) { return v > 0; }

// Bilinear sample for grayscale image (0..255). Outside => 0.
uint8_t sample_bilinear(const uint8_t* img, int w, int h, float x, float y) {
    int x0 = (int)std::floor(x);
    int y0 = (int)std::floor(y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    if (x0 < 0 || y0 < 0 || x1 >= w || y1 >= h) return 0;

    float fx = x - x0;
    float fy = y - y0;

    int i00 = img[y0*w + x0];
    int i10 = img[y0*w + x1];
    int i01 = img[y1*w + x0];
    int i11 = img[y1*w + x1];

    float v0 = i00 + fx * (i10 - i00);
    float v1 = i01 + fx * (i11 - i01);
    float v  = v0  + fy * (v1 - v0);

    return clamp_u8((int)std::lround(v));
}

// Nearest sample for mask. Outside => 0.
uint8_t sample_nearest(const uint8_t* img, int w, int h, float x, float y) {
    int xi = (int)std::lround(x);
    int yi = (int)std::lround(y);
    if (xi < 0 || yi < 0 || xi >= w || yi >= h) return 0;
    return img[yi*w + xi];
}

// Rotate around (cx, cy) by angle radians (positive = CCW).
// Output same size w*h.
std::unique_ptr<uint8_t[]> rotate_gray_bilinear(
    const uint8_t* img, int w, int h, float angle, float cx, float cy
){
    auto out = std::make_unique<uint8_t[]>(w*h);
    float c = std::cos(angle);
    float s = std::sin(angle);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            // inverse mapping: src = R^-1 * (dst - center) + center
            float dx = x - cx;
            float dy = y - cy;
            float sx =  c*dx + s*dy + cx;
            float sy = -s*dx + c*dy + cy;

            out[y*w + x] = sample_bilinear(img, w, h, sx, sy);
        }
    }
    return out;
}

std::unique_ptr<uint8_t[]> rotate_mask_nearest(
    const uint8_t* mask, int w, int h, float angle, float cx, float cy
){
    auto out = std::make_unique<uint8_t[]>(w*h);
    float c = std::cos(angle);
    float s = std::sin(angle);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float dx = x - cx;
            float dy = y - cy;
            float sx =  c*dx + s*dy + cx;
            float sy = -s*dx + c*dy + cy;

            uint8_t v = sample_nearest(mask, w, h, sx, sy);
            out[y*w + x] = is_fg(v) ? 255 : 0;
        }
    }
    return out;
}

// Compute centroid + covariance of mask foreground pixels.
bool mask_stats_pca(
    const uint8_t* mask, int w, int h,
    float& mx, float& my,
    float& sxx, float& syy, float& sxy,
    int& count
){
    double sumx = 0, sumy = 0;
    count = 0;

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (is_fg(mask[y*w + x])) {
                sumx += x;
                sumy += y;
                count++;
            }
        }
    }
    if (count < 50) return false; // muy poca máscara => no confiable

    mx = (float)(sumx / count);
    my = (float)(sumy / count);

    double cxx=0, cyy=0, cxy=0;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (is_fg(mask[y*w + x])) {
                double dx = x - mx;
                double dy = y - my;
                cxx += dx*dx;
                cyy += dy*dy;
                cxy += dx*dy;
            }
        }
    }

    sxx = (float)(cxx / count);
    syy = (float)(cyy / count);
    sxy = (float)(cxy / count);
    return true;
}

struct BBox { int x0,y0,x1,y1; };

bool bbox_from_mask(const uint8_t* mask, int w, int h, BBox& b) {
    int minx=w, miny=h, maxx=-1, maxy=-1;
    for (int y=0;y<h;++y){
        for (int x=0;x<w;++x){
            if (is_fg(mask[y*w+x])) {
                minx = std::min(minx, x);
                miny = std::min(miny, y);
                maxx = std::max(maxx, x);
                maxy = std::max(maxy, y);
            }
        }
    }
    if (maxx < 0) return false;
    b.x0=minx; b.y0=miny; b.x1=maxx; b.y1=maxy;
    return true;
}

// Simple nearest resize for both gray and mask (mask will stay 0/255).
std::unique_ptr<uint8_t[]> resize_nearest(const uint8_t* img, int w0, int h0, int w1, int h1) {
    auto out = std::make_unique<uint8_t[]>(w1*h1);
    float sx = (float)w0 / w1;
    float sy = (float)h0 / h1;
    for (int y=0;y<h1;++y){
        for (int x=0;x<w1;++x){
            int ix = std::clamp((int)(x*sx), 0, w0-1);
            int iy = std::clamp((int)(y*sy), 0, h0-1);
            out[y*w1 + x] = img[iy*w0 + ix];
        }
    }
    return out;
}

std::unique_ptr<uint8_t[]> resize_bilinear(const uint8_t* img, int w0, int h0, int w1, int h1) {
    auto out = std::make_unique<uint8_t[]>(w1*h1);
    float sx = (float)(w0 - 1) / std::max(1, w1 - 1);
    float sy = (float)(h0 - 1) / std::max(1, h1 - 1);

    for (int y = 0; y < h1; ++y) {
        float fy = y * sy;
        int y0 = (int)std::floor(fy);
        int y1i = std::min(y0 + 1, h0 - 1);
        float ty = fy - y0;

        for (int x = 0; x < w1; ++x) {
            float fx = x * sx;
            int x0 = (int)std::floor(fx);
            int x1i = std::min(x0 + 1, w0 - 1);
            float tx = fx - x0;

            int i00 = img[y0*w0 + x0];
            int i10 = img[y0*w0 + x1i];
            int i01 = img[y1i*w0 + x0];
            int i11 = img[y1i*w0 + x1i];

            float v0 = i00 + tx * (i10 - i00);
            float v1 = i01 + tx * (i11 - i01);
            float v  = v0  + ty * (v1 - v0);

            out[y*w1 + x] = clamp_u8((int)std::lround(v));
        }
    }
    return out;
}

// Crop a ROI and return new image (wR*hR).
std::unique_ptr<uint8_t[]> crop(const uint8_t* img, int w, int h, int x0, int y0, int wR, int hR) {
    auto out = std::make_unique<uint8_t[]>(wR*hR);
    for (int y=0;y<hR;++y){
        int sy = std::clamp(y0 + y, 0, h-1);
        for (int x=0;x<wR;++x){
            int sx = std::clamp(x0 + x, 0, w-1);
            out[y*wR + x] = img[sy*w + sx];
        }
    }
    return out;
}

} // namespace

AlignedCrop128 alinearYRecortarOreja128(
    const uint8_t* img128,
    const uint8_t* mask128,
    int w, int h,
    float margin_frac,
    bool recenter
){
    // Fallback: si no hay máscara válida, devuelve copias tal cual.
    AlignedCrop128 out;
    out.img  = std::make_unique<uint8_t[]>(w*h);
    out.mask = std::make_unique<uint8_t[]>(w*h);
    std::copy(img128,  img128  + w*h, out.img.get());
    std::copy(mask128, mask128 + w*h, out.mask.get());

    float mx=0, my=0, sxx=0, syy=0, sxy=0;
    int n=0;
    if (!mask_stats_pca(mask128, w, h, mx, my, sxx, syy, sxy, n)) {
        return out;
    }

    // Angle of principal axis (2D PCA)
    float theta = 0.5f * std::atan2(2.0f*sxy, (sxx - syy));

    // rotate by -theta to align
    auto r_img  = rotate_gray_bilinear(img128,  w, h, -theta, mx, my);
    auto r_mask = rotate_mask_nearest(mask128,  w, h, -theta, mx, my);

    // Optional recenter: move centroid to center (64,64)
    if (recenter) {
        float mx2=0, my2=0, sxx2=0, syy2=0, sxy2=0;
        int n2=0;
        if (mask_stats_pca(r_mask.get(), w, h, mx2, my2, sxx2, syy2, sxy2, n2)) {
            float tx = (w * 0.5f) - mx2;
            float ty = (h * 0.5f) - my2;

            // Apply translation via sampling
            auto t_img  = std::make_unique<uint8_t[]>(w*h);
            auto t_mask = std::make_unique<uint8_t[]>(w*h);

            for (int y=0;y<h;++y){
                for (int x=0;x<w;++x){
                    float sx = x - tx;
                    float sy = y - ty;
                    t_img[y*w + x]  = sample_bilinear(r_img.get(), w, h, sx, sy);
                    uint8_t mv = sample_nearest(r_mask.get(), w, h, sx, sy);
                    t_mask[y*w + x] = is_fg(mv) ? 255 : 0;
                }
            }
            r_img.swap(t_img);
            r_mask.swap(t_mask);
        }
    }

    // Coherent crop based on bbox of aligned mask + relative margin
    BBox b;
    if (!bbox_from_mask(r_mask.get(), w, h, b)) {
        // if bbox fails, return rotated
        std::copy(r_img.get(),  r_img.get()  + w*h, out.img.get());
        std::copy(r_mask.get(), r_mask.get() + w*h, out.mask.get());
        return out;
    }

    int bw = (b.x1 - b.x0 + 1);
    int bh = (b.y1 - b.y0 + 1);
    int mxp = (int)std::lround(bw * margin_frac);
    int myp = (int)std::lround(bh * margin_frac);

    int x0 = std::max(0, b.x0 - mxp);
    int y0 = std::max(0, b.y0 - myp);
    int x1 = std::min(w-1, b.x1 + mxp);
    int y1 = std::min(h-1, b.y1 + myp);

    int wR = std::max(8, x1 - x0 + 1);
    int hR = std::max(8, y1 - y0 + 1);

    auto c_img  = crop(r_img.get(),  w, h, x0, y0, wR, hR);
    auto c_mask = crop(r_mask.get(), w, h, x0, y0, wR, hR);

    // Resize back to 128x128 (use nearest for mask; nearest also ok for gray here)
    // Si quieres más calidad en gray, puedes hacer bilinear resize luego, pero no es obligatorio.
    out.img  = resize_bilinear(c_img.get(),  wR, hR, w, h); // NUEVO
    out.mask = resize_nearest (c_mask.get(), wR, hR, w, h); // igual

    // Ensure mask is strictly 0/255
    for (int i=0;i<w*h;++i) out.mask[i] = is_fg(out.mask[i]) ? 255 : 0;

    return out;
}
