#include "metricas/metricas.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <numeric>

std::pair<double, double> calcularPSNR_SNR(const uint8_t* ref, const uint8_t* res, int ancho, int alto) {
    const int total = ancho * alto;
    double suma_ref2 = 0.0, suma_error2 = 0.0;
    for (int i = 0; i < total; ++i) {
        const double r = ref[i], d = r - res[i];
        suma_ref2 += r * r;
        suma_error2 += d * d;
    }
    const double mse = suma_error2 / std::max(1, total);
    const double snr = (suma_error2 == 0.0) ? INFINITY : 10.0 * std::log10(suma_ref2 / suma_error2);
    const double psnr = (mse == 0.0) ? INFINITY : 10.0 * std::log10(255.0 * 255.0 / mse);
    return { psnr, snr };
}

double calcularEntropia(const uint8_t* imagen, int ancho, int alto) {
    std::vector<int> hist(256, 0);
    const int total = ancho * alto;
    for (int i = 0; i < total; ++i) hist[imagen[i]]++;
    double entropia = 0.0;
    for (int h : hist) {
        if (h > 0) {
            const double p = static_cast<double>(h) / total;
            entropia -= p * std::log2(p);
        }
    }
    return entropia;
}

double calcularSSIM(const uint8_t* img1, const uint8_t* img2, int ancho, int alto) {
    const int total = ancho * alto;
    double s1 = 0, s2 = 0, s1_2 = 0, s2_2 = 0, s12 = 0;
    for (int i = 0; i < total; ++i) {
        const double x = img1[i], y = img2[i];
        s1 += x; s2 += y;
        s1_2 += x * x; s2_2 += y * y;
        s12 += x * y;
    }
    const double mu1 = s1 / total, mu2 = s2 / total;
    const double var1 = s1_2 / total - mu1 * mu1;
    const double var2 = s2_2 / total - mu2 * mu2;
    const double cov = s12 / total - mu1 * mu2;
    constexpr double C1 = 6.5025, C2 = 58.5225;
    return ((2 * mu1 * mu2 + C1) * (2 * cov + C2)) /
        ((mu1 * mu1 + mu2 * mu2 + C1) * (var1 + var2 + C2));
}

double calcularIoU(const uint8_t* pred, const uint8_t* ref, int ancho, int alto) {
    const int total = ancho * alto;
    int inter = 0, uni = 0;
    for (int i = 0; i < total; ++i) {
        const bool p = pred[i] > 127, r = ref[i] > 127;
        if (p && r) inter++;
        if (p || r) uni++;
    }
    return (uni == 0) ? 1.0 : static_cast<double>(inter) / uni;
}

double calcularDensidadInformacion(const uint8_t* img, int ancho, int alto) {
    const int total = ancho * alto;
    int activos = 0;
    for (int i = 0; i < total; ++i) if (img[i] > 0) activos++;
    return static_cast<double>(activos) / std::max(1, total);
}

double calcularVarianzaExplicada(const std::vector<double>& valoresPropios, int k) {
    const double sumaTotal = std::accumulate(valoresPropios.begin(), valoresPropios.end(), 0.0);
    const double sumaTopK = std::accumulate(valoresPropios.begin(),
        valoresPropios.begin() + std::min<int>(k, valoresPropios.size()), 0.0);
    return (sumaTotal > 0.0) ? (sumaTopK / sumaTotal) : 0.0;
}
