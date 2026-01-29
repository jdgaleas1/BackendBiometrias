#include "stft.h"
#include "../../utils/config.h"
#include "../../utils/additional.h"
#include <cmath>
#include <complex>
#include <vector>
#include <iostream>
#include <iomanip>  

/**
 * FFT iterativa Cooley-Tukey con precision double
 * Algoritmo in-place con bit-reversal y butterfly operations
 *
 * @param data Vector de numeros complejos (debe ser potencia de 2)
 */
void fft_iterative(std::vector<std::complex<double>>& data) {
    const int n = static_cast<int>(data.size());
    if (n <= 1) return;

    // Verificar que n es potencia de 2
    if ((n & (n - 1)) != 0) {
        std::cerr << "! Error: FFT size must be power of 2" << std::endl;
        return;
    }

    // Bit-reversal reordering
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(data[i], data[j]);
        }
    }

    // Cooley-Tukey decimation-in-time radix-2 FFT
    for (int len = 2; len <= n; len <<= 1) {
        const double angle = -2.0 * M_PI / len;
        const std::complex<double> wlen(std::cos(angle), std::sin(angle));

        for (int i = 0; i < n; i += len) {
            std::complex<double> w(1.0, 0.0);
            for (int j = 0; j < len / 2; ++j) {
                const std::complex<double> u = data[i + j];
                const std::complex<double> v = data[i + j + len / 2] * w;
                data[i + j] = u + v;
                data[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

// ============================================================================
// UTILIDADES
// ============================================================================

int nextPowerOf2(int n) {
    int power = 1;
    while (power < n) power <<= 1;
    return power;
}

// ============================================================================
// STFT - VERSION CON VECTORES Y DOUBLE
// ============================================================================

std::vector<std::vector<AudioSample>> applySTFT(
    const std::vector<AudioSample>& audio,
    int sampleRate)
{
    if (audio.empty()) {
        std::cerr << "! Error: Audio invalido en STFT" << std::endl;
        return {};
    }

    const int totalSamples = static_cast<int>(audio.size());

    // Obtener configuracion
    auto& cfg = CONFIG_STFT;
    const int frameSize = sampleRate * cfg.frameSizeMs / 1000;
    const int frameStride = sampleRate * cfg.frameStrideMs / 1000;

    if (frameSize <= 0 || frameStride <= 0 || frameSize > totalSamples) {
        std::cerr << "! Error: Parametros STFT invalidos" << std::endl;
        return {};
    }

    const int numFrames = (totalSamples - frameSize) / frameStride + 1;
    const int fftSize = nextPowerOf2(frameSize);
    const int numBins = fftSize / 2;

    if (CONFIG_PREP.verbose) {
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "[ETAPA 3/6] SEGMENTACION - STFT (Short-Time Fourier Transform)" << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        std::cout << "\n# Parametros de STFT" << std::endl;
        std::cout << "   +" << std::string(66, '-') << "+" << std::endl;
        std::cout << "   | Parametro              | Valor configurado      | Muestras         " << std::endl;
        std::cout << "   +" << std::string(66, '-') << "+" << std::endl;
        std::cout << "   | Tamanho de ventana     | " << std::setw(19) << cfg.frameSizeMs << "ms | " << std::setw(10) << frameSize << " samp. " << std::endl;
        std::cout << "   | Stride (hop)           | " << std::setw(19) << cfg.frameStrideMs << "ms | " << std::setw(10) << frameStride << " samp. " << std::endl;
        std::cout << "   | FFT size (zero-pad)    | " << std::setw(22) << fftSize << " | (potencia 2)     |" << std::endl;
        std::cout << "\n# Resultado de segmentacion" << std::endl;
        std::cout << "   Frames generados: " << numFrames << std::endl;
        std::cout << "   Bins de frecuencia: " << numBins << std::endl;
    }

    if (numFrames <= 0) {
        std::cerr << "! Error: Numero de frames invalido" << std::endl;
        return {};
    }

    // Inicializar espectrograma con vectores (gestion automatica)
    std::vector<std::vector<AudioSample>> spectrogram(
        numFrames,
        std::vector<AudioSample>(numBins, 0.0)
    );

    // Procesar frames en paralelo (si OpenMP esta activado)
#if ENABLE_OPENMP
#pragma omp parallel
#endif
    {
        // Buffer local para cada thread
        std::vector<std::complex<double>> fftInput(fftSize);

#if ENABLE_OPENMP
#pragma omp for
#endif
        for (int f = 0; f < numFrames; ++f) {
            const int start = f * frameStride;

            // Extraer frame y aplicar ventana Hann
            for (int j = 0; j < fftSize; ++j) {
                double sample = 0.0;

                if (j < frameSize) {
                    const int idx = start + j;
                    if (idx < totalSamples) {
                        sample = audio[idx];
                    }

                    // Ventana Hann en double precision
                    const double window = 0.5 * (1.0 - std::cos(2.0 * M_PI * j / (frameSize - 1)));
                    sample *= window;
                }

                fftInput[j] = std::complex<double>(sample, 0.0);
            }

            // Aplicar FFT
            fft_iterative(fftInput);

            // Calcular magnitudes espectrales (sqrt(re^2 + im^2))
            for (int k = 0; k < numBins; ++k) {
                const double re = fftInput[k].real();
                const double im = fftInput[k].imag();
                spectrogram[f][k] = std::sqrt(re * re + im * im);
            }
        }
    }

    if (CONFIG_PREP.verbose) {
        std::cout << "   & STFT completado: " << numFrames << " frames x " << numBins << " bins" << std::endl;
    }

    return spectrogram;
}