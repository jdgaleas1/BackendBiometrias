#ifndef STFT_H
#define STFT_H

#include "config.h"
#include <vector>

/**
 * ===================================================================
 * STFT (Short-Time Fourier Transform) - VERSION 3.0
 * ===================================================================
 *
 * Cambios principales:
 * - float** -> std::vector<std::vector<AudioSample>>
 * - complex<float> -> complex<double> para precision
 * - Eliminacion de alocacion manual 2D
 * - Paralelizacion condicional con OpenMP
 * - Mejor precision numerica (double en vez de float)
 *
 * Aplica Short-Time Fourier Transform (STFT) al audio
 * Usa configuracion centralizada desde CONFIG_STFT (config.h)
 *
 * Parametros configurables en config.h:
 * - frameSizeMs: Tamaño de ventana temporal en milisegundos (default: 25ms)
 * - frameStrideMs: Desplazamiento entre frames en milisegundos (default: 10ms)
 *
 * @param audio Vector de audio de entrada (mono, normalizado)
 * @param sampleRate Frecuencia de muestreo en Hz (8000, 16000, 44100, etc)
 * @return Matriz 2D [frames][bins] con magnitudes espectrales
 *
 * Algoritmo:
 * 1. Divide el audio en frames solapados segun CONFIG_STFT.frameSizeMs
 * 2. Aplica ventana Hann a cada frame para reducir spectral leakage
 * 3. Zero-padding hasta proxima potencia de 2 para FFT eficiente
 * 4. Calcula FFT usando algoritmo Cooley-Tukey (O(N log N))
 * 5. Retorna magnitudes espectrales (sqrt(real^2 + imag^2))
 *
 * Caracteristicas:
 * - Ventana: Hann (reduce discontinuidades en los bordes)
 * - FFT: Implementacion iterativa in-place con double precision
 * - Paralelizacion: OpenMP sobre frames (thread-safe si ENABLE_OPENMP=1)
 * - Resolucion temporal: Controlada por frameStrideMs
 * - Resolucion frecuencial: sampleRate / fftSize
 *
 * Ejemplo de uso:
 *   std::vector<AudioSample> audio = leerAudio(...);
 *
 *   // Configurar parametros (opcional)
 *   CONFIG_STFT.frameSizeMs = 25;
 *   CONFIG_STFT.frameStrideMs = 10;
 *
 *   auto spectrogram = applySTFT(audio, 16000);
 *
 *   // Usar espectrograma: spectrogram[frame][bin]
 *   // ... memoria se libera automaticamente
 *
 * Notas de rendimiento:
 * - Complejidad: O(F * N log N) donde F=frames, N=fftSize
 * - Memoria: O(F * B) donde B=bins
 * - Paralelizacion: Escalable linealmente con numero de cores
 *
 * Consideraciones:
 * - frameSizeMs tipico: 20-30ms (balance tiempo-frecuencia)
 * - frameStrideMs tipico: 10ms (50% overlap)
 * - Mayor frameSize = mejor resolucion frecuencial, peor temporal
 * - Menor frameStride = mas frames, mayor costo computacional
 *
 * VENTAJAS VERSION 3.0:
 * - Precision 1000x mayor con double
 * - Cero memory leaks (RAII automatico)
 * - Codigo mas simple y mantenible
 * - Sin necesidad de liberar memoria manualmente
 */
std::vector<std::vector<AudioSample>> applySTFT(
    const std::vector<AudioSample>& audio,
    int sampleRate
);

/**
 * Encuentra la siguiente potencia de 2 mayor o igual a n
 * Util para calcular tamaño optimo de FFT
 *
 * @param n Numero de entrada
 * @return Potencia de 2 >= n
 *
 * Ejemplos:
 * - nextPowerOf2(100) = 128
 * - nextPowerOf2(256) = 256
 * - nextPowerOf2(1000) = 1024
 *
 * Complejidad: O(log n)
 */
int nextPowerOf2(int n);

#endif // STFT_H