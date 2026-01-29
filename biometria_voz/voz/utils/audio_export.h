#ifndef AUDIO_EXPORT_H
#define AUDIO_EXPORT_H

#include <vector>
#include "../utils/config.h"  // Para AudioSample typedef

// ============================================================================
// API DE EXPORTACION - VERSION REFACTORIZADA
// Cambio principal: float** → std::vector<std::vector<AudioSample>>
// Beneficio: Precision double hasta el ultimo momento, sin memory leaks
// ============================================================================

// Exporta audio como archivo WAV de 32-bit float
// Nota: Conversion a float ocurre INTERNAMENTE solo al escribir
// El audio en memoria permanece en AudioSample (double) hasta el final
void saveAudioToWav(const char* wavFile, const AudioSample* audioData, 
                    int numSamples, int sampleRate, int numChannels);

// Exporta espectrograma (STFT) a CSV con metadatos
// Formato: fila=frame temporal, columna=bin frecuencial
// Incluye header con frecuencias en Hz
// Conversion a float ocurre SOLO al escribir CSV (precision completa en memoria)
bool exportarEspectrogramaCSV(const std::vector<std::vector<AudioSample>>& spectrogram,
                               int sampleRate, 
                               int fftSize,
                               const char* filepath);

// Exporta coeficientes MFCC por frame a CSV
// Formato: fila=frame, columna=coeficiente MFCC (0 a numCoeffs-1)
// Conversion a float ocurre SOLO al escribir CSV
bool exportarMFCC_CSV(const std::vector<std::vector<AudioSample>>& mfcc,
                      const char* filepath);

// Exporta estadisticas MFCC finales (vector de features agregadas) a CSV
// Formato: vector simple con valores (mean de cada coeficiente)
// Conversion a float ocurre SOLO al escribir CSV
bool exportarEstadisticasMFCC_CSV(const std::vector<AudioSample>& stats, 
                                   const char* filepath);

#endif // AUDIO_EXPORT_H