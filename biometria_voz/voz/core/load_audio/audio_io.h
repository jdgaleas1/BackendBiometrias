#ifndef AUDIO_IO_H
#define AUDIO_IO_H

#include <cstdint>
#include <vector>
#include "config.h"  // Para AudioSample
// ============================================================================
// MODULO DE CARGA DE AUDIO - DECODIFICACION MULTIFORMATO
// ============================================================================
// Este modulo proporciona funciones para cargar archivos de audio de
// multiples formatos y convertirlos a formato float normalizado mono.

// VALIDACION DE VOZ ANTES DE ENTRAR AL FUJO:
// -Es demasiado silencioso (RMS < 0.005)
// -Es demasiado ruidoso (clipping > 5%)
// -Rechazará audios que duren menos de 5 segundos (muy cortos)

// Formatos soportados:
// - MP3  (.mp3)  - via minimp3
// - WAV  (.wav)  - via dr_wav
// - AIFF (.aiff) - via dr_wav
// - FLAC (.flac) - via dr_flac
//
// Caracteristicas:
// - Decodificacion automatica segun extension
// - Conversion automatica a mono
// - Normalizacion a rango [-1.0, 1.0]
// - Validaciones completas de parametros
// - Estadisticas de audio
// - Deteccion de clipping y silencio
//
// Limites de seguridad:
// - Sample rate: 8 kHz - 48 kHz
// - Canales: 1-2 (mono/estereo)
// - Duracion maxima: 5 minutos
// - Muestras minimas: 1000
// ============================================================================

// ============================================================================
// FUNCION PRINCIPAL
// ============================================================================

/**
 * Decodifica cualquier archivo de audio soportado a formato float mono
 *
 * Esta funcion:
 * 1. Detecta el formato automaticamente por extension
 * 2. Decodifica el audio usando la libreria apropiada
 * 3. Valida parametros (sample rate, canales, duracion)
 * 4. Convierte a mono si es necesario (promediando canales)
 * 5. Normaliza a rango [-1.0, 1.0]
 * 6. Calcula estadisticas basicas
 * 7. Detecta problemas (clipping, silencio)
 *
 * Formatos soportados:
 * - MP3  (.mp3)  - Decodificacion via minimp3
 * - WAV  (.wav)  - Soporta PCM 8/16/24/32 bits, float 32/64 bits
 * - AIFF (.aiff) - Igual que WAV
 * - FLAC (.flac) - Compresion sin perdidas
 *
 * Limites de seguridad:
 * - Sample rate: 8000 - 48000 Hz
 * - Canales: 1-2
 * - Duracion maxima: 300 segundos (5 minutos)
 * - Muestras minimas: 1000
 *
 * @param filePath Ruta completa del archivo de audio
 *                 Debe incluir extension (.mp3, .wav, .aiff, .flac)
 *                 Ejemplo: "/ruta/audio.mp3" o "audio.wav"
 *
 * @param sampleRate [salida] Frecuencia de muestreo en Hz
 *                   Valores tipicos: 8000, 16000, 22050, 44100, 48000
 *
 * @param numChannels [salida] Numero de canales
 *                    Siempre sera 1 tras conversion a mono
 *
 * @param numSamples [salida] Numero total de muestras mono
 *
 * @return Vector de muestras de audio en formato AudioSample (double)
 *         Vector vacio si falla la decodificacion
 *         El vector se gestiona automaticamente (RAII) - no requiere liberacion manual
 *
 * IMPORTANTE CAMBIO VERSION 3.0:
 * - Antes retornaba float* que requeria delete[]
 * - Ahora retorna std::vector<AudioSample> (double) con gestion automatica
 * - Cero riesgo de memory leaks
 * - Mejor precision numerica (double vs float)
 *
 * Ejemplo de uso:
 * @code
 *   int sampleRate, numChannels, numSamples;
 *   auto audio = loadAudio("grabacion.mp3", sampleRate, numChannels, numSamples);
 *
 *   if (!audio.empty()) {
 *       std::cout << "Sample rate: " << sampleRate << " Hz" << std::endl;
 *       std::cout << "Duracion: " << (numSamples / (double)sampleRate) << " seg" << std::endl;
 *
 *       // ... procesar audio ...
 *       // NO se requiere liberacion manual - RAII automatico
 *   } else {
 *       std::cerr << "Error al cargar audio" << std::endl;
 *   }
 * @endcode
 *
 * Mensajes de error comunes:
 * - "! Error: Formato no soportado" → Extension no reconocida
 * - "! Error: Sample rate fuera de rango" → Frecuencia invalida
 * - "! Error: Archivo demasiado largo" → Supera 5 minutos
 * - "! Error critico: Fallo asignacion de memoria" → Archivo muy grande
 * - "% Warning: Clipping detectado" → Audio saturado
 * - "% Warning: Audio muy silencioso" → RMS muy bajo
 *
 * Complejidad: O(N) donde N = numero de muestras
 *
 * Thread-safety: La decodificacion usa OpenMP para paralelizacion
 *                pero la funcion no es thread-safe para el mismo archivo
 */
std::vector<AudioSample> loadAudio(const char* filePath,
    int& sampleRate,
    int& numChannels,
    int& numSamples);

// ============================================================================
// NOTAS DE IMPLEMENTACION
// ============================================================================
// 
// FORMATOS SOPORTADOS EN DETALLE:
// 
// 1. MP3 (.mp3):
//    - Usa minimp3 (header-only, sin dependencias)
//    - Soporta todos los bitrates y sample rates comunes
//    - Decodificacion rapida y eficiente
//    - Normaliza de int16 a float automaticamente
// 
// 2. WAV (.wav) y AIFF (.aiff):
//    - Usa dr_wav (header-only)
//    - Soporta:
//      * PCM 8, 16, 24, 32 bits
//      * Float 32, 64 bits
//      * Multiples sample rates
//    - Decodificacion directa a float
// 
// 3. FLAC (.flac):
//    - Usa dr_flac (header-only)
//    - Compresion sin perdidas
//    - Menor tamaño que WAV con misma calidad
//    - Decodificacion directa a float
// 
// CONVERSION A MONO:
// - Si el audio tiene 2 canales (estereo):
//   mono[i] = (izquierdo[i] + derecho[i]) / 2
// - Si tiene 1 canal (mono):
//   No se modifica
// - Usa OpenMP para paralelizacion
// 
// NORMALIZACION:
// - Los formatos se decodifican a float en rango [-1.0, 1.0]
// - MP3: Se divide entre 32768 (max de int16)
// - WAV/FLAC: Las librerias normalizan automaticamente
// 
// VALIDACIONES DE SEGURIDAD:
// - Sample rate: [8000, 48000] Hz
//   * 8 kHz: Minimo para voz telefonica
//   * 48 kHz: Maximo para audio profesional
// 
// - Canales: [1, 2]
//   * Mono o estereo solamente
//   * Configuraciones superiores no soportadas
// 
// - Duracion: <= 300 segundos (5 minutos)
//   * Previene archivos muy grandes
//   * Para archivos mas largos, procesar por chunks
// 
// - Muestras minimas: >= 1000
//   * Asegura que haya contenido util
//   * Rechaza archivos casi vacios
// 
// DETECCION DE PROBLEMAS:
// - Clipping: |muestra| >= 0.99
//   * Indica saturacion/distorsion
//   * Puede requerir normalizacion previa
// 
// - Silencio: RMS < 0.001
//   * Audio muy bajo o silencioso
//   * Puede requerir amplificacion
// 
// MANEJO DE MEMORIA:
// - Usa new (std::nothrow) para todas las asignaciones
// - Verifica nullptr antes de usar memoria
// - Libera memoria en todos los casos de error
// - El usuario debe liberar con delete[] el resultado
// 
// RENDIMIENTO:
// - Paralelizacion con OpenMP en:
//   * Conversion int16 → float (MP3)
//   * Conversion a mono
//   * Calculo de estadisticas
// - Tiempo de carga tipico:
//   * 1 seg de audio @ 44.1 kHz: ~10 ms
//   * 5 min de audio @ 44.1 kHz: ~500 ms
// ============================================================================

#endif // AUDIO_IO_H