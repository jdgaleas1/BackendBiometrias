#ifndef PREPROCESAR_H
#define PREPROCESAR_H

#include <vector>
#include "config.h"  // Para AudioSample

/**
 * ===================================================================
 * API PUBLICA DE PREPROCESAMIENTO DE AUDIO - VERSION 3.0 REFACTORIZADA
 * ===================================================================
 *
 * Esta API unifica todas las funciones de preprocesamiento de audio
 * necesarias para el pipeline de reconocimiento de voz biometrico.
 *
 * CAMBIOS VERSION 3.0:
 * - Migracion completa a std::vector<AudioSample> (double)
 * - Eliminacion de punteros raw y gestion manual de memoria
 * - API moderna con RAII automatico
 * - Mejor precision numerica en todas las operaciones
 *
 * ORDEN RECOMENDADO DE APLICACION (PIPELINE ACTUAL):
 * 1. Normalizacion (RMS/Peak/AGC)
 * 2. VAD Avanzado
 *
 * IMPACTO EN ACCURACY (PIPELINE ACTUAL):
 * - TIER 1 (>15% mejora): VAD Avanzado, Normalizacion
 *
 * EJEMPLO DE PIPELINE COMPLETO:
 *   // 1. Normalizar volumen
 *   auto normalized = normalizeRMS(raw_audio, 0.1);
 *
 *   // 2. Detectar y extraer solo voz activa
 *   auto voice = applyVAD(normalized, sampleRate);
 *
 *   // 4. Continuar con extraccion de caracteristicas (MFCC)
 *   // auto mfcc = extractMFCC(voice, sampleRate);
 */

 // ===================================================================
 // NORMALIZACION - Compensa diferencias de volumen y ganancia
 // ===================================================================

 /**
  * Normalizacion RMS (Root Mean Square) - RECOMENDADO
  * Ajusta el audio a un nivel RMS objetivo manteniendo la dinamica relativa.
  *
  * @param audio Buffer de entrada
  * @param targetRMS Nivel RMS objetivo (0.05-0.2, default: 0.1)
  * @return Audio normalizado (auto-gestionado)
  */
std::vector<AudioSample> normalizeRMS(const std::vector<AudioSample>& audio,
    AudioSample targetRMS = 0.1);

/**
 * Normalizacion por pico maximo
 * Escala para que el pico mas alto sea ±targetPeak.
 *
 * @param audio Buffer de entrada
 * @param targetPeak Amplitud maxima objetivo (default: 0.95)
 * @return Audio normalizado (auto-gestionado)
 */
std::vector<AudioSample> normalizePeak(const std::vector<AudioSample>& audio,
    AudioSample targetPeak = 0.95);

/**
 * Control automatico de ganancia (AGC)
 * Normaliza dinamicamente por segmentos temporales.
 *
 * @param audio Buffer de entrada
 * @param sampleRate Frecuencia de muestreo (Hz)
 * @param windowMs Ventana de analisis en ms (default: 500)
 * @param targetRMS RMS objetivo por ventana (default: 0.1)
 * @return Audio con AGC aplicado (auto-gestionado)
 */
std::vector<AudioSample> automaticGainControl(const std::vector<AudioSample>& audio,
    int sampleRate,
    int windowMs = 500,
    AudioSample targetRMS = 0.1);

/**
 * UTILIDADES DE ANALISIS
 */
AudioSample calcularRMS(const std::vector<AudioSample>& audio);
AudioSample encontrarPico(const std::vector<AudioSample>& audio);

// ===================================================================
// VAD - Voice Activity Detection
// ===================================================================

/**
 * VAD Avanzado - Deteccion de actividad vocal
 * Consume la segmentacion generada por Denoising para decision mas precisa
 *
 * @param audio Buffer de entrada (post-denoising)
 * @param sampleRate Frecuencia de muestreo (Hz)
 * @return Audio con solo segmentos de voz (auto-gestionado)
 *
 * NOTA: Este VAD usa la segmentacion del denoising previo para
 *       clasificar frames como voz/ruido de manera mas precisa
 */
std::vector<AudioSample> applyVAD(const std::vector<AudioSample>& audio,
    int sampleRate);

#endif // PREPROCESAR_H