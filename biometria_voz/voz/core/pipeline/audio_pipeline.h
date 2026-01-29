#ifndef AUDIO_PIPELINE_H
#define AUDIO_PIPELINE_H

#include "config.h"
#include <vector>
#include <filesystem>
#include <optional>

// ============================================================================
// MODULO PIPELINE - PROCESAMIENTO COMPLETO DE AUDIO - VERSION 3.0
// ============================================================================
// 
// Cambios principales:
// - float → AudioSample (double)
// - Uso de std::optional para manejo de errores robusto
// - Sin punteros raw - todo con vectores
// - Pipeline completamente automatizado
// - Cero memory leaks garantizados
//
// Pipeline optimizado:
//   Archivo → Normalize → Denoising → VAD → STFT → MFCC → Estadisticas → Features
//
// UNICA FUNCION PUBLICA: procesarAudioCompleto()
//   - El flujo del sistema SOLO debe llamar a esta funcion
//   - Internamente maneja augmentation si esta configurado
//   - Retorna N vectores de features (N=1 sin augmentation, N>1 con augmentation)
// ============================================================================

/**
 * Procesa un archivo de audio completo (UNICA FUNCION PUBLICA DEL PIPELINE)
 *
 * Pipeline completo:
 *   Archivo → Normalize → Denoising → VAD → STFT → MFCC → Stats → Features
 *
 * Esta funcion:
 * 1. Carga el audio desde archivo
 * 2. Decide si aplicar augmentation segun CONFIG
 * 3. Si augmentation:
 *    - Genera (numVariaciones + 1) versiones del audio
 *    - Procesa cada version internamente
 *    - Retorna N vectores de features
 * 4. Si NO augmentation:
 *    - Procesa solo el original
 *    - Retorna 1 vector de features
 *
 * @param audioPath Ruta del archivo (.mp3, .wav, .flac, .aiff)
 * @param outFeatures Vector de salida:
 *                    - SIN augmentation: 1 vector de N features
 *                    - CON augmentation: M vectores de N features
 * @return true si se proceso al menos 1 muestra exitosamente
 *
 * Configuracion de augmentation:
 *   CONFIG_DATASET.usarAugmentation = true/false  // Master switch
 *   CONFIG_AUG.numVariaciones = N                 // 0 = sin augmentation
 *
 * Se activa augmentation SOLO si:
 *   CONFIG_DATASET.usarAugmentation == true  AND
 *   CONFIG_AUG.numVariaciones > 0
 *
 * Ejemplo SIN augmentation:
 *   CONFIG_DATASET.usarAugmentation = false;
 *
 *   std::vector<std::vector<AudioSample>> features;
 *   bool ok = procesarAudioCompleto("audio.wav", features);
 *   // features.size() == 1
 *   // features[0].size() == CONFIG_MFCC.numCoefficients
 *
 * Ejemplo CON augmentation:
 *   CONFIG_DATASET.usarAugmentation = true;
 *   CONFIG_AUG.numVariaciones = 4;
 *
 *   std::vector<std::vector<AudioSample>> features;
 *   bool ok = procesarAudioCompleto("audio.wav", features);
 *   // features.size() == 5 (original + 4 variaciones)
 *   // features[i].size() == CONFIG_MFCC.numCoefficients
 *
 * VENTAJAS VERSION 3.0:
 * - Precision double en todo el pipeline
 * - Manejo robusto de errores con optional
 * - Cero memory leaks (RAII automatico)
 * - Codigo mas limpio y mantenible
 * - Logging condicional (CONFIG_PREP.verbose)
 */
bool procesarAudioCompleto(
    const std::filesystem::path& audioPath,
    std::vector<std::vector<AudioSample>>& outFeatures
);

#endif // AUDIO_PIPELINE_H