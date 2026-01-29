#ifndef AUDIO_AUGMENTATION_H
#define AUDIO_AUGMENTATION_H

#include <vector>
#include "config.h"  // Para AudioSample

// Forward declaration - la definicion real esta en config.h
struct ConfigAugmentation;

// ================================================================================
// MODULO DE DATA AUGMENTATION PARA VOZ - VERSION 3.0 REFACTORIZADA
// ================================================================================
// Este modulo implementa tecnicas tradicionales de perturbacion de audio
// para aumentar la robustez de sistemas biometricos de voz.
//
// CAMBIOS VERSION 3.0:
// - Migracion completa a std::vector<AudioSample> (double)
// - Eliminacion de punteros raw y gestion manual de memoria
// - API moderna con RAII automatico
// - Mejor precision numerica en todas las operaciones
//
// Tecnicas implementadas:
// 1. Ruido blanco aditivo (simula ruido ambiental)
// 2. Escalado de volumen (simula distancia del microfono)
// 3. Cambio de velocidad (simula variaciones en tempo de habla)
// 4. Combinaciones de las anteriores
//
// NOTA IMPORTANTE: La configuracion se obtiene de config.h (CONFIG_AUG)
//                  Si no se pasa config=nullptr, usa la configuracion global
//
// Uso tipico:
//   auto variaciones = generarVariacionesAudio(audio, 4);
//   for (const auto& variacion : variaciones) {
//       // Procesar cada variacion (auto-liberacion)
//   }
// ================================================================================

// ================================================================================
// ENUMERACION DE TIPOS DE PERTURBACION
// ================================================================================

enum TipoPerturbacion {
    PERTURBACION_ORIGINAL = 0,      // Sin perturbacion (audio original)
    PERTURBACION_RUIDO = 1,         // Ruido blanco aditivo (SNR ~30dB)
    PERTURBACION_VOLUMEN = 2,       // Escalado de volumen (±10%)
    PERTURBACION_RUIDO_VOLUMEN = 3, // Ruido + volumen combinado
    PERTURBACION_VELOCIDAD = 4      // Cambio de velocidad (±5%)
};

// ================================================================================
// FUNCIONES DE GENERACION DE VARIACIONES
// ================================================================================

/**
 * Genera multiples variaciones de un audio aplicando diferentes perturbaciones
 *
 * @param audioOriginal Audio de entrada (vector de muestras)
 * @param numVariacionesExtra Numero de variaciones perturbadas a generar (sin contar el original)
 * @param config Configuracion de augmentation (nullptr usa CONFIG_AUG global)
 * @return Vector de vectores, donde [0] es el original, [1..N] son las variaciones
 *
 * IMPORTANTE:
 * - El audio original se incluye como variaciones[0]
 * - Liberacion automatica de memoria (RAII)
 * - Cada variacion puede tener diferente longitud (especialmente con cambio de velocidad)
 *
 * Ejemplo:
 *   std::vector<AudioSample> audio = cargarAudio("voz.wav");
 *   auto variaciones = generarVariacionesAudio(audio, 4);
 *   // variaciones[0] = original
 *   // variaciones[1] = con ruido
 *   // variaciones[2] = volumen modificado
 *   // variaciones[3] = ruido + volumen
 *   // variaciones[4] = velocidad modificada
 */
std::vector<std::vector<AudioSample>> generarVariacionesAudio(
    const std::vector<AudioSample>& audioOriginal,
    int numVariacionesExtra,
    ConfigAugmentation* config = nullptr
);

// ================================================================================
// FUNCIONES DE PERTURBACION INDIVIDUALES
// ================================================================================

/**
 * Aplica ruido blanco aditivo al audio
 *
 * @param audio Audio de entrada
 * @param intensidad Intensidad del ruido (desviacion estandar)
 * @param seed Semilla para reproducibilidad
 * @return Nuevo vector con audio ruidoso (auto-gestionado)
 *
 * Formula: audio_out[i] = audio[i] + N(0, intensidad²)
 * donde N es distribucion normal
 */
std::vector<AudioSample> aplicarRuidoBlanco(const std::vector<AudioSample>& audio,
    AudioSample intensidad,
    unsigned int seed);

/**
 * Aplica escalado de volumen al audio
 *
 * @param audio Audio de entrada
 * @param factor Factor de escalado (1.0 = sin cambio, 0.5 = mitad volumen, 2.0 = doble)
 * @return Nuevo vector con volumen escalado (auto-gestionado)
 *
 * Formula: audio_out[i] = audio[i] * factor
 *
 * Incluye soft-clipping para evitar saturacion:
 * - Valores > 1.0 se comprimen suavemente
 * - Valores < -1.0 se comprimen suavemente
 */
std::vector<AudioSample> aplicarEscaladoVolumen(const std::vector<AudioSample>& audio,
    AudioSample factor);

/**
 * Aplica cambio de velocidad mediante interpolacion lineal
 *
 * @param audio Audio de entrada
 * @param factor Factor de velocidad (1.0 = sin cambio, 0.95 = mas lento, 1.05 = mas rapido)
 * @return Nuevo vector con velocidad modificada (auto-gestionado)
 *
 * NOTA: El numero de muestras de salida cambia: N_out ≈ N_in / factor
 *
 * Implementacion:
 * - Usa interpolacion lineal para precision
 * - Factor < 1.0: estira el audio (mas lento, mas muestras)
 * - Factor > 1.0: comprime el audio (mas rapido, menos muestras)
 */
std::vector<AudioSample> aplicarCambioVelocidad(const std::vector<AudioSample>& audio,
    AudioSample factor);

/**
 * Aplica una perturbacion especifica segun el tipo
 *
 * @param audio Audio de entrada
 * @param tipo Tipo de perturbacion a aplicar
 * @param config Configuracion de augmentation (usa valores de aqui para intensidades)
 * @param seed Semilla para reproducibilidad
 * @return Audio perturbado (auto-gestionado)
 *
 * Funcion auxiliar que delega a las funciones especificas segun el tipo
 */
std::vector<AudioSample> aplicarPerturbacion(const std::vector<AudioSample>& audio,
    TipoPerturbacion tipo,
    const ConfigAugmentation& config,
    unsigned int seed);

#endif // AUDIO_AUGMENTATION_H