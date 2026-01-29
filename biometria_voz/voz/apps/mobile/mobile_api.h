#ifndef MOBILE_API_H
#define MOBILE_API_H

#include <stdint.h>
#include <stddef.h>

// ============================================================================
// API C para FFI (Flutter/Dart)
// ============================================================================

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// GESTION DE LIBRERIA
// ============================================================================

/**
 * Inicializar la libreria biometrica
 * @param db_path Ruta a la base de datos SQLite local
 * @param model_path Ruta al directorio de modelos SVM
 * @param dataset_path Ruta al dataset procesado
 * @return 0 si exito, -1 si error
 */
int voz_mobile_init(const char* db_path, 
                    const char* model_path,
                    const char* dataset_path);

/**
 * Liberar recursos de la libreria
 */
void voz_mobile_cleanup();

/**
 * Obtener version de la libreria
 * @return String con la version (ej: "1.0.0")
 */
const char* voz_mobile_version();

// ============================================================================
// USUARIOS
// ============================================================================

/**
 * Obtener ID de usuario por identificador unico
 * @param identificador Cedula o identificador unico
 * @return ID del usuario, -1 si no existe
 */
int voz_mobile_obtener_id_usuario(const char* identificador);

/**
 * Crear nuevo usuario
 * @param identificador Cedula o identificador unico
 * @return ID del usuario creado, -1 si error
 */
int voz_mobile_crear_usuario(const char* identificador);

/**
 * Verificar si usuario existe
 * @param identificador Cedula o identificador unico
 * @return 1 si existe, 0 si no existe
 */
int voz_mobile_usuario_existe(const char* identificador);

// ============================================================================
// FRASES DINAMICAS
// ============================================================================

/**
 * Obtener frase aleatoria activa
 * @param buffer Buffer donde se copiara la frase
 * @param buffer_size Tamaño del buffer
 * @return ID de la frase seleccionada, -1 si error
 */
int voz_mobile_obtener_frase_aleatoria(char* buffer, size_t buffer_size);

/**
 * Obtener frase por ID
 * @param id_frase ID de la frase
 * @param buffer Buffer donde se copiara la frase
 * @param buffer_size Tamaño del buffer
 * @return 0 si exito, -1 si error
 */
int voz_mobile_obtener_frase_por_id(int id_frase, char* buffer, size_t buffer_size);

/**
 * Insertar nuevas frases en la base de datos local
 * @param frases_json JSON con array de frases: [{"frase": "...", "categoria": "..."}, ...]
 * @return Cantidad de frases insertadas, -1 si error
 */
int voz_mobile_insertar_frases(const char* frases_json);

// ============================================================================
// REGISTRO BIOMETRICO
// ============================================================================

/**
 * Registrar biometria de voz
 * @param identificador Cedula del usuario
 * @param audio_path Ruta al archivo de audio WAV (temporal)
 * @param id_frase ID de la frase pronunciada
 * @param resultado_json Buffer donde se copiara el resultado JSON
 * @param buffer_size Tamaño del buffer de resultado
 * @return 0 si exito, -1 si error
 */
int voz_mobile_registrar_biometria(const char* identificador,
                                    const char* audio_path,
                                    int id_frase,
                                    char* resultado_json,
                                    size_t buffer_size);

// ============================================================================
// AUTENTICACION
// ============================================================================

/**
 * Autenticar usuario por voz
 * @param identificador Cedula del usuario
 * @param audio_path Ruta al archivo de audio WAV (temporal)
 * @param id_frase ID de la frase pronunciada
 * @param resultado_json Buffer donde se copiara el resultado JSON
 * @param buffer_size Tamaño del buffer de resultado
 * @return 1 si autenticado, 0 si rechazado, -1 si error
 */
int voz_mobile_autenticar(const char* identificador,
                          const char* audio_path,
                          int id_frase,
                          char* resultado_json,
                          size_t buffer_size);

// ============================================================================
// SINCRONIZACION
// ============================================================================

/**
 * Push: enviar vectores pendientes al servidor
 * @param server_url URL del servidor (ej: "http://localhost:8080")
 * @param resultado_json Buffer donde se copiara el resultado JSON
 * @param buffer_size Tamaño del buffer
 * @return 0 si exito, -1 si error
 */
int voz_mobile_sync_push(const char* server_url, char* resultado_json, size_t buffer_size);

/**
 * Pull: descargar cambios del servidor (frases, estados)
 * @param server_url URL del servidor
 * @param desde Timestamp desde cuando obtener cambios (opcional, "" para todas)
 * @param resultado_json Buffer donde se copiara el resultado JSON
 * @param buffer_size Tamaño del buffer
 * @return 0 si exito, -1 si error
 */
int voz_mobile_sync_pull(const char* server_url, const char* desde, char* resultado_json, size_t buffer_size);

/**
 * Pull modelo: descargar modelo re-entrenado del servidor
 * @param server_url URL del servidor
 * @param identificador Cedula del usuario
 * @param resultado_json Buffer donde se copiara el resultado JSON
 * @param buffer_size Tamaño del buffer
 * @return 0 si exito, -1 si error
 */
int voz_mobile_sync_modelo(const char* server_url, const char* identificador, char* resultado_json, size_t buffer_size);

/**
 * Obtener UUID del dispositivo
 * @param buffer Buffer donde se copiara el UUID
 * @param buffer_size Tamaño del buffer
 * @return 0 si exito, -1 si error
 */
int voz_mobile_obtener_uuid_dispositivo(char* buffer, size_t buffer_size);

/**
 * Establecer UUID del dispositivo
 * @param uuid UUID a establecer
 * @return 0 si exito, -1 si error
 */
int voz_mobile_establecer_uuid_dispositivo(const char* uuid);

// ============================================================================
// UTILIDADES
// ============================================================================

/**
 * Obtener ultimo error ocurrido
 * @param buffer Buffer donde se copiara el mensaje de error
 * @param buffer_size Tamaño del buffer
 */
void voz_mobile_obtener_ultimo_error(char* buffer, size_t buffer_size);

/**
 * Obtener estadisticas del modelo
 * @param stats_json Buffer donde se copiara el JSON con estadisticas
 * @param buffer_size Tamaño del buffer
 * @return 0 si exito, -1 si error
 */
int voz_mobile_obtener_estadisticas(char* stats_json, size_t buffer_size);

#ifdef __cplusplus
}
#endif

#endif // MOBILE_API_H
