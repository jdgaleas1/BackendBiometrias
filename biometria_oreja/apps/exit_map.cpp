// exit_map.cpp

#include "exit_map.h"
#include <string>

static ExitMapped genericMapped(const std::string& proc, int exit_code) {
    ExitMapped m;
    m.http_status = 500;
    m.title = "PROCESO_FALLIDO";
    m.message = "El proceso '" + proc + "' finalizó con exit_code=" + std::to_string(exit_code) + ".";
    return m;
}

ExitMapped mapExitCode(const std::string& proc, int exit_code) {

    // =========================
    // MAPEO ESPECÍFICO: agregar_usuario (tabla REAL)
    // =========================
    if (proc == "agregar_usuario") {
        switch (exit_code) {
            case 0:
                return ExitMapped{200, "OK", "Proceso completado correctamente."};

            case 2:
                return ExitMapped{409, "DUPLICADO", "El usuario ya existe en la base de datos: (PostgREST 409)."};
            case 5:
                return ExitMapped{400, "ENTRADA_INVALIDA", "Faltan campos mínimos: identificador_unico, nombres, apellidos."};

            // Errores internos de integración (servidor -> exe)
            case 10:
                return ExitMapped{500, "FS_INTEGRACION", "No existe datos.json en WORK_DIR (fallo de integración servidor->exe)."};
            case 3:
                return ExitMapped{500, "FS_INTEGRACION", "datos.json vacío o no se pudo leer (fallo de integración)."};
            case 4:
                return ExitMapped{500, "FS_INTEGRACION", "datos.json no es JSON válido (fallo de integración)."};
            case 8:
                return ExitMapped{502, "RESPUESTA_INVALIDA", "PostgREST respondió con cuerpo no-JSON al crear usuario."};
            case 9:
                return ExitMapped{502, "RESPUESTA_INCOMPLETA", "PostgREST no devolvió id_usuario (revisa Prefer: return=representation)."};
            case 6:
                return ExitMapped{502, "DEPENDENCIA_EXTERNA", "PostgREST/BD sin respuesta o backend caído."};
            case 7:
                return ExitMapped{502, "ERROR_POSTGREST", "PostgREST devolvió un error no clasificado al crear usuario."};

            default:
                return genericMapped(proc, exit_code);
        }
    }

    // =========================
    // MAPEO ESPECÍFICO: agregar_usuario_biometria (OREJA)
    // Basado en la tabla que definiste en el exe:
    // 0 OK
    // 10..24 errores
    // =========================
    if (proc == "agregar_usuario_biometria") {
        switch (exit_code) {
            case 0:
                return ExitMapped{200, "OK", "Registro biométrico completado correctamente."};

            // ---- Errores de entrada / preparación (cliente o integración FS) ----
            case 10:
                return ExitMapped{500, "FS_INTEGRACION", "WORK_DIR inválido o no existe (integración servidor->exe)."};
            case 11:
                return ExitMapped{400, "ENTRADA_INVALIDA", "Se requieren EXACTAMENTE 7 imágenes (.jpg) para registrar la oreja."};
            case 12:
                return ExitMapped{400, "ENTRADA_INVALIDA", "No se pudo cargar una o más imágenes (archivo corrupto o formato inválido)."};
            case 14:
                return ExitMapped{500, "PIPELINE_FALLIDO", "No se pudo procesar una o más imágenes. Verifique que las imágenes sean válidas y claras."};
            case 15:
                return ExitMapped{500, "FEATURES_VACIAS", "No se pudieron extraer características biométricas de las imágenes. Intente con imágenes de mejor calidad."};
            case 16:
                return ExitMapped{500, "PCA_FALLIDA", "Error interno del sistema al procesar las características. Contacte al administrador."};
            case 17:
                return ExitMapped{500, "FS_INTEGRACION", "Error interno del sistema (archivos de integración faltantes). Contacte al administrador."};
            case 18:
                return ExitMapped{500, "MODELO_FALTANTE", "El sistema no puede procesar registros debido a un problema de configuración. Contacte al administrador."}; // podría ser 500

            // ---- Reglas de negocio / conflicto ----
            case 19:
                return ExitMapped{409, "CONFLICTO", "Biometría duplicada probable: coincide con una clase existente."};
            case 20:
                return ExitMapped{409, "DUPLICADO", "El usuario ya está registrado en el modelo (la clase ya existe)."};

            // ---- Entrenamiento / guardado / evaluación ----
            case 21:
                return ExitMapped{500, "ENTRENAMIENTO_FALLIDO", "Error interno durante el entrenamiento del modelo. No se completó el registro. Contacte al administrador."};
            case 22:
                return ExitMapped{500, "EVALUACION_FALLIDA", "Error interno durante la evaluación del modelo. No se completó el registro. Contacte al administrador."};
            case 23:
                return ExitMapped{422, "CAIDA_EXCESIVA", "No se pudo completar el registro porque afectaría negativamente el desempeño del sistema. Intente con imágenes de mejor calidad o contacte al administrador."};

            // ---- BD / PostgREST credencial ----
            case 24:
                return ExitMapped{502, "DEPENDENCIA_EXTERNA", "El registro biométrico se completó pero hubo un error al guardar la credencial. Contacte al administrador."};

            // ---- QC global (si QC_ENFORCE=1) ----
            case 13:
                return ExitMapped{422, "CALIDAD_INSUFICIENTE", "No se alcanzó el mínimo de imágenes con calidad suficiente (QC_GLOBAL_FAIL)."};

            default:
                return genericMapped(proc, exit_code);
        }
    }

    // =========================
    // Por defecto (si no hay mapeo específico)
    // =========================
    if (exit_code == 0) {
        return ExitMapped{200, "OK", "Proceso completado correctamente."};
    }

    return genericMapped(proc, exit_code);
}

json buildErrorBody(const std::string& proc,
                    int exit_code,
                    const ExitMapped& mapped,
                    const std::string& stderr_tail,
                    const std::string& stdout_tail) {
    json j;
    j["ok"] = false;
    j["proc"] = proc;
    j["code"] = exit_code;
    j["error"] = {
        {"title", mapped.title},
        {"message", mapped.message},
        {"http_status", mapped.http_status}
    };

    if (!stderr_tail.empty()) j["stderr_tail"] = stderr_tail;
    if (!stdout_tail.empty()) j["stdout_tail"] = stdout_tail;

    return j;
}

json buildPublicErrorBody(int http_status, const std::string& title, const std::string& message) {
    json j;
    j["ok"] = false;
    j["error"] = {
        {"http_status", http_status},
        {"title", title},
        {"message", message}
    };
    return j;
}
