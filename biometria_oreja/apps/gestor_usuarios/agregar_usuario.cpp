#include "httplib.h"
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <filesystem>
#include <cctype>
#include <chrono>
#include <iomanip>

using json = nlohmann::json;
namespace fs = std::filesystem;

// ====================== HELPERS ENV ======================
static std::string getEnv(const char* k, const std::string& def) {
    const char* v = std::getenv(k);
    return (v && *v) ? std::string(v) : def;
}

static std::string trim(std::string s) {
    auto issp = [](unsigned char c){ return std::isspace(c); };
    while (!s.empty() && issp((unsigned char)s.front())) s.erase(s.begin());
    while (!s.empty() && issp((unsigned char)s.back())) s.pop_back();
    return s;
}

// ====================== TIMESTAMP ======================
static std::string nowTs() {
    using namespace std::chrono;
    auto now = system_clock::now();
    auto t = system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

// ====================== LOGGER HELPERS (PRESENTACIÓN) ======================

static std::string mkLine(const std::string& rid, const std::string& msg) {
    return std::string("[rid=") + rid + "] " + msg;
}

// Log a stderr (siempre visible en docker logs)
static void logRaw(const std::string& text) {
    std::cerr << text;
    std::cerr.flush();
}

static void logLine(const std::string& rid, const std::string& msg) {
    logRaw(mkLine(rid, msg) + "\n");
}

// ===== ESTILOS PARA DEFENSA =====

// Título principal (solo al inicio)
static void logTitle(const std::string& rid, const std::string& title) {
    logLine(rid, "============================================================");
    logLine(rid, "  " + title);
    logLine(rid, "============================================================");
}

// Separador de fase
static void logPhase(const std::string& rid, int num, const std::string& name, const std::string& objetivo = "") {
    logLine(rid, "------------------------------------------------------------");
    logLine(rid, "[FASE " + std::to_string(num) + "] " + name);
    if (!objetivo.empty()) {
        logLine(rid, "Objetivo: " + objetivo);
    }
    logLine(rid, "------------------------------------------------------------");
}

// Key-value con indentación
static void logKV(const std::string& rid, const std::string& key, const std::string& val, int indent = 2) {
    std::string pad(indent, ' ');
    logLine(rid, pad + "- " + key + ": " + val);
}

// Mensajes de estado
static void logOK(const std::string& rid, const std::string& msg, int indent = 2) {
    std::string pad(indent, ' ');
    logLine(rid, pad + "✓ " + msg);
}

static void logWARN(const std::string& rid, const std::string& msg, int indent = 2) {
    std::string pad(indent, ' ');
    logLine(rid, pad + "⚠ " + msg);
}

static void logERR(const std::string& rid, const std::string& msg, int indent = 2) {
    std::string pad(indent, ' ');
    logLine(rid, pad + "✗ " + msg);
}

// Paso dentro de una fase
static void logStep(const std::string& rid, const std::string& step, const std::string& desc) {
    logLine(rid, "  [" + step + "] " + desc);
}

// Separador final
static void logEnd(const std::string& rid, const std::string& msg) {
    logLine(rid, "============================================================");
    logLine(rid, "  " + msg);
    logLine(rid, "============================================================");
}

// Bloque de datos (JSON formateado, SQL, etc.)
static void logBlock(const std::string& rid, const std::string& title, const std::string& content) {
    logLine(rid, "  ┌─ " + title + " ─");
    std::istringstream iss(content);
    std::string line;
    while (std::getline(iss, line)) {
        logLine(rid, "  │ " + line);
    }
    logLine(rid, "  └─");
}

// ====================== HTTP CLIENT ======================
static httplib::Client makeClient() {
    const std::string host = getEnv("POSTGREST_HOST", "biometria_api");
    int port = 3000;
    try {
        port = std::stoi(getEnv("POSTGREST_PORT", "3000"));
    } catch (...) {
        port = 3000;
    }
    httplib::Client cli(host, port);
    cli.set_read_timeout(60, 0);
    cli.set_write_timeout(60, 0);
    cli.set_connection_timeout(10, 0);
    return cli;
}

// ====================== HELPERS IO ======================
static std::string readAllText(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return "";
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// ====================== MAIN ======================
int main(int argc, char** argv) {
    // ===== PARSEO ARGS =====
    std::string rid = "no-rid";
    bool debug = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--rid" && i + 1 < argc) rid = argv[++i];
        else if (a == "--debug") debug = true;
    }

    const std::string baseDir = getEnv("WORK_DIR", "nuevo_usuario");

    // ============================================================
    // INICIO
    // ============================================================
    logTitle(rid, "REGISTRO DE USUARIO (agregar_usuario)");
    
    logLine(rid, "");
    logKV(rid, "RID", rid, 0);
    logKV(rid, "WORK_DIR", baseDir, 0);
    logKV(rid, "DEBUG", debug ? "habilitado" : "deshabilitado", 0);
    logKV(rid, "POSTGREST_HOST", getEnv("POSTGREST_HOST", "biometria_api"), 0);
    logKV(rid, "POSTGREST_PORT", getEnv("POSTGREST_PORT", "3000"), 0);
    logLine(rid, "");

    // ============================================================
    // FASE 1: VALIDACIÓN FILESYSTEM
    // ============================================================
    logPhase(rid, 1, "VALIDACION DE FILESYSTEM", 
             "Verificar que existe WORK_DIR y datos.json");

    const std::string pathDatos = baseDir + "/datos.json";

    if (!fs::exists(baseDir)) {
        logERR(rid, "WORK_DIR no existe: " + baseDir);
        logEnd(rid, "PROCESO FINALIZADO CON ERROR (exit_code=10)");
        return 10;
    }
    logOK(rid, "WORK_DIR existe");

    if (!fs::exists(pathDatos)) {
        logERR(rid, "Archivo datos.json no encontrado en: " + pathDatos);
        logEnd(rid, "PROCESO FINALIZADO CON ERROR (exit_code=10)");
        return 10;
    }
    logOK(rid, "Archivo datos.json encontrado");
    logKV(rid, "Ruta completa", pathDatos);

    // ============================================================
    // FASE 2: CARGA Y PARSEO JSON
    // ============================================================
    logPhase(rid, 2, "CARGA Y PARSEO DE DATOS", 
             "Leer datos.json y validar estructura JSON");

    const std::string raw = readAllText(pathDatos);
    if (raw.empty()) {
        logERR(rid, "datos.json está vacío o no se pudo leer");
        logEnd(rid, "PROCESO FINALIZADO CON ERROR (exit_code=3)");
        return 3;
    }
    logOK(rid, "Archivo leído correctamente");
    logKV(rid, "Bytes leídos", std::to_string(raw.size()));

    json j;
    try {
        j = json::parse(raw);
    } catch (const std::exception& e) {
        logERR(rid, "JSON inválido: " + std::string(e.what()));
        logEnd(rid, "PROCESO FINALIZADO CON ERROR (exit_code=4)");
        return 4;
    }
    logOK(rid, "JSON parseado correctamente");

    // ============================================================
    // FASE 3: EXTRACCIÓN Y VALIDACIÓN DE CAMPOS
    // ============================================================
    logPhase(rid, 3, "EXTRACCION DE CAMPOS OBLIGATORIOS",
             "Verificar que existen: identificador_unico, nombres, apellidos");

    const std::string identificador = j.value("identificador_unico", "");
    const std::string nombres = j.value("nombres", "");
    const std::string apellidos = j.value("apellidos", "");
    const std::string sexo = j.value("sexo", "");
    const std::string fecha_nacimiento = j.value("fecha_nacimiento", "");

    // Mostrar datos extraídos
    logBlock(rid, "Datos del usuario", 
        "identificador_unico: " + (identificador.empty() ? "(vacío)" : identificador) + "\n" +
        "nombres:             " + (nombres.empty() ? "(vacío)" : nombres) + "\n" +
        "apellidos:           " + (apellidos.empty() ? "(vacío)" : apellidos) + "\n" +
        "sexo:                " + (sexo.empty() ? "(no especificado)" : sexo) + "\n" +
        "fecha_nacimiento:    " + (fecha_nacimiento.empty() ? "(no especificado)" : fecha_nacimiento)
    );

    // Validación
    if (trim(identificador).empty() || trim(nombres).empty() || trim(apellidos).empty()) {
        logERR(rid, "Faltan campos mínimos obligatorios");
        logWARN(rid, std::string("identificador_unico: ") + (identificador.empty() ? "FALTA" : "OK"));
        logWARN(rid, std::string("nombres: ") + (nombres.empty() ? "FALTA" : "OK"));
        logWARN(rid, std::string("apellidos: ") + (apellidos.empty() ? "FALTA" : "OK"));
        logEnd(rid, "PROCESO FINALIZADO CON ERROR (exit_code=5)");
        return 5;
    }

    logOK(rid, "Todos los campos obligatorios están presentes");

    // ============================================================
    // FASE 4: CONSTRUCCIÓN DEL PAYLOAD
    // ============================================================
    logPhase(rid, 4, "CONSTRUCCION DEL PAYLOAD",
             "Preparar JSON para inserción en tabla usuarios");

    json payload = {
        {"identificador_unico", identificador},
        {"nombres", nombres},
        {"apellidos", apellidos},
        {"fecha_nacimiento", fecha_nacimiento},
        {"sexo", sexo},
        {"estado", "activo"}
    };

    if (debug) {
        logBlock(rid, "Payload JSON (completo)", payload.dump(2));
    } else {
        logKV(rid, "identificador_unico", identificador);
        logKV(rid, "nombres", nombres);
        logKV(rid, "apellidos", apellidos);
        logKV(rid, "estado", "activo");
    }

    // ============================================================
    // FASE 5: INSERCIÓN EN BASE DE DATOS
    // ============================================================
    logPhase(rid, 5, "INSERCION EN BASE DE DATOS (PostgREST)",
             "POST /usuarios con header Prefer: return=representation");

    auto cli = makeClient();

    httplib::Headers headers;
    headers.emplace("Content-Type", "application/json");
    headers.emplace("Prefer", "return=representation");

    logStep(rid, "HTTP", "POST /usuarios");
    logKV(rid, "Content-Type", "application/json");
    logKV(rid, "Prefer", "return=representation");
    logKV(rid, "Payload size", std::to_string(payload.dump().size()) + " bytes");

    auto r = cli.Post("/usuarios", headers, payload.dump(), "application/json");

    if (!r) {
        logERR(rid, "Sin respuesta de PostgREST (timeout o red caída)");
        logEnd(rid, "PROCESO FINALIZADO CON ERROR (exit_code=6)");
        return 6;
    }

    logStep(rid, "RESP", "Respuesta recibida");
    logKV(rid, "HTTP Status", std::to_string(r->status));
    logKV(rid, "Body size", std::to_string(r->body.size()) + " bytes");

    // ============================================================
    // FASE 6: VALIDACIÓN DE RESPUESTA
    // ============================================================
    logPhase(rid, 6, "VALIDACION DE RESPUESTA",
             "Verificar status code y extraer id_usuario");

    // Manejo de duplicados
    if (r->status == 409) {
        logWARN(rid, "Usuario duplicado detectado (409 Conflict)");
        if (debug) logBlock(rid, "Body (detalle)", r->body);
        logEnd(rid, "PROCESO FINALIZADO: DUPLICADO (exit_code=2)");
        return 2;
    }

    // Validación de errores comunes
    if (r->status == 400) {
        logERR(rid, "Payload inválido (400 Bad Request)");
        if (debug) logBlock(rid, "Body (detalle)", r->body);
        logEnd(rid, "PROCESO FINALIZADO CON ERROR (exit_code=3)");
        return 3;
    }

    if (r->status == 401 || r->status == 403) {
        logERR(rid, "Error de autenticación/autorización (" + std::to_string(r->status) + ")");
        logEnd(rid, "PROCESO FINALIZADO CON ERROR (exit_code=4)");
        return 4;
    }

    if (r->status >= 500) {
        logERR(rid, "Error del servidor PostgREST (" + std::to_string(r->status) + ")");
        if (debug) logBlock(rid, "Body (detalle)", r->body);
        logEnd(rid, "PROCESO FINALIZADO CON ERROR (exit_code=6)");
        return 6;
    }

    if (r->status != 201 && r->status != 200) {
        logERR(rid, "Status inesperado: " + std::to_string(r->status));
        if (debug) logBlock(rid, "Body (detalle)", r->body);
        logEnd(rid, "PROCESO FINALIZADO CON ERROR (exit_code=7)");
        return 7;
    }

    logOK(rid, "Status OK: " + std::to_string(r->status));

    // ============================================================
    // FASE 7: EXTRACCIÓN DEL ID_USUARIO
    // ============================================================
    logPhase(rid, 7, "EXTRACCION DEL ID ASIGNADO",
             "Parsear response body y obtener id_usuario");

    json created;
    try {
        created = json::parse(r->body);
    } catch (const std::exception& e) {
        logERR(rid, "Body no es JSON válido: " + std::string(e.what()));
        if (debug) logBlock(rid, "Body (raw)", r->body);
        logEnd(rid, "PROCESO FINALIZADO CON ERROR (exit_code=8)");
        return 8;
    }

    if (!created.is_array() || created.empty() || !created[0].contains("id_usuario")) {
        logERR(rid, "Respuesta no contiene id_usuario");
        logWARN(rid, "Revisa que PostgREST esté configurado con Prefer: return=representation");
        if (debug) logBlock(rid, "Body (parsed)", created.dump(2));
        logEnd(rid, "PROCESO FINALIZADO CON ERROR (exit_code=9)");
        return 9;
    }

    int id_usuario = created[0]["id_usuario"];
    std::string estado = created[0].value("estado", "activo");

    logOK(rid, "Usuario creado exitosamente en la base de datos");
    logKV(rid, "ID asignado", std::to_string(id_usuario));
    logKV(rid, "Estado", estado);
    logKV(rid, "Identificador único", identificador);

    // ============================================================
    // SALIDA STDOUT (para que servidor lo parsee)
    // ============================================================
    std::cout << id_usuario << "\n";

    // ============================================================
    // RESUMEN FINAL
    // ============================================================
    logLine(rid, "");
    logEnd(rid, "REGISTRO COMPLETADO EXITOSAMENTE");
    logLine(rid, "");
    logLine(rid, "  📊 RESUMEN:");
    logKV(rid, "ID usuario", std::to_string(id_usuario), 4);
    logKV(rid, "Identificador único", identificador, 4);
    logKV(rid, "Nombres", nombres, 4);
    logKV(rid, "Apellidos", apellidos, 4);
    logKV(rid, "Estado", estado, 4);
    logKV(rid, "Exit code", "0 (éxito)", 4);
    logLine(rid, "");

    return 0;
}