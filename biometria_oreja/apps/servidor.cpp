#include "httplib.h"
#include "json.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <vector>
#include <sstream>
#include <cctype>
#include <cstdio>
#include <array>
#include <shared_mutex>
#include <ctime>
#include <iomanip>

#include "utilidades/logger.h"
#include "server_utils.h"
#include "server_env.h"
#include "proc_utils.h"
#include "exit_map.h"
#include "report_format.h"

using json = nlohmann::json;
namespace fs = std::filesystem;

// ===================== Config =====================
#if defined(_WIN32)
static constexpr const char *BASE_URL = "http://localhost:3001";
static constexpr const char *CMD_AGREGAR_USUARIO = "agregar_usuario.exe";
static constexpr const char *CMD_AGREGAR_USUARIO_BIOMETRIA = "agregar_usuario_biometria.exe";
static constexpr const char *CMD_PREDECIR = "predecir.exe";
#else
// Ajusta si tu PostgREST en local está en otro host/puerto dentro del docker network
static constexpr const char *BASE_URL = "http://biometria_api:3000";
static constexpr const char *CMD_AGREGAR_USUARIO = "./agregar_usuario";
static constexpr const char *CMD_AGREGAR_USUARIO_BIOMETRIA = "./agregar_usuario_biometria";
static constexpr const char *CMD_PREDECIR = "./predecir";
#endif

// Lectores: autenticación (muchos a la vez)
// Escritor: registro biométrico / reentreno (uno a la vez)
static std::shared_mutex g_model_rw;

#ifndef _WIN32
#include <sys/wait.h>
#endif

#if defined(_WIN32)
#define popen _popen
#define pclose _pclose
#endif

static httplib::Client makeClient()
{
    httplib::Client cli(BASE_URL);
    cli.set_connection_timeout(5, 0);
    cli.set_read_timeout(20, 0);
    cli.set_write_timeout(20, 0);
    return cli;
}

static void logRequestBasics(const std::string &tag, const std::string &rid, const httplib::Request &req)
{
    LOGI(tag, rid,
         "Request: method=" + req.method +
             " path=" + req.path +
             " ip=" + req.remote_addr +
             " body_bytes=" + std::to_string(req.body.size()) +
             " files=" + std::to_string(req.files.size()) +
             " params=" + std::to_string(req.params.size()));
}

static int execStreamToServer(const std::string &rid,
                              const std::string &cmd,
                              const std::string &savePath,
                              const std::string &tag = "OREJA_BIO")
{
    // Abrimos archivo para guardar copia
    std::ofstream out(savePath, std::ios::app);
    if (!out.is_open())
    {
        // si no se puede abrir, igual streameamos
        LOGW("OREJA", rid, "WARN: no se pudo abrir savePath=" + savePath);
    }

    // popen captura stdout del comando; por eso combinamos 2>&1 en cmd
    FILE *pipe = popen(cmd.c_str(), "r");
    if (!pipe)
        return 127;

    std::array<char, 4096> buffer{};
    while (fgets(buffer.data(), (int)buffer.size(), pipe))
    {
        std::string line(buffer.data());

        // guarda en archivo
        if (out.is_open())
            out << line;

        // imprime en logs del servidor (docker logs)
        // Nota: no metas saltos dobles; line ya trae \n usualmente
        if (!line.empty() && line.back() == '\n')
            line.pop_back();
        LOGI("OREJA", rid, "[" + tag + "] " + line);
    }

    int status = pclose(pipe);
    return systemExitCode(status); // usa tu helper existente
}

double getEnvDouble(const std::string &name, double def)
{
    const char *v = std::getenv(name.c_str());
    if (!v)
        return def;
    try
    {
        return std::stod(v);
    }
    catch (...)
    {
        return def;
    }
}

static bool tryGetEnvDouble(const std::string &name, double &out)
{
    const char *v = std::getenv(name.c_str());
    if (!v || !*v)
        return false;
    try
    {
        out = std::stod(v);
        return true;
    }
    catch (...)
    {
        return false;
    }
}

static bool cargarUmbralDesdeArchivo(const std::string &ruta, double &out)
{
    std::ifstream f(ruta);
    if (!f.is_open())
        return false;
    std::string line;
    while (std::getline(f, line))
    {
        if (line.rfind("threshold=", 0) == 0)
        {
            try
            {
                out = std::stod(line.substr(10));
                return true;
            }
            catch (...)
            {
                return false;
            }
        }
    }
    return false;
}

static std::string vectorToByteArray(const std::vector<double> &vec)
{
    std::stringstream ss;
    ss << "\\x";
    const uint8_t *bytes = reinterpret_cast<const uint8_t *>(vec.data());
    size_t numBytes = vec.size() * sizeof(double);
    for (size_t i = 0; i < numBytes; ++i)
    {
        ss << std::hex << std::setw(2) << std::setfill('0')
           << static_cast<int>(bytes[i]);
    }
    return ss.str();
}

static std::string nowUtcIso()
{
    std::time_t t = std::time(nullptr);
    std::stringstream ss;
    ss << std::put_time(std::gmtime(&t), "%Y-%m-%dT%H:%M:%SZ");
    return ss.str();
}

// ===================== MAIN =====================
int main()
{
    // Para defensa técnica, usa DEBUG. Para “normal”, INFO.
    setLogLevel(LOG_DEBUG);
    // Si quieres persistir logs a archivo (montado con /app/out):
    // setLogFile("/app/out/log_servidor.txt");

    httplib::Server servidor;

    // --------------------- GET / ---------------------
    servidor.Get("/", [](const httplib::Request &req, httplib::Response &res)
                 {
        const std::string rid = makeRequestId();
        LOG_SCOPE("OREJA", rid, "GET /");
        logRequestBasics("OREJA", rid, req);

        res.status = 200;
        res.set_content("Servidor biometrico activo", "text/plain");

        LOGI("OREJA", rid, "Respuesta: status=200 mensaje='Servidor biometrico activo'"); });

    // --------------------- POST /registrar_usuario ---------------------
    // Flujo:
    // 1) Recibe JSON con datos
    // 2) Guarda nuevo_usuario/datos.json
    // 3) Ejecuta agregar_usuario
    // 4) Lee id_usuario_interno.txt
    // 5) Verifica en PostgREST que el usuario existe (para poder afirmar “guardado en BD”)
    servidor.Post("/registrar_usuario", [](const httplib::Request &req, httplib::Response &res)
                  {
        const std::string rid = makeRequestId();
        LOG_SCOPE("USUARIOS", rid, "POST /registrar_usuario");

        LOGI("USUARIOS", rid, "\n" + repTitle("REGISTRO DE USUARIO"));
        LOGI("USUARIOS", rid, repSection("INICIO DE PROCESO"));

        logRequestBasics("USUARIOS", rid, req);

        const bool AUDIT_MODE = (getEnvStr("AUDIT_MODE", "0") == "1");
        const std::string TMP_DIR = getEnvStr("TMP_DIR", "/tmp/biometria_oreja");

        // carpeta por request
        const std::string WORK_DIR = TMP_DIR + "/usr_" + rid;

        if (req.body.empty()) {
            LOGI("USUARIOS", rid, repSection("ENTRADA DE DATOS"));
            LOGW("USUARIOS", rid, repFAIL("Body vacío: no se recibió JSON"));

            LOGI("USUARIOS", rid, repEndFail("Proceso terminado por entrada inválida"));

            res.status = 400;
            json pub = buildPublicErrorBody(
                400,
                "ENTRADA_INVALIDA",
                "Se requiere JSON con los datos del usuario"
            );
            res.set_content(pub.dump(4), "application/json");
            return;
        }

        json datos;
        try {
            datos = json::parse(req.body);
        } catch (...) {
            LOGI("USUARIOS", rid, repSection("ENTRADA DE DATOS"));
            LOGW("USUARIOS", rid, repFAIL("JSON inválido: error de parseo"));

            LOGI("USUARIOS", rid, repEndFail("Proceso terminado por entrada inválida"));

            res.status = 400;
            json pub = buildPublicErrorBody(
                400,
                "ENTRADA_INVALIDA",
                "El cuerpo enviado no es un JSON válido"
            );
            res.set_content(pub.dump(4), "application/json");
            return;
        }

        LOGI("USUARIOS", rid, repSection("ENTRADA DE DATOS"));
        LOGI("USUARIOS", rid, "\n" + resumenUsuarioJson(datos));

        LOGI("USUARIOS", rid, repSection("PROCESAMIENTO (FS / PREPARACION)"));

        // preparar carpeta
        try {
            if (fs::exists(WORK_DIR)) fs::remove_all(WORK_DIR);
            fs::create_directories(WORK_DIR);
        } catch (const std::exception& e) {
            LOGE("USUARIOS", rid, std::string("FS ERROR: no se pudo crear WORK_DIR: ") + e.what());
            res.status = 500;
            res.set_content("No se pudo preparar directorio de trabajo", "text/plain");
            return;
        }

        const std::string rutaDatos = WORK_DIR + "/datos.json";

        // Guardar JSON
        {
            const std::string dump = datos.dump(4);
            std::ofstream ofs(rutaDatos);
            if (!ofs.is_open()) {
                LOGE("USUARIOS", rid, "FS ERROR: no se pudo escribir " + rutaDatos + " -> 500");
                res.status = 500;
                res.set_content("No se pudo escribir datos.json", "text/plain");
                return;
            }
            ofs << dump;
            LOGI("USUARIOS", rid, "FS OK: datos.json escrito ruta=" + rutaDatos +
                                " bytes=" + std::to_string(dump.size()));
        }

        LOGI("USUARIOS", rid, repSection("EJECUCION"));

        // Ejecutar binario con stdout/stderr separados
        const std::string outStd = WORK_DIR + "/agregar_usuario.out"; // aquí debe venir SOLO el id
        const std::string outErr = WORK_DIR + "/agregar_usuario.log"; // logs técnicos del exe

        // Le pasamos WORK_DIR por env para que el exe lea WORK_DIR/datos.json (y no "nuevo_usuario")
        // Si tu exe no usa WORK_DIR, NO rompe: igual lo dejamos listo.
        std::string cmd = std::string("WORK_DIR=\"") + WORK_DIR + "\" " +
                        std::string(CMD_AGREGAR_USUARIO) +
                        " --rid " + rid +
                        " 1> " + outStd +
                        " 2> " + outErr;

        LOGI("USUARIOS", rid, "EXEC: agregar_usuario stdout=" + outStd + " stderr=" + outErr);
        LOGI("USUARIOS", rid, "EXEC: ejecutar agregar_usuario (WORK_DIR=" + WORK_DIR + ")");
        LOGD("USUARIOS", rid, "EXEC CMD: " + cmd);

        int status = std::system(cmd.c_str());
        int exit_code = systemExitCode(status);

        LOGI("USUARIOS", rid,
            "EXEC: agregar_usuario status(raw)=" + std::to_string(status) +
            " exit_code=" + std::to_string(exit_code));

        if (exit_code != 0) {
            ExitMapped mapped = mapExitCode("agregar_usuario", exit_code);

            // Log técnico completo (solo en docker logs)
            std::string err_tail = leerUltimasLineas(outErr, 80);

            LOGE("USUARIOS", rid,
                "EXEC ERROR: agregar_usuario falló exit_code=" + std::to_string(exit_code) +
                " -> http=" + std::to_string(mapped.http_status) +
                " title=" + mapped.title);

            if (mapped.http_status == 409) {
                LOGW("USUARIOS", rid, "Negado por duplicado (409). Detalles en DEBUG.");
                LOGD("USUARIOS", rid, "EXEC STDERR (tail) path=" + outErr + "\n" + truncN(err_tail, 2000));
            } else {
                LOGE("USUARIOS", rid, "EXEC STDERR (tail) path=" + outErr + "\n" + truncN(err_tail, 2000));
            }

            // ===== Reporte bonito (para lectura humana) =====
            LOGI("USUARIOS", rid, repSection("CONTROLES"));
            LOGW("USUARIOS", rid, repFAIL("Ejecución agregar_usuario falló (exit_code=" + std::to_string(exit_code) + ")"));
            LOGW("USUARIOS", rid, repFAIL("Clasificación: http=" + std::to_string(mapped.http_status) + " title=" + mapped.title));

            LOGI("USUARIOS", rid, repSection("FINALIZACION"));
            LOGW("USUARIOS", rid, repFAIL("Proceso finalizado por error"));
            LOGW("USUARIOS", rid, repKeyVal("http_status", std::to_string(mapped.http_status)));
            LOGW("USUARIOS", rid, repKeyVal("title", mapped.title));
            LOGW("USUARIOS", rid, repKeyVal("exit_code", std::to_string(exit_code)));
            
            // Respuesta mínima para el usuario (Postman)
            json pub = buildPublicErrorBody(mapped.http_status, mapped.title, mapped.message);
            res.status = mapped.http_status;
            res.set_content(pub.dump(4), "application/json");

            if (!AUDIT_MODE) {
                try { fs::remove_all(WORK_DIR); } catch (...) {}
            } else {
                LOGW("USUARIOS", rid, "AUDIT_MODE=1 -> se conservan temporales en " + WORK_DIR);
            }
            return;
        }

        LOGI("USUARIOS", rid, repSection("CONTROLES"));
        LOGI("USUARIOS", rid, repOK("JSON recibido: parse OK"));
        LOGI("USUARIOS", rid, repOK("datos.json escrito correctamente"));
        LOGI("USUARIOS", rid, repOK("Ejecución agregar_usuario exit_code=0 (sin duplicado ni errores)"));

        // ---- Leer id_usuario desde stdout del ejecutable ----
        int id_usuario = -1;
        {
            std::ifstream f(outStd);
            std::string line;
            if (f.is_open()) std::getline(f, line);

            line = trunc(line); // si trunc() corta y quita saltos, ok. Si no tienes trunc, reemplaza por trim.
            if (!line.empty() && esEntero(line)) {
                id_usuario = std::stoi(line);
                LOGI("USUARIOS", rid, "Parse OK: id_usuario leído desde stdout=" + std::to_string(id_usuario));
            } else {
                LOGW("USUARIOS", rid, "Parse WARN: stdout vacío/no entero. Intentando fallback archivo id_usuario_interno.txt");

                // Fallback si tu exe viejo aún escribe archivo
                const std::string rutaId = WORK_DIR + "/id_usuario_interno.txt";
                std::ifstream idfile(rutaId);
                if (idfile.is_open()) {
                    idfile >> id_usuario;
                }
                if (id_usuario <= 0) {
                    LOGE("USUARIOS", rid, "FS/Parse ERROR: no se obtuvo id_usuario (stdout ni archivo) -> 500");
                    res.status = 500;
                    res.set_content("No se pudo obtener id_usuario del registro", "text/plain");
                    if (!AUDIT_MODE) {
                        try { fs::remove_all(WORK_DIR); } catch (...) {}
                    }
                    return;
                }

                LOGI("USUARIOS", rid, "FS OK: id_usuario leído desde archivo=" + std::to_string(id_usuario));
            }
        }

        // Verificación BD real
        bool verificado_bd = false;
        std::string estado_bd = "desconocido";
        {
            httplib::Client cli = makeClient();
            const std::string url = "/usuarios?id_usuario=eq." + std::to_string(id_usuario);
            auto r = cli.Get(url.c_str());

            if (!r) {
                LOGW("USUARIOS", rid, "BD WARN: sin respuesta verificando " + url);
            } else {
                LOGI("USUARIOS", rid, "BD: GET " + url + " status=" + std::to_string(r->status) +
                                    " body_bytes=" + std::to_string(r->body.size()));
                if (r->status == 200) {
                    try {
                        json check = json::parse(r->body);
                        verificado_bd = (check.is_array() && !check.empty());
                        if (verificado_bd && check[0].contains("estado")) {
                            estado_bd = check[0]["estado"].is_string()
                                ? check[0]["estado"].get<std::string>()
                                : check[0]["estado"].dump();
                        }
                    } catch (...) {
                        LOGW("USUARIOS", rid, "BD WARN: JSON inválido en verificación");
                    }
                }
            }
        }

        LOGI("USUARIOS", rid, repSection("FINALIZACION"));
        LOGI("USUARIOS", rid, repOK("Usuario creado y guardado en BD"));
        LOGI("USUARIOS", rid, repKeyVal("id_usuario", std::to_string(id_usuario)));
        LOGI("USUARIOS", rid, repKeyVal("estado", estado_bd));

        json respuesta = {
            {"mensaje", "Usuario registrado correctamente"},
            {"id_usuario", id_usuario},
            {"verificado_bd", verificado_bd},
            {"estado", estado_bd}
        };

        res.status = 201;
        res.set_content(respuesta.dump(4), "application/json");

        if (AUDIT_MODE) {
            LOGW("USUARIOS", rid, "AUDIT_MODE=1 -> se conservan temporales en " + WORK_DIR);
        } else {
            try { fs::remove_all(WORK_DIR); } catch (...) {}
        } });

    // --------------------- POST /oreja/registrar ---------------------
    // Flujo:
    // 1) multipart 5 imágenes + identificador (consistente con dataset offline 5 train)
    // 2) busca id_usuario en BD
    // 3) guarda imágenes en nuevo_usuario/
    // 4) escribe id_usuario.txt + id_usuario_interno.txt
    // 5) ejecuta agregar_usuario_biometria
    servidor.Post("/oreja/registrar", [](const httplib::Request &req, httplib::Response &res)
                  {
        const std::string rid = makeRequestId();
        LOG_SCOPE("OREJA", rid, "POST /oreja/registrar");
        logRequestBasics("OREJA", rid, req);

        const bool AUDIT_MODE = (getEnvStr("AUDIT_MODE", "0") == "1");
        const std::string TMP_DIR = getEnvStr("TMP_DIR", "/tmp/biometria_oreja");

        // Usaremos una carpeta por request (evita colisiones y ya no dependes de "nuevo_usuario" fijo)
        const std::string WORK_DIR = TMP_DIR + "/reg_" + rid;

        if (req.files.size() != 5 || !req.has_param("identificador")) {
            LOGW("OREJA", rid, "Rechazado: requiere 5 imágenes + param identificador -> 400");
            res.status = 400;
            res.set_content("Se requieren 5 imágenes y el campo 'identificador'", "text/plain");
            return;
        }

        const std::string identificador = safeParam(req, "identificador");
        LOGI("OREJA", rid, "Entrada validada: identificador=" + identificador + " files=" + std::to_string(req.files.size()));

        // Log técnico de archivos recibidos
        size_t total = 0;
        int idx = 0;
        for (const auto& kv : req.files) {
            const auto& f = kv.second;
            total += f.content.size();
            LOGD("OREJA", rid,
                "File[" + std::to_string(idx++) + "] field=" + kv.first +
                " filename=" + f.filename +
                " content_type=" + f.content_type +
                " bytes=" + std::to_string(f.content.size()));
        }
        LOGI("OREJA", rid, "Files resumen: total_bytes=" + std::to_string(total));

        // 1) DB: obtener id_usuario
        int id_usuario = -1;
        {
            httplib::Client cli = makeClient();
            const std::string urlUsuario = "/usuarios?identificador_unico=eq." + identificador;

            auto r = cli.Get(urlUsuario.c_str());
            if (!r) {
                LOGE("OREJA", rid, "BD ERROR: sin respuesta consultando " + urlUsuario + " -> 500");
                res.status = 500;
                res.set_content("Error al consultar usuario (SIN_RESPUESTA)", "text/plain");
                return;
            }

            LOGI("OREJA", rid, "BD: GET " + urlUsuario + " status=" + std::to_string(r->status) +
                            " body_bytes=" + std::to_string(r->body.size()));

            if (r->status != 200) {
                LOGE("OREJA", rid, "BD ERROR: status inesperado=" + std::to_string(r->status) + " -> 500");
                res.status = 500;
                res.set_content("Error al consultar usuario", "text/plain");
                return;
            }

            json data_user;
            try { data_user = json::parse(r->body); }
            catch (...) {
                LOGE("OREJA", rid, "BD ERROR: JSON inválido en /usuarios -> 500");
                res.status = 500;
                res.set_content("Respuesta inválida del servidor de usuarios (JSON)", "text/plain");
                return;
            }

            if (!data_user.is_array() || data_user.empty() || !data_user[0].contains("id_usuario")) {
                LOGW("OREJA", rid, "No encontrado: usuario con identificador=" + identificador + " -> 404");
                res.status = 404;
                res.set_content("Usuario no encontrado", "text/plain");
                return;
            }

            id_usuario = data_user[0]["id_usuario"];
            LOGI("OREJA", rid, "Usuario encontrado: identificador=" + identificador + " id_usuario=" + std::to_string(id_usuario));
        }

        // Crear workdir por request
        try {
            if (fs::exists(WORK_DIR)) fs::remove_all(WORK_DIR);
            fs::create_directories(WORK_DIR);
        } catch (const std::exception& e) {
            LOGE("OREJA", rid, std::string("FS ERROR: no se pudo crear workdir: ") + e.what());
            res.status = 500;
            res.set_content("No se pudo preparar directorio de trabajo", "text/plain");
            return;
        }

        // Guardar imágenes
        {
            int i = 0;
            for (const auto& [campo, archivo] : req.files) {
                (void)campo;
                const std::string nombre = WORK_DIR + "/img_" + std::to_string(i++) + ".jpg";
                std::ofstream ofs(nombre, std::ios::binary);
                if (!ofs.is_open()) {
                    LOGE("OREJA", rid, "FS ERROR: no se pudo escribir " + nombre + " -> 500");
                    res.status = 500;
                    res.set_content("No se pudo guardar una imagen", "text/plain");
                    return;
                }
                ofs.write(archivo.content.c_str(), (std::streamsize)archivo.content.size());
                if (!ofs.good()) {
                    LOGE("OREJA", rid, "FS ERROR: fallo al escribir " + nombre + " -> 500");
                    res.status = 500;
                    res.set_content("Error al escribir la imagen en disco", "text/plain");
                    return;
                }
            }
            LOGI("OREJA", rid, "FS OK: imágenes guardadas en carpeta=" + WORK_DIR +
                            " count=" + std::to_string(req.files.size()));
        }

        // Guardar ids como tu ejecutable espera (si tu binario usa estos nombres)
        {
            std::ofstream f1(WORK_DIR + "/id_usuario.txt");
            std::ofstream f2(WORK_DIR + "/id_usuario_interno.txt");
            if (!f1.is_open() || !f2.is_open()) {
                LOGE("OREJA", rid, "FS ERROR: no se pudo escribir id_usuario*.txt -> 500");
                res.status = 500;
                res.set_content("No se pudo escribir archivos id_usuario*.txt", "text/plain");
                return;
            }
            f1 << identificador;
            f2 << id_usuario;
            LOGI("OREJA", rid, "FS OK: ids escritos identificador=" + identificador + " id_usuario=" + std::to_string(id_usuario));
        }

        // Separate stdout and stderr for clean logging
        const std::string outStd = WORK_DIR + "/agregar_usuario_biometria.out";
        const std::string outErr = WORK_DIR + "/agregar_usuario_biometria.log";

        // Importante: usa bash -lc para que funcionen exports y redirecciones bien
        std::string cmd =
            "bash -lc '"
            "export WORK_DIR=\"" + WORK_DIR + "\"; "
            "export MODEL_DIR=\"" + getEnvStr("MODEL_DIR", "out") + "\"; "
            "export AUDIT_MODE=\"" + std::string(AUDIT_MODE ? "1" : "0") + "\"; "
            "export LOG_DETAIL=\"" + getEnvStr("LOG_DETAIL","2") + "\"; "
            "export QC_MIN_PASS=\"" + getEnvStr("QC_MIN_PASS","6") + "\"; "
            "export QC_ENFORCE=\"" + getEnvStr("QC_ENFORCE","0") + "\"; "
            + std::string(CMD_AGREGAR_USUARIO_BIOMETRIA) + " --rid " + rid +
            " 1> " + outStd +
            " 2> " + outErr + "'";

        LOGI("OREJA", rid, "EXEC: stdout=" + outStd + " stderr=" + outErr);
        LOGD("OREJA", rid, "EXEC: cmd=" + cmd);

        int exit_code = systemExitCode(std::system(cmd.c_str()));
        LOGI("OREJA", rid, "EXEC: agregar_usuario_biometria terminó exit_code=" + std::to_string(exit_code));


        if (exit_code != 0) {
            // Leer log del ejecutable (stderr) para clasificar el fallo
            std::string err_tail = leerUltimasLineas(outErr, 80);

            // Caso especial: biometría duplicada (regla de negocio)
            // El exe lo escribe textual: "ALERTA: Biometría duplicada probable. Coincide con clase existente: X"
            auto p = err_tail.find("ALERTA: Biometría duplicada probable");
            if (p != std::string::npos) {
                // Intentar extraer la clase
                int clase_existente = -1;
                {
                    auto q = err_tail.rfind("clase existente:");
                    if (q != std::string::npos) {
                        std::string num = err_tail.substr(q + std::string("clase existente:").size());
                        // recorta espacios
                        while (!num.empty() && std::isspace((unsigned char)num.front())) num.erase(num.begin());
                        // corta hasta fin de línea
                        auto nl = num.find('\n');
                        if (nl != std::string::npos) num = num.substr(0, nl);
                        try { clase_existente = std::stoi(num); } catch (...) {}
                    }
                }

                LOGW("OREJA", rid, "Registro rechazado: biometría duplicada probable. clase_existente=" +
                                std::to_string(clase_existente) + " (ver " + outErr + ")");

                res.status = 409;
                json j = {
                    {"error", "Biometría duplicada: parece pertenecer a un usuario ya registrado"},
                    {"rid", rid},
                    {"clase_detectada", clase_existente}
                };
                res.set_content(j.dump(4), "application/json");

                if (AUDIT_MODE) {
                    LOGW("OREJA", rid, "AUDIT_MODE=1 -> se conservan temporales en " + WORK_DIR);
                } else {
                    try { fs::remove_all(WORK_DIR); } catch (...) {}
                }

                
                return;
            }

            // Caso general: error real
            LOGE("OREJA", rid,
                "EXEC ERROR: agregar_usuario_biometria exit_code=" + std::to_string(exit_code) +
                " stderr_path=" + outErr);

            // Log TODO el stderr al docker log para debugging de admin (NUNCA se envía al usuario)
            std::ifstream stderrFile(outErr);
            if (stderrFile.is_open()) {
                std::string line;
                LOGE("OREJA", rid, "========== STDERR COMPLETO (SOLO DOCKER LOG) ==========");
                while (std::getline(stderrFile, line)) {
                    LOGE("OREJA", rid, line);
                }
                LOGE("OREJA", rid, "========== FIN STDERR ==========");
                stderrFile.close();
            } else {
                LOGE("OREJA", rid, "No se pudo abrir " + outErr + " para logging completo");
            }

            ExitMapped mapped = mapExitCode("agregar_usuario_biometria", exit_code);

            res.status = mapped.http_status;

            // Clean error response - NO internal logs exposed
            json j = buildErrorBody(
                "agregar_usuario_biometria",
                exit_code,
                mapped,
                "",   // stderr_tail - REMOVED to prevent leaking internal logs
                ""   // stdout_tail
            );

            // Include only essential info for debugging
            j["rid"] = rid;

            res.set_content(j.dump(4), "application/json");

            if (AUDIT_MODE) {
                LOGW("OREJA", rid, "AUDIT_MODE=1 -> se conservan temporales en " + WORK_DIR);
            } else {
                try { fs::remove_all(WORK_DIR); } catch (...) {}
            }
            return;
        }

        // 4) Log COMPLETO del stderr al docker log (para debugging de admin)
        // IMPORTANTE: Esto SOLO va a docker logs, NUNCA al usuario
        {
            std::ifstream bioLog(outErr);
            if (bioLog.is_open()) {
                std::string linea;
                LOGI("OREJA", rid, "========== STDERR COMPLETO (REGISTRO EXITOSO) ==========");
                while (std::getline(bioLog, linea)) {
                    // Log todas las líneas al docker log para debugging
                    LOGI("OREJA", rid, linea);
                }
                LOGI("OREJA", rid, "========== FIN STDERR ==========");
                bioLog.close();
            } else {
                LOGW("OREJA", rid, "No se pudo abrir " + outErr + " para logging completo");
            }
        }

        LOGI("OREJA", rid, "Registro biométrico COMPLETADO: identificador=" + identificador +
                        " id_usuario=" + std::to_string(id_usuario) +
                        " imgs=" + std::to_string(req.files.size()));

        res.status = 200;
        res.set_content("Credencial biométrica registrada correctamente.", "text/plain");

        if (AUDIT_MODE) {
            LOGW("OREJA", rid, "AUDIT_MODE=1 -> se conservan temporales en " + WORK_DIR);
        } else {
            try { fs::remove_all(WORK_DIR); } catch (...) {}
        } });

    // --------------------- POST /oreja/autenticar ---------------------
    // Flujo:
    // 1) recibe imagen (archivo) + etiqueta
    // 2) valida usuario activo y credencial oreja activa
    // 3) ejecuta predecir
    // 4) parse clase;score_top1;score_claimed
    // 5) decisión por umbral + coincidencia
    // 6) registra validación
    servidor.Post("/oreja/autenticar", [](const httplib::Request &req, httplib::Response &res)
                  {
                      const std::string rid = makeRequestId();
                      LOG_SCOPE("OREJA", rid, "POST /oreja/autenticar");
                      logRequestBasics("OREJA", rid, req);

                      if (!req.has_file("archivo") || !req.has_param("etiqueta"))
                      {
                          LOGW("OREJA", rid, "Rechazado: falta archivo o etiqueta -> 400");
                          res.status = 400;
                          res.set_content("Se requiere imagen (campo 'archivo') y cedula (campo 'etiqueta')", "text/plain");
                          return;
                      }

                      const std::string etiqueta = safeParam(req, "etiqueta");
                      const auto &file = req.get_file_value("archivo");
                      LOGI("OREJA", rid, "Entrada validada: etiqueta=" + etiqueta + " filename=" + file.filename + " bytes=" + std::to_string(file.content.size()));

                      const std::string BASE_TMP = tmpDir();
                      const std::string REQ_DIR = BASE_TMP + "/req_" + rid;

                      fs::create_directories(REQ_DIR);

                      const std::string rutaImagen = REQ_DIR + "/imagen.jpg";

                      {
                          std::ofstream ofs(rutaImagen, std::ios::binary);
                          if (!ofs.is_open())
                          {
                              LOGE("OREJA", rid, "FS ERROR: no se pudo escribir imagen tmp -> 500");
                              res.status = 500;
                              res.set_content("No se pudo guardar la imagen en el servidor", "text/plain");
                              return;
                          }
                          ofs.write(file.content.c_str(), (std::streamsize)file.content.size());
                          if (!ofs.good())
                          {
                              LOGE("OREJA", rid, "FS ERROR: error escribiendo imagen -> 500");
                              res.status = 500;
                              res.set_content("Error al escribir la imagen en disco", "text/plain");
                              return;
                          }
                          LOGD("OREJA", rid, "FS OK: imagen guardada ruta=" + rutaImagen);
                      }

                      httplib::Client cli = makeClient();

                      // 1) Usuario
                      int id_usuario_real = -1;
                      {
                          const std::string urlUsuario = "/usuarios?identificador_unico=eq." + etiqueta;
                          auto r = cli.Get(urlUsuario.c_str());
                          if (!r)
                          {
                              LOGE("OREJA", rid, "BD ERROR: sin respuesta consultando usuario -> 500");
                              res.status = 500;
                              res.set_content("Error al consultar usuario", "text/plain");
                              return;
                          }
                          LOGI("OREJA", rid, "BD: GET " + urlUsuario + " status=" + std::to_string(r->status) + " body_bytes=" + std::to_string(r->body.size()));
                          if (r->status != 200)
                          {
                              LOGE("OREJA", rid, "BD ERROR: status usuario=" + std::to_string(r->status) + " -> 500");
                              res.status = 500;
                              res.set_content("Error al consultar usuario", "text/plain");
                              return;
                          }

                          json data_user;
                          try
                          {
                              data_user = json::parse(r->body);
                          }
                          catch (...)
                          {
                              LOGE("OREJA", rid, "BD ERROR: JSON inválido usuario -> 500");
                              res.status = 500;
                              res.set_content("Respuesta inválida del servidor de usuarios (JSON)", "text/plain");
                              return;
                          }

                          if (!data_user.is_array() || data_user.empty())
                          {
                              LOGW("OREJA", rid, "No encontrado: usuario etiqueta=" + etiqueta + " -> 404");
                              res.status = 404;
                              res.set_content("Usuario no encontrado", "text/plain");
                              return;
                          }

                          const json &usuario = data_user[0];
                          const std::string estado = usuario.value("estado", "activo");
                          if (estado != "activo")
                          {
                              LOGW("OREJA", rid, "Rechazado: usuario no activo estado=" + estado + " -> 403");
                              res.status = 403;
                              res.set_content("Usuario no esta activo", "text/plain");
                              return;
                          }

                          id_usuario_real = usuario["id_usuario"];
                          LOGI("OREJA", rid, "Usuario OK: etiqueta=" + etiqueta + " id_usuario=" + std::to_string(id_usuario_real));
                      }

                      // 2) Credencial oreja activa
                      {
                          const std::string urlCred =
                              "/credenciales_biometricas?id_usuario=eq." + std::to_string(id_usuario_real) +
                              "&tipo_biometria=eq.oreja&estado=eq.activo";

                          auto r = cli.Get(urlCred.c_str());
                          if (!r)
                          {
                              LOGE("OREJA", rid, "BD ERROR: sin respuesta consultando credenciales -> 500");
                              res.status = 500;
                              res.set_content("Error al consultar credenciales biometricas", "text/plain");
                              return;
                          }
                          LOGI("OREJA", rid, "BD: GET " + urlCred + " status=" + std::to_string(r->status) + " body_bytes=" + std::to_string(r->body.size()));
                          if (r->status != 200)
                          {
                              LOGE("OREJA", rid, "BD ERROR: status credenciales=" + std::to_string(r->status) + " -> 500");
                              res.status = 500;
                              res.set_content("Error al consultar credenciales biometricas", "text/plain");
                              return;
                          }

                          json data_cred;
                          try
                          {
                              data_cred = json::parse(r->body);
                          }
                          catch (...)
                          {
                              LOGE("OREJA", rid, "BD ERROR: JSON inválido credenciales -> 500");
                              res.status = 500;
                              res.set_content("Respuesta inválida de credenciales (JSON)", "text/plain");
                              return;
                          }

                          if (!data_cred.is_array() || data_cred.empty())
                          {
                              LOGW("OREJA", rid, "Rechazado: no tiene credencial oreja activa -> 403");
                              res.status = 403;
                              res.set_content("El usuario no tiene credencial biometrica de tipo oreja activa", "text/plain");
                              return;
                          }

                          LOGI("OREJA", rid, "Credencial OK: oreja activa encontrada para id_usuario=" + std::to_string(id_usuario_real));
                      }

                      std::shared_lock<std::shared_mutex> model_read_lock(g_model_rw);

                      // 3) Ejecutar predecir

                      const std::string outPred = REQ_DIR + "/prediccion.txt"; // si lo usas
                      const std::string outLog = REQ_DIR + "/predecir.log";    // si lo usas

                      // stdout -> prediccion (clase;score_top1;score_claimed)
                      // stderr -> log del predecir
                      const std::string comando =
                          std::string("cd /app && ") + CMD_PREDECIR + " " + rutaImagen +
                          " --rid " + rid +
                          " --claim " + etiqueta +
                          " 1> " + outPred +
                          " 2> " + outLog;

                      LOGI("OREJA", rid, "EXEC: predecir stdout=" + outPred + " stderr=" + outLog);
                      LOGI("OREJA", rid, "EXEC: ejecutar predecir cmd=" + comando);
                      int exit_code = systemExitCode(std::system(comando.c_str()));
                      LOGI("OREJA", rid, "EXEC: predecir finalizó exit_code=" + std::to_string(exit_code));

                      // Mostrar stderr completo en docker logs (para debugging y tutores)
                      std::ifstream stderrFile(outLog);
                      if (stderrFile.is_open())
                      {
                          std::string line;
                          LOGI("OREJA", rid, "========== STDERR PREDECIR (PIPELINE COMPLETO) ==========");
                          while (std::getline(stderrFile, line))
                          {
                              LOGI("OREJA", rid, line);
                          }
                          LOGI("OREJA", rid, "========== FIN STDERR PREDECIR ==========");
                          stderrFile.close();
                      }
                      else
                      {
                          LOGW("OREJA", rid, "No se pudo abrir " + outLog + " para logging completo");
                      }

                      if (exit_code != 0)
                      {
                          ExitMapped mapped = mapExitCode("predecir", exit_code);

                          LOGE("OREJA", rid,
                               "EXEC ERROR: predecir falló exit_code=" + std::to_string(exit_code) +
                                   " -> " + std::to_string(mapped.http_status));

                          res.status = mapped.http_status;
                          res.set_content("Error ejecutando predecir", "text/plain");
                          return;
                      }

                      // 4) Parse resultado
                      // Formato actualizado: clase;score_top1;score_claimed
                      std::ifstream pred(outPred);
                      std::string resultado;
                      std::getline(pred, resultado);

                      if (resultado.empty())
                      {
                          LOGE("OREJA", rid, "Parse ERROR: predicción vacía -> 500");
                          res.status = 500;
                          res.set_content("Archivo de prediccion vacio", "text/plain");
                          return;
                      }

                      // Parsear formato: clase;score_top1;score_claimed
                      auto pos1 = resultado.find(';');
                      if (pos1 == std::string::npos)
                      {
                          LOGE("OREJA", rid, "Parse ERROR: formato inválido '" + trunc(resultado) + "' -> 500");
                          res.status = 500;
                          res.set_content("Formato de predicción inválido (esperado clase;score_top1;score_claimed)", "text/plain");
                          return;
                      }

                      auto pos2 = resultado.find(';', pos1 + 1);
                      if (pos2 == std::string::npos)
                      {
                          LOGE("OREJA", rid, "Parse ERROR: formato inválido (falta score_claimed) '" + trunc(resultado) + "' -> 500");
                          res.status = 500;
                          res.set_content("Formato de predicción inválido (esperado clase;score_top1;score_claimed)", "text/plain");
                          return;
                      }

                      std::string parteClase = resultado.substr(0, pos1);
                      std::string parteScore = resultado.substr(pos1 + 1, pos2 - pos1 - 1);
                      std::string parteScoreClaimed = resultado.substr(pos2 + 1);

                      if (!esEntero(parteClase) || !esDoubleSimple(parteScore) || !esDoubleSimple(parteScoreClaimed))
                      {
                          LOGE("OREJA", rid, "Parse ERROR: valores inválidos '" + trunc(resultado) + "' -> 500");
                          res.status = 500;
                          res.set_content("Predicción inválida", "text/plain");
                          return;
                      }

                      int clase_predicha = std::stoi(parteClase);
                      double score_top1 = std::stod(parteScore);
                      double score_claimed = std::stod(parteScoreClaimed);

                      // 5) Decisión de autenticación 1:1
                      auto parseDoubleSafe = [](const std::string &s, double &out) -> bool
                      {
                          try
                          {
                              size_t idx = 0;
                              out = std::stod(s, &idx);
                              return idx == s.size();
                          }
                          catch (...)
                          {
                              return false;
                          }
                      };

                      // UMBRAL OPTIMIZADO basado en métricas de verificación 1:1
                      // Para obtener el umbral óptimo personalizado, ejecutar:
                      //   ./calcular_umbrales_1_1 --test test_norm --out out
                      // Esto calculará el umbral EER (Equal Error Rate) desde tus datos reales
                      //
                      // Umbral por defecto: 0.5 (coseno; rango [-1, 1])
                      // - Alta seguridad: ~0.65-0.80
                      // - Balance (EER): ~0.45-0.60 (default)
                      // - Alta usabilidad: ~0.25-0.40
                      //
                      // Puedes configurar con variable de entorno: UMBRAL_AUTENTICACION
                      // O pasar como parámetro query: ?umbral=0.5
                      // Si no hay env, usa umbral_eer.txt generado por procesar_dataset
                      double UMBRAL_VERIFICACION = 0.5;
                      double envUmbral = 0.0;
                      if (tryGetEnvDouble("UMBRAL_AUTENTICACION", envUmbral))
                      {
                          UMBRAL_VERIFICACION = envUmbral;
                      }
                      else
                      {
                          const std::string modelDir = getEnvStr("MODEL_DIR", "out");
                          const std::string rutaUmbral = modelDir + "/umbral_eer.txt";
                          double fileUmbral = 0.0;
                          if (cargarUmbralDesdeArchivo(rutaUmbral, fileUmbral))
                          {
                              UMBRAL_VERIFICACION = fileUmbral;
                              LOGI("OREJA", rid, "Umbral EER cargado desde archivo: " + rutaUmbral + " -> " + std::to_string(UMBRAL_VERIFICACION));
                          }
                      }

                      if (req.has_param("umbral"))
                      {
                          const std::string ustr = safeParam(req, "umbral");
                          double u = 0.0;
                          if (!parseDoubleSafe(ustr, u))
                          {
                              res.status = 400;
                              res.set_content("Parámetro 'umbral' inválido. Ej: ?umbral=0.5", "text/plain");
                              return;
                          }
                          // límites razonables para evitar valores absurdos
                          if (u < 0.0)
                              u = 0.0;
                          if (u > 10.0)
                              u = 10.0;
                          UMBRAL_VERIFICACION = u;

                          LOGI("OREJA", rid, "Umbral recibido por query: umbral=" + std::to_string(UMBRAL_VERIFICACION));
                      }

                      bool coincide = (std::to_string(clase_predicha) == etiqueta);
                      bool pasaUmbral = (score_claimed >= UMBRAL_VERIFICACION);
                      bool autenticado = (clase_predicha != -1) && coincide && pasaUmbral;

                      LOGI("OREJA", rid,
                           "Decisión: etiqueta=" + etiqueta +
                               " id_real=" + std::to_string(id_usuario_real) +
                               " predicho=" + std::to_string(clase_predicha) +
                               " score_top1=" + std::to_string(score_top1) +
                               " score_claimed=" + std::to_string(score_claimed) +
                               " umbral=" + std::to_string(UMBRAL_VERIFICACION) +
                               " coincide=" + std::string(coincide ? "si" : "no") +
                               " pasaUmbral=" + std::string(pasaUmbral ? "si" : "no") +
                               " autenticado=" + std::string(autenticado ? "si" : "no"));

                      // 6) Registrar validación (mejor esfuerzo)
                      {
                          json validacion = {
                              {"id_usuario", id_usuario_real},
                              {"tipo_biometria", "oreja"},
                              {"resultado", autenticado ? "exito" : "fallo"}};

                          auto r = cli.Post("/validaciones_biometricas", validacion.dump(), "application/json");
                          if (!r)
                          {
                              LOGW("OREJA", rid, "BD WARN: no se pudo registrar validación (SIN_RESPUESTA)");
                          }
                          else
                          {
                              LOGI("OREJA", rid, "BD: POST /validaciones_biometricas status=" + std::to_string(r->status));
                          }
                      }

                      json respuesta = {
                          {"id_usuario", id_usuario_real},
                          {"id_usuario_predicho", clase_predicha},
                          {"score_top1", score_top1},
                          {"score_claimed", score_claimed},
                          {"umbral", UMBRAL_VERIFICACION},
                          {"autenticado", autenticado},
                          {"mensaje", autenticado ? "Identidad verificada correctamente"
                                                  : "Identidad no coincide o confianza insuficiente"}};

                      res.status = autenticado ? 200 : 401;
                      res.set_content(respuesta.dump(4), "application/json");
                      LOGI("OREJA", rid, "Respuesta: status=" + std::to_string(res.status) + " autenticado=" + std::string(autenticado ? "si" : "no"));

                      if (!auditMode())
                      {
                          try
                          {
                              fs::remove_all(REQ_DIR);
                          }
                          catch (const std::exception &e)
                          {
                              LOGW("OREJA", rid,
                                   std::string("No se pudo eliminar temporales: ") + e.what());
                          }
                      }
                  });

    // --------------------- POST /oreja/sync/push ---------------------
    // Body JSON: { uuid_dispositivo, caracteristicas: [{id_usuario,id_credencial?,vector_features[],dimension}] }
    servidor.Post("/oreja/sync/push", [](const httplib::Request &req, httplib::Response &res)
                  {
        const std::string rid = makeRequestId();
        LOG_SCOPE("SYNC", rid, "POST /oreja/sync/push");
        logRequestBasics("SYNC", rid, req);

        if (req.body.empty()) {
            res.status = 400;
            res.set_content("Body JSON requerido", "text/plain");
            return;
        }

        json body;
        try { body = json::parse(req.body); }
        catch (...) {
            res.status = 400;
            res.set_content("JSON inválido", "text/plain");
            return;
        }

        if (!body.contains("uuid_dispositivo") || !body.contains("caracteristicas") || !body["caracteristicas"].is_array()) {
            res.status = 400;
            res.set_content("Faltan campos requeridos: uuid_dispositivo, caracteristicas[]", "text/plain");
            return;
        }

        const std::string uuidDispositivo = body["uuid_dispositivo"].get<std::string>();
        const auto& items = body["caracteristicas"];

        json response = {
            {"ok", true},
            {"ids_procesados", json::array()},
            {"procesados", 0},
            {"total", (int)items.size()}
        };

        httplib::Client cli = makeClient();
        httplib::Headers headers = { {"Prefer", "return=representation"} };

        int procesados = 0;
        for (const auto& item : items) {
            if (!item.contains("id_usuario") || !item.contains("vector_features") || !item.contains("dimension")) {
                continue;
            }

            int idUsuario = item["id_usuario"].get<int>();
            int idCredencial = item.value("id_credencial", 0);
            int dimension = item["dimension"].get<int>();

            std::vector<double> features;
            try { features = item["vector_features"].get<std::vector<double>>(); }
            catch (...) { continue; }

            json payload;
            payload["id_usuario"] = idUsuario;
            if (idCredencial > 0) payload["id_credencial"] = idCredencial;
            payload["vector_features"] = vectorToByteArray(features);
            payload["dimension"] = dimension;
            payload["origen"] = "mobile";
            payload["uuid_dispositivo"] = uuidDispositivo;

            auto r = cli.Post("/caracteristicas_oreja", headers, payload.dump(), "application/json");
            if (r && r->status == 201) {
                procesados++;
                try {
                    json data = json::parse(r->body);
                    if (data.is_array() && !data.empty() && data[0].contains("id_caracteristica")) {
                        response["ids_procesados"].push_back(data[0]["id_caracteristica"]);
                    }
                } catch (...) {}
            }
        }

        response["procesados"] = procesados;
        res.status = 200;
        res.set_content(response.dump(4), "application/json"); });

    // --------------------- GET /oreja/sync/pull ---------------------
    // Query: ?desde=<timestamp>
    servidor.Get("/oreja/sync/pull", [](const httplib::Request &req, httplib::Response &res)
                 {
        const std::string rid = makeRequestId();
        LOG_SCOPE("SYNC", rid, "GET /oreja/sync/pull");
        logRequestBasics("SYNC", rid, req);

        const std::string desde = req.has_param("desde") ? safeParam(req, "desde") : "";

        json response = {
            {"ok", true},
            {"usuarios", json::array()},
            {"credenciales", json::array()},
            {"timestamp_actual", nowUtcIso()}
        };

        httplib::Client cli = makeClient();

        // Usuarios
        {
            std::string url = "/usuarios";
            if (!desde.empty()) url += "?updated_at=gt." + desde;
            auto r = cli.Get(url.c_str());
            if (r && r->status == 200) {
                try {
                    json data = json::parse(r->body);
                    for (const auto& u : data) {
                        json item;
                        item["id_usuario"] = u.value("id_usuario", 0);
                        item["identificador_unico"] = u.value("identificador_unico", "");
                        item["estado"] = u.value("estado", "activo");
                        item["updated_at"] = u.value("updated_at", "");
                        response["usuarios"].push_back(item);
                    }
                } catch (...) {
                    response["ok"] = false;
                }
            } else {
                response["ok"] = false;
            }
        }

        // Credenciales
        {
            std::string url = "/credenciales_biometricas";
            if (!desde.empty()) url += "?updated_at=gt." + desde;
            auto r = cli.Get(url.c_str());
            if (r && r->status == 200) {
                try {
                    json data = json::parse(r->body);
                    for (const auto& c : data) {
                        json item;
                        item["id_credencial"] = c.value("id_credencial", 0);
                        item["id_usuario"] = c.value("id_usuario", 0);
                        item["tipo_biometria"] = c.value("tipo_biometria", "oreja");
                        item["estado"] = c.value("estado", "activo");
                        item["updated_at"] = c.value("updated_at", "");
                        response["credenciales"].push_back(item);
                    }
                } catch (...) {
                    response["ok"] = false;
                }
            } else {
                response["ok"] = false;
            }
        }

        res.status = response["ok"].get<bool>() ? 200 : 502;
        res.set_content(response.dump(4), "application/json"); });

    // --------------------- GET /oreja/sync/modelo ---------------------
    // Query: ?archivo=<nombre>
    servidor.Get("/oreja/sync/modelo", [](const httplib::Request &req, httplib::Response &res)
                 {
        const std::string rid = makeRequestId();
        LOG_SCOPE("SYNC", rid, "GET /oreja/sync/modelo");
        logRequestBasics("SYNC", rid, req);

        const std::string archivo = req.has_param("archivo") ? safeParam(req, "archivo") : "modelo_svm.svm";
        const std::vector<std::string> allow = {
            "modelo_svm.svm",
            "modelo_pca.dat",
            "modelo_lda.dat",
            "zscore_params.dat",
            "umbral_svm.txt",
            "templates_k1.csv",
            "umbrales_metricas_tecnicas.csv"
        };

        if (std::find(allow.begin(), allow.end(), archivo) == allow.end()) {
            res.status = 400;
            res.set_content("Archivo no permitido", "text/plain");
            return;
        }

        const std::string modelDir = getEnvStr("MODEL_DIR", "out");
        const std::string path = modelDir + "/" + archivo;
        if (!fs::exists(path)) {
            res.status = 404;
            res.set_content("Archivo no encontrado", "text/plain");
            return;
        }

        std::ifstream f(path, std::ios::binary);
        if (!f.is_open()) {
            res.status = 500;
            res.set_content("No se pudo abrir el archivo", "text/plain");
            return;
        }

        std::ostringstream ss;
        ss << f.rdbuf();
        res.set_header("Content-Disposition", "attachment; filename=" + archivo);
        res.set_content(ss.str(), "application/octet-stream"); });

    // --------------------- POST /eliminar?identificador= ---------------------
    servidor.Post("/eliminar", [](const httplib::Request &req, httplib::Response &res)
                  {
        const std::string rid = makeRequestId();
        LOG_SCOPE("USUARIOS", rid, "POST /eliminar");

        LOGI("USUARIOS", rid, "\n" + repTitle("ELIMINACION DE USUARIO"));
        LOGI("USUARIOS", rid, repSection("INICIO DE PROCESO"));
        logRequestBasics("USUARIOS", rid, req);

        // ===== Entrada =====
        LOGI("USUARIOS", rid, repSection("ENTRADA DE DATOS"));

        if (!req.has_param("identificador")) {
            LOGW("USUARIOS", rid, repFAIL("Falta parámetro obligatorio: identificador"));

            LOGI("USUARIOS", rid, repSection("FINALIZACION"));
            LOGW("USUARIOS", rid, repFAIL("Proceso terminado por entrada inválida"));
            LOGW("USUARIOS", rid, "RESULTADO: FAIL http=400 title=ENTRADA_INVALIDA");

            res.status = 400;
            json pub = buildPublicErrorBody(400, "ENTRADA_INVALIDA", "Falta parámetro 'identificador'");
            res.set_content(pub.dump(4), "application/json");
            return;
        }

        const std::string identificador = safeParam(req, "identificador");
        LOGI("USUARIOS", rid, repKeyVal("identificador", identificador));

        if (identificador.empty()) {
            LOGW("USUARIOS", rid, repFAIL("Parámetro identificador vacío"));

            LOGI("USUARIOS", rid, repSection("FINALIZACION"));
            LOGW("USUARIOS", rid, repFAIL("Proceso terminado por entrada inválida"));
            LOGW("USUARIOS", rid, "RESULTADO: FAIL http=400 title=ENTRADA_INVALIDA");

            res.status = 400;
            json pub = buildPublicErrorBody(400, "ENTRADA_INVALIDA", "El parámetro 'identificador' no puede estar vacío");
            res.set_content(pub.dump(4), "application/json");
            return;
        }

        // ===== Consulta usuario =====
        LOGI("USUARIOS", rid, repSection("CONSULTA EN BD (PostgREST)"));

        httplib::Client cli = makeClient();
        const std::string urlUsuario = "/usuarios?identificador_unico=eq." + identificador;

        auto r = cli.Get(urlUsuario.c_str());
        if (!r) {
            LOGE("USUARIOS", rid, repFAIL("Sin respuesta de BD en GET /usuarios"));
            LOGE("USUARIOS", rid, repKeyVal("url", urlUsuario));

            LOGI("USUARIOS", rid, repSection("FINALIZACION"));
            LOGW("USUARIOS", rid, repFAIL("Proceso finalizado por dependencia externa (BD)"));
            LOGW("USUARIOS", rid, "RESULTADO: FAIL http=502 title=DEPENDENCIA_EXTERNA");

            res.status = 502;
            json pub = buildPublicErrorBody(502, "DEPENDENCIA_EXTERNA", "No se pudo consultar el usuario (BD sin respuesta)");
            res.set_content(pub.dump(4), "application/json");
            return;
        }

        LOGI("USUARIOS", rid, "BD: GET " + urlUsuario + " status=" + std::to_string(r->status) +
                            " body_bytes=" + std::to_string(r->body.size()));

        if (r->status != 200) {
            LOGE("USUARIOS", rid, repFAIL("GET /usuarios retornó status inesperado"));
            LOGE("USUARIOS", rid, repKeyVal("status", std::to_string(r->status)));

            LOGI("USUARIOS", rid, repSection("FINALIZACION"));
            LOGW("USUARIOS", rid, repFAIL("Proceso finalizado por error consultando BD"));
            LOGW("USUARIOS", rid, "RESULTADO: FAIL http=502 title=DEPENDENCIA_EXTERNA");

            res.status = 502;
            json pub = buildPublicErrorBody(502, "DEPENDENCIA_EXTERNA", "No se pudo consultar el usuario (status inesperado)");
            res.set_content(pub.dump(4), "application/json");
            return;
        }

        json data;
        try { data = json::parse(r->body); }
        catch (...) {
            LOGE("USUARIOS", rid, repFAIL("Respuesta BD inválida: JSON parse falló"));
            LOGE("USUARIOS", rid, repKeyVal("url", urlUsuario));

            LOGI("USUARIOS", rid, repSection("FINALIZACION"));
            LOGW("USUARIOS", rid, repFAIL("Proceso finalizado por respuesta inválida de BD"));
            LOGW("USUARIOS", rid, "RESULTADO: FAIL http=502 title=DEPENDENCIA_EXTERNA");

            res.status = 502;
            json pub = buildPublicErrorBody(502, "DEPENDENCIA_EXTERNA", "Respuesta inválida de BD (JSON)");
            res.set_content(pub.dump(4), "application/json");
            return;
        }

        if (!data.is_array() || data.empty() || !data[0].contains("id_usuario")) {
            LOGW("USUARIOS", rid, repWARN("Usuario no encontrado en BD"));
            LOGW("USUARIOS", rid, repKeyVal("identificador", identificador));

            LOGI("USUARIOS", rid, repSection("FINALIZACION"));
            LOGW("USUARIOS", rid, repFAIL("Proceso finalizado: usuario no existe"));
            LOGW("USUARIOS", rid, "RESULTADO: FAIL http=404 title=NO_ENCONTRADO");

            res.status = 404;
            json pub = buildPublicErrorBody(404, "NO_ENCONTRADO", "Usuario no encontrado");
            res.set_content(pub.dump(4), "application/json");
            return;
        }

        const int id_usuario = data[0]["id_usuario"];
        const std::string estadoActual = data[0].value("estado", "activo");

        LOGI("USUARIOS", rid, repSub("Usuario localizado"));
        LOGI("USUARIOS", rid, repKeyVal("id_usuario", std::to_string(id_usuario)));
        LOGI("USUARIOS", rid, repKeyVal("estado_actual", estadoActual));

        // ===== Controles =====
        LOGI("USUARIOS", rid, repSection("CONTROLES"));

        if (estadoActual == "eliminado") {
            LOGI("USUARIOS", rid, repOK("Usuario ya estaba marcado como eliminado (sin cambios)"));

            LOGI("USUARIOS", rid, repSection("FINALIZACION"));
            LOGI("USUARIOS", rid, repOK("Sin cambios: estado ya era 'eliminado'"));
            LOGI("USUARIOS", rid, "RESULTADO: OK http=200 accion=sin_cambios");

            res.status = 200;
            json ok = { {"ok", true}, {"mensaje", "El usuario ya estaba marcado como eliminado"} };
            res.set_content(ok.dump(4), "application/json");
            return;
        }

        // ===== Patch =====
        LOGI("USUARIOS", rid, repSection("ACTUALIZACION EN BD (PATCH)"));

        json update = { {"estado", "eliminado"} };
        const std::string urlPatch = "/usuarios?id_usuario=eq." + std::to_string(id_usuario);

        auto p = cli.Patch(urlPatch.c_str(), update.dump(), "application/json");
        if (!p) {
            LOGE("USUARIOS", rid, repFAIL("Sin respuesta de BD en PATCH /usuarios"));
            LOGE("USUARIOS", rid, repKeyVal("url", urlPatch));

            LOGI("USUARIOS", rid, repSection("FINALIZACION"));
            LOGW("USUARIOS", rid, repFAIL("Proceso finalizado por dependencia externa (BD)"));
            LOGW("USUARIOS", rid, "RESULTADO: FAIL http=502 title=DEPENDENCIA_EXTERNA");

            res.status = 502;
            json pub = buildPublicErrorBody(502, "DEPENDENCIA_EXTERNA", "No se pudo eliminar el usuario (BD sin respuesta)");
            res.set_content(pub.dump(4), "application/json");
            return;
        }

        LOGI("USUARIOS", rid, "BD: PATCH " + urlPatch + " status=" + std::to_string(p->status));

        if (p->status != 204 && p->status != 200) {
            LOGE("USUARIOS", rid, repFAIL("PATCH /usuarios status inesperado"));
            LOGE("USUARIOS", rid, repKeyVal("status", std::to_string(p->status)));

            LOGI("USUARIOS", rid, repSection("FINALIZACION"));
            LOGW("USUARIOS", rid, repFAIL("Proceso finalizado: no se pudo actualizar estado"));
            LOGW("USUARIOS", rid, "RESULTADO: FAIL http=502 title=DEPENDENCIA_EXTERNA");

            res.status = 502;
            json pub = buildPublicErrorBody(502, "DEPENDENCIA_EXTERNA", "No se pudo eliminar el usuario (status inesperado)");
            res.set_content(pub.dump(4), "application/json");
            return;
        }

        // ===== Finalización OK =====
        LOGI("USUARIOS", rid, repSection("FINALIZACION"));
        LOGI("USUARIOS", rid, repOK("Usuario marcado como eliminado"));
        LOGI("USUARIOS", rid, repKeyVal("id_usuario", std::to_string(id_usuario)));
        LOGI("USUARIOS", rid, repKeyVal("identificador", identificador));
        LOGI("USUARIOS", rid, "RESULTADO: OK http=200 accion=eliminar");

        res.status = 200;
        json ok = { {"ok", true}, {"mensaje", "Usuario eliminado"} };
        res.set_content(ok.dump(4), "application/json"); });

    // --------------------- POST /restaurar?identificador= ---------------------
    servidor.Post("/restaurar", [](const httplib::Request &req, httplib::Response &res)
                  {
        const std::string rid = makeRequestId();
        LOG_SCOPE("USUARIOS", rid, "POST /restaurar");

        LOGI("USUARIOS", rid, "\n" + repTitle("RESTAURACION DE USUARIO"));
        LOGI("USUARIOS", rid, repSection("INICIO DE PROCESO"));
        logRequestBasics("USUARIOS", rid, req);

        // ===== Entrada =====
        LOGI("USUARIOS", rid, repSection("ENTRADA DE DATOS"));

        if (!req.has_param("identificador")) {
            LOGW("USUARIOS", rid, repFAIL("Falta parámetro obligatorio: identificador"));

            LOGI("USUARIOS", rid, repSection("FINALIZACION"));
            LOGW("USUARIOS", rid, repFAIL("Proceso terminado por entrada inválida"));
            LOGW("USUARIOS", rid, "RESULTADO: FAIL http=400 title=ENTRADA_INVALIDA");

            res.status = 400;
            json pub = buildPublicErrorBody(400, "ENTRADA_INVALIDA", "Falta parámetro 'identificador'");
            res.set_content(pub.dump(4), "application/json");
            return;
        }

        const std::string identificador = safeParam(req, "identificador");
        LOGI("USUARIOS", rid, repKeyVal("identificador", identificador));

        if (identificador.empty()) {
            LOGW("USUARIOS", rid, repFAIL("Parámetro identificador vacío"));

            LOGI("USUARIOS", rid, repSection("FINALIZACION"));
            LOGW("USUARIOS", rid, repFAIL("Proceso terminado por entrada inválida"));
            LOGW("USUARIOS", rid, "RESULTADO: FAIL http=400 title=ENTRADA_INVALIDA");

            res.status = 400;
            json pub = buildPublicErrorBody(400, "ENTRADA_INVALIDA", "El parámetro 'identificador' no puede estar vacío");
            res.set_content(pub.dump(4), "application/json");
            return;
        }

        // ===== Consulta usuario =====
        LOGI("USUARIOS", rid, repSection("CONSULTA EN BD (PostgREST)"));

        httplib::Client cli = makeClient();
        const std::string urlUsuario = "/usuarios?identificador_unico=eq." + identificador;

        auto r = cli.Get(urlUsuario.c_str());
        if (!r) {
            LOGE("USUARIOS", rid, repFAIL("Sin respuesta de BD en GET /usuarios"));
            LOGE("USUARIOS", rid, repKeyVal("url", urlUsuario));

            LOGI("USUARIOS", rid, repSection("FINALIZACION"));
            LOGW("USUARIOS", rid, repFAIL("Proceso finalizado por dependencia externa (BD)"));
            LOGW("USUARIOS", rid, "RESULTADO: FAIL http=502 title=DEPENDENCIA_EXTERNA");

            res.status = 502;
            json pub = buildPublicErrorBody(502, "DEPENDENCIA_EXTERNA", "No se pudo consultar el usuario (BD sin respuesta)");
            res.set_content(pub.dump(4), "application/json");
            return;
        }

        LOGI("USUARIOS", rid, "BD: GET " + urlUsuario + " status=" + std::to_string(r->status) +
                            " body_bytes=" + std::to_string(r->body.size()));

        if (r->status != 200) {
            LOGE("USUARIOS", rid, repFAIL("GET /usuarios retornó status inesperado"));
            LOGE("USUARIOS", rid, repKeyVal("status", std::to_string(r->status)));

            LOGI("USUARIOS", rid, repSection("FINALIZACION"));
            LOGW("USUARIOS", rid, repFAIL("Proceso finalizado por error consultando BD"));
            LOGW("USUARIOS", rid, "RESULTADO: FAIL http=502 title=DEPENDENCIA_EXTERNA");

            res.status = 502;
            json pub = buildPublicErrorBody(502, "DEPENDENCIA_EXTERNA", "No se pudo consultar el usuario (status inesperado)");
            res.set_content(pub.dump(4), "application/json");
            return;
        }

        json data;
        try { data = json::parse(r->body); }
        catch (...) {
            LOGE("USUARIOS", rid, repFAIL("Respuesta BD inválida: JSON parse falló"));
            LOGE("USUARIOS", rid, repKeyVal("url", urlUsuario));

            LOGI("USUARIOS", rid, repSection("FINALIZACION"));
            LOGW("USUARIOS", rid, repFAIL("Proceso finalizado por respuesta inválida de BD"));
            LOGW("USUARIOS", rid, "RESULTADO: FAIL http=502 title=DEPENDENCIA_EXTERNA");

            res.status = 502;
            json pub = buildPublicErrorBody(502, "DEPENDENCIA_EXTERNA", "Respuesta inválida de BD (JSON)");
            res.set_content(pub.dump(4), "application/json");
            return;
        }

        if (!data.is_array() || data.empty() || !data[0].contains("id_usuario")) {
            LOGW("USUARIOS", rid, repWARN("Usuario no encontrado en BD"));
            LOGW("USUARIOS", rid, repKeyVal("identificador", identificador));

            LOGI("USUARIOS", rid, repSection("FINALIZACION"));
            LOGW("USUARIOS", rid, repFAIL("Proceso finalizado: usuario no existe"));
            LOGW("USUARIOS", rid, "RESULTADO: FAIL http=404 title=NO_ENCONTRADO");

            res.status = 404;
            json pub = buildPublicErrorBody(404, "NO_ENCONTRADO", "Usuario no encontrado");
            res.set_content(pub.dump(4), "application/json");
            return;
        }

        const int id_usuario = data[0]["id_usuario"];
        const std::string estadoActual = data[0].value("estado", "activo");

        LOGI("USUARIOS", rid, repSub("Usuario localizado"));
        LOGI("USUARIOS", rid, repKeyVal("id_usuario", std::to_string(id_usuario)));
        LOGI("USUARIOS", rid, repKeyVal("estado_actual", estadoActual));

        // ===== Controles =====
        LOGI("USUARIOS", rid, repSection("CONTROLES"));

        if (estadoActual == "activo") {
            LOGI("USUARIOS", rid, repOK("Usuario ya estaba activo (sin cambios)"));

            LOGI("USUARIOS", rid, repSection("FINALIZACION"));
            LOGI("USUARIOS", rid, repOK("Sin cambios: estado ya era 'activo'"));
            LOGI("USUARIOS", rid, "RESULTADO: OK http=200 accion=sin_cambios");

            res.status = 200;
            json ok = { {"ok", true}, {"mensaje", "El usuario ya estaba activo"} };
            res.set_content(ok.dump(4), "application/json");
            return;
        }

        // ===== Patch =====
        LOGI("USUARIOS", rid, repSection("ACTUALIZACION EN BD (PATCH)"));

        json update = { {"estado", "activo"} };
        const std::string urlPatch = "/usuarios?id_usuario=eq." + std::to_string(id_usuario);

        auto p = cli.Patch(urlPatch.c_str(), update.dump(), "application/json");
        if (!p) {
            LOGE("USUARIOS", rid, repFAIL("Sin respuesta de BD en PATCH /usuarios"));
            LOGE("USUARIOS", rid, repKeyVal("url", urlPatch));

            LOGI("USUARIOS", rid, repSection("FINALIZACION"));
            LOGW("USUARIOS", rid, repFAIL("Proceso finalizado por dependencia externa (BD)"));
            LOGW("USUARIOS", rid, "RESULTADO: FAIL http=502 title=DEPENDENCIA_EXTERNA");

            res.status = 502;
            json pub = buildPublicErrorBody(502, "DEPENDENCIA_EXTERNA", "No se pudo restaurar el usuario (BD sin respuesta)");
            res.set_content(pub.dump(4), "application/json");
            return;
        }

        LOGI("USUARIOS", rid, "BD: PATCH " + urlPatch + " status=" + std::to_string(p->status));

        if (p->status != 204 && p->status != 200) {
            LOGE("USUARIOS", rid, repFAIL("PATCH /usuarios status inesperado"));
            LOGE("USUARIOS", rid, repKeyVal("status", std::to_string(p->status)));

            LOGI("USUARIOS", rid, repSection("FINALIZACION"));
            LOGW("USUARIOS", rid, repFAIL("Proceso finalizado: no se pudo actualizar estado"));
            LOGW("USUARIOS", rid, "RESULTADO: FAIL http=502 title=DEPENDENCIA_EXTERNA");

            res.status = 502;
            json pub = buildPublicErrorBody(502, "DEPENDENCIA_EXTERNA", "No se pudo restaurar el usuario (status inesperado)");
            res.set_content(pub.dump(4), "application/json");
            return;
        }

        // ===== Finalización OK =====
        LOGI("USUARIOS", rid, repSection("FINALIZACION"));
        LOGI("USUARIOS", rid, repOK("Usuario restaurado a estado activo"));
        LOGI("USUARIOS", rid, repKeyVal("id_usuario", std::to_string(id_usuario)));
        LOGI("USUARIOS", rid, repKeyVal("identificador", identificador));
        LOGI("USUARIOS", rid, "RESULTADO: OK http=200 accion=restaurar");

        res.status = 200;
        json ok = { {"ok", true}, {"mensaje", "Usuario restaurado exitosamente"} };
        res.set_content(ok.dump(4), "application/json"); });

    LOGI("OREJA", "", "Servidor biometria oreja escuchando en 0.0.0.0:8085");
    servidor.listen("0.0.0.0", 8085);
    return 0;
}
