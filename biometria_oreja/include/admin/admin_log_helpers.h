#pragma once
#include <fstream>
#include <string>
#include <vector>

#include "admin/admin_types.h"

// Crea /workDir/agregar_usuario_biometria.log (append)
std::ofstream crearLogStream(const std::string& workDir);

// "[BIO] [rid=XYZ] mensaje"
std::string mkLine(const std::string& rid, const std::string& msg);

// Escritura cruda a archivo + stderr (thread-safe por omp critical)
void logRaw(std::ofstream& log, const std::string& text);
void logRawLine(std::ofstream& log, const std::string& line);

// === Overloads de compatibilidad (para no tocar tus 500 llamadas) ===
void logRaw(std::ofstream& log, const std::string& rid, const std::string& textBlock);
void logRawLine(std::ofstream& log, const std::string& rid, const std::string& msg);
void logBlank(std::ofstream& log, const std::string& rid);

void logMensaje(std::ofstream& log, const std::string& rid, const std::string& msg);
void logBlank(std::ofstream& log);

// Decoradores
void logTechTitle(std::ofstream& log, const std::string& rid, const std::string& title);

void logSection(std::ofstream& log, const std::string& rid, int LOG_DETAIL, const std::string& title);
void logPrettyTitle(std::ofstream& log, const Ctx& ctx, const std::string& title);

void logBox(std::ofstream& log, const std::string& rid, int LOG_DETAIL,
            const std::string& title,
            const std::vector<std::string>& lines = {});

// Fase/pasos
void logPhase(std::ofstream& log, const std::string& rid, int LOG_DETAIL,
              int phaseNum, const std::string& phaseName,
              const std::string& objective,
              const std::vector<std::string>& lines = {});

void logStep(std::ofstream& log, const std::string& rid, int LOG_DETAIL,
             const std::string& stepId, const std::string& title);

// Mensajes con indent configurable (default = 4)
void logKV(std::ofstream& log, const std::string& rid, int LOG_DETAIL,
           const std::string& k, const std::string& v, int indent = 4);

void logOK(std::ofstream& log, const std::string& rid, int LOG_DETAIL,
           const std::string& msg, int indent = 4);

void logWARN(std::ofstream& log, const std::string& rid, int LOG_DETAIL,
             const std::string& msg, int indent = 4);

void logERR(std::ofstream& log, const std::string& rid, int LOG_DETAIL,
            const std::string& msg, int indent = 4);

// LOG_DETAIL >= level => imprime
void logDet(std::ofstream& log, const std::string& rid, int LOG_DETAIL, int level, const std::string& msg);

// Línea especial que tu servidor “extrae”
void log7B(std::ofstream& log, const std::string& rid, const std::string& msg);
