#ifndef LOGGER_H
#define LOGGER_H

#include <string>
#include <fstream>
#include <mutex>
#include <chrono>

enum LogLevel {
    LOG_DEBUG = 0,
    LOG_INFO  = 1,
    LOG_WARN  = 2,
    LOG_ERROR = 3
};

// Config
void setLogLevel(LogLevel level);
void setLogFile(const std::string& path);   // opcional: si no se llama, log solo a stdout
std::string makeRequestId();

// Log principal
void logMessage(LogLevel level,
                const std::string& tag,
                const std::string& rid,
                const std::string& msg);

// Scope RAII: mide duración y deja log de entrada/salida del bloque
class LogScope {
public:
    LogScope(const std::string& tag,
             const std::string& rid,
             const std::string& name);
    ~LogScope();

private:
    std::string tag_;
    std::string rid_;
    std::string name_;
    std::chrono::steady_clock::time_point t0_;
};

// Macros cómodos
#define LOGD(tag, rid, msg) logMessage(LOG_DEBUG, tag, rid, msg)
#define LOGI(tag, rid, msg) logMessage(LOG_INFO,  tag, rid, msg)
#define LOGW(tag, rid, msg) logMessage(LOG_WARN,  tag, rid, msg)
#define LOGE(tag, rid, msg) logMessage(LOG_ERROR, tag, rid, msg)

#define LOG_SCOPE(tag, rid, name) LogScope _logscope_##__LINE__(tag, rid, name)

#endif
