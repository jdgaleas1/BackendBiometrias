#include "server_env.h"

#include <cstdlib>

std::string getEnvStr(const char* key, const std::string& def_value) {
    const char* v = std::getenv(key);
    if (!v || !*v) return def_value;
    return std::string(v);
}

std::string getEnvStr(const std::string& key, const std::string& def_value) {
    return getEnvStr(key.c_str(), def_value);
}

bool auditMode() {
    // MISMA lógica que antes si tú comparabas == "1"
    // Si quieres 100% idéntico a tu servidor actual, usa solo (v == "1")
    const std::string v = getEnvStr("AUDIT_MODE", "0");
    return (v == "1"); // <- modo estricto, sin alterar comportamiento
}

std::string tmpDir() {
    return getEnvStr("TMP_DIR", "/tmp/biometria_oreja");
}
