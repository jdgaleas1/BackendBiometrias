#include "server_utils.h"

#include <algorithm>
#include <cctype>

// ========== Copia exacta (solo removido "static") ==========

std::string trunc(const std::string& s, size_t n) {
    if (s.size() <= n) return s;
    return s.substr(0, n) + "...";
}

std::string jsonCampo(const json& j, const char* k) {
    if (!j.contains(k)) return "";
    if (j[k].is_string()) return j[k].get<std::string>();
    return j[k].dump();
}

std::string repLine(char ch, int n) {
    return std::string(n, ch) + "\n";
}

std::string repKV(const std::string& k, const std::string& v) {
    return "  - " + k + ": " + v + "\n";
}

std::string repQC(const std::string& metrica, const std::string& valor,
                  const std::string& umbral, const std::string& detalle) {
    std::string out;
    out += "  * QC [" + metrica + "] valor=" + valor + " umbral=" + umbral;
    if (!detalle.empty()) out += " (" + detalle + ")";
    out += "\n";
    return out;
}

std::string truncN(const std::string& s, size_t max_chars) {
    if (s.size() <= max_chars) return s;
    return s.substr(0, max_chars) + "\n... (truncado)\n";
}

std::string resumenUsuarioJson(const json& datos) {
    std::string out;
    out += repLine('=');
    out += "  RESUMEN DE DATOS DE USUARIO (JSON)\n";
    out += repLine('=');

    // Identificador único (prioridad)
    std::string idu = jsonCampo(datos, "identificador_unico");
    if (idu.empty()) idu = jsonCampo(datos, "identificador");
    if (idu.empty()) idu = jsonCampo(datos, "cedula"); // fallback por si aún existe

    out += repKV("cedula", idu);

    // Campos que sí usas
    out += repKV("nombres", jsonCampo(datos, "nombres"));
    out += repKV("apellidos", jsonCampo(datos, "apellidos"));

    // Opcionales: solo imprimir si existen (para que no ensucie el log)

    auto sexo = jsonCampo(datos, "sexo");
    if (!sexo.empty()) out += repKV("sexo", sexo);

    auto fn = jsonCampo(datos, "fecha_nacimiento");
    if (!fn.empty()) out += repKV("fecha_nacimiento", fn);

    out += repLine('=');
    return out;
}

bool esEntero(const std::string& s) {
    if (s.empty()) return false;
    size_t i = 0;
    if (s[0] == '-' || s[0] == '+') i = 1;
    if (i >= s.size()) return false;
    for (; i < s.size(); ++i) {
        if (!std::isdigit(static_cast<unsigned char>(s[i]))) return false;
    }
    return true;
}

bool esDoubleSimple(const std::string& s) {
    if (s.empty()) return false;
    bool dot = false;
    size_t i = 0;
    if (s[0] == '-' || s[0] == '+') i = 1;
    if (i >= s.size()) return false;

    for (; i < s.size(); ++i) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        if (c == '.') {
            if (dot) return false;
            dot = true;
        } else if (!std::isdigit(c)) {
            return false;
        }
    }
    return true;
}

std::string safeParam(const httplib::Request& req, const std::string& name) {
    if (!req.has_param(name)) return "";
    return req.get_param_value(name);
}
