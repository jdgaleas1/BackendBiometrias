#include <iostream>
#include <utility>
#include "../../../external/json.hpp"
#include "../../../external/httplib.h"
#include "../../../utils/config.h"

using json = nlohmann::json;

int main() {
    /*
    auto [host, port] = obtenerPostgRESTConfig();
    httplib::Client cli(host.c_str(), port);

    // 1. Crear un nuevo usuario (POST)
    json nuevo = {
        {"correo", "ejemplo@correo.com"},
        {"clase_oreja", 12},
        {"clase_voz", 34}
    };

    auto res_post = cli.Post("/usuarios", nuevo.dump(), "application/json");

    if (res_post && res_post->status == 201) {
        std::cout << "✅ Usuario creado.\n";
    }
    else {
        std::cerr << "❌ Error creando usuario: " << (res_post ? res_post->body : "no response") << "\n";
    }

    // 2. Obtener info de usuario por correo (GET)
    std::string correo = "ejemplo@correo.com";
    std::string url = "/usuarios?correo=eq." + httplib::detail::encode_url(correo);

    auto res_get = cli.Get(url.c_str());

    if (res_get && res_get->status == 200) {
        auto data = json::parse(res_get->body);
        if (!data.empty()) {
            int id = data[0]["id"];
            std::cout << "🔍 Usuario encontrado. ID: " << id << "\n";

            // 3. Suspender usuario (PATCH)
            json update = { {"estado", "suspendido"} };
            std::string patch_url = "/usuarios?id=eq." + std::to_string(id);

            auto res_patch = cli.Patch(patch_url.c_str(), update.dump(), "application/json");

            if (res_patch && res_patch->status == 204) {
                std::cout << "⛔ Usuario suspendido correctamente.\n";
            }
            else {
                std::cerr << "❌ Error suspendiendo usuario\n";
            }
        }
        else {
            std::cout << "⚠️ Usuario no encontrado.\n";
        }
    }
    else {
        std::cerr << "❌ Error al buscar usuario\n";
    }

    return 0;
    */
}
