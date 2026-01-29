#include <iostream>
#include <fstream>
#include <filesystem>
#include "../../../external/httplib.h"
#include "../../../core/asr/whisper_asr.h"

int main() {
    httplib::Server svr;

    svr.Post("/verificar", [](const httplib::Request& req, httplib::Response& res) {
        auto file = req.get_file_value("audio");
        std::string expected = req.get_param_value("frase");  //  Obtener frase esperada

        std::string filename = "grabacion.wav";
        std::ofstream ofs(filename, std::ios::binary);
        ofs.write(file.content.c_str(), file.content.size());
        ofs.close();

        std::cout << "Audio recibido. Procesando...\n";

        bool resultado = transcribeAndCompare(filename, expected);  //  Pasar ambos parámetros
        res.set_content(resultado ? "true" : "false", "text/plain");
        });

    std::cout << "ASR server activo en http://10.43.164.242:8080\n";

    svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("Servidor ASR activo ", "text/plain");
        });

    svr.listen("0.0.0.0", 8080);
    return 0;
}
