#include "frases_controller.h"


FrasesController::FrasesController() {}

nlohmann::json FrasesController::actualizarEstadoFrase(int id_texto, int activo) {
    return service.actualizarEstadoFrase(id_texto, activo);
}
nlohmann::json FrasesController::agregarFrase(const std::string& frase) {
    return service.insertarFrase(frase);
}

nlohmann::json FrasesController::listarFrases() {
    return service.obtenerTodas();
}

nlohmann::json FrasesController::obtenerFraseAleatoria() {
    return service.obtenerFraseAleatoria();
}
nlohmann::json FrasesController::obtenerFrasePorId(int id) {
    // Implementar consulta específica por ID

    return service.obtenerFrasePorId(id);
}

nlohmann::json FrasesController::eliminarFrase(int id_texto) {
    return service.eliminarFrase(id_texto);
}