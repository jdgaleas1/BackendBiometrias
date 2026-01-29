#ifndef REGISTRAR_SERVICE_H
#define REGISTRAR_SERVICE_H

#include <string>
#include <vector>
#include <map>
#include "../../external/json.hpp"
#include "../../utils/config.h"

using json = nlohmann::json;


struct ResultadoRegistro {
    bool exito = false;
    int userId = -1;
    std::string userName;
    int totalAudios = 0;
    int audiosExitosos = 0;
    int audiosFallidos = 0;
    std::string error;
};

struct ResultadoEntrenamientoModelo {
    bool exito = false;
    std::string mensaje;
    std::string error;
    int numClases = 0;
};

class RegistrarService {
public:
    RegistrarService(const std::string& mappingPath = "model/speaker_mapping.txt",
                     const std::string& trainDataPath = "processed_dataset_bin/train_dataset.bin");

    // Registrar usuario completo (nombre + biometria) - DEPRECADO
    json registrarUsuarioCompleto(const std::string& nombre, 
                                  const std::vector<std::string>& audioPaths);

    // Registrar SOLO biometria usando cedula de usuario existente
    json registrarBiometriaPorCedula(const std::string& cedula, 
                                     const std::vector<std::string>& audioPaths);

    // Entrenar modelo (llamar DESPUÉS de registrar)
    ResultadoEntrenamientoModelo entrenarModelo();

    // Legacy (mantener compatibilidad)
    ResultadoRegistro registrarUsuario(const std::string& nombre, 
                                       const std::vector<std::string>& audiosPaths);

private:
    std::string mappingPath;
    std::string trainDataPath;
    std::map<int, std::string> mapeoUsuarios;

    void cargarMapeos();
    bool procesarAudio(const std::string& audioPath, std::vector<AudioSample>& features);
    int obtenerSiguienteId();
};

#endif // REGISTRAR_SERVICE_H
