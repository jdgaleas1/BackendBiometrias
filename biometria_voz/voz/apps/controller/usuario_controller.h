#ifndef USUARIO_CONTROLLER_H
#define USUARIO_CONTROLLER_H

#include <string>
#include <vector>
#include <memory>
#include <random>
#include "../service/autenticacion_service.h"
#include "../service/listar_service.h"
#include "../service/registrar_service.h"
#include "../../external/json.hpp"

using json = nlohmann::json;

class UsuarioController {
private:
    std::unique_ptr<AutenticacionService> authService;
    std::unique_ptr<RegistrarService> registerService;
    std::unique_ptr<ListarService> listService;

    // Configuración
    std::string modelPath;
    std::string mappingPath;
    std::string trainDataPath;
    std::string tempDir;

public:
    UsuarioController();
    ~UsuarioController();

    // Endpoints principales
    json autenticar(const std::string& audioPath, const std::string& identificador, int idFrase, 
                    const std::string& ipCliente = "127.0.0.1", const std::string& userAgent = "");
    json registrarUsuario(const std::string& nombre, const std::vector<std::string>& audiosPaths);
    json registrarBiometria(const std::string& cedula, const std::vector<std::string>& audioPaths);
    json listarUsuarios();
    json eliminarUsuario(int userId);
    json entrenarModelo();
    
    // Utilidades
    void recargarConfiguracion();
};

#endif