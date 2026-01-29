#include "listar_service.h"
#include "../../external/json.hpp"
#include "../../core/classification/svm.h"
#include <fstream>
#include <sstream>
#include <filesystem>

using json = nlohmann::json;
namespace fs = std::filesystem;

ListarService::ListarService(const std::string& mappingPath) {
    cargarDatos(mappingPath);
}

void ListarService::recargarDatos(const std::string& mappingPath) {
    cargarDatos(mappingPath);
}

void ListarService::cargarDatos(const std::string& mappingPath) {
    mapeoUsuarios.clear();
    usuariosSuspendidos.clear();

    // Cargar desde metadata.json (fuente de verdad)
    std::string metadataPath = obtenerRutaModelo() + "metadata.json";
    std::ifstream metaIn(metadataPath);
    
    if (metaIn.is_open()) {
        try {
            json metadata;
            metaIn >> metadata;
            metaIn.close();

            // Extraer lista de clases activas
            auto& clases = metadata["classes"];
            
            // Por ahora usamos el ID como nombre
            for (int userId : clases) {
                mapeoUsuarios[userId] = "Usuario_" + std::to_string(userId);
            }

        } catch (const std::exception& e) {
            std::cerr << "! Error parseando metadata.json: " << e.what() << std::endl;
        }
    } else {
        std::cerr << "! Error: No se pudo abrir " << metadataPath << std::endl;
    }
}

ResultadoListado ListarService::listarUsuarios() {
    ResultadoListado resultado;

    try {
        for (const auto& [id, nombre] : mapeoUsuarios) {
            Usuario usuario;
            usuario.id = id;
            usuario.nombre = nombre;
            resultado.usuarios.push_back(usuario);
        }

        resultado.exito = true;
        resultado.total = resultado.usuarios.size();

    }
    catch (const std::exception& e) {
        resultado.exito = false;
        resultado.error = std::string("Error listando usuarios: ") + e.what();
    }

    return resultado;
}

bool ListarService::eliminarUsuario(int userId) {
    try {
        std::string modelPath = obtenerRutaModelo();
        std::string metadataPath = modelPath + "metadata.json";
        
        // 1. Leer metadata.json
        std::ifstream metaIn(metadataPath);
        if (!metaIn.is_open()) {
            std::cerr << "! Error: No se pudo abrir metadata.json" << std::endl;
            return false;
        }

        json metadata;
        metaIn >> metadata;
        metaIn.close();

        // 2. Buscar el userId en el array de clases
        auto& clases = metadata["classes"];
        auto it = std::find(clases.begin(), clases.end(), userId);
        
        if (it == clases.end()) {
            std::cerr << "! Error: Usuario " << userId << " no encontrado en metadata" << std::endl;
            return false;
        }

        // 3. Eliminar el archivo class_X.bin
        std::string classFile = modelPath + "class_" + std::to_string(userId) + ".bin";
        if (fs::exists(classFile)) {
            fs::remove(classFile);
            std::cout << "   & Eliminado: " << classFile << std::endl;
        } else {
            std::cerr << "! Advertencia: Archivo no encontrado: " << classFile << std::endl;
        }

        // 4. Eliminar userId del array de clases
        clases.erase(it);
        
        // 5. Actualizar num_classes
        metadata["num_classes"] = clases.size();

        // 6. Guardar metadata actualizado
        std::ofstream metaOut(metadataPath);
        if (!metaOut.is_open()) {
            std::cerr << "! Error: No se pudo escribir metadata.json" << std::endl;
            return false;
        }

        metaOut << metadata.dump(4);  // Pretty print
        metaOut.close();

        std::cout << "   & Usuario " << userId << " eliminado exitosamente" << std::endl;
        std::cout << "   & Clases restantes: " << clases.size() << std::endl;

        // 7. Recargar datos en memoria desde metadata.json
        recargarDatos("");  // El path ya no importa, ahora lee de metadata.json

        return true;

    } catch (const std::exception& e) {
        std::cerr << "! Error eliminando usuario: " << e.what() << std::endl;
        return false;
    }
}