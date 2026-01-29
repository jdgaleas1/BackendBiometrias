#ifndef LISTAR_SERVICE_H
#define LISTAR_SERVICE_H

#include <string>
#include <vector>
#include <map>
#include <set>

struct Usuario {
    int id;
    std::string nombre;
};

struct ResultadoListado {
    bool exito;
    std::vector<Usuario> usuarios;
    int total;
    std::string error;
};

class ListarService {
private:
    std::map<int, std::string> mapeoUsuarios;
    std::set<int> usuariosSuspendidos;

public:
    ListarService(const std::string& mappingPath);

    ResultadoListado listarUsuarios();
    bool eliminarUsuario(int userId);
    void recargarDatos(const std::string& mappingPath);

private:
    void cargarDatos(const std::string& mappingPath);
};

#endif