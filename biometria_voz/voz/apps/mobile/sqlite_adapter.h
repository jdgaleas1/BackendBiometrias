#ifndef SQLITE_ADAPTER_H
#define SQLITE_ADAPTER_H

#include <string>
#include <vector>
#include <optional>

#include "../../external/sqlite3.h"
#include "../../external/json.hpp"

using json = nlohmann::json;

// ============================================================================
// Estructuras de Datos
// ============================================================================

struct Usuario {
    int id_usuario;
    std::string identificador_unico;
    std::string estado;
    std::string fecha_registro;
};

struct CredencialBiometrica {
    int id_credencial;
    int id_usuario;
    std::string tipo_biometria;
    std::string estado;
    std::string fecha_registro;
};

struct FraseDinamica {
    int id_frase;
    std::string frase;
    std::string categoria;
    bool activa;
};

struct ValidacionBiometrica {
    int id_validacion;
    int id_credencial;
    std::string resultado;
    double confianza;
    std::string fecha_validacion;
};

struct CaracteristicaHablante {
    int id_caracteristica;
    int id_usuario;
    int id_credencial;
    std::vector<double> vector_features;
    int dimension;
    std::string origen;
    std::string uuid_dispositivo;
    std::string fecha_captura;
    int sincronizado;
};

// ============================================================================
// Adaptador SQLite para App Movil
// ============================================================================

class SQLiteAdapter {
private:
    sqlite3* db;
    std::string dbPath;
    bool conectado;

    void verificarConexion();
    void registrarEnColaSincronizacion(const std::string& tabla, 
                                       const std::string& accion,
                                       const json& datos);

public:
    SQLiteAdapter(const std::string& path);
    ~SQLiteAdapter();

    // Inicializacion
    bool inicializarEsquema();
    bool conectar();
    void desconectar();
    bool estaConectado() const { return conectado; }

    // ========================================================================
    // USUARIOS
    // ========================================================================
    std::optional<Usuario> obtenerUsuarioPorIdentificador(const std::string& identificador);
    std::optional<Usuario> obtenerUsuarioPorId(int idUsuario);
    int insertarUsuario(const std::string& identificador, const std::string& estado = "activo");
    bool actualizarEstadoUsuario(int idUsuario, const std::string& estado);
    std::vector<Usuario> listarUsuarios();

    // ========================================================================
    // CREDENCIALES BIOMETRICAS
    // ========================================================================
    std::optional<CredencialBiometrica> obtenerCredencialPorUsuario(
        int idUsuario, 
        const std::string& tipoBiometria);
    
    int insertarCredencial(int idUsuario, const std::string& tipoBiometria);
    bool actualizarEstadoCredencial(int idCredencial, const std::string& estado);
    std::vector<CredencialBiometrica> listarCredencialesPorUsuario(int idUsuario);

    // ========================================================================
    // FRASES DINAMICAS
    // ========================================================================
    std::vector<FraseDinamica> obtenerFrasesActivas();
    std::optional<FraseDinamica> obtenerFrasePorId(int idFrase);
    int insertarFrase(const std::string& frase, const std::string& categoria = "general");
    bool desactivarFrase(int idFrase);

    // ========================================================================
    // VALIDACIONES BIOMETRICAS
    // ========================================================================
    int insertarValidacion(int idCredencial, const std::string& resultado, 
                          double confianza);
    std::vector<ValidacionBiometrica> listarValidacionesPorCredencial(int idCredencial);

    // ========================================================================
    // SINCRONIZACION
    // ========================================================================
    json obtenerColaSincronizacion();
    bool marcarComoSincronizado(int idSync);
    int contarPendientesSincronizacion();

    // ========================================================================
    // CONFIG SYNC
    // ========================================================================
    void guardarConfigSync(const std::string& clave, const std::string& valor);
    std::string obtenerConfigSync(const std::string& clave);

    // ========================================================================
    // CARACTERISTICAS HABLANTES
    // ========================================================================
    int insertarCaracteristicaLocal(int idUsuario, int idCredencial, 
                                    const std::vector<double>& features,
                                    const std::string& uuidDispositivo = "");
    std::vector<CaracteristicaHablante> obtenerCaracteristicasPendientes();
    bool marcarCaracteristicaSincronizada(int idCaracteristica);
    std::vector<CaracteristicaHablante> obtenerCaracteristicasPorUsuario(int idUsuario);

    // ========================================================================
    // UTILIDADES
    // ========================================================================
    json ejecutarConsultaJSON(const std::string& sql);
    bool ejecutarComando(const std::string& sql);
    std::string obtenerUltimoError() const;
};

#endif // SQLITE_ADAPTER_H
