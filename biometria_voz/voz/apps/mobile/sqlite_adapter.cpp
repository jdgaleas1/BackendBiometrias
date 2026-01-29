#include "sqlite_adapter.h"
#include <iostream>
#include <sstream>

// ============================================================================
// CONSTRUCTOR Y DESTRUCTOR
// ============================================================================

SQLiteAdapter::SQLiteAdapter(const std::string& path) 
    : db(nullptr), dbPath(path), conectado(false) {
}

SQLiteAdapter::~SQLiteAdapter() {
    desconectar();
}

// ============================================================================
// CONEXION Y INICIALIZACION
// ============================================================================

bool SQLiteAdapter::conectar() {
    int rc = sqlite3_open(dbPath.c_str(), &db);
    if (rc != SQLITE_OK) {
        std::cerr << "! Error abriendo SQLite: " << sqlite3_errmsg(db) << std::endl;
        conectado = false;
        return false;
    }
    
    conectado = true;
    return inicializarEsquema();
}

void SQLiteAdapter::desconectar() {
    if (db) {
        sqlite3_close(db);
        db = nullptr;
    }
    conectado = false;
}

void SQLiteAdapter::verificarConexion() {
    if (!conectado || !db) {
        throw std::runtime_error("Base de datos SQLite no conectada");
    }
}

bool SQLiteAdapter::inicializarEsquema() {
    const char* sql = R"(
        CREATE TABLE IF NOT EXISTS usuarios (
            id_usuario INTEGER PRIMARY KEY AUTOINCREMENT,
            identificador_unico TEXT UNIQUE NOT NULL,
            estado TEXT DEFAULT 'activo',
            fecha_registro DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS credenciales_biometricas (
            id_credencial INTEGER PRIMARY KEY AUTOINCREMENT,
            id_usuario INTEGER NOT NULL,
            tipo_biometria TEXT NOT NULL,
            estado TEXT DEFAULT 'activo',
            fecha_registro DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (id_usuario) REFERENCES usuarios(id_usuario)
        );

        CREATE TABLE IF NOT EXISTS frases_dinamicas (
            id_frase INTEGER PRIMARY KEY AUTOINCREMENT,
            frase TEXT NOT NULL,
            categoria TEXT DEFAULT 'general',
            activa INTEGER DEFAULT 1,
            fecha_creacion DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS validaciones_biometricas (
            id_validacion INTEGER PRIMARY KEY AUTOINCREMENT,
            id_credencial INTEGER NOT NULL,
            resultado TEXT NOT NULL,
            confianza REAL,
            fecha_validacion DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (id_credencial) REFERENCES credenciales_biometricas(id_credencial)
        );

        CREATE TABLE IF NOT EXISTS config_sync (
            clave TEXT PRIMARY KEY,
            valor TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS caracteristicas_hablantes (
            id_caracteristica INTEGER PRIMARY KEY AUTOINCREMENT,
            id_usuario INTEGER NOT NULL,
            id_credencial INTEGER,
            vector_features BLOB NOT NULL,
            dimension INTEGER NOT NULL,
            origen TEXT DEFAULT 'mobile',
            uuid_dispositivo TEXT,
            fecha_captura DATETIME DEFAULT CURRENT_TIMESTAMP,
            sincronizado INTEGER DEFAULT 0,
            FOREIGN KEY (id_usuario) REFERENCES usuarios(id_usuario),
            FOREIGN KEY (id_credencial) REFERENCES credenciales_biometricas(id_credencial)
        );

        CREATE INDEX IF NOT EXISTS idx_usuarios_identificador 
            ON usuarios(identificador_unico);
        
        CREATE INDEX IF NOT EXISTS idx_credenciales_usuario 
            ON credenciales_biometricas(id_usuario);
        
        CREATE INDEX IF NOT EXISTS idx_caracteristicas_usuario 
            ON caracteristicas_hablantes(id_usuario);

        CREATE INDEX IF NOT EXISTS idx_caracteristicas_sincronizado 
            ON caracteristicas_hablantes(sincronizado);
    )";

    char* errMsg = nullptr;
    int rc = sqlite3_exec(db, sql, nullptr, nullptr, &errMsg);
    
    if (rc != SQLITE_OK) {
        std::cerr << "! Error inicializando esquema: " << errMsg << std::endl;
        sqlite3_free(errMsg);
        return false;
    }
    
    return true;
}

// ============================================================================
// USUARIOS
// ============================================================================

std::optional<Usuario> SQLiteAdapter::obtenerUsuarioPorIdentificador(
    const std::string& identificador) {
    
    verificarConexion();
    
    std::string sql = "SELECT id_usuario, identificador_unico, estado, "
                      "fecha_registro FROM usuarios WHERE identificador_unico = ?";
    
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        return std::nullopt;
    }
    
    sqlite3_bind_text(stmt, 1, identificador.c_str(), -1, SQLITE_TRANSIENT);
    
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        Usuario u;
        u.id_usuario = sqlite3_column_int(stmt, 0);
        u.identificador_unico = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        u.estado = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        u.fecha_registro = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        
        sqlite3_finalize(stmt);
        return u;
    }
    
    sqlite3_finalize(stmt);
    return std::nullopt;
}

std::optional<Usuario> SQLiteAdapter::obtenerUsuarioPorId(int idUsuario) {
    verificarConexion();
    
    std::string sql = "SELECT id_usuario, identificador_unico, estado, "
                      "fecha_registro FROM usuarios WHERE id_usuario = ?";
    
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        return std::nullopt;
    }
    
    sqlite3_bind_int(stmt, 1, idUsuario);
    
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        Usuario u;
        u.id_usuario = sqlite3_column_int(stmt, 0);
        u.identificador_unico = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        u.estado = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        u.fecha_registro = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        
        sqlite3_finalize(stmt);
        return u;
    }
    
    sqlite3_finalize(stmt);
    return std::nullopt;
}

int SQLiteAdapter::insertarUsuario(const std::string& identificador, 
                                   const std::string& estado) {
    verificarConexion();
    
    std::string sql = "INSERT INTO usuarios (identificador_unico, estado) VALUES (?, ?)";
    
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        return -1;
    }
    
    sqlite3_bind_text(stmt, 1, identificador.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, estado.c_str(), -1, SQLITE_TRANSIENT);
    
    if (sqlite3_step(stmt) != SQLITE_DONE) {
        sqlite3_finalize(stmt);
        return -1;
    }
    
    int idUsuario = static_cast<int>(sqlite3_last_insert_rowid(db));
    sqlite3_finalize(stmt);
    
    // Encolar para sincronizacion
    json datos;
    datos["tabla"] = "usuarios";
    datos["accion"] = "INSERT";
    datos["id_usuario"] = idUsuario;
    datos["identificador_unico"] = identificador;
    datos["estado"] = estado;
    
    registrarEnColaSincronizacion("usuarios", "INSERT", datos);
    
    return idUsuario;
}

bool SQLiteAdapter::actualizarEstadoUsuario(int idUsuario, const std::string& estado) {
    verificarConexion();
    
    std::string sql = "UPDATE usuarios SET estado = ? WHERE id_usuario = ?";
    
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        return false;
    }
    
    sqlite3_bind_text(stmt, 1, estado.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int(stmt, 2, idUsuario);
    
    bool exito = (sqlite3_step(stmt) == SQLITE_DONE);
    sqlite3_finalize(stmt);
    
    if (exito) {
        json datos;
        datos["tabla"] = "usuarios";
        datos["accion"] = "UPDATE";
        datos["id_usuario"] = idUsuario;
        datos["estado"] = estado;
        
        registrarEnColaSincronizacion("usuarios", "UPDATE", datos);
    }
    
    return exito;
}

std::vector<Usuario> SQLiteAdapter::listarUsuarios() {
    verificarConexion();
    
    std::vector<Usuario> usuarios;
    std::string sql = "SELECT id_usuario, identificador_unico, estado, "
                      "fecha_registro FROM usuarios";
    
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        return usuarios;
    }
    
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        Usuario u;
        u.id_usuario = sqlite3_column_int(stmt, 0);
        u.identificador_unico = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        u.estado = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        u.fecha_registro = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        usuarios.push_back(u);
    }
    
    sqlite3_finalize(stmt);
    return usuarios;
}

// ============================================================================
// CREDENCIALES BIOMETRICAS
// ============================================================================

std::optional<CredencialBiometrica> SQLiteAdapter::obtenerCredencialPorUsuario(
    int idUsuario, const std::string& tipoBiometria) {
    
    verificarConexion();
    
    std::string sql = "SELECT id_credencial, id_usuario, tipo_biometria, estado, "
                      "fecha_registro FROM credenciales_biometricas "
                      "WHERE id_usuario = ? AND tipo_biometria = ?";
    
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        return std::nullopt;
    }
    
    sqlite3_bind_int(stmt, 1, idUsuario);
    sqlite3_bind_text(stmt, 2, tipoBiometria.c_str(), -1, SQLITE_TRANSIENT);
    
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        CredencialBiometrica c;
        c.id_credencial = sqlite3_column_int(stmt, 0);
        c.id_usuario = sqlite3_column_int(stmt, 1);
        c.tipo_biometria = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        c.estado = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        c.fecha_registro = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
        
        sqlite3_finalize(stmt);
        return c;
    }
    
    sqlite3_finalize(stmt);
    return std::nullopt;
}

int SQLiteAdapter::insertarCredencial(int idUsuario, const std::string& tipoBiometria) {
    verificarConexion();
    
    std::string sql = "INSERT INTO credenciales_biometricas (id_usuario, tipo_biometria) "
                      "VALUES (?, ?)";
    
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        return -1;
    }
    
    sqlite3_bind_int(stmt, 1, idUsuario);
    sqlite3_bind_text(stmt, 2, tipoBiometria.c_str(), -1, SQLITE_TRANSIENT);
    
    if (sqlite3_step(stmt) != SQLITE_DONE) {
        sqlite3_finalize(stmt);
        return -1;
    }
    
    int idCredencial = static_cast<int>(sqlite3_last_insert_rowid(db));
    sqlite3_finalize(stmt);
    
    json datos;
    datos["tabla"] = "credenciales_biometricas";
    datos["accion"] = "INSERT";
    datos["id_credencial"] = idCredencial;
    datos["id_usuario"] = idUsuario;
    datos["tipo_biometria"] = tipoBiometria;
    
    registrarEnColaSincronizacion("credenciales_biometricas", "INSERT", datos);
    
    return idCredencial;
}

// ============================================================================
// FRASES DINAMICAS
// ============================================================================

std::vector<FraseDinamica> SQLiteAdapter::obtenerFrasesActivas() {
    verificarConexion();
    
    std::vector<FraseDinamica> frases;
    std::string sql = "SELECT id_frase, frase, categoria, activa "
                      "FROM frases_dinamicas WHERE activa = 1";
    
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        return frases;
    }
    
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        FraseDinamica f;
        f.id_frase = sqlite3_column_int(stmt, 0);
        f.frase = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        f.categoria = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        f.activa = sqlite3_column_int(stmt, 3) == 1;
        frases.push_back(f);
    }
    
    sqlite3_finalize(stmt);
    return frases;
}

std::optional<FraseDinamica> SQLiteAdapter::obtenerFrasePorId(int idFrase) {
    verificarConexion();
    
    std::string sql = "SELECT id_frase, frase, categoria, activa "
                      "FROM frases_dinamicas WHERE id_frase = ?";
    
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        return std::nullopt;
    }
    
    sqlite3_bind_int(stmt, 1, idFrase);
    
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        FraseDinamica f;
        f.id_frase = sqlite3_column_int(stmt, 0);
        f.frase = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        f.categoria = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        f.activa = sqlite3_column_int(stmt, 3) == 1;
        
        sqlite3_finalize(stmt);
        return f;
    }
    
    sqlite3_finalize(stmt);
    return std::nullopt;
}

int SQLiteAdapter::insertarFrase(const std::string& frase, 
                                 const std::string& categoria) {
    verificarConexion();
    
    std::string sql = "INSERT INTO frases_dinamicas (frase, categoria) VALUES (?, ?)";
    
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        return -1;
    }
    
    sqlite3_bind_text(stmt, 1, frase.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, categoria.c_str(), -1, SQLITE_TRANSIENT);
    
    if (sqlite3_step(stmt) != SQLITE_DONE) {
        sqlite3_finalize(stmt);
        return -1;
    }
    
    int idFrase = static_cast<int>(sqlite3_last_insert_rowid(db));
    sqlite3_finalize(stmt);
    
    return idFrase;
}

// ============================================================================
// VALIDACIONES BIOMETRICAS
// ============================================================================

int SQLiteAdapter::insertarValidacion(int idCredencial, const std::string& resultado,
                                      double confianza) {
    verificarConexion();
    
    std::string sql = "INSERT INTO validaciones_biometricas "
                      "(id_credencial, resultado, confianza) VALUES (?, ?, ?)";
    
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        return -1;
    }
    
    sqlite3_bind_int(stmt, 1, idCredencial);
    sqlite3_bind_text(stmt, 2, resultado.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_double(stmt, 3, confianza);
    
    if (sqlite3_step(stmt) != SQLITE_DONE) {
        sqlite3_finalize(stmt);
        return -1;
    }
    
    int idValidacion = static_cast<int>(sqlite3_last_insert_rowid(db));
    sqlite3_finalize(stmt);
    
    json datos;
    datos["tabla"] = "validaciones_biometricas";
    datos["accion"] = "INSERT";
    datos["id_validacion"] = idValidacion;
    datos["id_credencial"] = idCredencial;
    datos["resultado"] = resultado;
    datos["confianza"] = confianza;
    
    registrarEnColaSincronizacion("validaciones_biometricas", "INSERT", datos);
    
    return idValidacion;
}

// ============================================================================
// SINCRONIZACION
// ============================================================================

void SQLiteAdapter::registrarEnColaSincronizacion(const std::string& tabla,
                                                   const std::string& accion,
                                                   const json& datos) {
    std::string sql = "INSERT INTO cola_sincronizacion (tabla, accion, datos_json) "
                      "VALUES (?, ?, ?)";
    
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        return;
    }
    
    std::string datosStr = datos.dump();
    
    sqlite3_bind_text(stmt, 1, tabla.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, accion.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 3, datosStr.c_str(), -1, SQLITE_TRANSIENT);
    
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
}

json SQLiteAdapter::obtenerColaSincronizacion() {
    verificarConexion();
    
    json cola = json::array();
    std::string sql = "SELECT id_sync, tabla, accion, datos_json, fecha_creacion "
                      "FROM cola_sincronizacion WHERE sincronizado = 0 "
                      "ORDER BY fecha_creacion ASC";
    
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        return cola;
    }
    
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        json item;
        item["id_sync"] = sqlite3_column_int(stmt, 0);
        item["tabla"] = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        item["accion"] = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        
        std::string datosStr = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        item["datos"] = json::parse(datosStr);
        item["fecha_creacion"] = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 4));
        
        cola.push_back(item);
    }
    
    sqlite3_finalize(stmt);
    return cola;
}

bool SQLiteAdapter::marcarComoSincronizado(int idSync) {
    verificarConexion();
    
    std::string sql = "UPDATE cola_sincronizacion SET sincronizado = 1 WHERE id_sync = ?";
    
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        return false;
    }
    
    sqlite3_bind_int(stmt, 1, idSync);
    
    bool exito = (sqlite3_step(stmt) == SQLITE_DONE);
    sqlite3_finalize(stmt);
    
    return exito;
}

int SQLiteAdapter::contarPendientesSincronizacion() {
    verificarConexion();
    
    std::string sql = "SELECT COUNT(*) FROM cola_sincronizacion WHERE sincronizado = 0";
    
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        return 0;
    }
    
    int count = 0;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        count = sqlite3_column_int(stmt, 0);
    }
    
    sqlite3_finalize(stmt);
    return count;
}

// ============================================================================
// UTILIDADES
// ============================================================================

json SQLiteAdapter::ejecutarConsultaJSON(const std::string& sql) {
    verificarConexion();
    
    json resultado = json::array();
    sqlite3_stmt* stmt;
    
    if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) == SQLITE_OK) {
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            json fila;
            int numCols = sqlite3_column_count(stmt);
            
            for (int i = 0; i < numCols; i++) {
                std::string nombreCol = sqlite3_column_name(stmt, i);
                
                switch (sqlite3_column_type(stmt, i)) {
                    case SQLITE_INTEGER:
                        fila[nombreCol] = sqlite3_column_int(stmt, i);
                        break;
                    case SQLITE_FLOAT:
                        fila[nombreCol] = sqlite3_column_double(stmt, i);
                        break;
                    case SQLITE_TEXT:
                        fila[nombreCol] = reinterpret_cast<const char*>(
                            sqlite3_column_text(stmt, i));
                        break;
                    case SQLITE_NULL:
                        fila[nombreCol] = nullptr;
                        break;
                }
            }
            resultado.push_back(fila);
        }
        sqlite3_finalize(stmt);
    }
    
    return resultado;
}

bool SQLiteAdapter::ejecutarComando(const std::string& sql) {
    verificarConexion();
    
    char* errMsg = nullptr;
    int rc = sqlite3_exec(db, sql.c_str(), nullptr, nullptr, &errMsg);
    
    if (rc != SQLITE_OK) {
        std::cerr << "! Error SQL: " << errMsg << std::endl;
        sqlite3_free(errMsg);
        return false;
    }
    
    return true;
}

std::string SQLiteAdapter::obtenerUltimoError() const {
    if (db) {
        return sqlite3_errmsg(db);
    }
    return "Sin conexion a base de datos";
}

// ============================================================================
// CONFIG SYNC
// ============================================================================

void SQLiteAdapter::guardarConfigSync(const std::string& clave, const std::string& valor) {
    verificarConexion();
    
    std::string sql = "INSERT OR REPLACE INTO config_sync (clave, valor) VALUES (?, ?)";
    
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        throw std::runtime_error("Error preparando INSERT config_sync");
    }
    
    sqlite3_bind_text(stmt, 1, clave.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 2, valor.c_str(), -1, SQLITE_TRANSIENT);
    
    if (sqlite3_step(stmt) != SQLITE_DONE) {
        sqlite3_finalize(stmt);
        throw std::runtime_error("Error guardando configuracion sync");
    }
    
    sqlite3_finalize(stmt);
}

std::string SQLiteAdapter::obtenerConfigSync(const std::string& clave) {
    verificarConexion();
    
    std::string sql = "SELECT valor FROM config_sync WHERE clave = ?";
    
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        throw std::runtime_error("Error preparando SELECT config_sync");
    }
    
    sqlite3_bind_text(stmt, 1, clave.c_str(), -1, SQLITE_TRANSIENT);
    
    std::string valor;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        valor = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
    }
    
    sqlite3_finalize(stmt);
    return valor;
}

// ============================================================================
// CARACTERISTICAS HABLANTES
// ============================================================================

int SQLiteAdapter::insertarCaracteristicaLocal(int idUsuario, int idCredencial,
                                                const std::vector<double>& features,
                                                const std::string& uuidDispositivo) {
    verificarConexion();
    
    // Serializar vector a BLOB
    size_t blobSize = features.size() * sizeof(double);
    const uint8_t* blobData = reinterpret_cast<const uint8_t*>(features.data());
    
    std::string sql = R"(
        INSERT INTO caracteristicas_hablantes 
        (id_usuario, id_credencial, vector_features, dimension, uuid_dispositivo, sincronizado)
        VALUES (?, ?, ?, ?, ?, 0)
    )";
    
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        throw std::runtime_error("Error preparando INSERT caracteristicas_hablantes");
    }
    
    sqlite3_bind_int(stmt, 1, idUsuario);
    sqlite3_bind_int(stmt, 2, idCredencial);
    sqlite3_bind_blob(stmt, 3, blobData, blobSize, SQLITE_TRANSIENT);
    sqlite3_bind_int(stmt, 4, static_cast<int>(features.size()));
    sqlite3_bind_text(stmt, 5, uuidDispositivo.c_str(), -1, SQLITE_TRANSIENT);
    
    if (sqlite3_step(stmt) != SQLITE_DONE) {
        sqlite3_finalize(stmt);
        throw std::runtime_error("Error insertando caracteristica local");
    }
    
    int idCaracteristica = static_cast<int>(sqlite3_last_insert_rowid(db));
    sqlite3_finalize(stmt);
    
    return idCaracteristica;
}

std::vector<CaracteristicaHablante> SQLiteAdapter::obtenerCaracteristicasPendientes() {
    verificarConexion();
    
    std::vector<CaracteristicaHablante> caracteristicas;
    
    std::string sql = R"(
        SELECT id_caracteristica, id_usuario, id_credencial, 
               vector_features, dimension, origen, uuid_dispositivo, 
               fecha_captura, sincronizado
        FROM caracteristicas_hablantes
        WHERE sincronizado = 0
    )";
    
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        throw std::runtime_error("Error preparando SELECT caracteristicas pendientes");
    }
    
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        CaracteristicaHablante car;
        car.id_caracteristica = sqlite3_column_int(stmt, 0);
        car.id_usuario = sqlite3_column_int(stmt, 1);
        car.id_credencial = sqlite3_column_int(stmt, 2);
        
        // Deserializar BLOB a vector
        const void* blobData = sqlite3_column_blob(stmt, 3);
        int blobBytes = sqlite3_column_bytes(stmt, 3);
        int numElements = blobBytes / sizeof(double);
        
        const double* doubleData = reinterpret_cast<const double*>(blobData);
        car.vector_features.assign(doubleData, doubleData + numElements);
        
        car.dimension = sqlite3_column_int(stmt, 4);
        car.origen = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 5));
        car.uuid_dispositivo = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 6));
        car.fecha_captura = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 7));
        car.sincronizado = sqlite3_column_int(stmt, 8);
        
        caracteristicas.push_back(car);
    }
    
    sqlite3_finalize(stmt);
    return caracteristicas;
}

bool SQLiteAdapter::marcarCaracteristicaSincronizada(int idCaracteristica) {
    verificarConexion();
    
    std::string sql = "UPDATE caracteristicas_hablantes SET sincronizado = 1 WHERE id_caracteristica = ?";
    
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        return false;
    }
    
    sqlite3_bind_int(stmt, 1, idCaracteristica);
    
    bool exito = (sqlite3_step(stmt) == SQLITE_DONE);
    sqlite3_finalize(stmt);
    
    return exito;
}

std::vector<CaracteristicaHablante> SQLiteAdapter::obtenerCaracteristicasPorUsuario(int idUsuario) {
    verificarConexion();
    
    std::vector<CaracteristicaHablante> caracteristicas;
    
    std::string sql = R"(
        SELECT id_caracteristica, id_usuario, id_credencial, 
               vector_features, dimension, origen, uuid_dispositivo, 
               fecha_captura, sincronizado
        FROM caracteristicas_hablantes
        WHERE id_usuario = ?
    )";
    
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        throw std::runtime_error("Error preparando SELECT caracteristicas por usuario");
    }
    
    sqlite3_bind_int(stmt, 1, idUsuario);
    
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        CaracteristicaHablante car;
        car.id_caracteristica = sqlite3_column_int(stmt, 0);
        car.id_usuario = sqlite3_column_int(stmt, 1);
        car.id_credencial = sqlite3_column_int(stmt, 2);
        
        // Deserializar BLOB a vector
        const void* blobData = sqlite3_column_blob(stmt, 3);
        int blobBytes = sqlite3_column_bytes(stmt, 3);
        int numElements = blobBytes / sizeof(double);
        
        const double* doubleData = reinterpret_cast<const double*>(blobData);
        car.vector_features.assign(doubleData, doubleData + numElements);
        
        car.dimension = sqlite3_column_int(stmt, 4);
        car.origen = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 5));
        car.uuid_dispositivo = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 6));
        car.fecha_captura = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 7));
        car.sincronizado = sqlite3_column_int(stmt, 8);
        
        caracteristicas.push_back(car);
    }
    
    sqlite3_finalize(stmt);
    return caracteristicas;
}

