"""Almacén de usuarios del modo de desarrollo.

En producción la identidad vive en Firebase Authentication (§4.10 del
documento de Etapa 3): este almacén existe solo para el modo
``SAMUAPP_AUTH_MODE=dev``, en el que la SPA necesita un punto de entrada para
emitir tokens de prueba sin depender de un proyecto de Firebase configurado.

Las contraseñas nunca se guardan en claro: se almacenan como
PBKDF2-HMAC-SHA256 con sal aleatoria por usuario. Aun así, el almacén es un
sustituto de desarrollo: no implementa verificación de correo, recuperación de
contraseña ni revocación, y no debe usarse en producción.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

__all__ = ["UsuarioRegistrado", "UsuariosStore", "hash_contrasena"]

logger = logging.getLogger(__name__)

#: Iteraciones de PBKDF2-HMAC-SHA256. Costo suficiente para un almacen de
#: desarrollo sin volver lento el arranque del servicio.
ITERACIONES_PBKDF2 = 100_000

#: Longitud de la sal aleatoria, en bytes.
LONGITUD_SAL = 16

#: Prefijo de los identificadores de usuario generados por el servicio.
PREFIJO_UID = "op_"

ESQUEMA_USUARIOS = """
CREATE TABLE IF NOT EXISTS usuarios (
    uid              TEXT PRIMARY KEY,
    nombre           TEXT NOT NULL,
    correo           TEXT NOT NULL UNIQUE,
    hash_contrasena  TEXT NOT NULL,
    salt             TEXT NOT NULL,
    rol              TEXT NOT NULL,
    creado_en        TEXT NOT NULL,
    verificado       INTEGER NOT NULL DEFAULT 1
)
"""


def hash_contrasena(contrasena: str, salt: str) -> str:
    """Deriva el hash de una contraseña con PBKDF2-HMAC-SHA256."""
    derivada = hashlib.pbkdf2_hmac(
        "sha256",
        contrasena.encode("utf-8"),
        bytes.fromhex(salt),
        ITERACIONES_PBKDF2,
    )
    return derivada.hex()


@dataclass(frozen=True)
class UsuarioRegistrado:
    """Usuario almacenado, con su rol y su estado de verificación."""

    uid: str
    nombre: str
    correo: str
    rol: str
    creado_en: str
    verificado: bool


class UsuariosStore:
    """Repositorio de usuarios del modo de desarrollo."""

    def __init__(self, ruta: Path | str) -> None:
        self._ruta = Path(ruta)

    @property
    def ruta(self) -> Path:
        """Ruta del archivo SQLite."""
        return self._ruta

    def _conectar(self) -> sqlite3.Connection:
        """Abre una conexión con las pragmas de trabajo del servicio."""
        self._ruta.parent.mkdir(parents=True, exist_ok=True)
        conexion = sqlite3.connect(self._ruta, timeout=30.0)
        conexion.row_factory = sqlite3.Row
        conexion.execute("PRAGMA journal_mode=WAL")
        conexion.execute("PRAGMA foreign_keys=ON")
        return conexion

    def crear_tablas(self) -> None:
        """Crea la tabla de usuarios. Es idempotente."""
        with self._conectar() as conexion:
            conexion.execute(ESQUEMA_USUARIOS)
        logger.info(
            "Repositorio de usuarios listo",
            extra={"evento": "usuarios_store_inicializado", "ruta": str(self._ruta)},
        )

    # --- lectura ------------------------------------------------------------

    def buscar_por_correo(self, correo: str) -> dict[str, object] | None:
        """Devuelve la fila del usuario con ese correo, o `None`."""
        with self._conectar() as conexion:
            fila = conexion.execute(
                "SELECT * FROM usuarios WHERE correo = ?", (correo,)
            ).fetchone()
        return dict(fila) if fila is not None else None

    def buscar_por_uid(self, uid: str) -> dict[str, object] | None:
        """Devuelve la fila del usuario con ese `uid`, o `None`."""
        with self._conectar() as conexion:
            fila = conexion.execute(
                "SELECT * FROM usuarios WHERE uid = ?", (uid,)
            ).fetchone()
        return dict(fila) if fila is not None else None

    def contar(self) -> int:
        """Número de usuarios registrados."""
        with self._conectar() as conexion:
            total = conexion.execute("SELECT COUNT(*) FROM usuarios").fetchone()[0]
        return int(total)

    # --- escritura ----------------------------------------------------------

    def crear(
        self,
        nombre: str,
        correo: str,
        contrasena: str,
        rol: str,
        verificado: bool = True,
    ) -> str:
        """Registra un usuario nuevo y devuelve su `uid`."""
        uid = f"{PREFIJO_UID}{secrets.token_hex(8)}"
        salt = secrets.token_hex(LONGITUD_SAL)
        ahora = datetime.now(UTC).isoformat(timespec="milliseconds")
        with self._conectar() as conexion:
            conexion.execute(
                """
                INSERT INTO usuarios (
                    uid, nombre, correo, hash_contrasena, salt, rol, creado_en, verificado
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uid,
                    nombre,
                    correo,
                    hash_contrasena(contrasena, salt),
                    salt,
                    rol,
                    ahora,
                    1 if verificado else 0,
                ),
            )
        return uid

    def verificar_contrasena(self, uid: str, contrasena: str) -> bool:
        """Comprueba la contraseña de un usuario en tiempo constante."""
        fila = self.buscar_por_uid(uid)
        if fila is None:
            return False
        esperado = str(fila["hash_contrasena"])
        calculado = hash_contrasena(contrasena, str(fila["salt"]))
        return hmac.compare_digest(esperado, calculado)
