"""Emision de tokens de sesion y siembra de usuarios de demostracion.

El contrato de identidad de SAMUAPP es Firebase Authentication (§4.10): el
token lo emite Identity Platform y el servicio lo verifica en cada peticion.
Para poder probar la SPA sin un proyecto de Firebase, el modo
``SAMUAPP_AUTH_MODE=dev`` emite tokens con la forma ``dev:<rol>:<uid>``, que
es exactamente lo que entiende `VerificadorDeDesarrollo`.

Este módulo **solo** produce esos tokens de desarrollo. No contiene
verificacion de correo, recuperacion de contrasena ni revocacion: ninguna de
esas capacidades existe fuera de Firebase Authentication.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from core.contracts import Rol

__all__ = [
    "PREFIJO_TOKEN_DEV",
    "USUARIOS_DEMO",
    "UsuarioDemo",
    "analizar_token_dev",
    "crear_token_dev",
    "semilla_usuarios_demo",
]

logger = logging.getLogger(__name__)

#: Prefijo de los tokens emitidos por el modo de desarrollo.
PREFIJO_TOKEN_DEV = "dev"

#: Contrasena de los usuarios de demostracion. Es una credencial publica de
#: un entorno de pruebas: solo existe mientras `SAMUAPP_AUTH_MODE=dev`, y el
#: servicio se niega a emitir tokens de desarrollo en cualquier otro modo.
CONTRASENA_DEMO = "Samuapp2026!"


@dataclass(frozen=True)
class UsuarioDemo:
    """Usuario de demostracion sembrado en el almacen."""

    nombre: str
    correo: str
    rol: Rol


#: Usuarios de demostracion que se siembran la primera vez que arranca el
#: servicio en modo de desarrollo.
USUARIOS_DEMO: tuple[UsuarioDemo, ...] = (
    UsuarioDemo("Operador de demostracion", "operador@samuapp.cl", Rol.OPERADOR),
    UsuarioDemo("Supervisor de demostracion", "supervisor@samuapp.cl", Rol.SUPERVISOR),
    UsuarioDemo("Administrador de demostracion", "admin@samuapp.cl", Rol.ADMIN),
)


def crear_token_dev(uid: str, rol: Rol) -> str:
    """Emite un token de desarrollo con la forma `dev:<rol>:<uid>`."""
    return f"{PREFIJO_TOKEN_DEV}:{rol.value}:{uid}"


def analizar_token_dev(token: str) -> tuple[Rol, str] | None:
    """Extrae el rol y el `uid` de un token de desarrollo.

    Devuelve `None` si el token no tiene la forma esperada.
    """
    if not token:
        return None
    partes = token.split(":")
    if len(partes) != 3 or partes[0] != PREFIJO_TOKEN_DEV:
        return None
    try:
        rol = Rol(partes[1].strip().lower())
    except ValueError:
        return None
    uid = partes[2].strip()
    if not uid:
        return None
    return rol, uid


def calcular_expiracion(minutos: int) -> int:
    """Marca de expiracion del token, en milisegundos epoch."""
    minutos = max(1, int(minutos))
    instante = datetime.now(UTC) + timedelta(minutes=minutos)
    return int(instante.timestamp() * 1000)


def semilla_usuarios_demo(almacen: object, contrasena: str = CONTRASENA_DEMO) -> int:
    """Siembra los usuarios de demostracion si el almacen esta vacio.

    Devuelve el numero de usuarios creados. Es idempotente: si ya hay usuarios
    registrados, no se siembra nada.
    """
    contar = getattr(almacen, "contar", None)
    if contar is None:
        return 0
    if int(contar()) > 0:
        return 0

    creados = 0
    for usuario in USUARIOS_DEMO:
        try:
            almacen.crear(
                usuario.nombre,
                usuario.correo,
                contrasena,
                usuario.rol.value,
            )
        except Exception as exc:  # noqa: BLE001 - la siembra nunca detiene el arranque
            logger.warning(
                "No se pudo sembrar el usuario de demostracion",
                extra={"evento": "semilla_usuarios", "detalle": str(exc)},
            )
            continue
        creados += 1

    logger.info(
        "Usuarios de demostracion sembrados",
        extra={"evento": "semilla_usuarios", "creados": creados},
    )
    return creados
