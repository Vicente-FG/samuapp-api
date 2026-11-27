"""Verificación de tokens y autorizacion por rol.

La verificación esta **desacoplada del proveedor** mediante la interfaz
`VerificadorToken`, de modo que el nucleo de dominio no dependa de Firebase:

* `VerificadorFirebase` valida el ID token de Firebase. La integracion real
  usa `firebase-admin` con import perezoso: si la dependencia opcional no esta
  instalada, el servicio arranca igual y las rutas protegidas responden 503
  con un mensaje claro (`VerificadorNoDisponible`).
* `VerificadorDeDesarrollo` acepta un token con forma `dev:<rol>:<uid>`. Solo
  se habilita con `SAMUAPP_AUTH_MODE=dev`, que **nunca** es el valor por
  defecto.

El `uid` se toma siempre del token, nunca del cuerpo de la peticion: es la
regla de seguridad que impide suplantar a otro operador.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from core.contracts import NIVELES_ROL, Rol

__all__ = [
    "ROL_NIVELES",
    "Rol",
    "TokenInvalido",
    "Usuario",
    "VerificadorDeDesarrollo",
    "VerificadorFirebase",
    "VerificadorNoDisponible",
    "VerificadorToken",
    "crear_verificador",
    "enmascarar_token",
    "nivel_de_rol",
]

logger = logging.getLogger(__name__)

#: Nivel jerarquico de cada rol: el supervisor hereda del operador y el admin
#: hereda del supervisor. Se define en el nucleo de dominio (`core.contracts`)
#: porque el modelo de roles es un concepto de negocio, no de la plataforma.
ROL_NIVELES: dict[Rol, int] = NIVELES_ROL


def nivel_de_rol(rol: Rol) -> int:
    """Nivel jerarquico de un rol."""
    return ROL_NIVELES[rol]


@dataclass(frozen=True)
class Usuario:
    """Identidad autenticada, derivada exclusivamente del token."""

    uid: str
    rol: Rol
    email: str | None = None


class TokenInvalido(Exception):
    """El token es ausente, vencido o invalido."""


class VerificadorNoDisponible(Exception):
    """El verificador no puede operar (dependencia opcional ausente o sin config)."""


class VerificadorToken(ABC):
    """Interfaz de verificación de tokens, independiente del proveedor."""

    @abstractmethod
    def verificar(self, token: str) -> Usuario:
        """Valida el token y devuelve la identidad. Lanza `TokenInvalido`."""


def enmascarar_token(token: str | None) -> str:
    """Enmascara un token para poder escribirlo en el log.

    Nunca se registra el token completo: solo se conservan los últimos cuatro
    caracteres, que permiten correlacionar sin exponer la credencial.
    """
    if not token:
        return "<ausente>"
    if len(token) <= 4:
        return "*" * len(token)
    return f"{'*' * 8}{token[-4:]}"


class VerificadorDeDesarrollo(VerificadorToken):
    """Verificador para pruebas locales: token con forma `dev:<rol>:<uid>`."""

    PREFIJO = "dev"

    def verificar(self, token: str) -> Usuario:
        """Valida el token de desarrollo y devuelve la identidad."""
        partes = token.split(":")
        if len(partes) != 3 or partes[0] != self.PREFIJO:
            raise TokenInvalido(
                "El token de desarrollo debe tener la forma 'dev:<rol>:<uid>'."
            )
        _, rol_texto, uid = partes
        try:
            rol = Rol(rol_texto.strip().lower())
        except ValueError as exc:
            raise TokenInvalido(f"Rol desconocido en el token: {rol_texto!r}") from exc
        if not uid.strip():
            raise TokenInvalido("El token de desarrollo no trae uid.")
        return Usuario(uid=uid.strip(), rol=rol)


class VerificadorFirebase(VerificadorToken):
    """Verificador del ID token de Firebase Authentication.

    La dependencia `firebase-admin` es opcional: se importa de forma perezosa
    para que el servicio arranque (y las pruebas corran) sin ella.
    """

    def __init__(
        self,
        project_id: str | None = None,
        check_revocado: bool = True,
    ) -> None:
        self._project_id = project_id
        self._check_revocado = check_revocado
        self._app: object | None = None

    def _obtener_app(self) -> object:
        """Inicializa `firebase-admin` de forma perezosa."""
        if self._app is not None:
            return self._app
        try:
            import firebase_admin
            from firebase_admin import credentials
        except ImportError as exc:
            raise VerificadorNoDisponible(
                "La dependencia opcional 'firebase-admin' no esta instalada. "
                "Instale firebase-admin para usar SAMUAPP_AUTH_MODE=firebase, "
                "o configure SAMUAPP_AUTH_MODE=dev para pruebas locales."
            ) from exc

        try:
            try:
                self._app = firebase_admin.get_app()
            except ValueError:
                credenciales = credentials.ApplicationDefault()
                opciones = {"projectId": self._project_id} if self._project_id else None
                self._app = firebase_admin.initialize_app(credenciales, opciones)
        except Exception as exc:  # noqa: BLE001 - se reempaqueta como error de dominio
            raise VerificadorNoDisponible(
                f"No se pudo inicializar firebase-admin: {exc}"
            ) from exc
        return self._app

    def verificar(self, token: str) -> Usuario:
        """Valida el ID token de Firebase y devuelve la identidad del usuario."""
        app = self._obtener_app()
        import firebase_admin

        try:
            decodificado = firebase_admin.auth.verify_id_token(
                token, app=app, check_revoked=self._check_revocado
            )
        except VerificadorNoDisponible:
            raise
        except Exception as exc:  # noqa: BLE001 - cualquier fallo es token invalido
            raise TokenInvalido(f"Token de Firebase invalido: {exc}") from exc

        uid = str(decodificado.get("uid") or "").strip()
        if not uid:
            raise TokenInvalido("El token de Firebase no trae 'uid'.")

        rol_texto = str(
            decodificado.get("rol") or decodificado.get("role") or Rol.OPERADOR.value
        ).strip().lower()
        try:
            rol = Rol(rol_texto)
        except ValueError:
            logger.warning(
                "El token no declara un rol conocido; se asume 'operador'",
                extra={"evento": "rol_desconocido", "rol": rol_texto},
            )
            rol = Rol.OPERADOR

        email = decodificado.get("email")
        return Usuario(uid=uid, rol=rol, email=str(email) if email else None)


def crear_verificador(
    auth_mode: str,
    project_id: str | None = None,
    check_revocado: bool = True,
) -> VerificadorToken:
    """Construye el verificador según `SAMUAPP_AUTH_MODE`."""
    if auth_mode == "dev":
        logger.warning(
            "SAMUAPP_AUTH_MODE=dev activo: la verificacion de tokens es de "
            "desarrollo y no debe usarse en produccion",
            extra={"evento": "auth_modo_desarrollo"},
        )
        return VerificadorDeDesarrollo()
    return VerificadorFirebase(project_id=project_id, check_revocado=check_revocado)
