"""Configuración del servicio y logging estructurado.

Toda la configuración se lee de variables de entorno con prefijo `SAMUAPP_`:

| Variable                    | Default            | Uso                                    |
| --------------------------- | ------------------ | -------------------------------------- |
| `SAMUAPP_MODELOS_DIR`       | `./modelos/`       | Directorio de artefactos del modelo.   |
| `SAMUAPP_DB_PATH`           | `./data/predicciones.db` | Ruta de la base SQLite.           |
| `SAMUAPP_AUTH_MODE`         | `firebase`         | `firebase` o `dev` (solo pruebas).     |
| `SAMUAPP_MODO_DEGRADADO`    | `false`            | Habilita la respuesta heurística.      |
| `SAMUAPP_TIMEOUT_MS`        | `2000`             | Timeout máximo de la inferencia.       |
| `SAMUAPP_VERSION_CODIGO`    | `0.1.0`            | Versión de código persistida.          |
| `SAMUAPP_LOG_LEVEL`         | `INFO`             | Nivel de los logs JSON.                |
| `SAMUAPP_CORS_ORIGENES`     | *(vacío)*          | Origenes CORS separados por coma.      |

Ningun secreto se lee ni se registra: los tokens se enmascaran antes de
escribirlos en el log.
"""

from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "Settings",
    "configurar_logging",
    "contexto_trace_id",
    "registrar_evento",
    "trace_id_actual",
]

#: Identificador de la peticion en curso. Lo heredan todos los logs del ambito.
contexto_trace_id: ContextVar[str] = ContextVar("samuapp_trace_id", default="")

#: Campos adicionales que el formador JSON agrega al registro cuando existen.
CAMPOS_EXTRA: tuple[str, ...] = (
    "duracion_ms",
    "operador_uid",
    "prediction_id",
    "model_version",
    "artifact_version",
    "code_version",
    "http_metodo",
    "http_ruta",
    "http_status",
    "comuna",
    "region",
    "detalle",
    "rol",
)


def trace_id_actual() -> str:
    """Devuelve el identificador de traza de la peticion en curso."""
    return contexto_trace_id.get()


def registrar_evento(
    nivel: int,
    evento: str,
    mensaje: str,
    **campos: Any,
) -> None:
    """Registra un evento estructurado en el log JSON."""
    logging.log(nivel, mensaje, extra={"evento": evento, **campos})


class FormatterJson(logging.Formatter):
    """Formador que emite un objeto JSON por linea."""

    def format(self, record: logging.LogRecord) -> str:
        """Serializa el registro de log como JSON de una linea."""
        payload: dict[str, Any] = {
            "nivel": record.levelname,
            "evento": getattr(record, "evento", record.name),
            "mensaje": record.getMessage(),
            "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds"),
        }
        trace_id = contexto_trace_id.get()
        if trace_id:
            payload["trace_id"] = trace_id
        for campo in CAMPOS_EXTRA:
            valor = getattr(record, campo, None)
            if valor is not None:
                payload[campo] = valor
        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configurar_logging(nivel: str = "INFO") -> None:
    """Configura el log raiz con el formador JSON."""
    raiz = logging.getLogger()
    for handler in list(raiz.handlers):
        raiz.removeHandler(handler)
    handler = logging.StreamHandler()
    handler.setFormatter(FormatterJson())
    raiz.addHandler(handler)
    raiz.setLevel(str(nivel).upper())


class Settings(BaseSettings):
    """Configuración del backend, leida de variables de entorno."""

    model_config = SettingsConfigDict(
        env_prefix="SAMUAPP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    #: Directorio con los artefactos publicados del modelo.
    modelos_dir: Path = Path("./modelos/")
    #: Ruta del archivo SQLite con el historico de predicciones.
    db_path: Path = Path("./data/predicciones.db")
    #: Mecanismo de verificación de tokens: `firebase` o `dev`.
    auth_mode: Literal["firebase", "dev"] = "firebase"
    #: Habilita la respuesta heurística cuando no hay artefactos disponibles.
    modo_degradado: bool = False
    #: Timeout máximo de la inferencia, en milisegundos.
    timeout_ms: int = 2000
    #: Versión de código del servicio, persistida con cada prediccion.
    version_codigo: str = "0.1.0"
    #: Nivel de los logs JSON.
    log_level: str = "INFO"
    #: Origenes CORS permitidos, separados por coma. Vacío: solo mismo origen.
    cors_origenes: str = ""
    #: Project ID de Firebase (opcional: usa las credenciales por defecto).
    firebase_project_id: str | None = None
    #: Verifica la revocacion del token de Firebase en cada peticion.
    firebase_check_revocado: bool = True
    #: Número de inferencias concurrentes admitidas antes de responder 429.
    max_concurrencia: int = 32
    #: Exige un token valido para consultar el endpoint de salud.
    health_requiere_token: bool = False

    @property
    def origenes_cors(self) -> list[str]:
        """Lista de origenes CORS configurados."""
        return [origen.strip() for origen in self.cors_origenes.split(",") if origen.strip()]

    @property
    def timeout_segundos(self) -> float:
        """Timeout de la inferencia en segundos."""
        return max(0.001, self.timeout_ms / 1000.0)
