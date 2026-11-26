"""Interfaz de acceso versionado a los artefactos del modelo.

El `Model Registry` (C-11 del documento de Etapa 3) abstrae el lugar fisico
donde viven los artefactos. La implementacion incluida en este repositorio es
local: lee el directorio configurado por `SAMUAPP_MODELOS_DIR`.

El registry mantiene el bundle en memoria: los modelos se cargan **una sola
vez** y nunca se recargan por peticion.
"""

from __future__ import annotations

import abc
import logging
from pathlib import Path

from core.models.bundle import ErrorDeArtefactos, ModelBundle

__all__ = ["ModelRegistry", "ModelRegistryLocal"]

logger = logging.getLogger(__name__)


class ModelRegistry(abc.ABC):
    """Contrato de acceso a la versión vigente de los artefactos."""

    @abc.abstractmethod
    def leer_actual(self) -> ModelBundle:
        """Devuelve el bundle vigente. Lanza `ErrorDeArtefactos` si no hay."""

    @abc.abstractmethod
    def version_actual(self) -> str | None:
        """Versión del modelo vigente, o `None` si todavia no se pudo cargar."""

    @abc.abstractmethod
    def listar_versiones(self) -> list[str]:
        """Versiones publicadas disponibles en el registro."""


class ModelRegistryLocal(ModelRegistry):
    """Registro local en disco.

    Guarda en memoria el bundle ya cargado y también el error de carga: si el
    directorio no tiene artefactos validos, no se reintenta en cada peticion.
    """

    def __init__(self, directorio: Path | str) -> None:
        self._directorio = Path(directorio)
        self._bundle: ModelBundle | None = None
        self._error: ErrorDeArtefactos | None = None

    @property
    def directorio(self) -> Path:
        """Directorio desde el que se leen los artefactos."""
        return self._directorio

    def leer_actual(self) -> ModelBundle:
        """Devuelve el bundle vigente, cargandolo la primera vez."""
        if self._error is not None:
            raise self._error
        if self._bundle is None:
            try:
                self._bundle = ModelBundle.cargar(self._directorio)
            except ErrorDeArtefactos as exc:
                self._error = exc
                logger.error(
                    "No se pudieron cargar los artefactos del modelo",
                    extra={
                        "evento": "model_bundle_error",
                        "directorio": str(self._directorio),
                        "detalle": str(exc),
                    },
                )
                raise
        return self._bundle

    def version_actual(self) -> str | None:
        """Versión del modelo vigente, o `None` si no se pudo cargar."""
        try:
            return self.leer_actual().model_version
        except ErrorDeArtefactos:
            return None

    def listar_versiones(self) -> list[str]:
        """Versiones publicadas. En el registro local es la versión vigente."""
        version = self.version_actual()
        return [version] if version else []

    def reiniciar(self) -> None:
        """Olvida el bundle cacheado para forzar una recarga en el proximo acceso."""
        self._bundle = None
        self._error = None
