"""Carga, verificación de integridad y versionado de los artefactos del modelo.

Los artefactos publicados viven en el directorio configurado por
`SAMUAPP_MODELOS_DIR` (por defecto `./modelos/`):

* `lgb_es_fatal_1_26M.joblib` -> clasificador de `pFatal`.
* `lgb_victimas_graves_1_26M.joblib` -> regresor de `victimasGravesPred`.
* `features_xgb.joblib` -> orden real de las columnas de características.
* `encodings.parquet` -> tabla congelada de target encoding.
* `tasa_global.json` -> tasa global del dataset de entrenamiento.
* `MANIFEST.json` -> SHA-256 y tamano de cada artefacto.

El bundle se carga **una sola vez** en el arranque y se mantiene en memoria:
nunca se recarga por peticion.

Contrato de integridad: al cargar se recalcula el SHA-256 de cada artefacto
declarado en el MANIFEST y, ante discrepancia, se falla de forma explícita
(`failed-precondition`, HTTP 412) en lugar de servir predicciones silenciosamente
corruptas. Si el MANIFEST no existe se permite la carga, pero se registra un
warning explícito de "integridad no verificada".
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from core.features.encodings import (
    MetadatosTasaGlobal,
    cargar_desde_parquet,
    cargar_metadatos_tasa_global,
)
from core.features.pipeline import (
    ORDEN_POR_DEFECTO,
    FeatureTransformer,
    ValorCaracteristica,
    valor_numerico,
)

__all__ = [
    "ARCHIVO_CLASIFICADOR",
    "ARCHIVO_ENCODINGS",
    "ARCHIVO_MANIFIESTO",
    "ARCHIVO_ORDEN_COLUMNAS",
    "ARCHIVO_REGRESOR",
    "ARCHIVO_TASA_GLOBAL",
    "ArtefactoCargado",
    "ArtefactosNoDisponibles",
    "ErrorDeArtefactos",
    "IntegridadDeArtefactos",
    "ModelBundle",
]

logger = logging.getLogger(__name__)

#: Nombres exactos de los artefactos esperados.
ARCHIVO_CLASIFICADOR = "lgb_es_fatal_1_26M.joblib"
ARCHIVO_REGRESOR = "lgb_victimas_graves_1_26M.joblib"
ARCHIVO_ORDEN_COLUMNAS = "features_xgb.joblib"
ARCHIVO_ENCODINGS = "encodings.parquet"
ARCHIVO_TASA_GLOBAL = "tasa_global.json"
ARCHIVO_MANIFIESTO = "MANIFEST.json"

#: Tamano de bloque de lectura para el calculo del SHA-256.
TAMANO_BLOQUE = 1024 * 1024


class ErrorDeArtefactos(Exception):
    """Error base de carga de artefactos. Se traduce a `failed-precondition`."""


class ArtefactosNoDisponibles(ErrorDeArtefactos):
    """Falta un artefacto obligatorio o no se puede cargar."""


class IntegridadDeArtefactos(ErrorDeArtefactos):
    """El SHA-256 recalculado no coincide con el declarado en el MANIFEST."""


@dataclass(frozen=True)
class ArtefactoCargado:
    """Estado de un artefacto publicado y su verificación de integridad."""

    nombre: str
    sha256: str
    bytes: int
    verificado: bool


@dataclass(frozen=True)
class ModelBundle:
    """Conjunto versionado de modelos y tablas de referencia, listo para inferir."""

    clasificador: Any
    regresor: Any
    transformer: FeatureTransformer
    model_version: str
    artifact_version: str
    artefactos: tuple[ArtefactoCargado, ...]
    directorio: Path
    integridad_verificada: bool

    @classmethod
    def cargar(cls, directorio: Path | str) -> ModelBundle:
        """Carga los artefactos desde disco y verifica su integridad."""
        ruta = Path(directorio)
        if not ruta.is_dir():
            raise ArtefactosNoDisponibles(
                f"No existe el directorio de artefactos: {ruta}"
            )

        manifiesto = cls._leer_manifiesto(ruta)
        artefactos = cls._verificar_integridad(ruta, manifiesto)

        clasificador = cls._cargar_modelo(ruta / ARCHIVO_CLASIFICADOR, "clasificador")
        regresor = cls._cargar_modelo(ruta / ARCHIVO_REGRESOR, "regresor")
        orden = cls._cargar_orden_columnas(ruta / ARCHIVO_ORDEN_COLUMNAS)

        ruta_encodings = ruta / ARCHIVO_ENCODINGS
        ruta_tasa = ruta / ARCHIVO_TASA_GLOBAL
        if not ruta_encodings.is_file():
            raise ArtefactosNoDisponibles(
                f"Falta el artefacto de target encoding: {ruta_encodings}"
            )
        if not ruta_tasa.is_file():
            raise ArtefactosNoDisponibles(
                f"Falta el artefacto de tasa global: {ruta_tasa}"
            )

        metadatos = cargar_metadatos_tasa_global(ruta_tasa)
        referencias = cargar_desde_parquet(ruta_encodings, metadatos.tasa_global)

        model_version = cls._model_version(manifiesto, ruta / ARCHIVO_CLASIFICADOR)
        artifact_version = cls._artifact_version(manifiesto, metadatos)

        bundle = cls(
            clasificador=clasificador,
            regresor=regresor,
            transformer=FeatureTransformer(referencias=referencias, orden_columnas=orden),
            model_version=model_version,
            artifact_version=artifact_version,
            artefactos=artefactos,
            directorio=ruta,
            integridad_verificada=manifiesto is not None,
        )
        logger.info(
            "Artefactos del modelo cargados",
            extra={
                "evento": "model_bundle_cargado",
                "model_version": model_version,
                "artifact_version": artifact_version,
                "integridad_verificada": bundle.integridad_verificada,
            },
        )
        return bundle

    # --- carga de piezas ----------------------------------------------------

    @staticmethod
    def _leer_manifiesto(ruta: Path) -> dict[str, Any] | None:
        """Lee el MANIFEST.json; `None` si no existe (integridad no verificada)."""
        ruta_manifiesto = ruta / ARCHIVO_MANIFIESTO
        if not ruta_manifiesto.is_file():
            logger.warning(
                "No existe MANIFEST.json: integridad no verificada",
                extra={
                    "evento": "manifiesto_ausente",
                    "directorio": str(ruta),
                },
            )
            return None
        try:
            contenido = json.loads(ruta_manifiesto.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ArtefactosNoDisponibles(
                f"El MANIFEST.json no es JSON valido: {ruta_manifiesto}"
            ) from exc
        if not isinstance(contenido, dict):
            raise ArtefactosNoDisponibles(
                f"El MANIFEST.json debe ser un objeto JSON: {ruta_manifiesto}"
            )
        return contenido

    @staticmethod
    def _calcular_sha256(ruta: Path) -> str:
        """SHA-256 de un archivo, leído por bloques."""
        digest = hashlib.sha256()
        with ruta.open("rb") as archivo:
            for bloque in iter(lambda: archivo.read(TAMANO_BLOQUE), b""):
                digest.update(bloque)
        return digest.hexdigest()

    @classmethod
    def _verificar_integridad(
        cls, ruta: Path, manifiesto: dict[str, Any] | None
    ) -> tuple[ArtefactoCargado, ...]:
        """Recalcula el SHA-256 de cada artefacto declarado en el MANIFEST."""
        if manifiesto is None:
            return ()
        declarados = manifiesto.get("artefactos")
        if not isinstance(declarados, dict):
            raise ArtefactosNoDisponibles(
                "El MANIFEST.json debe traer el objeto 'artefactos'."
            )

        artefactos: list[ArtefactoCargado] = []
        for nombre, ficha in declarados.items():
            ruta_artefacto = ruta / str(nombre)
            if not ruta_artefacto.is_file():
                raise ArtefactosNoDisponibles(
                    f"El MANIFEST declara el artefacto {nombre!r} pero no existe en {ruta}."
                )
            sha256 = cls._calcular_sha256(ruta_artefacto)
            esperado = str(ficha.get("sha256", ""))
            if esperado and sha256 != esperado:
                raise IntegridadDeArtefactos(
                    f"Checksum SHA-256 no coincide para {nombre!r}: "
                    f"esperado {esperado}, calculado {sha256}."
                )
            tamano = ruta_artefacto.stat().st_size
            esperado_bytes = ficha.get("bytes")
            if esperado_bytes is not None and int(esperado_bytes) != tamano:
                raise IntegridadDeArtefactos(
                    f"El tamano de {nombre!r} no coincide: "
                    f"esperado {esperado_bytes}, calculado {tamano}."
                )
            artefactos.append(
                ArtefactoCargado(
                    nombre=str(nombre),
                    sha256=sha256,
                    bytes=tamano,
                    verificado=bool(esperado),
                )
            )
        return tuple(artefactos)

    @staticmethod
    def _cargar_modelo(ruta: Path, descripcion: str) -> Any:
        """Carga un modelo `.joblib` verificando que exista."""
        if not ruta.is_file():
            raise ArtefactosNoDisponibles(
                f"Falta el artefacto del {descripcion}: {ruta}"
            )
        try:
            return joblib.load(ruta)
        except Exception as exc:  # noqa: BLE001 - se reempaqueta como error de dominio
            raise ArtefactosNoDisponibles(
                f"No se pudo cargar el artefacto del {descripcion}: {ruta} ({exc})"
            ) from exc

    @staticmethod
    def _cargar_orden_columnas(ruta: Path) -> tuple[str, ...]:
        """Carga el orden real de columnas desde `features_xgb.joblib`."""
        if not ruta.is_file():
            logger.warning(
                "No existe features_xgb.joblib: se usa el orden por defecto",
                extra={"evento": "orden_columnas_por_defecto"},
            )
            return ORDEN_POR_DEFECTO
        try:
            contenido = joblib.load(ruta)
        except Exception as exc:  # noqa: BLE001 - se reempaqueta como error de dominio
            raise ArtefactosNoDisponibles(
                f"No se pudo cargar el orden de columnas: {ruta} ({exc})"
            ) from exc
        if isinstance(contenido, Mapping):
            contenido = contenido.get("columnas", contenido.get("features"))
        try:
            columnas = tuple(str(nombre) for nombre in contenido)
        except TypeError as exc:
            raise ArtefactosNoDisponibles(
                f"El orden de columnas de {ruta} no es una lista de nombres."
            ) from exc
        if not columnas:
            raise ArtefactosNoDisponibles(
                f"El orden de columnas de {ruta} esta vacio."
            )
        logger.info(
            "Orden de columnas del entrenamiento cargado",
            extra={
                "evento": "orden_columnas_cargado",
                "total": len(columnas),
            },
        )
        return columnas

    @staticmethod
    def _model_version(manifiesto: dict[str, Any] | None, ruta_clasificador: Path) -> str:
        """Versión del modelo: la del MANIFEST o el nombre del artefacto."""
        if manifiesto:
            version = manifiesto.get("model_version")
            if version:
                return str(version)
        return ruta_clasificador.stem

    @staticmethod
    def _artifact_version(
        manifiesto: dict[str, Any] | None, metadatos: MetadatosTasaGlobal
    ) -> str:
        """Versión de las tablas de referencia."""
        if manifiesto:
            version = manifiesto.get("artifact_version")
            if version:
                return str(version)
        if metadatos.artifact_version:
            return metadatos.artifact_version
        if metadatos.fecha:
            return f"encodings@{metadatos.fecha}"
        return "encodings@local"

    # --- inferencia ---------------------------------------------------------

    def _preparar_entrada(
        self, modelo: Any, caracteristicas: Mapping[str, ValorCaracteristica]
    ) -> Any:
        """Construye la matriz de entrada respetando el orden del modelo."""
        import numpy as np

        nombres = self.transformer.orden_columnas
        valores = [valor_numerico(nombre, caracteristicas[nombre]) for nombre in nombres]
        mapa = dict(zip(nombres, valores, strict=True))

        esperados = getattr(modelo, "feature_names_in_", None)
        if esperados is not None:
            import pandas as pd

            faltantes = [str(nombre) for nombre in esperados if str(nombre) not in mapa]
            if faltantes:
                raise ArtefactosNoDisponibles(
                    "El modelo espera caracteristicas que el pipeline no produce: "
                    f"{faltantes}"
                )
            columnas = [str(nombre) for nombre in esperados]
            return pd.DataFrame([[mapa[columna] for columna in columnas]], columns=columnas)
        return np.asarray([valores], dtype=float)

    @staticmethod
    def _probabilidad_fatalidad(entrada: Any) -> float:
        """Probabilidad de fatalidad en [0, 1] según la interfaz del modelo."""
        import numpy as np

        if hasattr(entrada, "columns"):
            probabilidades = np.asarray(entrada.iloc[0].values, dtype=float)
        else:
            probabilidades = np.asarray(entrada, dtype=float).ravel()
        if probabilidades.size == 0:
            raise ArtefactosNoDisponibles("El clasificador no devolvio probabilidades.")
        if probabilidades.size > 1:
            probabilidades = probabilidades / probabilidades.sum()
        return float(min(1.0, max(0.0, probabilidades[-1])))

    def predecir(
        self, caracteristicas: Mapping[str, ValorCaracteristica]
    ) -> tuple[float, float]:
        """Ejecuta ambos modelos y devuelve `(pFatal, victimasGravesPred)`."""
        import numpy as np

        entrada_clasificador = self._preparar_entrada(self.clasificador, caracteristicas)
        entrada_regresor = self._preparar_entrada(self.regresor, caracteristicas)

        if hasattr(self.clasificador, "predict_proba"):
            p_fatal = self._probabilidad_fatalidad(
                self.clasificador.predict_proba(entrada_clasificador)
            )
        else:
            p_fatal = self._probabilidad_fatalidad(
                self.clasificador.predict(entrada_clasificador)
            )

        predicciones = np.asarray(self.regresor.predict(entrada_regresor)).ravel()
        if predicciones.size == 0:
            raise ArtefactosNoDisponibles("El regresor no devolvio predicciones.")
        victimas_graves = max(0.0, float(predicciones[0]))
        return p_fatal, victimas_graves
