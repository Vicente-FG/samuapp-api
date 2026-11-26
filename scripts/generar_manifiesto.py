"""Genera el `MANIFEST.json` con el SHA-256 de cada artefacto publicado.

El manifiesto es el contrato de integridad del backend: en el arranque el
`ModelBundle` recalcula el SHA-256 de cada artefacto declarado y falla de forma
explicita (`failed-precondition`, HTTP 412) ante cualquier discrepancia, en
lugar de servir predicciones silenciosamente corruptas.

Uso::

    python -m scripts.generar_manifiesto --destino ./modelos
    python -m scripts.generar_manifiesto --destino ./modelos \
        --model-version "lgb_es_fatal_1_26M@2026-08-22" \
        --artifact-version "encodings@2026-08-22"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

__all__ = ["generar_manifiesto", "main"]

logger = logging.getLogger(__name__)

#: Nombres de los artefactos que se verifican, en orden de presentacion.
ARTEFACTOS: tuple[str, ...] = (
    "lgb_es_fatal_1_26M.joblib",
    "lgb_victimas_graves_1_26M.joblib",
    "features_xgb.joblib",
    "encodings.parquet",
    "tasa_global.json",
)

#: Tamano de bloque de lectura para el calculo del SHA-256.
TAMANO_BLOQUE = 1024 * 1024


def _sha256(ruta: Path) -> str:
    """SHA-256 de un archivo, leido por bloques."""
    digest = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(TAMANO_BLOQUE), b""):
            digest.update(bloque)
    return digest.hexdigest()


def generar_manifiesto(
    destino: Path | str,
    model_version: str | None = None,
    artifact_version: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Escribe `MANIFEST.json` en el directorio de artefactos."""
    ruta_destino = Path(destino)
    if not ruta_destino.is_dir():
        raise FileNotFoundError(f"No existe el directorio de artefactos: {ruta_destino}")

    artefactos: dict[str, Any] = {}
    faltantes: list[str] = []
    for nombre in ARTEFACTOS:
        ruta = ruta_destino / nombre
        if not ruta.is_file():
            faltantes.append(nombre)
            continue
        artefactos[nombre] = {"sha256": _sha256(ruta), "bytes": ruta.stat().st_size}

    if faltantes:
        logger.warning(
            "Artefactos ausentes del manifiesto",
            extra={"evento": "manifiesto_incompleto", "faltantes": faltantes},
        )

    manifiesto: dict[str, Any] = {
        "model_version": model_version or "lgb_es_fatal_1_26M",
        "artifact_version": artifact_version or "encodings@local",
        "artefactos": artefactos,
    }
    if extra:
        manifiesto.update(extra)

    ruta_manifiesto = ruta_destino / "MANIFEST.json"
    ruta_manifiesto.write_text(
        json.dumps(manifiesto, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(
        "Manifiesto generado",
        extra={
            "evento": "manifiesto_generado",
            "artefactos": len(artefactos),
            "manifiesto": str(ruta_manifiesto),
        },
    )
    return ruta_manifiesto


def main() -> int:
    """Punto de entrada de linea de comandos."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raiz = Path(__file__).resolve().parents[1]
    analizador = argparse.ArgumentParser(
        description="Genera el MANIFEST.json con los checksums de los artefactos."
    )
    analizador.add_argument(
        "--destino",
        default=str(raiz / "modelos"),
        help="Directorio de artefactos (por defecto ./modelos/).",
    )
    analizador.add_argument(
        "--model-version",
        default="lgb_es_fatal_1_26M@2026-08-22",
        help="Version del modelo publicada en el manifiesto.",
    )
    analizador.add_argument(
        "--artifact-version",
        default="encodings@2026-08-22",
        help="Version de las tablas de referencia publicada en el manifiesto.",
    )
    argumentos = analizador.parse_args()

    ruta = generar_manifiesto(
        argumentos.destino,
        model_version=argumentos.model_version,
        artifact_version=argumentos.artifact_version,
    )
    logger.info("Manifiesto listo en %s", ruta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
