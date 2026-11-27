"""Prueba de regresion del manifiesto de artefactos.

El `MANIFEST.json` es el contrato de integridad del servicio: si el SHA-256
declarado no coincide con el archivo que se despliega, el arranque falla con
`failed-precondition` (412) en lugar de servir predicciones corruptas.

Esta prueba detecta la clase de error mas habitual: regenerar un artefacto
(sin regenerar el manifiesto) o editar el manifiesto a mano. Es la garantia de
que lo que se versiona es exactamente lo que se verifica.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.generar_manifiesto import ARTEFACTOS

RAIZ = Path(__file__).resolve().parents[2]
DIRECTORIO_MODELOS = RAIZ / "modelos"


def _sha256(ruta: Path) -> str:
    """SHA-256 de un archivo, leido por bloques."""
    digest = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()


@pytest.fixture(scope="module")
def manifiesto() -> dict:
    """Manifiesto de artefactos publicado."""
    ruta = DIRECTORIO_MODELOS / "MANIFEST.json"
    if not ruta.is_file():
        pytest.skip("No hay artefactos publicados en este entorno.")
    return json.loads(ruta.read_text(encoding="utf-8"))


def test_el_manifiesto_declara_los_cinco_artefactos(manifiesto: dict) -> None:
    """El manifiesto describe exactamente los artefactos que se verifican."""
    declarados = set(manifiesto["artefactos"])
    assert declarados == set(ARTEFACTOS)


def test_los_artefactos_publicados_coinciden_con_el_manifiesto(manifiesto: dict) -> None:
    """Cada artefacto del repositorio tiene el SHA-256 que declara el manifiesto."""
    for nombre, meta in manifiesto["artefactos"].items():
        ruta = DIRECTORIO_MODELOS / nombre
        if not ruta.is_file():
            pytest.skip(f"El artefacto {nombre} no esta disponible en este entorno.")

        assert ruta.stat().st_size == meta["bytes"], f"{nombre} cambio de tamano"
        assert _sha256(ruta) == meta["sha256"], (
            f"El SHA-256 de {nombre} no coincide con el manifiesto: regenera el "
            "manifiesto con `python -m scripts.generar_manifiesto --destino modelos`."
        )


def test_el_manifiesto_declara_las_versiones_publicadas(manifiesto: dict) -> None:
    """La version del modelo y la de artefactos estan declaradas."""
    assert manifiesto["model_version"].startswith("lgb_es_fatal_1_26M@")
    assert manifiesto["artifact_version"].startswith("encodings@")
