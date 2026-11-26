"""Genera un juego de artefactos de DEMOSTRACION para probar el flujo completo.

.. warning::

   El modelo que produce este script es **sintético y de uso exclusivo para
   pruebas**. No sirve para reportar desempeno,Exactitud ni resultados de la
   investigacion: los artefactos reales los entrega el equipo que entreno los
   modelos sobre los 1.261.124 siniestros de Carabineros (2010-2025).

El script escribe en `./modelos-demo/` (o en el directorio indicado con
``--destino``). **Nunca escribe en `./modelos/`**: ese directorio aloja los
artefactos reales publicados por el equipo de ciencia de datos.

* `lgb_es_fatal_1_26M.joblib` -> clasificador de `pFatal`.
* `lgb_victimas_graves_1_26M.joblib` -> regresor de `victimasGravesPred`.
* `features_xgb.joblib` -> orden de las 40 columnas de características.
* `encodings.parquet` -> tabla de target encoding de ejemplo.
* `tasa_global.json` -> tasa global sintetica.
* `MANIFEST.json` -> SHA-256 y tamano de cada artefacto.

Uso::

    python -m scripts.entrenar_modelo_demo
    python -m scripts.entrenar_modelo_demo --destino ./modelos-demo --muestras 2000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from core.contracts import Involucrado, Siniestro
from core.features.encodings import (
    TIPO_COMUNA,
    TIPO_REGION,
    TIPO_URBANO,
    TablaTargetEncoding,
)
from core.features.pipeline import ORDEN_POR_DEFECTO, FeatureTransformer
from core.features.taxonomy import TipoVehiculo

__all__ = ["generar_artefactos_demo", "main"]

logger = logging.getLogger(__name__)

#: Semilla del generador pseudoaleatorio: el demo es reproducible.
SEMILLA = 20261126

#: Fecha de publicación del set de demostracion.
FECHA_DEMO = "2026-11-26"

#: Versión declarada del modelo de demostracion.
VERSION_MODELO_DEMO = f"demo_lgb_es_fatal_1_26M@{FECHA_DEMO}"

#: Versión declarada de las tablas de referencia de demostracion.
VERSION_ARTIFACTOS_DEMO = f"encodings@demo-{FECHA_DEMO}"

#: Comunas y regiones de ejemplo (subconjunto del padron CONASET).
COMUNAS_DEMO: dict[str, list[str]] = {
    "Tarapacá": ["Colchane", "Pica", "Iquique", "Alto Hospicio"],
    "Antofagasta": ["Calama", "Antofagasta", "Tocopilla", "Taltal"],
    "Atacama": ["Copiapó", "Vallenar", "Chañaral"],
    "Metropolitana": ["Puente Alto", "Maipú", "Las Condes", "Quilicura"],
}

#: Tipos de vehículo usados en la sintesis, con su peso relativo.
TIPOS_SINTETICOS: list[tuple[TipoVehiculo, int]] = [
    (TipoVehiculo.AUTO, 46),
    (TipoVehiculo.MOTO, 18),
    (TipoVehiculo.CAMIONETA, 12),
    (TipoVehiculo.BUS, 6),
    (TipoVehiculo.CAMION, 5),
    (TipoVehiculo.FURGON, 4),
    (TipoVehiculo.MOTONETA, 3),
    (TipoVehiculo.JEEP, 2),
    (TipoVehiculo.MINIBUS, 2),
    (TipoVehiculo.TRACTOR, 1),
    (TipoVehiculo.AMBULANCIA, 1),
]

#: Vehículos que la heurística de sintesis considera pesados.
PESADOS_SINTETICOS = {
    TipoVehiculo.CAMION,
    TipoVehiculo.CAMIONETA,
    TipoVehiculo.BUS,
    TipoVehiculo.MINIBUS,
    TipoVehiculo.TRACTOR,
}

#: Fecha inicial de la ventana temporal sintetica.
FECHA_INICIAL = date(2024, 1, 1)


def _sigmoide(valor: float) -> float:
    """Funcion logistica estándar."""
    import math

    return 1.0 / (1.0 + math.exp(-valor))


def _generar_siniestro(
    generador: random.Random, indice: int
) -> tuple[Siniestro, float, float]:
    """Sintetiza un siniestro con su probabilidad de fatalidad y victimas graves."""
    region = generador.choice(list(COMUNAS_DEMO))
    comuna = generador.choice(COMUNAS_DEMO[region])
    urbano = generador.random() < 0.72
    hora = generador.randrange(24)
    fecha = FECHA_INICIAL + timedelta(days=generador.randrange(900))

    cantidad = generador.choices([1, 2, 3, 4], weights=[52, 27, 14, 7])[0]
    pesos = [peso for _, peso in TIPOS_SINTETICOS]
    tipos = [tipo for tipo, _ in TIPOS_SINTETICOS]

    involucrados: list[Involucrado] = []
    for _ in range(cantidad):
        tipo = generador.choices(tipos, weights=pesos)[0]
        sexo = generador.choices(["M", "F", "ND"], weights=[63, 34, 3])[0]
        edad = generador.choices(
            [generador.randrange(0, 17), generador.randrange(18, 64), generador.randrange(65, 96)],
            weights=[18, 67, 15],
        )[0]
        involucrados.append(Involucrado(sexo=sexo, edad=edad, tipo=tipo))

    tiene_pesado = any(inv.tipo in PESADOS_SINTETICOS for inv in involucrados)
    tiene_mayor = any(inv.edad >= 75 for inv in involucrados)
    madrugada = 2 <= hora <= 6

    logit = (
        -3.55
        + (0.0 if urbano else 1.25)
        + (0.85 if madrugada else 0.0)
        + (0.95 if tiene_pesado else 0.0)
        + (0.75 if tiene_mayor else 0.0)
        + 0.22 * (cantidad - 1)
    )
    probabilidad = _sigmoide(logit)
    fatal = 1 if generador.random() < probabilidad else 0

    victimas = max(
        0,
        int(
            round(
                0.45
                + 1.45 * probabilidad
                + 0.30 * (cantidad - 1)
                + (0.45 if tiene_mayor else 0.0)
            )
        ),
    )
    siniestro = Siniestro(
        fecha=fecha,
        hora=hora,
        comuna=comuna,
        region=region,
        urbano=urbano,
        involucrados=involucrados,
    )
    return siniestro, float(fatal), float(victimas)


def _construir_tabla_encodings(
    siniestros: list[Siniestro], fatales: list[float]
) -> tuple[TablaTargetEncoding, list[dict[str, Any]]]:
    """Construye la tabla de target encoding a partir de la muestra sintetica."""
    from collections import defaultdict

    conteo_comuna: dict[str, list[float]] = defaultdict(list)
    conteo_region: dict[str, list[float]] = defaultdict(list)
    conteo_urbano: dict[bool, list[float]] = defaultdict(list)
    for siniestro, fatal in zip(siniestros, fatales, strict=True):
        conteo_comuna[siniestro.comuna].append(fatal)
        conteo_region[siniestro.region].append(fatal)
        conteo_urbano[siniestro.urbano].append(fatal)

    filas: list[dict[str, Any]] = []
    entradas: dict[tuple[str, str | bool], Any] = {}
    from core.features.encodings import EntradaEncoding

    grupos: list[tuple[str, dict[str, list[float]]]] = [
        (TIPO_COMUNA, dict(conteo_comuna)),
        (TIPO_REGION, dict(conteo_region)),
    ]
    for tipo, grupos_valores in grupos:
        for valor, observaciones in grupos_valores.items():
            n_c = float(len(observaciones))
            tasa_c = sum(observaciones) / n_c if n_c else 0.0
            filas.append(
                {"clave": f"{tipo}|{valor}", "n_c": n_c, "tasa_c": tasa_c}
            )
            entradas[(tipo, valor)] = EntradaEncoding(n_c=n_c, tasa_c=tasa_c)

    for es_urbano, observaciones in conteo_urbano.items():
        n_c = float(len(observaciones))
        tasa_c = sum(observaciones) / n_c if n_c else 0.0
        filas.append(
            {"clave": f"{TIPO_URBANO}|{es_urbano}", "n_c": n_c, "tasa_c": tasa_c}
        )
        entradas[(TIPO_URBANO, es_urbano)] = EntradaEncoding(n_c=n_c, tasa_c=tasa_c)

    total = float(len(fatales))
    tasa_global = sum(fatales) / total if total else 0.0
    return TablaTargetEncoding(tasa_global=tasa_global, entradas=entradas), filas


def generar_artefactos_demo(destino: Path | str, muestras: int = 2000) -> dict[str, str]:
    """Genera el set completo de artefactos de demostracion.

    Devuelve un diccionario con la ruta de cada artefacto escrito.
    """
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    ruta_destino = Path(destino)
    ruta_destino.mkdir(parents=True, exist_ok=True)

    generador = random.Random(SEMILLA)
    siniestros: list[Siniestro] = []
    fatales: list[float] = []
    victimas: list[float] = []
    for indice in range(muestras):
        siniestro, fatal, cantidad_victimas = _generar_siniestro(generador, indice)
        siniestros.append(siniestro)
        fatales.append(fatal)
        victimas.append(cantidad_victimas)

    tabla, filas_encodings = _construir_tabla_encodings(siniestros, fatales)
    transformador = FeatureTransformer(referencias=tabla)

    matriz = np.asarray(
        [transformador.vector(transformador.transformar(s)) for s in siniestros],
        dtype=float,
    )
    etiquetas_fatal = np.asarray(fatales, dtype=int)
    etiquetas_victimas = np.asarray(victimas, dtype=float)

    clasificador = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=5,
        random_state=SEMILLA,
        n_jobs=-1,
    )
    clasificador.fit(matriz, etiquetas_fatal)

    regresor = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=5,
        random_state=SEMILLA,
        n_jobs=-1,
    )
    regresor.fit(matriz, etiquetas_victimas)

    ruta_clasificador = ruta_destino / "lgb_es_fatal_1_26M.joblib"
    ruta_regresor = ruta_destino / "lgb_victimas_graves_1_26M.joblib"
    ruta_orden = ruta_destino / "features_xgb.joblib"
    ruta_encodings = ruta_destino / "encodings.parquet"
    ruta_tasa = ruta_destino / "tasa_global.json"
    ruta_manifiesto = ruta_destino / "MANIFEST.json"

    joblib.dump(clasificador, ruta_clasificador)
    joblib.dump(regresor, ruta_regresor)
    joblib.dump(list(ORDEN_POR_DEFECTO), ruta_orden)
    pd.DataFrame(filas_encodings, columns=["clave", "n_c", "tasa_c"]).to_parquet(
        ruta_encodings, index=False
    )
    ruta_tasa.write_text(
        json.dumps(
            {
                "tasa_global": tabla.tasa_global,
                "fecha": FECHA_DEMO,
                "artifact_version": VERSION_ARTIFACTOS_DEMO,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    artefactos: dict[str, Any] = {}
    for ruta in (
        ruta_clasificador,
        ruta_regresor,
        ruta_orden,
        ruta_encodings,
        ruta_tasa,
    ):
        artefactos[ruta.name] = {
            "sha256": hashlib.sha256(ruta.read_bytes()).hexdigest(),
            "bytes": ruta.stat().st_size,
        }

    manifiesto = {
        "model_version": VERSION_MODELO_DEMO,
        "artifact_version": VERSION_ARTIFACTOS_DEMO,
        "generado_por": "scripts.entrenar_modelo_demo",
        "advertencia": (
            "Artefactos de demostracion: modelo sintetico, no sirve para "
            "reportar desempeno. El modelo real lo entrega el equipo."
        ),
        "artefactos": artefactos,
    }
    ruta_manifiesto.write_text(
        json.dumps(manifiesto, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logger.info(
        "Artefactos de demostracion generados",
        extra={
            "evento": "artefactos_demo",
            "directorio": str(ruta_destino),
            "muestras": muestras,
            "tasa_global": tabla.tasa_global,
        },
    )
    return {
        "clasificador": str(ruta_clasificador),
        "regresor": str(ruta_regresor),
        "orden_columnas": str(ruta_orden),
        "encodings": str(ruta_encodings),
        "tasa_global": str(ruta_tasa),
        "manifiesto": str(ruta_manifiesto),
    }


def main() -> int:
    """Punto de entrada de linea de comandos."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )
    raiz = Path(__file__).resolve().parents[1]
    analizador = argparse.ArgumentParser(
        description="Genera los artefactos de demostracion de SAMUAPP."
    )
    analizador.add_argument(
        "--destino",
        default=str(raiz / "modelos-demo"),
        help=(
            "Directorio donde se escriben los artefactos "
            "(por defecto ./modelos-demo/, nunca ./modelos/)."
        ),
    )
    analizador.add_argument(
        "--muestras",
        type=int,
        default=2000,
        help="Cantidad de siniestros sinteticos del set de entrenamiento.",
    )
    argumentos = analizador.parse_args()

    rutas = generar_artefactos_demo(argumentos.destino, argumentos.muestras)
    logger.info("Artefactos de demostracion listos en %s", argumentos.destino)
    for nombre, ruta in rutas.items():
        logger.info("  %s: %s", nombre, ruta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
