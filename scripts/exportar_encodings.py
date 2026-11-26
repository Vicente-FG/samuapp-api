"""Exporta la tabla de target encoding congelada desde el dataset de entrenamiento.

Reproduce exactamente el calculo del entrenamiento (repositorio de ciencia de
datos, `notebooks/probar_modelos.py`):

* `ES_FATAL = (Muertos > 0)`.
* `tasa_global = mean(ES_FATAL)` sobre el dataset completo (1.261.124
  siniestros).
* Por comuna, region y binario urbano:
  `tasa_te = (n_fatales + ALPHA * tasa_global) / (n + ALPHA)` con `ALPHA = 50`.

El resultado se publica como artefacto congelado:

* `encodings.parquet` con las columnas `clave`, `n_c` y `tasa_c`.
* `tasa_global.json` con `{"tasa_global": <float>}`.

Uso::

    python -m scripts.exportar_encodings --dataset <ruta df_train.parquet>
    python -m scripts.exportar_encodings --dataset <ruta> --destino ./modelos

.. note::

   El `tasa_c` que se guarda es la tasa observada de la categoria
   (`n_fatales / n_c`). La tasa bayesiana que consume el backend,
   `(n_c * tasa_c + ALPHA * tasa_global) / (n_c + ALPHA)`, es
   identica a la del entrenamiento porque `n_c * tasa_c = n_fatales`.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from core.features.encodings import TIPO_COMUNA, TIPO_REGION, TIPO_URBANO

__all__ = ["exportar_encodings", "main"]

logger = logging.getLogger(__name__)

#: Columnas que se leen del dataset: solo las necesarias para el encoding.
COLUMNAS_NECESARIAS = ["Comuna", "Región", "Urbano", "Muertos"]

#: Mapa de valores textuales de `Urbano` a 1 (urbano) o 0 (rural).
MAPA_URBANO = {"Si": 1, "Sí": 1, "Urbano": 1, "No": 0, "Rural": 0}

#: Fecha de publicacion de la tabla congelada.
FECHA_ACTUAL = "2026-08-22"


def _urbano_binario(serie: Any) -> Any:
    """Convierte la columna `Urbano` a 1 (urbano) o 0 (rural)."""
    valores = set(serie.dropna().unique())
    if valores.issubset({True, False}):
        return serie.fillna(False).astype("int8")
    return serie.map(MAPA_URBANO).fillna(0).astype("int8")


def _tabla_te(serie_clave: Any, es_fatal: Any, tipo: str) -> list[dict[str, Any]]:
    """Construye las filas `clave`, `n_c` y `tasa_c` de una agrupacion."""
    import pandas as pd

    marco = pd.DataFrame(
        {"clave": serie_clave.to_numpy(), "es_fatal": es_fatal.to_numpy()}
    )
    agrupado = marco.groupby("clave").agg(
        n=("es_fatal", "count"), n_fatales=("es_fatal", "sum")
    )
    filas: list[dict[str, Any]] = []
    for valor, fila in agrupado.iterrows():
        n_c = float(fila["n"])
        tasa_c = float(fila["n_fatales"]) / n_c if n_c else 0.0
        filas.append({"clave": f"{tipo}|{valor}", "n_c": n_c, "tasa_c": tasa_c})
    return filas


def exportar_encodings(dataset: Path | str, destino: Path | str) -> dict[str, Any]:
    """Construye `encodings.parquet` y `tasa_global.json` desde el dataset."""
    import pandas as pd

    ruta_dataset = Path(dataset)
    ruta_destino = Path(destino)
    ruta_destino.mkdir(parents=True, exist_ok=True)

    if not ruta_dataset.is_file():
        raise FileNotFoundError(f"No existe el dataset de entrenamiento: {ruta_dataset}")

    logger.info("Leyendo el dataset de entrenamiento", extra={"dataset": str(ruta_dataset)})
    marco = pd.read_parquet(ruta_dataset, columns=COLUMNAS_NECESARIAS)
    total = len(marco)

    es_fatal = (marco["Muertos"] > 0).astype(int)
    tasa_global = float(es_fatal.mean())
    logger.info(
        "Tasa global de fatalidad",
        extra={"evento": "tasa_global", "tasa_global": tasa_global, "filas": total},
    )

    marcador = es_fatal
    filas: list[dict[str, Any]] = []
    filas.extend(_tabla_te(marco["Comuna"], marcador, TIPO_COMUNA))
    filas.extend(_tabla_te(marco["Región"], marcador, TIPO_REGION))
    filas.extend(
        _tabla_te(_urbano_binario(marco["Urbano"]), marcador, TIPO_URBANO)
    )

    tabla = pd.DataFrame(filas, columns=["clave", "n_c", "tasa_c"])
    ruta_encodings = ruta_destino / "encodings.parquet"
    tabla.to_parquet(ruta_encodings, index=False)

    ruta_tasa = ruta_destino / "tasa_global.json"
    ruta_tasa.write_text(
        json.dumps(
            {
                "tasa_global": tasa_global,
                "fecha": FECHA_ACTUAL,
                "siniestros": total,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info(
        "Tabla de target encoding exportada",
        extra={
            "evento": "encodings_exportados",
            "filas": len(tabla),
            "tasa_global": tasa_global,
            "encodings": str(ruta_encodings),
        },
    )
    return {
        "encodings": str(ruta_encodings),
        "tasa_global": str(ruta_tasa),
        "tasa_global_valor": tasa_global,
        "filas": len(tabla),
    }


def main() -> int:
    """Punto de entrada de linea de comandos."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raiz = Path(__file__).resolve().parents[1]
    analizador = argparse.ArgumentParser(
        description="Exporta la tabla de target encoding congelada."
    )
    analizador.add_argument(
        "--dataset",
        required=True,
        help="Ruta del parquet de entrenamiento (df_train.parquet).",
    )
    analizador.add_argument(
        "--destino",
        default=str(raiz / "modelos"),
        help="Directorio donde se escriben los artefactos (por defecto ./modelos/).",
    )
    argumentos = analizador.parse_args()

    resultado = exportar_encodings(argumentos.dataset, argumentos.destino)
    logger.info("Artefactos exportados: %s", resultado)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
