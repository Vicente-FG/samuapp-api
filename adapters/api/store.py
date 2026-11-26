"""Repositorio de predicciones sobre SQLite de la biblioteca estándar.

No se usa ORM: la persistencia es una tabla por prediccion con indices por
operador, comuna y fecha, que son los tres filtros del historico (HU-05).

Por cada prediccion se guarda: `predictionId`, `operatorUid`, `createdAt`
(UTC ISO-8601), `input` (JSON), `inputHash` (SHA-256 del payload canonico),
`features` (JSON de las 40 características), `output` (JSON), `recomendacion`
(JSON), `modelVersion`, `artifactVersion`, `codeVersion`, `latencyMs`, `mode`,
`siniestroRef` y `traceId`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = ["PredictionStore", "RegistroPrediccion", "hash_payload"]

logger = logging.getLogger(__name__)

#: Sentido de ordenacion del historico: del más reciente al más antiguo.
ORDEN_POR_DEFECTO = "created_at DESC, prediction_id DESC"

#: Máximo de registros que devuelve un listado sin paginacion explícita.
LIMITE_MAXIMO = 200

ESQUEMA_PREDICCIONES = """
CREATE TABLE IF NOT EXISTS predicciones (
    prediction_id      TEXT PRIMARY KEY,
    operator_uid       TEXT NOT NULL,
    created_at         TEXT NOT NULL,
    input_json         TEXT NOT NULL,
    input_hash         TEXT NOT NULL,
    features_json      TEXT NOT NULL,
    output_json        TEXT NOT NULL,
    recomendacion_json TEXT NOT NULL,
    model_version      TEXT NOT NULL,
    artifact_version   TEXT NOT NULL,
    code_version       TEXT NOT NULL,
    latency_ms         INTEGER NOT NULL,
    mode               TEXT NOT NULL,
    siniestro_ref      TEXT,
    trace_id           TEXT NOT NULL,
    comuna              TEXT NOT NULL,
    region             TEXT NOT NULL
)
"""

ESQUEMA_ENTREGAS = """
CREATE TABLE IF NOT EXISTS entregas (
    prediction_id TEXT PRIMARY KEY,
    ack_by        TEXT NOT NULL,
    ack_at        TEXT NOT NULL,
    nota          TEXT,
    FOREIGN KEY (prediction_id) REFERENCES predicciones (prediction_id)
)
"""

INDICES = (
    "CREATE INDEX IF NOT EXISTS idx_predicciones_operador ON predicciones (operator_uid)",
    "CREATE INDEX IF NOT EXISTS idx_predicciones_comuna ON predicciones (comuna)",
    "CREATE INDEX IF NOT EXISTS idx_predicciones_created ON predicciones (created_at)",
)


def hash_payload(payload: Any) -> str:
    """SHA-256 del payload canonico (JSON con claves ordenadas)."""
    canonico = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    )
    return hashlib.sha256(canonico.encode("utf-8")).hexdigest()


def _ahora_utc() -> str:
    """Marca de tiempo actual en ISO-8601 UTC."""
    return datetime.now(UTC).isoformat(timespec="milliseconds")


@dataclass(frozen=True)
class RegistroPrediccion:
    """Registro completo de una prediccion persistida."""

    prediction_id: str
    operator_uid: str
    created_at: str
    input: dict[str, Any]
    input_hash: str
    features: dict[str, Any]
    output: dict[str, Any]
    recomendacion: dict[str, Any]
    model_version: str
    artifact_version: str
    code_version: str
    latency_ms: int
    mode: str
    siniestro_ref: str | None
    trace_id: str
    comuna: str
    region: str


class PredictionStore:
    """Repositorio de predicciones y confirmaciones de entrega."""

    def __init__(self, ruta: Path | str) -> None:
        self._ruta = Path(ruta)

    @property
    def ruta(self) -> Path:
        """Ruta del archivo SQLite."""
        return self._ruta

    # --- infraestructura ----------------------------------------------------

    def _conectar(self) -> sqlite3.Connection:
        """Abre una conexion con las pragmas de trabajo del servicio."""
        self._ruta.parent.mkdir(parents=True, exist_ok=True)
        conexion = sqlite3.connect(self._ruta, timeout=30.0)
        conexion.row_factory = sqlite3.Row
        conexion.execute("PRAGMA journal_mode=WAL")
        conexion.execute("PRAGMA foreign_keys=ON")
        return conexion

    def crear_tablas(self) -> None:
        """Crea las tablas y los indices. Es idempotente."""
        with self._conectar() as conexion:
            conexion.execute(ESQUEMA_PREDICCIONES)
            conexion.execute(ESQUEMA_ENTREGAS)
            for indice in INDICES:
                conexion.execute(indice)
        logger.info(
            "Repositorio de predicciones listo",
            extra={"evento": "store_inicializado", "ruta": str(self._ruta)},
        )

    # --- escritura ----------------------------------------------------------

    def guardar(self, registro: RegistroPrediccion) -> None:
        """Persiste una prediccion. Reemplaza la fila si el id ya existe."""
        with self._conectar() as conexion:
            conexion.execute(
                """
                INSERT OR REPLACE INTO predicciones (
                    prediction_id, operator_uid, created_at, input_json, input_hash,
                    features_json, output_json, recomendacion_json, model_version,
                    artifact_version, code_version, latency_ms, mode, siniestro_ref,
                    trace_id, comuna, region
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    registro.prediction_id,
                    registro.operator_uid,
                    registro.created_at,
                    json.dumps(registro.input, ensure_ascii=False, default=str),
                    registro.input_hash,
                    json.dumps(registro.features, ensure_ascii=False, default=str),
                    json.dumps(registro.output, ensure_ascii=False, default=str),
                    json.dumps(registro.recomendacion, ensure_ascii=False, default=str),
                    registro.model_version,
                    registro.artifact_version,
                    registro.code_version,
                    registro.latency_ms,
                    registro.mode,
                    registro.siniestro_ref,
                    registro.trace_id,
                    registro.comuna,
                    registro.region,
                ),
            )

    def registrar_entrega(
        self, prediction_id: str, ack_by: str, nota: str | None = None
    ) -> dict[str, Any]:
        """Registra la confirmacion de entrega de una hoja de despacho."""
        ack_at = _ahora_utc()
        with self._conectar() as conexion:
            conexion.execute(
                """
                INSERT OR REPLACE INTO entregas (prediction_id, ack_by, ack_at, nota)
                VALUES (?, ?, ?, ?)
                """,
                (prediction_id, ack_by, ack_at, nota),
            )
        return {
            "predictionId": prediction_id,
            "recibido": True,
            "ackAt": ack_at,
            "ackBy": ack_by,
        }

    # --- lectura -------------------------------------------------------------

    def obtener(self, prediction_id: str) -> dict[str, Any] | None:
        """Devuelve la prediccion completa, o `None` si no existe."""
        with self._conectar() as conexion:
            fila = conexion.execute(
                "SELECT * FROM predicciones WHERE prediction_id = ?",
                (prediction_id,),
            ).fetchone()
        if fila is None:
            return None
        registro = dict(fila)
        registro["input"] = json.loads(registro.pop("input_json"))
        registro["features"] = json.loads(registro.pop("features_json"))
        registro["output"] = json.loads(registro.pop("output_json"))
        registro["recomendacion"] = json.loads(registro.pop("recomendacion_json"))
        return registro

    def obtener_entrega(self, prediction_id: str) -> dict[str, Any] | None:
        """Devuelve la confirmacion de entrega de una prediccion, o `None`."""
        with self._conectar() as conexion:
            fila = conexion.execute(
                "SELECT * FROM entregas WHERE prediction_id = ?",
                (prediction_id,),
            ).fetchone()
        return dict(fila) if fila is not None else None

    def listar(
        self,
        operator_uid: str | None = None,
        comuna: str | None = None,
        fecha_desde: str | None = None,
        fecha_hasta: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Lista el historico con filtros y paginacion.

        Los filtros son por `operatorUid`, `comuna` y rango de fechas sobre el
        día de registro (`createdAt`).
        """
        limite = max(1, min(int(limit), LIMITE_MAXIMO))
        desplazamiento = max(0, int(offset))

        condiciones: list[str] = []
        parametros: list[Any] = []
        if operator_uid:
            condiciones.append("operator_uid = ?")
            parametros.append(operator_uid)
        if comuna:
            condiciones.append("comuna = ?")
            parametros.append(comuna)
        if fecha_desde:
            condiciones.append("substr(created_at, 1, 10) >= ?")
            parametros.append(fecha_desde)
        if fecha_hasta:
            condiciones.append("substr(created_at, 1, 10) <= ?")
            parametros.append(fecha_hasta)

        donde = f" WHERE {' AND '.join(condiciones)}" if condiciones else ""

        with self._conectar() as conexion:
            total = int(
                conexion.execute(
                    f"SELECT COUNT(*) FROM predicciones{donde}", parametros
                ).fetchone()[0]
            )
            filas = conexion.execute(
                f"""
                SELECT prediction_id, created_at, operator_uid, comuna, region,
                       output_json, model_version
                FROM predicciones{donde}
                ORDER BY {ORDEN_POR_DEFECTO}
                LIMIT ? OFFSET ?
                """,
                [*parametros, limite, desplazamiento],
            ).fetchall()

        items: list[dict[str, Any]] = []
        for fila in filas:
            salida = json.loads(fila["output_json"])
            items.append(
                {
                    "predictionId": fila["prediction_id"],
                    "createdAt": fila["created_at"],
                    "operatorUid": fila["operator_uid"],
                    "comuna": fila["comuna"],
                    "region": fila["region"],
                    "pFatal": salida.get("pFatal"),
                    "victimasGravesPred": salida.get("victimasGravesPred"),
                    "nivelAlerta": salida.get("nivelAlerta"),
                    "modelVersion": fila["model_version"],
                }
            )
        return {
            "items": items,
            "total": total,
            "limit": limite,
            "offset": desplazamiento,
        }
