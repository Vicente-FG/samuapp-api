"""Pruebas unitarias del repositorio de predicciones sobre SQLite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from adapters.api.store import PredictionStore, RegistroPrediccion, hash_payload


def _registro(
    prediction_id: str = "pr_ABC12345",
    operator_uid: str = "operador1",
    comuna: str = "Colchane",
    region: str = "Tarapacá",
    created_at: str = "2026-11-20T15:04:05.000Z",
) -> RegistroPrediccion:
    """Construye un registro de prediccion de pruebas."""
    entrada: dict[str, Any] = {
        "data": {
            "fecha": "2026-11-20",
            "hora": 3,
            "comuna": comuna,
            "region": region,
            "urbano": False,
            "involucrados": [{"sexo": "M", "edad": 34, "tipo": "Camioneta"}],
        },
        "notas": "Volcamiento en ruta",
    }
    return RegistroPrediccion(
        prediction_id=prediction_id,
        operator_uid=operator_uid,
        created_at=created_at,
        input=entrada,
        input_hash=hash_payload(entrada),
        features={"Anio": 2026.0, "Hora_num": 3.0},
        output={
            "pFatal": 0.412,
            "victimasGravesPred": 2.08,
            "nivelAlerta": "ALERTA_HOSPITAL",
        },
        recomendacion={
            "ambulancias": 3,
            "paramedico": True,
            "helicoptero": False,
            "alertaHospital": True,
            "fundamento": "P(fatal) 41,2 % (30–60 %).",
        },
        model_version="demo_lgb_es_fatal_1_26M@2026-11-26",
        artifact_version="encodings@demo-2026-11-26",
        code_version="0.1.0",
        latency_ms=187,
        mode="modelo",
        siniestro_ref=None,
        trace_id="trace-1234",
        comuna=comuna,
        region=region,
    )


@pytest.fixture()
def almacen(tmp_path: Path) -> PredictionStore:
    """Repositorio SQLite vacío, con las tablas ya creadas."""
    repositorio = PredictionStore(tmp_path / "data" / "predicciones.db")
    repositorio.crear_tablas()
    return repositorio


def test_guardar_y_recuperar_una_prediccion(almacen: PredictionStore) -> None:
    """La prediccion guardada se recupera integra."""
    almacen.guardar(_registro())

    registro = almacen.obtener("pr_ABC12345")

    assert registro is not None
    assert registro["prediction_id"] == "pr_ABC12345"
    assert registro["operator_uid"] == "operador1"
    assert registro["created_at"] == "2026-11-20T15:04:05.000Z"
    assert registro["input"]["data"]["comuna"] == "Colchane"
    assert registro["features"]["Anio"] == 2026.0
    assert registro["output"]["pFatal"] == 0.412
    assert registro["recomendacion"]["ambulancias"] == 3
    assert registro["model_version"] == "demo_lgb_es_fatal_1_26M@2026-11-26"
    assert registro["code_version"] == "0.1.0"
    assert registro["latency_ms"] == 187
    assert registro["mode"] == "modelo"
    assert registro["trace_id"] == "trace-1234"


def test_obtener_una_prediccion_inexistente(almacen: PredictionStore) -> None:
    """Una prediccion que no existe devuelve `None`."""
    assert almacen.obtener("pr_NOEXISTE") is None


def test_filtrar_por_operador(almacen: PredictionStore) -> None:
    """El filtro por `operatorUid` devuelve solo las suyas."""
    almacen.guardar(_registro("pr_UNO", operator_uid="operador1"))
    almacen.guardar(_registro("pr_DOS", operator_uid="operador2"))
    almacen.guardar(_registro("pr_TRES", operator_uid="operador1", comuna="Pica"))

    pagina = almacen.listar(operator_uid="operador1")

    assert pagina["total"] == 2
    assert {item["predictionId"] for item in pagina["items"]} == {"pr_UNO", "pr_TRES"}


def test_filtrar_por_comuna(almacen: PredictionStore) -> None:
    """El filtro por comuna devuelve solo las de esa comuna."""
    almacen.guardar(_registro("pr_UNO", comuna="Colchane"))
    almacen.guardar(_registro("pr_DOS", comuna="Pica"))
    almacen.guardar(_registro("pr_TRES", comuna="Colchane"))

    pagina = almacen.listar(comuna="Colchane")

    assert pagina["total"] == 2
    assert all(item["comuna"] == "Colchane" for item in pagina["items"])


def test_filtrar_por_rango_de_fechas(almacen: PredictionStore) -> None:
    """El filtro por rango de fechas usa el día de registro."""
    almacen.guardar(_registro("pr_UNO", created_at="2026-11-01T10:00:00.000Z"))
    almacen.guardar(_registro("pr_DOS", created_at="2026-11-15T10:00:00.000Z"))
    almacen.guardar(_registro("pr_TRES", created_at="2026-12-01T10:00:00.000Z"))

    pagina = almacen.listar(fecha_desde="2026-11-10", fecha_hasta="2026-11-30")

    assert pagina["total"] == 1
    assert pagina["items"][0]["predictionId"] == "pr_DOS"


def test_paginacion(almacen: PredictionStore) -> None:
    """La paginacion devuelve la pagina pedida y el total completo."""
    for indice in range(5):
        almacen.guardar(
            _registro(
                f"pr_0000000{indice}",
                created_at=f"2026-11-2{indice}T10:00:00.000Z",
            )
        )

    primera = almacen.listar(limit=2, offset=0)
    segunda = almacen.listar(limit=2, offset=2)

    assert primera["total"] == 5
    assert len(primera["items"]) == 2
    assert primera["limit"] == 2
    assert primera["offset"] == 0
    assert len(segunda["items"]) == 2
    assert segunda["offset"] == 2
    assert {item["predictionId"] for item in primera["items"]}.isdisjoint(
        item["predictionId"] for item in segunda["items"]
    )


def test_crear_tablas_es_idempotente(almacen: PredictionStore) -> None:
    """Crear las tablas dos veces no falla ni borra los datos."""
    almacen.guardar(_registro())
    almacen.crear_tablas()

    assert almacen.obtener("pr_ABC12345") is not None


def test_hash_payload_es_estable_y_depende_del_contenido() -> None:
    """El hash del payload canonico no depende del orden de las claves."""
    primero = {"a": 1, "b": {"c": 2, "d": 3}}
    segundo = {"b": {"d": 3, "c": 2}, "a": 1}

    assert hash_payload(primero) == hash_payload(segundo)
    assert hash_payload(primero) != hash_payload({"a": 1, "b": {"c": 2, "d": 4}})
    assert len(hash_payload(primero)) == 64


def test_registrar_y_leer_la_confirmacion_de_entrega(
    almacen: PredictionStore,
) -> None:
    """La confirmacion de entrega queda registrada y consultable."""
    almacen.guardar(_registro())

    confirmacion = almacen.registrar_entrega("pr_ABC12345", "operador1", "Recibido")

    assert confirmacion["recibido"] is True
    assert confirmacion["ackBy"] == "operador1"
    entrega = almacen.obtener_entrega("pr_ABC12345")
    assert entrega is not None
    assert entrega["nota"] == "Recibido"


def test_el_listado_expone_el_resumen_publico(almacen: PredictionStore) -> None:
    """El listado expone las claves que consume la SPA."""
    almacen.guardar(_registro())

    item = almacen.listar()["items"][0]

    assert set(item) == {
        "predictionId",
        "createdAt",
        "operatorUid",
        "comuna",
        "region",
        "pFatal",
        "victimasGravesPred",
        "nivelAlerta",
        "modelVersion",
    }
    assert json.dumps(item, ensure_ascii=False)
