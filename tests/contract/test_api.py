"""Pruebas de contrato de la API REST de SAMUAPP.

Se ejercita la aplicación completa con `TestClient`, en modo de desarrollo
(`SAMUAPP_AUTH_MODE=dev`) y contra los artefactos de demostracion generados
en un directorio temporal.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from adapters.api.main import crear_app
from scripts.entrenar_modelo_demo import generar_artefactos_demo

#: Token de desarrollo del rol operador.
TOKEN_OPERADOR = "dev:operador:operador1"
#: Token de desarrollo del rol supervisor.
TOKEN_SUPERVISOR = "dev:supervisor:supervisor1"

#: Payload de ejemplo publicado en el contrato de la API.
PAYLOAD_EJEMPLO: dict = {
    "data": {
        "fecha": "2026-11-20",
        "hora": 3,
        "comuna": "Colchane",
        "region": "Tarapacá",
        "urbano": False,
        "involucrados": [
            {"sexo": "M", "edad": 34, "tipo": "Camioneta"},
            {"sexo": "M", "edad": 28, "tipo": "Camión"},
            {"sexo": "F", "edad": 61, "tipo": "Auto"},
        ],
    },
    "notas": "Volcamiento en ruta",
}

#: Claves que la SPA espera en la respuesta de una prediccion.
CLAVES_RESPUESTA = {
    "predictionId",
    "pFatal",
    "victimasGravesPred",
    "nivelAlerta",
    "recomendacion",
    "modelVersion",
    "artifactVersion",
    "latencyMs",
    "mode",
}


@pytest.fixture(scope="module")
def artefactos_demo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Genera el set de artefactos de demostracion una sola vez."""
    destino = tmp_path_factory.mktemp("artefactos-demo") / "modelos"
    generar_artefactos_demo(destino, muestras=800)
    return destino


@pytest.fixture()
def cliente(
    artefactos_demo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    """Aplicación de pruebas, con autenticación de desarrollo y artefactos demo."""
    monkeypatch.setenv("SAMUAPP_MODELOS_DIR", str(artefactos_demo))
    monkeypatch.setenv("SAMUAPP_DB_PATH", str(tmp_path / "data" / "predicciones.db"))
    monkeypatch.setenv("SAMUAPP_AUTH_MODE", "dev")
    monkeypatch.setenv("SAMUAPP_MODO_DEGRADADO", "false")
    monkeypatch.setenv("SAMUAPP_LOG_LEVEL", "WARNING")
    monkeypatch.setenv("SAMUAPP_TIMEOUT_MS", "5000")

    aplicacion = crear_app()
    with TestClient(aplicacion) as cliente_test:
        yield cliente_test


def _cabeceras(token: str = TOKEN_OPERADOR) -> dict[str, str]:
    """Cabeceras de autenticación con el token de desarrollo."""
    return {"Authorization": f"Bearer {token}"}


def test_crear_prediccion_devuelve_el_contrato_publicado(cliente: TestClient) -> None:
    """POST /api/v1/predictions responde 200 con las claves del contrato."""
    respuesta = cliente.post("/api/v1/predictions", json=PAYLOAD_EJEMPLO, headers=_cabeceras())

    assert respuesta.status_code == 200, respuesta.text
    cuerpo = respuesta.json()

    assert set(cuerpo) == CLAVES_RESPUESTA
    assert cuerpo["predictionId"].startswith("pr_")
    assert 0.0 <= cuerpo["pFatal"] <= 1.0
    assert cuerpo["victimasGravesPred"] >= 0.0
    assert cuerpo["nivelAlerta"] in {"ESTANDAR", "REFORZADO", "ALERTA_HOSPITAL"}
    assert cuerpo["mode"] == "modelo"
    assert cuerpo["modelVersion"]
    assert cuerpo["artifactVersion"]
    assert cuerpo["latencyMs"] >= 0
    assert respuesta.headers["X-Trace-Id"]


def test_la_recomendacion_tiene_las_claves_del_contrato(cliente: TestClient) -> None:
    """El bloque `recomendacion` trae las claves que consume la SPA."""
    respuesta = cliente.post("/api/v1/predictions", json=PAYLOAD_EJEMPLO, headers=_cabeceras())

    recomendacion = respuesta.json()["recomendacion"]

    assert set(recomendacion) == {
        "ambulancias",
        "paramedico",
        "helicoptero",
        "alertaHospital",
        "fundamento",
    }
    assert recomendacion["ambulancias"] >= 1
    assert recomendacion["fundamento"]


def test_payload_invalido_responde_errores_por_campo(cliente: TestClient) -> None:
    """Hora 25, edad 120 y cero involucrados devuelven 400 por campo."""
    invalido = {
        "data": {
            "fecha": "2026-11-20",
            "hora": 25,
            "comuna": "Colchane",
            "region": "Tarapacá",
            "urbano": False,
            "involucrados": [{"sexo": "M", "edad": 120, "tipo": "Auto"}],
        }
    }

    respuesta = cliente.post("/api/v1/predictions", json=invalido, headers=_cabeceras())

    assert respuesta.status_code == 400
    error = respuesta.json()["error"]
    assert error["code"] == "invalid-argument"
    assert "hora" in error["detalles"]
    assert "involucrados[0].edad" in error["detalles"]


def test_payload_sin_involucrados_es_invalido(cliente: TestClient) -> None:
    """Sin involucrados no hay severity: el payload se rechaza."""
    invalido = {
        "data": {
            "fecha": "2026-11-20",
            "hora": 3,
            "comuna": "Colchane",
            "region": "Tarapacá",
            "urbano": False,
            "involucrados": [],
        }
    }

    respuesta = cliente.post("/api/v1/predictions", json=invalido, headers=_cabeceras())

    assert respuesta.status_code == 400
    error = respuesta.json()["error"]
    assert error["code"] == "invalid-argument"
    assert "involucrados" in error["detalles"]


def test_sin_token_responde_unauthenticated(cliente: TestClient) -> None:
    """Sin cabecera de autorizacion la respuesta es 401."""
    respuesta = cliente.post("/api/v1/predictions", json=PAYLOAD_EJEMPLO)

    assert respuesta.status_code == 401
    assert respuesta.json()["error"]["code"] == "unauthenticated"


def test_token_invalido_responde_unauthenticated(cliente: TestClient) -> None:
    """Un token que no cumple el formato de desarrollo es 401."""
    respuesta = cliente.post(
        "/api/v1/predictions", json=PAYLOAD_EJEMPLO, headers=_cabeceras("basura")
    )

    assert respuesta.status_code == 401
    assert respuesta.json()["error"]["code"] == "unauthenticated"


def test_operador_no_puede_listar_el_historico_global(cliente: TestClient) -> None:
    """El listado global exige supervisor o admin."""
    respuesta = cliente.get("/api/v1/predictions", headers=_cabeceras(TOKEN_OPERADOR))

    assert respuesta.status_code == 403
    assert respuesta.json()["error"]["code"] == "permission-denied"


def test_supervisor_puede_listar_el_historico_global(cliente: TestClient) -> None:
    """El supervisor si puede listar el historico completo."""
    respuesta = cliente.get("/api/v1/predictions", headers=_cabeceras(TOKEN_SUPERVISOR))

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert set(cuerpo) == {"items", "total", "limit", "offset"}


def test_health_reporta_el_estado_de_los_artefactos(cliente: TestClient) -> None:
    """El endpoint de salud informa versiones e integridad de los artefactos."""
    respuesta = cliente.get("/api/v1/health")

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["estado"] == "ok"
    assert cuerpo["modelVersion"]
    assert cuerpo["artifactVersion"]
    assert cuerpo["integridadVerificada"] is True
    assert cuerpo["artefactos"]
    assert all(artefacto["verificado"] for artefacto in cuerpo["artefactos"])


def test_models_current_publica_las_versiones(cliente: TestClient) -> None:
    """El modelo vigente expone su versión y la de sus artefactos."""
    respuesta = cliente.get("/api/v1/models/current", headers=_cabeceras())

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["modelVersion"]
    assert cuerpo["artifactVersion"]
    assert len(cuerpo["caracteristicas"]) == 40


def test_la_prediccion_se_persiste_y_se_puede_consultar(cliente: TestClient) -> None:
    """La prediccion creada queda disponible para su propietario."""
    creada = cliente.post("/api/v1/predictions", json=PAYLOAD_EJEMPLO, headers=_cabeceras())
    prediction_id = creada.json()["predictionId"]

    detalle = cliente.get(
        f"/api/v1/predictions/{prediction_id}", headers=_cabeceras()
    )

    assert detalle.status_code == 200
    cuerpo = detalle.json()
    assert cuerpo["predictionId"] == prediction_id
    assert cuerpo["operatorUid"] == "operador1"
    assert len(cuerpo["features"]) == 40
    assert cuerpo["inputHash"]
    assert cuerpo["traceId"]


def test_otro_operador_no_puede_consultar_una_prediccion(cliente: TestClient) -> None:
    """Un operador no puede leer las predicciones de otro operador."""
    creada = cliente.post("/api/v1/predictions", json=PAYLOAD_EJEMPLO, headers=_cabeceras())
    prediction_id = creada.json()["predictionId"]

    respuesta = cliente.get(
        f"/api/v1/predictions/{prediction_id}",
        headers=_cabeceras("dev:operador:otro_operador"),
    )

    assert respuesta.status_code == 403
    assert respuesta.json()["error"]["code"] == "permission-denied"


def test_hoja_de_despacho_es_html_imprimible(cliente: TestClient) -> None:
    """La hoja de despacho se entrega como HTML con los datos del siniestro."""
    creada = cliente.post("/api/v1/predictions", json=PAYLOAD_EJEMPLO, headers=_cabeceras())
    prediction_id = creada.json()["predictionId"]

    respuesta = cliente.get(
        f"/api/v1/predictions/{prediction_id}/dispatch-sheet",
        headers=_cabeceras(),
    )

    assert respuesta.status_code == 200
    assert "text/html" in respuesta.headers["content-type"]
    assert prediction_id in respuesta.text
    assert "Colchane" in respuesta.text


def test_confirmacion_de_entrega_registra_el_ack(cliente: TestClient) -> None:
    """El ack de la hoja de despacho queda registrado."""
    creada = cliente.post("/api/v1/predictions", json=PAYLOAD_EJEMPLO, headers=_cabeceras())
    prediction_id = creada.json()["predictionId"]

    respuesta = cliente.post(
        f"/api/v1/predictions/{prediction_id}/dispatch-sheet/ack",
        headers=_cabeceras(),
    )

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["predictionId"] == prediction_id
    assert cuerpo["recibido"] is True
    assert cuerpo["ackBy"] == "operador1"


def test_sin_artefactos_responde_failed_precondition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sin artefactos y sin modo degradado, la API responde 412."""
    monkeypatch.setenv("SAMUAPP_MODELOS_DIR", str(tmp_path / "modelos-vacios"))
    monkeypatch.setenv("SAMUAPP_DB_PATH", str(tmp_path / "data" / "predicciones.db"))
    monkeypatch.setenv("SAMUAPP_AUTH_MODE", "dev")
    monkeypatch.setenv("SAMUAPP_MODO_DEGRADADO", "false")
    monkeypatch.setenv("SAMUAPP_LOG_LEVEL", "WARNING")

    with TestClient(crear_app()) as cliente_sin_artefactos:
        respuesta = cliente_sin_artefactos.post(
            "/api/v1/predictions", json=PAYLOAD_EJEMPLO, headers=_cabeceras()
        )

    assert respuesta.status_code == 412
    assert respuesta.json()["error"]["code"] == "failed-precondition"


def test_modo_degradado_responde_200_sin_artefactos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Con `SAMUAPP_MODO_DEGRADADO=true` la API responde 200 en modo degradado."""
    monkeypatch.setenv("SAMUAPP_MODELOS_DIR", str(tmp_path / "modelos-vacios"))
    monkeypatch.setenv("SAMUAPP_DB_PATH", str(tmp_path / "data" / "predicciones.db"))
    monkeypatch.setenv("SAMUAPP_AUTH_MODE", "dev")
    monkeypatch.setenv("SAMUAPP_MODO_DEGRADADO", "true")
    monkeypatch.setenv("SAMUAPP_LOG_LEVEL", "WARNING")

    with TestClient(crear_app()) as cliente_degradado:
        respuesta = cliente_degradado.post(
            "/api/v1/predictions", json=PAYLOAD_EJEMPLO, headers=_cabeceras()
        )

    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["mode"] == "degradado"
    assert 0.0 <= cuerpo["pFatal"] <= 1.0
    assert cuerpo["recomendacion"]["ambulancias"] >= 1
