"""Pruebas de contrato del inicio de sesion y del registro (HU-01).

Se ejercitan los dos endpoints de identidad del modo de desarrollo
(`SAMUAPP_AUTH_MODE=dev`): `POST /api/v1/auth/sesion` y
`POST /api/v1/auth/registro`, y se comprueba que el token emitido sirve para
llamar a un endpoint protegido.

No se generan artefactos de modelo: las rutas de identidad no los usan.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from adapters.api.main import crear_app
from adapters.api.sesion import CONTRASENA_DEMO, USUARIOS_DEMO

#: Payload de ejemplo del contrato de la API.
PAYLOAD_EJEMPLO: dict = {
    "data": {
        "fecha": "2026-11-20",
        "hora": 3,
        "comuna": "Colchane",
        "region": "Tarapacá",
        "urbano": False,
        "involucrados": [
            {"sexo": "M", "edad": 34, "tipo": "Camioneta"},
        ],
    }
}


@pytest.fixture()
def cliente(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Aplicación de pruebas en modo de desarrollo, con almacen temporal."""
    monkeypatch.setenv("SAMUAPP_MODELOS_DIR", str(tmp_path / "modelos"))
    monkeypatch.setenv("SAMUAPP_DB_PATH", str(tmp_path / "data" / "predicciones.db"))
    monkeypatch.setenv("SAMUAPP_AUTH_MODE", "dev")
    monkeypatch.setenv("SAMUAPP_LOG_LEVEL", "WARNING")

    aplicacion = crear_app()
    with TestClient(aplicacion) as cliente_test:
        yield cliente_test


@pytest.fixture()
def cliente_firebase(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Aplicación de pruebas en modo firebase, donde el servicio no emite tokens."""
    monkeypatch.setenv("SAMUAPP_MODELOS_DIR", str(tmp_path / "modelos"))
    monkeypatch.setenv("SAMUAPP_DB_PATH", str(tmp_path / "data" / "predicciones.db"))
    monkeypatch.setenv("SAMUAPP_AUTH_MODE", "firebase")
    monkeypatch.setenv("SAMUAPP_LOG_LEVEL", "WARNING")

    aplicacion = crear_app()
    with TestClient(aplicacion) as cliente_test:
        yield cliente_test


# --- CA-01.4: inicio de sesion valido ---------------------------------------


def test_inicio_de_sesion_valido_devuelve_token_con_uid_y_rol(cliente: TestClient) -> None:
    """Las credenciales de demostracion producen un token utilizable."""
    respuesta = cliente.post(
        "/api/v1/auth/sesion",
        json={"correo": "operador@samuapp.cl", "contrasena": CONTRASENA_DEMO},
    )

    assert respuesta.status_code == 200, respuesta.text
    cuerpo = respuesta.json()

    assert cuerpo["token"].startswith("dev:operador:")
    assert cuerpo["rol"] == "operador"
    assert cuerpo["uid"].startswith("op_")
    assert cuerpo["correo"] == "operador@samuapp.cl"
    assert cuerpo["expiraEn"] > 0
    assert cuerpo["modo"] == "dev"


def test_el_token_emitido_sirve_para_un_endpoint_protegido(cliente: TestClient) -> None:
    """El token de la sesion autentica la creacion de una prediccion."""
    sesion = cliente.post(
        "/api/v1/auth/sesion",
        json={"correo": "supervisor@samuapp.cl", "contrasena": CONTRASENA_DEMO},
    )
    token = sesion.json()["token"]

    respuesta = cliente.post(
        "/api/v1/predictions",
        json=PAYLOAD_EJEMPLO,
        headers={"Authorization": f"Bearer {token}"},
    )

    # El servicio responde 412 porque no hay artefactos de modelo en la prueba;
    # lo relevante es que la peticion supero la verificacion del token.
    assert respuesta.status_code != 401, respuesta.text
    assert respuesta.status_code != 403, respuesta.text


def test_cada_rol_de_demostracion_recibe_su_rol(cliente: TestClient) -> None:
    """Los tres usuarios de demostracion obtienen su rol correspondiente."""
    for usuario in USUARIOS_DEMO:
        respuesta = cliente.post(
            "/api/v1/auth/sesion",
            json={"correo": usuario.correo, "contrasena": CONTRASENA_DEMO},
        )
        assert respuesta.status_code == 200, respuesta.text
        assert respuesta.json()["rol"] == usuario.rol.value


# --- CA-01.5: credenciales invalidas ---------------------------------------


def test_contrasena_incorrecta_es_rechazada(cliente: TestClient) -> None:
    """Una contraseña incorrecta produce 401 con el sobre del contrato."""
    respuesta = cliente.post(
        "/api/v1/auth/sesion",
        json={"correo": "operador@samuapp.cl", "contrasena": "ContrasenaIncorrecta1"},
    )

    assert respuesta.status_code == 401, respuesta.text
    cuerpo = respuesta.json()
    assert cuerpo["error"]["code"] == "unauthenticated"
    assert "válidas" in cuerpo["error"]["message"] or "validas" in cuerpo["error"]["message"]


def test_el_mensaje_no_revela_si_el_correo_existe(cliente: TestClient) -> None:
    """Un correo inexistente produce exactamente el mismo mensaje de error."""
    existente = cliente.post(
        "/api/v1/auth/sesion",
        json={"correo": "operador@samuapp.cl", "contrasena": "ContrasenaIncorrecta1"},
    )
    inexistente = cliente.post(
        "/api/v1/auth/sesion",
        json={"correo": "nadie@samuapp.cl", "contrasena": "ContrasenaIncorrecta1"},
    )

    assert existente.status_code == inexistente.status_code == 401
    assert existente.json()["error"]["message"] == inexistente.json()["error"]["message"]


# --- CA-01.1 y CA-01.2: registro -------------------------------------------


def test_registro_exitoso_crea_la_cuenta_y_abre_sesion(cliente: TestClient) -> None:
    """Un operador nuevo puede registrarse y entrar con sus credenciales."""
    registro = cliente.post(
        "/api/v1/auth/registro",
        json={
            "nombre": "Nuevo Operador",
            "correo": "nuevo@samuapp.cl",
            "contrasena": "ContrasenaSegura2026",
            "rol": "operador",
        },
    )

    assert registro.status_code == 200, registro.text
    assert registro.json()["token"].startswith("dev:operador:")

    sesion = cliente.post(
        "/api/v1/auth/sesion",
        json={"correo": "nuevo@samuapp.cl", "contrasena": "ContrasenaSegura2026"},
    )
    assert sesion.status_code == 200, sesion.text


def test_registro_con_correo_duplicado_no_revela_la_existencia(cliente: TestClient) -> None:
    """CA-01.2: el mensaje de duplicado no anuncia que la cuenta existe."""
    respuesta = cliente.post(
        "/api/v1/auth/registro",
        json={
            "nombre": "Otro Operador",
            "correo": "operador@samuapp.cl",
            "contrasena": "ContrasenaSegura2026",
            "rol": "operador",
        },
    )

    assert respuesta.status_code == 400, respuesta.text
    cuerpo = respuesta.json()
    assert cuerpo["error"]["code"] == "invalid-argument"

    mensaje = cuerpo["error"]["message"].lower()
    for pista in ("existe", "registrado", "duplicad", "correo"):
        assert pista not in mensaje, mensaje

    # La cuenta original sigue intacta: el duplicado no la modifico.
    sesion = cliente.post(
        "/api/v1/auth/sesion",
        json={"correo": "operador@samuapp.cl", "contrasena": "ContrasenaSegura2026"},
    )
    assert sesion.status_code == 401


# --- CA-01.3: contraseña insegura ------------------------------------------


def test_registro_con_contrasena_corta_es_rechazado(cliente: TestClient) -> None:
    """CA-01.3: una contraseña de menos de 12 caracteres no crea la cuenta."""
    respuesta = cliente.post(
        "/api/v1/auth/registro",
        json={
            "nombre": "Operador Debil",
            "correo": "debil@samuapp.cl",
            "contrasena": "Corta123",
            "rol": "operador",
        },
    )

    assert respuesta.status_code == 400, respuesta.text
    assert respuesta.json()["error"]["code"] == "invalid-argument"

    # La cuenta no se creó: no se puede iniciar sesion con ella. El esquema
    # rechaza la contraseña corta antes de consultar el almacen.
    sesion = cliente.post(
        "/api/v1/auth/sesion",
        json={"correo": "debil@samuapp.cl", "contrasena": "Corta123"},
    )
    assert sesion.status_code in (400, 401)


# --- Modo firebase: el servicio no emite tokens ------------------------------


def test_en_modo_firebase_el_inicio_de_sesion_delega_al_proveedor(
    cliente_firebase: TestClient,
) -> None:
    """Con `SAMUAPP_AUTH_MODE=firebase` el servicio no emite tokens propios."""
    respuesta = cliente_firebase.post(
        "/api/v1/auth/sesion",
        json={"correo": "operador@samuapp.cl", "contrasena": CONTRASENA_DEMO},
    )

    assert respuesta.status_code == 412, respuesta.text
    assert respuesta.json()["error"]["code"] == "failed-precondition"
    assert "Firebase" in respuesta.json()["error"]["message"]


def test_en_modo_firebase_el_registro_delega_al_proveedor(
    cliente_firebase: TestClient,
) -> None:
    """El registro tambien queda delegado a Firebase Authentication."""
    respuesta = cliente_firebase.post(
        "/api/v1/auth/registro",
        json={
            "nombre": "Nuevo Operador",
            "correo": "nuevo@samuapp.cl",
            "contrasena": "ContrasenaSegura2026",
            "rol": "operador",
        },
    )

    assert respuesta.status_code == 412, respuesta.text
    assert respuesta.json()["error"]["code"] == "failed-precondition"
