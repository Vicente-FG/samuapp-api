"""Pruebas unitarias del target encoding bayesiano (alpha = 50)."""

from __future__ import annotations

import logging

import pytest

from core.features.encodings import (
    ALPHA,
    EntradaEncoding,
    TablaTargetEncoding,
    tasa_target_encoding,
)

#: Tasa global de ejemplo.
TASA_GLOBAL = 0.0206

TABLA = TablaTargetEncoding(
    tasa_global=TASA_GLOBAL,
    entradas={
        ("comuna", "colchane"): EntradaEncoding(n_c=10, tasa_c=0.05),
        ("comuna", "iquique"): EntradaEncoding(n_c=800, tasa_c=0.012),
        ("region", "tarapaca"): EntradaEncoding(n_c=100, tasa_c=0.03),
        ("region", "metropolitana"): EntradaEncoding(n_c=900, tasa_c=0.008),
        ("urbano", True): EntradaEncoding(n_c=500, tasa_c=0.02),
    },
)

#: Tabla con la clave binaria rural, como la publica el entrenamiento real.
TABLA_BINARIA = TablaTargetEncoding(
    tasa_global=TASA_GLOBAL,
    entradas={
        ("comuna", "colchane"): EntradaEncoding(n_c=10, tasa_c=0.05),
        ("region", "tarapaca"): EntradaEncoding(n_c=100, tasa_c=0.03),
        ("urbano", True): EntradaEncoding(n_c=500, tasa_c=0.02),
        ("urbano", False): EntradaEncoding(n_c=300, tasa_c=0.06),
    },
)


def test_alpha_es_cincuenta() -> None:
    """El parámetro de suavizado es el del entrenamiento: 50."""
    assert ALPHA == 50.0


def test_formula_bayesiana_calculada_a_mano() -> None:
    """tasa_te(c) = (n_c * tasa_c + ALPHA * tasa_global) / (n_c + ALPHA)."""
    # (100 * 0,10 + 50 * 0,02) / (100 + 50) = 11 / 150 = 0,0733333...
    esperado = (100 * 0.10 + 50 * 0.02) / (100 + 50)

    assert tasa_target_encoding(100, 0.10, 0.02) == pytest.approx(esperado)
    assert tasa_target_encoding(100, 0.10, 0.02) == pytest.approx(0.0733333, rel=1e-6)


def test_con_cero_casos_devuelve_la_tasa_global() -> None:
    """Una categoria sin observaciones colapsa a la tasa global."""
    assert tasa_target_encoding(0, 0.0, TASA_GLOBAL) == pytest.approx(TASA_GLOBAL)


def test_clave_urbano_verdadero() -> None:
    """Con `urbano = True` la clave es `("urbano", True)`."""
    tasa = TABLA.target_encode("Colchane", "Tarapacá", True)

    esperado = (500 * 0.02 + ALPHA * TASA_GLOBAL) / (500 + ALPHA)
    assert tasa == pytest.approx(esperado)


def test_clave_comuna_cuando_es_rural() -> None:
    """Sin clave binaria rural, se usa `("comuna", comuna)`."""
    sin_binario = TablaTargetEncoding(
        tasa_global=TASA_GLOBAL,
        entradas={
            ("comuna", "colchane"): EntradaEncoding(n_c=10, tasa_c=0.05),
            ("region", "tarapaca"): EntradaEncoding(n_c=100, tasa_c=0.03),
        },
    )

    tasa = sin_binario.target_encode("Colchane", "Tarapacá", False)

    esperado = (10 * 0.05 + ALPHA * TASA_GLOBAL) / (10 + ALPHA)
    assert tasa == pytest.approx(esperado)


def test_la_clave_binaria_rural_tiene_prioridad() -> None:
    """Con clave binaria publicada, manda la del entrenamiento (paridad)."""
    tasa = TABLA_BINARIA.target_encode("Colchane", "Tarapacá", False)

    esperado_binario = (300 * 0.06 + ALPHA * TASA_GLOBAL) / (300 + ALPHA)
    esperado_comuna = (10 * 0.05 + ALPHA * TASA_GLOBAL) / (10 + ALPHA)
    assert tasa == pytest.approx(esperado_binario)
    assert tasa != pytest.approx(esperado_comuna)


def test_respaldo_en_la_region_cuando_la_comuna_no_existe() -> None:
    """Si la comuna no esta en la tabla, se usa el respaldo de la región."""
    tasa = TABLA.target_encode("Pozo Almonte", "Tarapacá", False)

    esperado_region = (100 * 0.03 + ALPHA * TASA_GLOBAL) / (100 + ALPHA)
    assert tasa == pytest.approx(esperado_region)


def test_clave_inexistente_devuelve_la_tasa_global() -> None:
    """Una clave que no existe en la tabla devuelve la tasa global."""
    tasa = TABLA.target_encode("Desconocida", "Region Inventada", False)

    assert tasa == pytest.approx(TASA_GLOBAL)


def test_registra_un_warning_cuando_la_clave_no_existe(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """La ausencia de una clave queda registrada en el log."""
    with caplog.at_level(logging.WARNING):
        TABLA.target_encode("Desconocida", "Region Inventada", False)

    assert any("tasa global" in registro.message for registro in caplog.records)


def test_la_comparacion_ignora_tildes_y_mayusculas() -> None:
    """`Tarapacá`, `tarapaca` y `TARAPACA` resuelven a la misma entrada."""
    con_tilde = TABLA.target_encode_region("Tarapacá")
    sin_tilde = TABLA.target_encode_region("Tarapaca")
    mayusculas = TABLA.target_encode_region("TARAPACA")

    assert con_tilde == pytest.approx(sin_tilde)
    assert con_tilde == pytest.approx(mayusculas)
    assert con_tilde == pytest.approx(
        (100 * 0.03 + ALPHA * TASA_GLOBAL) / (100 + ALPHA)
    )


def test_tabla_vacia_devuelve_siempre_la_tasa_global() -> None:
    """Una tabla sin entradas colapsa a la tasa global."""
    tabla = TablaTargetEncoding(tasa_global=0.02, entradas={})

    assert tabla.target_encode("Colchane", "Tarapacá", False) == pytest.approx(0.02)
    assert tabla.target_encode("Colchane", "Tarapacá", True) == pytest.approx(0.02)
