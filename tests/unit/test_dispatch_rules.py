"""Pruebas unitarias del motor de reglas de despacho (RN-01..RN-08)."""

from __future__ import annotations

import pytest

from core.contracts import Involucrado, NivelAlerta, Siniestro
from core.rules.dispatch_rules import (
    estimar_riesgo_degradado,
    puntaje_degradado,
    recomendacion_degradada,
    recomendacion_modelo,
)


def _siniestro(
    hora: int = 12,
    urbano: bool = True,
    tipos: tuple[str, ...] = ("Auto",),
    edades: tuple[int, ...] = (35,),
) -> Siniestro:
    """Construye un siniestro de pruebas con los Involvement indicados."""
    involucrados = [
        Involucrado(sexo="M", edad=edad, tipo=tipo)
        for tipo, edad in zip(tipos, edades, strict=True)
    ]
    return Siniestro(
        fecha="2026-11-20",
        hora=hora,
        comuna="Colchane",
        region="Tarapacá",
        urbano=urbano,
        involucrados=involucrados,
    )


# --- modo degradado ---------------------------------------------------------


def test_modo_degradado_umbral_de_alerta_hospitalaria() -> None:
    """Puntaje >= 5: tres ambulancias, paramédico y alerta hospitalaria."""
    siniestro = _siniestro(
        hora=3, urbano=False, tipos=("Camion", "Auto"), edades=(30, 40)
    )

    assert puntaje_degradado(siniestro) >= 5
    recomendacion = recomendacion_degradada(siniestro)

    assert recomendacion.ambulancias == 3
    assert recomendacion.paramedico is True
    assert recomendacion.helicoptero is False
    assert recomendacion.alerta_hospital is True
    assert recomendacion.nivel_alerta is NivelAlerta.ALERTA_HOSPITAL
    assert recomendacion.fundamento == (
        "Modo degradado: heurística por geografía, franja horaria y tipo de vehículo."
    )


def test_modo_degradado_umbral_de_despacho_reforzado() -> None:
    """Puntaje >= 3: dos ambulancias y despacho reforzado."""
    siniestro = _siniestro(hora=3, urbano=True, tipos=("Auto",), edades=(80,))

    assert puntaje_degradado(siniestro) == 3
    recomendacion = recomendacion_degradada(siniestro)

    assert recomendacion.ambulancias == 2
    assert recomendacion.paramedico is True
    assert recomendacion.helicoptero is False
    assert recomendacion.alerta_hospital is False
    assert recomendacion.nivel_alerta is NivelAlerta.REFORZADO
    assert recomendacion.fundamento == (
        "Modo degradado: heurística con puntaje de riesgo intermedio."
    )


def test_modo_degradado_sin_factores_de_riesgo() -> None:
    """Puntaje bajo: una ambulancia y despacho estándar."""
    siniestro = _siniestro(hora=12, urbano=True, tipos=("Auto",), edades=(35,))

    assert puntaje_degradado(siniestro) == 0
    recomendacion = recomendacion_degradada(siniestro)

    assert recomendacion.ambulancias == 1
    assert recomendacion.paramedico is False
    assert recomendacion.helicoptero is False
    assert recomendacion.alerta_hospital is False
    assert recomendacion.nivel_alerta is NivelAlerta.ESTANDAR
    assert recomendacion.fundamento == (
        "Modo degradado: sin factores de riesgo identificados."
    )


def test_modo_degradado_umbral_de_veinticuatro_horas() -> None:
    """El rango de evening (18-21) aporta un punto al puntaje."""
    diurna = _siniestro(hora=12, urbano=True, tipos=("Auto",), edades=(35,))
    vespertina = _siniestro(hora=19, urbano=True, tipos=("Auto",), edades=(35,))

    assert puntaje_degradado(diurna) == 0
    assert puntaje_degradado(vespertina) == 1
    assert recomendacion_degradada(vespertina).ambulancias == 1


def test_heuristica_degradada_es_conservadora() -> None:
    """La heurística de riesgo no supera los topes y crece con el puntaje."""
    escenario_bajo = _siniestro(hora=12, urbano=True, tipos=("Auto",), edades=(35,))
    escenario_alto = _siniestro(
        hora=3, urbano=False, tipos=("Camion", "Bus"), edades=(80, 30)
    )

    p_bajo, vg_bajo = estimar_riesgo_degradado(escenario_bajo)
    p_alto, vg_alto = estimar_riesgo_degradado(escenario_alto)

    assert 0.0 < p_bajo < p_alto <= 0.60
    assert 0.0 < vg_bajo < vg_alto <= 3.0


# --- recomendacion del modelo ----------------------------------------------


def test_recomendacion_del_modelo_reproduce_el_ejemplo_del_contrato() -> None:
    """P(fatal) 41,2 % y VG 2,08 reproducen el ejemplo publicado en §4.9."""
    recomendacion = recomendacion_modelo(0.412, 2.08)

    assert recomendacion.ambulancias == 3
    assert recomendacion.paramedico is True
    assert recomendacion.helicoptero is False
    assert recomendacion.alerta_hospital is True
    assert recomendacion.nivel_alerta is NivelAlerta.ALERTA_HOSPITAL
    assert recomendacion.fundamento == (
        "P(fatal) 41,2 % (30–60 %) y 2,1 víctimas graves estimadas (1,0–3,0)."
    )


def test_rn08_el_minimo_de_ambulancias_es_uno() -> None:
    """RN-08: el mínimo de ambulancias es siempre 1."""
    recomendacion = recomendacion_modelo(0.01, 0.0)

    assert recomendacion.ambulancias == 1
    assert recomendacion.nivel_alerta is NivelAlerta.ESTANDAR
    assert recomendacion.paramedico is False


def test_rn01_ambulancias_segun_victimas_graves() -> None:
    """RN-01: `ambulancias = max(1, floor(VG * 1,5))`."""
    recomendacion = recomendacion_modelo(0.65, 4.0)

    # floor(4,0 * 1,5) = 6, y la banda máxima exige más de una ambulancia.
    assert recomendacion.ambulancias == 6
    assert recomendacion.helicoptero is True
    assert recomendacion.nivel_alerta is NivelAlerta.ALERTA_HOSPITAL


def test_banda_reforzada_por_probabilidad_de_fatalidad() -> None:
    """Una fatalidad entre 10 % y 30 % exige despacho reforzado."""
    recomendacion = recomendacion_modelo(0.20, 0.5)

    assert recomendacion.nivel_alerta is NivelAlerta.REFORZADO
    assert recomendacion.paramedico is True
    assert recomendacion.ambulancias == 1


def test_alerta_hospitalaria_por_victimas_graves() -> None:
    """RN-03: con más de 0,5 victimas graves se alerta al hospital."""
    recomendacion = recomendacion_modelo(0.35, 1.2)

    assert recomendacion.alerta_hospital is True
    assert recomendacion.nivel_alerta is NivelAlerta.ALERTA_HOSPITAL
    assert recomendacion.ambulancias == 3


@pytest.mark.parametrize(
    ("p_fatal", "victimas", "esperado"),
    [
        (0.412, 2.08, 3),
        (0.05, 0.10, 1),
        (0.20, 0.50, 1),
        (0.65, 4.00, 6),
    ],
)
def test_ambulancias_son_consistentes(
    p_fatal: float, victimas: float, esperado: int
) -> None:
    """El número de ambulancias es estable frente a los casos de la tabla."""
    assert recomendacion_modelo(p_fatal, victimas).ambulancias == esperado
