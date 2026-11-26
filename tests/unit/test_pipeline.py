"""Pruebas unitarias del pipeline de 40 características."""

from __future__ import annotations

import pytest

from core.contracts import Involucrado, PayloadPrediccion, Siniestro
from core.features.encodings import EntradaEncoding, TablaTargetEncoding
from core.features.pipeline import ORDEN_POR_DEFECTO, FeatureTransformer
from core.features.schema import EDAD_MAYOR, validar_siniestro
from core.features.taxonomy import (
    INDICADORES_VEHICULO,
    TipoVehiculo,
    normalizar_tipo_vehiculo,
)

#: Tabla de referencia mínima y determinista para las pruebas.
TABLA = TablaTargetEncoding(
    tasa_global=0.0206,
    entradas={
        ("comuna", "colchane"): EntradaEncoding(n_c=10, tasa_c=0.05),
        ("region", "tarapaca"): EntradaEncoding(n_c=100, tasa_c=0.03),
        ("urbano", True): EntradaEncoding(n_c=500, tasa_c=0.02),
    },
)


def _transformador(orden: tuple[str, ...] = ORDEN_POR_DEFECTO) -> FeatureTransformer:
    """Construye el transformador con la tabla de referencia de pruebas."""
    return FeatureTransformer(referencias=TABLA, orden_columnas=orden)


#: Orden real de columnas publicado en `features_xgb.joblib` por el equipo que
#: entreno los modelos sobre los 1.261.124 siniestros.
ORDEN_REAL_PUBLICADO: tuple[str, ...] = (
    "Hora_num",
    "Fin_de_semana",
    "Mixto_genero",
    "Tiene_Auto",
    "Tiene_Camioneta",
    "Tiene_Furgon",
    "Tiene_Jeep",
    "Tiene_Moto",
    "Tiene_Motoneta",
    "Tiene_Bici",
    "Tiene_Bus",
    "Tiene_Camion",
    "Tiene_Tracto",
    "Tiene_Taxi",
    "Tiene_Minibus",
    "Tiene_Ambulancia",
    "Tiene_Fuga",
    "Tiene_Tractor",
    "Tiene_Maquinaria",
    "Tiene_TraccionAnimal",
    "Tiene_Otros",
    "Cant_Involucrados_real",
    "N_hombres",
    "Prop_hombres",
    "N_mujeres",
    "Prop_mujeres",
    "Diversidad_vehiculos",
    "Diversidad_vehiculos_distintos",
    "Edad_promedio",
    "Edad_min",
    "Edad_max",
    "Tiene_menor",
    "Tiene_mayor",
    "Anio",
    "Mes_num",
    "Dia_semana",
    "Urbano_bin",
    "Comuna_target_enc",
    "Region_target_enc",
    "Urbano_target_enc",
)


def _siniestro_ejemplo() -> Siniestro:
    """Siniestro del ejemplo del contrato de la API."""
    return Siniestro(
        fecha="2026-11-20",
        hora=3,
        comuna="Colchane",
        region="Tarapacá",
        urbano=False,
        involucrados=[
            Involucrado(sexo="M", edad=34, tipo="Camioneta"),
            Involucrado(sexo="M", edad=28, tipo="Camión"),
            Involucrado(sexo="F", edad=61, tipo="Auto"),
        ],
    )


def test_produce_exactamente_cuarenta_caracteristicas() -> None:
    """El pipeline produce 40 características con los nombres esperados."""
    caracteristicas = _transformador().transformar(_siniestro_ejemplo())

    assert len(caracteristicas) == 40
    assert len(ORDEN_POR_DEFECTO) == 40
    assert tuple(caracteristicas) == ORDEN_POR_DEFECTO
    assert len(set(ORDEN_POR_DEFECTO)) == 40


def test_los_nombres_coinciden_con_la_tabla_de_contrato() -> None:
    """Los nombres coinciden exactamente con la tabla de 40 características."""
    esperados = (
        "Anio",
        "Mes_num",
        "Dia_semana",
        "Hora_num",
        "Fin_de_semana",
        "Franja_horaria",
        "Cant_Involucrados_real",
        "N_hombres",
        "N_mujeres",
        "Prop_hombres",
        "Prop_mujeres",
        "Mixto_genero",
        "Tiene_hombres",
        "Tiene_mujeres",
        "Tiene_Auto",
        "Tiene_Camioneta",
        "Tiene_Furgon",
        "Tiene_Jeep",
        "Tiene_Moto",
        "Tiene_Motoneta",
        "Tiene_Bici",
        "Tiene_Bus",
        "Tiene_Camion",
        "Tiene_Tracto",
        "Tiene_Taxi",
        "Tiene_Minibus",
        "Tiene_Ambulancia",
        "Tiene_Fuga",
        "Tiene_Tractor",
        "Urbano_bin",
        "Urbano_target_enc",
        "Comuna_target_enc",
        "Region_target_enc",
        "Edad_promedio",
        "Edad_min",
        "Edad_max",
        "Tiene_menor",
        "Tiene_mayor",
        "Diversidad_vehiculos",
        "Diversidad_vehiculos_distintos",
    )

    assert ORDEN_POR_DEFECTO == esperados


def test_valores_de_un_caso_concreto() -> None:
    """Los valores del caso de ejemplo son los del documento de Etapa 3."""
    caracteristicas = _transformador().transformar(_siniestro_ejemplo())

    # Temporales: 2026-11-20 es jueves (día de la semana 4).
    assert caracteristicas["Anio"] == 2026.0
    assert caracteristicas["Mes_num"] == 11.0
    assert caracteristicas["Dia_semana"] == 4.0
    assert caracteristicas["Hora_num"] == 3.0
    assert caracteristicas["Fin_de_semana"] == 0.0
    assert caracteristicas["Franja_horaria"] == "madrugada"

    # Composición: 3 involucrados, 2 hombres y 1 mujer.
    assert caracteristicas["Cant_Involucrados_real"] == 3.0
    assert caracteristicas["N_hombres"] == 2.0
    assert caracteristicas["N_mujeres"] == 1.0
    assert caracteristicas["Prop_hombres"] == pytest.approx(2 / 3)
    assert caracteristicas["Prop_mujeres"] == pytest.approx(1 / 3)
    assert caracteristicas["Mixto_genero"] == 1.0
    assert caracteristicas["Tiene_hombres"] == 1.0
    assert caracteristicas["Tiene_mujeres"] == 1.0

    # Vehículos: camioneta, camion (con tilde) y auto.
    assert caracteristicas["Tiene_Camioneta"] == 1.0
    assert caracteristicas["Tiene_Camion"] == 1.0
    assert caracteristicas["Tiene_Auto"] == 1.0
    assert caracteristicas["Tiene_Bus"] == 0.0
    assert caracteristicas["Tiene_Bici"] == 0.0

    # Geográficas: el siniestro es rural.
    assert caracteristicas["Urbano_bin"] == 0.0
    assert caracteristicas["Urbano_target_enc"] == pytest.approx(
        (10 * 0.05 + 50 * 0.0206) / (10 + 50)
    )
    assert caracteristicas["Comuna_target_enc"] == pytest.approx(
        (10 * 0.05 + 50 * 0.0206) / (10 + 50)
    )
    assert caracteristicas["Region_target_enc"] == pytest.approx(
        (100 * 0.03 + 50 * 0.0206) / (100 + 50)
    )

    # Edad y severidad: el umbral de `Tiene_mayor` es 60, como en el
    # entrenamiento del modelo publicado (la involucrada de 61 anos lo activa).
    assert caracteristicas["Edad_promedio"] == pytest.approx(41.0)
    assert caracteristicas["Edad_min"] == 28.0
    assert caracteristicas["Edad_max"] == 61.0
    assert caracteristicas["Tiene_menor"] == 0.0
    assert caracteristicas["Tiene_mayor"] == 1.0
    assert caracteristicas["Diversidad_vehiculos"] == 3.0
    assert caracteristicas["Diversidad_vehiculos_distintos"] == 3.0


def test_normaliza_el_tilde_de_camion() -> None:
    """`Camión` y `Camion` son el mismo tipo de vehículo."""
    assert normalizar_tipo_vehiculo("Camión") is TipoVehiculo.CAMION
    assert normalizar_tipo_vehiculo("Camion") is TipoVehiculo.CAMION
    assert normalizar_tipo_vehiculo("camión") is TipoVehiculo.CAMION
    assert normalizar_tipo_vehiculo("CAMIÓN") is TipoVehiculo.CAMION

    siniestro = Siniestro(
        fecha="2026-11-20",
        hora=12,
        comuna="Iquique",
        region="Tarapacá",
        urbano=True,
        involucrados=[Involucrado(sexo="M", edad=40, tipo="Camión")],
    )
    caracteristicas = _transformador().transformar(siniestro)

    assert caracteristicas["Tiene_Camion"] == 1.0
    assert caracteristicas["Tiene_Camioneta"] == 0.0
    assert caracteristicas["Diversidad_vehiculos"] == 1.0


def test_el_payload_acepta_camion_con_tilde() -> None:
    """El contrato acepta `Camión` tal como lo publica el documento."""
    payload = PayloadPrediccion(
        data={
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
        notas="Volcamiento en ruta",
    )

    assert payload.data.involucrados[1].tipo is TipoVehiculo.CAMION
    assert payload.notas == "Volcamiento en ruta"


def test_rechaza_cero_involucrados() -> None:
    """Sin involucrados no hay severity: la transformación no es definible."""
    siniestro = Siniestro(
        fecha="2026-11-20",
        hora=3,
        comuna="Colchane",
        region="Tarapacá",
        urbano=False,
        involucrados=[],
    )

    errores = validar_siniestro(siniestro)

    assert "involucrados" in errores
    assert "al menos" in errores["involucrados"]


def test_rechaza_rangos_fuera_de_limite() -> None:
    """La hora y la edad tienen rangos cerrados."""
    siniestro = Siniestro(
        fecha="2026-11-20",
        hora=25,
        comuna="Colchane",
        region="Tarapacá",
        urbano=False,
        involucrados=[Involucrado(sexo="M", edad=120, tipo="Auto")],
    )

    errores = validar_siniestro(siniestro)

    assert "hora" in errores
    assert "involucrados[0].edad" in errores


def test_el_orden_publicado_manda_sobre_el_orden_por_defecto() -> None:
    """Si el artefacto publica otro orden, ese orden manda."""
    orden_publicado = ("Edad_max",) + tuple(
        nombre for nombre in ORDEN_POR_DEFECTO if nombre != "Edad_max"
    )
    caracteristicas = _transformador(orden_publicado).transformar(_siniestro_ejemplo())

    assert tuple(caracteristicas) == orden_publicado
    assert caracteristicas["Edad_max"] == 61.0


def test_soporta_el_orden_real_del_modelo_publicado() -> None:
    """El orden real de `features_xgb.joblib` (40 columnas) es aceptado."""
    orden_real = ORDEN_REAL_PUBLICADO
    caracteristicas = _transformador(orden_real).transformar(_siniestro_ejemplo())

    assert len(orden_real) == 40
    assert tuple(caracteristicas) == orden_real
    assert "Franja_horaria" not in caracteristicas
    assert caracteristicas["Tiene_Camion"] == 1.0
    assert caracteristicas["Tiene_Maquinaria"] == 0.0
    assert caracteristicas["Tiene_TraccionAnimal"] == 0.0
    assert caracteristicas["Tiene_Otros"] == 0.0


def test_los_indicadores_de_vehiculo_son_dieciocho() -> None:
    """El pipeline calcula los 18 indicadores `Tiene_*` del modelo real."""
    siniestro = Siniestro(
        fecha="2026-11-20",
        hora=12,
        comuna="Iquique",
        region="Tarapacá",
        urbano=True,
        involucrados=[
            Involucrado(sexo="M", edad=40, tipo="Maquinaria"),
            Involucrado(sexo="F", edad=30, tipo="TraccionAnimal"),
            Involucrado(sexo="ND", edad=20, tipo="Otro"),
        ],
    )

    caracteristicas = _transformador(ORDEN_REAL_PUBLICADO).transformar(siniestro)

    assert caracteristicas["Tiene_Maquinaria"] == 1.0
    assert caracteristicas["Tiene_TraccionAnimal"] == 1.0
    assert caracteristicas["Tiene_Otros"] == 1.0
    assert caracteristicas["Tiene_Auto"] == 0.0
    assert len(INDICADORES_VEHICULO) == 18


def test_tiene_mayor_usa_el_umbral_del_entrenamiento() -> None:
    """El umbral es 60 anos, como en el modelo publicado."""
    sixty = Siniestro(
        fecha="2026-11-20",
        hora=12,
        comuna="Iquique",
        region="Tarapacá",
        urbano=True,
        involucrados=[Involucrado(sexo="M", edad=60, tipo="Auto")],
    )
    cincuenta_y_nueve = Siniestro(
        fecha="2026-11-20",
        hora=12,
        comuna="Iquique",
        region="Tarapacá",
        urbano=True,
        involucrados=[Involucrado(sexo="M", edad=59, tipo="Auto")],
    )

    assert _transformador().transformar(sixty)["Tiene_mayor"] == 1.0
    assert _transformador().transformar(cincuenta_y_nueve)["Tiene_mayor"] == 0.0
    assert EDAD_MAYOR == 60
