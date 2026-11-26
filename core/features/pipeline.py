"""Transformación del siniestro a las 40 características del modelo.

La paridad exacta con el entrenamiento es un requisito crítico (RNF-04): este
módulo es la única fuente de verdad de los nombres, el orden y las fórmulas de
las características.

Grupos (40 en total):

* Temporales (6): `Anio`, `Mes_num`, `Dia_semana`, `Hora_num`, `Fin_de_semana`,
  `Franja_horaria`.
* Composición del siniestro (8): `Cant_Involucrados_real`, `N_hombres`,
  `N_mujeres`, `Prop_hombres`, `Prop_mujeres`, `Mixto_genero`,
  `Tiene_hombres`, `Tiene_mujeres`.
* Tipos de vehículo (15): los indicadores `Tiene_*`.
* Geográficas (4): `Urbano_bin` y los tres target encoding.
* Edad y severidad (7): `Edad_promedio`, `Edad_min`, `Edad_max`,
  `Tiene_menor`, `Tiene_mayor`, `Diversidad_vehiculos` y
  `Diversidad_vehiculos_distintos`.

El orden por defecto es el de la tabla anterior. Si el artefacto
`features_xgb.joblib` trae el orden real de columnas del entrenamiento, ese
orden manda: es el requisito de paridad.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from core.contracts import Involucrado, Siniestro
from core.features.encodings import TablaTargetEncoding
from core.features.schema import EDAD_MAYOR, EDAD_MENOR
from core.features.taxonomy import (
    CODIGOS_FRANJA_HORARIA,
    INDICADORES_VEHICULO,
    TipoVehiculo,
    etiqueta_franja_horaria,
)

__all__ = [
    "ORDEN_POR_DEFECTO",
    "FeatureTransformer",
    "NOMBRES_CARACTERISTICAS",
    "valor_numerico",
]

#: Valor de una característica: numérica, salvo la etiqueta de franja horaria.
ValorCaracteristica = float | str


def valor_numerico(nombre: str, valor: ValorCaracteristica) -> float:
    """Convierte una característica a su valor numérico para el modelo.

    La única característica no numérica es `Franja_horaria`, que el pipeline
    expone como etiqueta y el modelo consume como código 0..3.
    """
    if isinstance(valor, str):
        if valor not in CODIGOS_FRANJA_HORARIA:
            raise ValueError(f"La caracteristica {nombre!r} no es numerica: {valor!r}")
        return float(CODIGOS_FRANJA_HORARIA[valor])
    return float(valor)

#: Nombres de las 40 características, en el orden por defecto.
ORDEN_POR_DEFECTO: tuple[str, ...] = (
    # Temporales (6)
    "Anio",
    "Mes_num",
    "Dia_semana",
    "Hora_num",
    "Fin_de_semana",
    "Franja_horaria",
    # Composición del siniestro (8)
    "Cant_Involucrados_real",
    "N_hombres",
    "N_mujeres",
    "Prop_hombres",
    "Prop_mujeres",
    "Mixto_genero",
    "Tiene_hombres",
    "Tiene_mujeres",
    # Tipos de vehículo (15)
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
    # Geográficas (4)
    "Urbano_bin",
    "Urbano_target_enc",
    "Comuna_target_enc",
    "Region_target_enc",
    # Edad y severidad (7)
    "Edad_promedio",
    "Edad_min",
    "Edad_max",
    "Tiene_menor",
    "Tiene_mayor",
    "Diversidad_vehiculos",
    "Diversidad_vehiculos_distintos",
)

#: Alias de solo lectura del orden por defecto.
NOMBRES_CARACTERISTICAS: tuple[str, ...] = ORDEN_POR_DEFECTO

#: Columnas adicionales que el pipeline calcula y que solo aparecen cuando el
#: artefacto `features_xgb.joblib` las publica.
#:
#: El modelo entrenado por el equipo de ciencia de datos usa 18 indicadores de
#: tipo de vehiculo (`Tiene_Maquinaria`, `Tiene_TraccionAnimal` y
#: `Tiene_Otros` ademas de los 15 del diseno) y no usa `Franja_horaria` ni
#: `Tiene_hombres`/`Tiene_mujeres`. El pipeline calcula el conjunto completo y
#: despues selecciona las columnas que pide el artefacto, de modo que el mismo
#: codigo sirve para el modelo real y para el orden por defecto.
COLUMNAS_ADICIONALES: tuple[str, ...] = (
    "Tiene_Maquinaria",
    "Tiene_TraccionAnimal",
    "Tiene_Otros",
)

#: Todas las caracteristicas que el pipeline es capaz de calcular (43).
CARACTERISTICAS_DISPONIBLES: tuple[str, ...] = ORDEN_POR_DEFECTO + COLUMNAS_ADICIONALES

#: Días de la semana considered fin de semana (lunes = 0).
DIAS_FIN_DE_SEMANA = frozenset({5, 6})


@dataclass(frozen=True)
class FeatureTransformer:
    """Convierte un siniestro en el vector de 40 características del modelo.

    Recibe la tabla de target encoding congelada y, opcionalmente, el orden
    real de columnas publicado en `features_xgb.joblib`.
    """

    referencias: TablaTargetEncoding
    orden_columnas: tuple[str, ...] = ORDEN_POR_DEFECTO

    def __post_init__(self) -> None:
        """Valida que el orden publicado use solo características calculables."""
        desconocidas = sorted(set(self.orden_columnas) - set(CARACTERISTICAS_DISPONIBLES))
        if desconocidas:
            raise ValueError(
                "El orden de columnas publicado contiene características que el "
                f"pipeline no produce: {desconocidas}"
            )
        if len(set(self.orden_columnas)) != len(self.orden_columnas):
            raise ValueError("El orden de columnas publicado tiene columnas duplicadas.")
        if not self.orden_columnas:
            raise ValueError("El orden de columnas publicado está vacío.")
        object.__setattr__(self, "orden_columnas", tuple(self.orden_columnas))

    @staticmethod
    def _tipos_de(involucrados: list[Involucrado]) -> list[TipoVehiculo]:
        """Tipos de vehículo declarados, en el orden de los involucrados."""
        return [inv.tipo for inv in involucrados if inv.tipo]

    def transformar(self, siniestro: Siniestro) -> dict[str, ValorCaracteristica]:
        """Deriva las 40 características de un siniestro.

        El diccionario devuelto mantiene el orden de `orden_columnas`.
        """
        fecha = siniestro.fecha
        involucrados = siniestro.involucrados
        total = len(involucrados)
        tipos = self._tipos_de(involucrados)

        hombres = sum(1 for inv in involucrados if inv.sexo.value == "M")
        mujeres = sum(1 for inv in involucrados if inv.sexo.value == "F")
        edades = [inv.edad for inv in involucrados]

        caracteristicas: dict[str, ValorCaracteristica] = {
            # Temporales
            "Anio": float(fecha.year),
            "Mes_num": float(fecha.month),
            "Dia_semana": float(fecha.weekday()),
            "Hora_num": float(siniestro.hora),
            "Fin_de_semana": 1.0 if fecha.weekday() in DIAS_FIN_DE_SEMANA else 0.0,
            "Franja_horaria": etiqueta_franja_horaria(siniestro.hora),
            # Composición del siniestro
            "Cant_Involucrados_real": float(total),
            "N_hombres": float(hombres),
            "N_mujeres": float(mujeres),
            "Prop_hombres": (hombres / total) if total else 0.0,
            "Prop_mujeres": (mujeres / total) if total else 0.0,
            "Mixto_genero": 1.0 if (hombres > 0 and mujeres > 0) else 0.0,
            "Tiene_hombres": 1.0 if hombres > 0 else 0.0,
            "Tiene_mujeres": 1.0 if mujeres > 0 else 0.0,
            # Geográficas
            "Urbano_bin": 1.0 if siniestro.urbano else 0.0,
            "Urbano_target_enc": self.referencias.target_encode(
                siniestro.comuna, siniestro.region, siniestro.urbano
            ),
            "Comuna_target_enc": self.referencias.target_encode_comuna(
                siniestro.comuna, siniestro.region
            ),
            "Region_target_enc": self.referencias.target_encode_region(
                siniestro.region
            ),
            # Edad y severidad
            "Edad_promedio": (sum(edades) / len(edades)) if edades else 0.0,
            "Edad_min": float(min(edades)) if edades else 0.0,
            "Edad_max": float(max(edades)) if edades else 0.0,
            "Tiene_menor": 1.0 if any(e < EDAD_MENOR for e in edades) else 0.0,
            "Tiene_mayor": 1.0 if any(e >= EDAD_MAYOR for e in edades) else 0.0,
            "Diversidad_vehiculos": float(len(tipos)),
            "Diversidad_vehiculos_distintos": float(len(set(tipos))),
        }

        # Tipos de vehículo: 15 indicadores. `Maquinaria`, `TraccionAnimal` y
        # `Otro` no tienen indicador propio y no aportan columna.
        for tipo, indicador in INDICADORES_VEHICULO.items():
            caracteristicas[indicador] = 1.0 if tipo in tipos else 0.0

        return {
            nombre: caracteristicas[nombre] for nombre in self.orden_columnas
        }

    def vector(self, caracteristicas: Mapping[str, ValorCaracteristica]) -> list[float]:
        """Vector numérico en el orden de `orden_columnas`."""
        return [
            valor_numerico(nombre, caracteristicas[nombre])
            for nombre in self.orden_columnas
        ]
