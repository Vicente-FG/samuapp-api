"""Vocabulario cerrado del dominio: sexos, tipos de vehículo y franjas horarias.

El dominio tiene 18 tipos de vehículo, pero el modelo solo deriva 15
indicadores `Tiene_*`. `Maquinaria`, `TraccionAnimal` y `Otro` no tienen
indicador propio: quedan absorbidos por la categoria residual del pipeline.

Este módulo normaliza el texto de entrada para que la ficha capturada en
terreno (que escribe "Camion" con o sin tilde) no rompa la transformación.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping
from enum import Enum
from types import MappingProxyType

__all__ = [
    "CODIGOS_FRANJA_HORARIA",
    "FRANJA_HORARIA_MADRUGADA",
    "FRANJA_HORARIA_MAÑANA",
    "FRANJA_HORARIA_NOCHE",
    "FRANJA_HORARIA_TARDE",
    "INDICADORES_VEHICULO",
    "RANGOS_FRANJA_HORARIA",
    "SEXOS",
    "TIPOS_PESADOS",
    "TIPOS_VEHICULO",
    "TipoVehiculo",
    "codigo_franja_horaria",
    "etiqueta_franja_horaria",
    "normalizar_tipo_vehiculo",
]


class TipoVehiculo(str, Enum):
    """Los 18 tipos de vehículo registrados en la ficha de accidentado."""

    AUTO = "Auto"
    CAMIONETA = "Camioneta"
    FURGON = "Furgon"
    JEEP = "Jeep"
    MOTO = "Moto"
    MOTONETA = "Motoneta"
    BICICLETA = "Bicicleta"
    BUS = "Bus"
    CAMION = "Camion"
    TRACTO = "Tracto"
    TAXI = "Taxi"
    MINIBUS = "Minibus"
    AMBULANCIA = "Ambulancia"
    FUGA = "Fuga"
    TRACTOR = "Tractor"
    MAQUINARIA = "Maquinaria"
    TRACCION_ANIMAL = "TraccionAnimal"
    OTRO = "Otro"


#: Los 18 tipos de vehiculo, en el orden de la ficha de accidentado.
TIPOS_VEHICULO: tuple[TipoVehiculo, ...] = tuple(TipoVehiculo)

#: Sexos que registra Carabineros. `ND` significa "no determinado".
SEXOS: tuple[str, ...] = ("M", "F", "ND")

#: Indicador `Tiene_*` derivado para cada tipo de vehículo con soporte.
#:
#: El modelo publicado en `features_xgb.joblib` usa **18** indicadores: los 15
#: del diseño del documento de Etapa 3 más `Tiene_Maquinaria`,
#: `Tiene_TraccionAnimal` y `Tiene_Otros`. El pipeline calcula el conjunto
#: completo y después selecciona las columnas que pide el artefacto, de modo
#: que sirve tanto para el modelo real como para el orden por defecto.
INDICADORES_VEHICULO: Mapping[TipoVehiculo, str] = MappingProxyType(
    {
        TipoVehiculo.AUTO: "Tiene_Auto",
        TipoVehiculo.CAMIONETA: "Tiene_Camioneta",
        TipoVehiculo.FURGON: "Tiene_Furgon",
        TipoVehiculo.JEEP: "Tiene_Jeep",
        TipoVehiculo.MOTO: "Tiene_Moto",
        TipoVehiculo.MOTONETA: "Tiene_Motoneta",
        TipoVehiculo.BICICLETA: "Tiene_Bici",
        TipoVehiculo.BUS: "Tiene_Bus",
        TipoVehiculo.CAMION: "Tiene_Camion",
        TipoVehiculo.TRACTO: "Tiene_Tracto",
        TipoVehiculo.TAXI: "Tiene_Taxi",
        TipoVehiculo.MINIBUS: "Tiene_Minibus",
        TipoVehiculo.AMBULANCIA: "Tiene_Ambulancia",
        TipoVehiculo.FUGA: "Tiene_Fuga",
        TipoVehiculo.TRACTOR: "Tiene_Tractor",
        TipoVehiculo.MAQUINARIA: "Tiene_Maquinaria",
        TipoVehiculo.TRACCION_ANIMAL: "Tiene_TraccionAnimal",
        TipoVehiculo.OTRO: "Tiene_Otros",
    }
)

#: Tipos considerados pesados por la regla de despacho degradada (RN degradada).
#: Decision documentada: se incluyen `Camioneta` y `Tractor` porque en las rutas
#: rurales chilenas concentran masa y capacidad de lesionado comparable a un
#: vehículo mayor, y se excluye `Fuga` porque no describe masa del vehículo.
TIPOS_PESADOS: frozenset[TipoVehiculo] = frozenset(
    {
        TipoVehiculo.CAMION,
        TipoVehiculo.TRACTO,
        TipoVehiculo.BUS,
        TipoVehiculo.MINIBUS,
        TipoVehiculo.CAMIONETA,
        TipoVehiculo.MAQUINARIA,
        TipoVehiculo.TRACTOR,
    }
)

#: Etiquetas de la franja horaria.
FRANJA_HORARIA_MADRUGADA = "madrugada"
FRANJA_HORARIA_MAÑANA = "mañana"
FRANJA_HORARIA_TARDE = "tarde"
FRANJA_HORARIA_NOCHE = "noche"

#: Bandas horarias: hora inicial, hora final (inclusive) y etiqueta.
RANGOS_FRANJA_HORARIA: tuple[tuple[int, int, str], ...] = (
    (0, 5, FRANJA_HORARIA_MADRUGADA),
    (6, 11, FRANJA_HORARIA_MAÑANA),
    (12, 17, FRANJA_HORARIA_TARDE),
    (18, 23, FRANJA_HORARIA_NOCHE),
)

#: Código numérico de cada franja. El pipeline expone la etiqueta y el modelo
#: consume el código: ambos deben coincidir con el entrenamiento (ver README,
#: seccion "Que falta para el modelo real").
CODIGOS_FRANJA_HORARIA: Mapping[str, int] = MappingProxyType(
    {
        FRANJA_HORARIA_MADRUGADA: 0,
        FRANJA_HORARIA_MAÑANA: 1,
        FRANJA_HORARIA_TARDE: 2,
        FRANJA_HORARIA_NOCHE: 3,
    }
)


def etiqueta_franja_horaria(hora: int) -> str:
    """Devuelve la etiqueta de franja horaria para una hora en [0, 23]."""
    for inicio, fin, etiqueta in RANGOS_FRANJA_HORARIA:
        if inicio <= hora <= fin:
            return etiqueta
    raise ValueError(f"La hora debe estar entre 0 y 23: {hora!r}")


def codigo_franja_horaria(etiqueta: str) -> int:
    """Devuelve el código numérico de una etiqueta de franja horaria."""
    return CODIGOS_FRANJA_HORARIA[etiqueta]


def _sin_tildes(texto: str) -> str:
    """Elimina diacriticos manteniendo el resto del texto intacto."""
    descompuesto = unicodedata.normalize("NFD", texto)
    return "".join(c for c in descompuesto if unicodedata.category(c) != "Mn")


def _clave_comparacion(texto: str) -> str:
    """Normaliza un texto a una clave de busqueda: sin tildes, sin espacios extra."""
    sin_tildes = _sin_tildes(texto).strip().lower()
    return " ".join(sin_tildes.split())


#: Alias aceptados por la normalización, para las variantes usuales en terreno.
_ALIAS_TIPOS: Mapping[str, TipoVehiculo] = MappingProxyType(
    {
        "automovil": TipoVehiculo.AUTO,
        "vehiculo particular": TipoVehiculo.AUTO,
        "motocicleta": TipoVehiculo.MOTO,
        "moto": TipoVehiculo.MOTO,
        "camion": TipoVehiculo.CAMION,
        "camioneta": TipoVehiculo.CAMIONETA,
        "furgon": TipoVehiculo.FURGON,
        "furgoneta": TipoVehiculo.FURGON,
        "tracto": TipoVehiculo.TRACTO,
        "tractocamion": TipoVehiculo.TRACTO,
        "tractor": TipoVehiculo.TRACTOR,
        "bus": TipoVehiculo.BUS,
        "taxi": TipoVehiculo.TAXI,
        "minibus": TipoVehiculo.MINIBUS,
        "microbus": TipoVehiculo.MINIBUS,
        "ambulancia": TipoVehiculo.AMBULANCIA,
        "fuga": TipoVehiculo.FUGA,
        "bicicleta": TipoVehiculo.BICICLETA,
        "maquinaria": TipoVehiculo.MAQUINARIA,
        "traccion animal": TipoVehiculo.TRACCION_ANIMAL,
        "traccionanimal": TipoVehiculo.TRACCION_ANIMAL,
        "otro": TipoVehiculo.OTRO,
        "otros": TipoVehiculo.OTRO,
    }
)


def _construir_mapa() -> dict[str, TipoVehiculo]:
    mapa: dict[str, TipoVehiculo] = {}
    for tipo in TipoVehiculo:
        mapa[_clave_comparacion(tipo.value)] = tipo
    mapa.update(_ALIAS_TIPOS)
    return mapa


#: Mapa de claves normalizadas a tipos de vehículo.
_MAPA_TIPOS: Mapping[str, TipoVehiculo] = MappingProxyType(_construir_mapa())


def normalizar_tipo_vehiculo(valor: str) -> TipoVehiculo | None:
    """Normaliza el nombre de un tipo de vehículo; `None` si no es valido.

    Acepta las variantes con y sin tilde (`"Camión"` y `"Camion"` son el mismo
    tipo), ignora mayúsculas y espacios sobrantes, y reconoce los alias
    frecuentes de la ficha de accidentado.
    """
    if not isinstance(valor, str):
        return None
    clave = _clave_comparacion(valor)
    if not clave:
        return None
    directo = _MAPA_TIPOS.get(clave)
    if directo is not None:
        return directo
    return _MAPA_TIPOS.get(clave.replace(" ", ""))
