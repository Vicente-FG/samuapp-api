"""Target encoding bayesiano congelado (alpha = 50).

En el entrenamiento el target encoding se calculo sobre el dataset completo
(1.261.124 siniestros) con la fórmula:

    tasa_te(c) = (n_c * tasa_c + ALPHA * tasa_global) / (n_c + ALPHA)

En producción **no se recalcula**: seria leakage y además imposible con datos
nuevos. Se lee del artefacto congelado `encodings.parquet` (columnas `clave`,
`n_c` y `tasa_c`) junto con `tasa_global.json`.

Las claves del artefacto tienen la forma `"<tipo>|<valor>"`, por ejemplo
`"comuna|Colchane"`, `"region|Tarapacá"` y `"urbano|True"`. La comparación de
comunas y regiones ignora tildes y mayúsculas, de modo que "Tarapacá" y
"Tarapaca" resuelven a la misma entrada.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from core.features.taxonomy import _clave_comparacion

__all__ = [
    "ALPHA",
    "EntradaEncoding",
    "MetadatosTasaGlobal",
    "TablaTargetEncoding",
    "cargar_desde_parquet",
    "cargar_metadatos_tasa_global",
    "cargar_tasa_global",
    "tasa_target_encoding",
]

logger = logging.getLogger(__name__)

#: Parámetro de suavizado bayesiano del entrenamiento. No debe cambiarse sin
#: reentrenar: alteraria la paridad de las 40 características.
ALPHA = 50.0

#: Separador entre el tipo y el valor dentro de la columna `clave`.
SEPARADOR_CLAVE = "|"

#: Tipos de clave admitidos en la tabla de referencia.
TIPO_URBANO = "urbano"
TIPO_COMUNA = "comuna"
TIPO_REGION = "region"

#: Nombre de la columna que lleva la tasa global del dataset de entrenamiento.
CAMPO_TASA_GLOBAL = "tasa_global"


def tasa_target_encoding(
    n_c: float,
    tasa_c: float,
    tasa_global: float,
    alpha: float = ALPHA,
) -> float:
    """Aplica la tasa bayesiana de una categoria.

    Con `n_c = 0` el resultado es exactamente `tasa_global`, que es el
    respaldo cuando una categoria no existe en la tabla congelada.
    """
    denominador = n_c + alpha
    if denominador <= 0:
        raise ValueError("El denominador bayesiano debe ser positivo.")
    return (n_c * tasa_c + alpha * tasa_global) / denominador


@dataclass(frozen=True)
class EntradaEncoding:
    """Fila de la tabla de target encoding: conteo y tasa observada."""

    n_c: float
    tasa_c: float


@dataclass(frozen=True)
class MetadatosTasaGlobal:
    """Contenido de `tasa_global.json`."""

    tasa_global: float
    fecha: str | None = None
    artifact_version: str | None = None


def _normalizar_valor(valor: object) -> str | bool:
    """Normaliza el valor de una clave para comparar sin tildes ni caja."""
    if isinstance(valor, bool):
        return valor
    return _clave_comparacion(str(valor))


def _interpretar_clave(texto: str) -> tuple[str, str | bool]:
    """Interpreta la columna `clave` del artefacto (`"tipo|valor"`)."""
    tipo, _, valor = str(texto).partition(SEPARADOR_CLAVE)
    tipo = tipo.strip().lower()
    if tipo == TIPO_URBANO:
        return tipo, str(valor).strip().lower() in {"true", "1", "si", "yes"}
    return tipo, _normalizar_valor(valor)


@dataclass(frozen=True)
class TablaTargetEncoding:
    """Tabla congelada de target encoding más la tasa global del entrenamiento."""

    tasa_global: float
    entradas: Mapping[tuple[str, str | bool], EntradaEncoding] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        """Normaliza las claves para que la busqueda sea insensible a tildes."""
        normalizadas: dict[tuple[str, str | bool], EntradaEncoding] = {}
        for clave, entrada in self.entradas.items():
            tipo = str(clave[0]).strip().lower()
            valor = _normalizar_valor(clave[1])
            normalizadas[(tipo, valor)] = entrada
        object.__setattr__(self, "entradas", MappingProxyType(normalizadas))

    def obtener(self, clave: tuple[str, str | bool]) -> EntradaEncoding | None:
        """Devuelve la entrada de la tabla para una clave, o `None` si no existe."""
        tipo = str(clave[0]).strip().lower()
        valor = _normalizar_valor(clave[1])
        return self.entradas.get((tipo, valor))

    def tasa_de_clave(self, clave: tuple[str, str | bool]) -> float:
        """Tasa bayesiana de una clave; si no existe, la tasa global.

        En el caso ausente se registra un warning explícito: es la senal de
        que la tabla de referencia quedo desactualizada respecto de las
        comunas o regiones que captura el operador.
        """
        entrada = self.obtener(clave)
        if entrada is None:
            logger.warning(
                "Clave ausente en la tabla de target encoding; se usa la tasa global",
                extra={
                    "evento": "target_encoding_clave_ausente",
                    "clave_tipo": str(clave[0]),
                    "clave_valor": str(clave[1]),
                    "tasa_global": self.tasa_global,
                },
            )
            return self.tasa_global
        return tasa_target_encoding(entrada.n_c, entrada.tasa_c, self.tasa_global)

    def target_encode(self, comuna: str, region: str, urbano: bool) -> float:
        """Target encoding principal del dominio.

        La clave depende de la urbanidad del siniestro:

        * `urbano = True`  -> clave `("urbano", True)` (equivalente a
          `("urbano", 1)`, porque en Python `True == 1`).
        * `urbano = False` -> clave `("urbano", False)`, con respaldo en
          `("comuna", comuna)` y luego `("region", region)`.

        **Por que la clave binaria y no la de comuna.** El modelo publicado se
        entrenó con el target encoding de `Urbano_bin` (véase
        `notebooks/probar_modelos.py` del repositorio de ciencia de datos):
        para un siniestro rural se usa la tasa agregada de los siniestros
        rurales, no la de su comuna. Mantener esa clave es el requisito de
        paridad (RNF-04); el respaldo por comuna y región es el que describe el
        documento de Etapa 3 y cubre las tablas que no publiquen la clave
        binaria.
        """
        entrada_urbana = self.obtener((TIPO_URBANO, bool(urbano)))
        if entrada_urbana is not None:
            return tasa_target_encoding(
                entrada_urbana.n_c, entrada_urbana.tasa_c, self.tasa_global
            )
        if not urbano:
            entrada_comuna = self.obtener((TIPO_COMUNA, comuna))
            if entrada_comuna is not None:
                return tasa_target_encoding(
                    entrada_comuna.n_c, entrada_comuna.tasa_c, self.tasa_global
                )
        return self._tasa_con_respaldo_region(comuna, region)

    def _tasa_con_respaldo_region(self, comuna: str, region: str) -> float:
        """Aplica el respaldo regional cuando la comuna no existe en la tabla."""
        entrada = self.obtener((TIPO_REGION, region))
        if entrada is None:
            logger.warning(
                "Comuna y region ausentes en la tabla de target encoding; "
                "se usa la tasa global",
                extra={
                    "evento": "target_encoding_geografia_ausente",
                    "comuna": comuna,
                    "region": region,
                    "tasa_global": self.tasa_global,
                },
            )
            return self.tasa_global
        logger.info(
            "Comuna ausente en la tabla de target encoding; se usa el respaldo regional",
            extra={
                "evento": "target_encoding_respaldo_region",
                "comuna": comuna,
                "region": region,
            },
        )
        return tasa_target_encoding(entrada.n_c, entrada.tasa_c, self.tasa_global)

    def target_encode_comuna(self, comuna: str, region: str) -> float:
        """Target encoding de la comuna, con respaldo en la región."""
        entrada = self.obtener((TIPO_COMUNA, comuna))
        if entrada is not None:
            return tasa_target_encoding(entrada.n_c, entrada.tasa_c, self.tasa_global)
        return self._tasa_con_respaldo_region(comuna, region)

    def target_encode_region(self, region: str) -> float:
        """Target encoding de la región."""
        return self.tasa_de_clave((TIPO_REGION, region))


def cargar_metadatos_tasa_global(ruta: Path) -> MetadatosTasaGlobal:
    """Lee `tasa_global.json` y devuelve la tasa global junto a sus metadatos.

    El archivo admite, ademas de `tasa_global`, las claves opcionales
    `fecha` (AAA-MM-DD de la publicacion de la tabla) y `artifact_version`
    (version explicita de las tablas de referencia).
    """
    try:
        contenido = json.loads(Path(ruta).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"No existe el artefacto de tasa global: {ruta}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"El artefacto de tasa global no es JSON valido: {ruta}") from exc
    if not isinstance(contenido, dict) or CAMPO_TASA_GLOBAL not in contenido:
        raise ValueError(
            f"El artefacto de tasa global debe traer el campo {CAMPO_TASA_GLOBAL!r}: {ruta}"
        )
    fecha = contenido.get("fecha")
    artifact_version = contenido.get("artifact_version")
    return MetadatosTasaGlobal(
        tasa_global=float(contenido[CAMPO_TASA_GLOBAL]),
        fecha=str(fecha) if fecha else None,
        artifact_version=str(artifact_version) if artifact_version else None,
    )


def cargar_tasa_global(ruta: Path) -> float:
    """Lee `tasa_global.json` y devuelve la tasa global del entrenamiento."""
    return cargar_metadatos_tasa_global(ruta).tasa_global


def cargar_desde_parquet(ruta: Path, tasa_global: float) -> TablaTargetEncoding:
    """Carga la tabla congelada de target encoding desde `encodings.parquet`.

    Se espera un parquet con las columnas `clave` (`"tipo|valor"`), `n_c` y
    `tasa_c`. Como alternativa se acepta el formato descompuesto, con las
    columnas `tipo`, `valor`, `n_c` y `tasa_c`.
    """
    import pandas as pd  # import local: solo se necesita al leer artefactos

    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f"No existe la tabla de target encoding: {ruta}")
    tabla = pd.read_parquet(ruta)
    columnas = {str(columna).lower(): columna for columna in tabla.columns}

    if "clave" in columnas:
        pares = [
            _interpretar_clave(valor) for valor in tabla[columnas["clave"]].tolist()
        ]
    elif "tipo" in columnas and "valor" in columnas:
        pares = []
        for tipo, valor in zip(
            tabla[columnas["tipo"]].tolist(),
            tabla[columnas["valor"]].tolist(),
            strict=True,
        ):
            tipo_norm = str(tipo).strip().lower()
            if tipo_norm == TIPO_URBANO:
                valor_norm: str | bool = str(valor).strip().lower() in {
                    "true",
                    "1",
                    "si",
                    "yes",
                }
            else:
                valor_norm = _normalizar_valor(valor)
            pares.append((tipo_norm, valor_norm))
    else:
        raise ValueError(
            "La tabla de target encoding debe traer la columna 'clave' "
            "(formato 'tipo|valor') o las columnas 'tipo' y 'valor': "
            f"{ruta}"
        )

    columna_n = columnas.get("n_c")
    columna_tasa = columnas.get("tasa_c")
    if columna_n is None or columna_tasa is None:
        raise ValueError(
            f"La tabla de target encoding debe traer las columnas 'n_c' y 'tasa_c': {ruta}"
        )

    entradas: dict[tuple[str, str | bool], EntradaEncoding] = {}
    for indice, (tipo, valor) in enumerate(pares):
        n_c = float(tabla[columna_n].iloc[indice])
        tasa_c = float(tabla[columna_tasa].iloc[indice])
        entradas[(tipo, valor)] = EntradaEncoding(n_c=n_c, tasa_c=tasa_c)

    logger.info(
        "Tabla de target encoding cargada",
        extra={
            "evento": "target_encoding_cargado",
            "entradas": len(entradas),
            "tasa_global": tasa_global,
        },
    )
    return TablaTargetEncoding(tasa_global=tasa_global, entradas=entradas)
