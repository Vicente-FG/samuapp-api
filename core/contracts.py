"""Contratos de dominio de SAMUAPP: DTOs de entrada y salida (Pydantic v2).

Este módulo define la frontera publica del sistema:

* Los DTOs de **entrada** (`PayloadPrediccion`, `Siniestro`, `Involucrado`)
  describen el único payload aceptado por `POST /api/v1/predictions`.
* Los DTOs de **salida** reproducen exactamente las claves que consume la SPA
  (`predictionId`, `pFatal`, `victimasGravesPred`, `nivelAlerta`, ...), que son
  el contrato funcional del proyecto.

No depende de FastAPI, Firebase ni de ningun proveedor de nube.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.features.taxonomy import TipoVehiculo, normalizar_tipo_vehiculo

__all__ = [
    "ConfirmacionEntregaPayload",
    "DispatchRecommendation",
    "EstadoSaludPayload",
    "Involucrado",
    "ListadoPrediccionesPayload",
    "ModeloActualPayload",
    "ModoPrediccion",
    "NivelAlerta",
    "PayloadPrediccion",
    "PrediccionDetallePayload",
    "PrediccionPayload",
    "ResumenPrediccionPayload",
    "Sexo",
    "Siniestro",
]


class Sexo(str, Enum):
    """Sexo registrado por Carabineros. `ND` es "no determinado"."""

    M = "M"
    F = "F"
    ND = "ND"


class NivelAlerta(str, Enum):
    """Niveles de alerta del despacho.

    `ESTANDAR` y `REFORZADO` provienen de las reglas RN-04 y RN-05; el nivel
    `ALERTA_HOSPITAL` corresponde a RN-06 y RN-07.
    """

    ESTANDAR = "ESTANDAR"
    REFORZADO = "REFORZADO"
    ALERTA_HOSPITAL = "ALERTA_HOSPITAL"


class ModoPrediccion(str, Enum):
    """Origen de la prediccion: modelo publicado o heurística de respaldo."""

    MODELO = "modelo"
    DEGRADADO = "degradado"


class Involucrado(BaseModel):
    """Registro de un involvedor del siniestro (columnas 06-08 de la ficha)."""

    model_config = ConfigDict(populate_by_name=True)

    sexo: Sexo
    edad: int
    tipo: TipoVehiculo

    @field_validator("sexo", mode="before")
    @classmethod
    def _normalizar_sexo(cls, valor: object) -> object:
        """Acepta el sexo en cualquier caja y descarta espacios sobrantes."""
        if isinstance(valor, str):
            return valor.strip().upper()
        return valor

    @field_validator("tipo", mode="before")
    @classmethod
    def _normalizar_tipo(cls, valor: object) -> object:
        """Normaliza el tipo de vehículo ("Camión" y "Camion" son el mismo)."""
        if isinstance(valor, TipoVehiculo):
            return valor
        tipo = normalizar_tipo_vehiculo(str(valor))
        if tipo is None:
            raise ValueError(f"Tipo de vehiculo no permitido: {valor!r}")
        return tipo


class Siniestro(BaseModel):
    """Siniestro capturado: los 8 campos de mínimo duro del modelo.

    Cada campo alimenta al menos una de las 40 características; ninguno es
    redundante ni reemplazable (ver §5.4.2.1 del documento de Etapa 3).
    """

    model_config = ConfigDict(populate_by_name=True)

    fecha: date
    hora: int
    comuna: str
    region: str
    urbano: bool
    involucrados: list[Involucrado]

    @field_validator("fecha", mode="before")
    @classmethod
    def _normalizar_fecha(cls, valor: object) -> object:
        """Acepta la fecha en ISO (`YYYY-MM-DD`) y en el formato local chileno.

        El formato `DD-MM-YYYY` se acepta por comodidad del operador en
        terreno. No se admite `MM-DD-YYYY` porque es ambiguo: una fecha como
        `03-04-2026` no se puede distinguir de `04-03-2026`.
        """
        if isinstance(valor, datetime):
            return valor.date()
        if isinstance(valor, date):
            return valor
        if isinstance(valor, str):
            texto = valor.strip()
            for formato in ("%Y-%m-%d", "%d-%m-%Y"):
                try:
                    return datetime.strptime(texto, formato).date()
                except ValueError:
                    continue
            raise ValueError(
                f"La fecha debe tener el formato YYYY-MM-DD (o DD-MM-YYYY): {texto!r}"
            )
        raise ValueError(f"La fecha no es valida: {valor!r}")

    @field_validator("comuna", "region")
    @classmethod
    def _recortar_texto(cls, valor: str) -> str:
        """Recorta espacios sobrantes de los campos de texto."""
        return valor.strip()


class PayloadPrediccion(BaseModel):
    """Cuerpo de `POST /api/v1/predictions`.

    `notas` y `siniestroRef` son campos de auditoria: no intervienen en
    ninguna de las 40 características y su ausencia no impide la prediccion.
    """

    model_config = ConfigDict(populate_by_name=True)

    data: Siniestro
    notas: str | None = None
    siniestro_ref: str | None = Field(default=None, alias="siniestroRef")

    @field_validator("notas", "siniestro_ref")
    @classmethod
    def _recortar_opcionales(cls, valor: str | None) -> str | None:
        """Convierte textos vacios en `None` y recorta los sobrantes."""
        if valor is None:
            return None
        texto = valor.strip()
        return texto or None


class DispatchRecommendation(BaseModel):
    """Recomendacion de recursos de despacho derivada de la prediccion.

    Es la salida del motor de reglas RN-01..RN-08 y también la salida de la
    heurística de modo degradado.
    """

    model_config = ConfigDict(populate_by_name=True)

    ambulancias: int
    paramedico: bool
    helicoptero: bool
    alerta_hospital: bool
    fundamento: str
    nivel_alerta: NivelAlerta


class RecomendacionPayload(BaseModel):
    """Bloque `recomendacion` de la respuesta publica (contrato de la SPA)."""

    model_config = ConfigDict(populate_by_name=True)

    ambulancias: int
    paramedico: bool
    helicoptero: bool
    alerta_hospital: bool = Field(alias="alertaHospital")
    fundamento: str


class PrediccionPayload(BaseModel):
    """Respuesta 200 de `POST /api/v1/predictions` (contrato de la SPA)."""

    model_config = ConfigDict(populate_by_name=True)

    prediction_id: str = Field(alias="predictionId")
    p_fatal: float = Field(alias="pFatal")
    victimas_graves_pred: float = Field(alias="victimasGravesPred")
    nivel_alerta: NivelAlerta = Field(alias="nivelAlerta")
    recomendacion: RecomendacionPayload
    model_version: str = Field(alias="modelVersion")
    artifact_version: str = Field(alias="artifactVersion")
    latency_ms: int = Field(alias="latencyMs")
    mode: ModoPrediccion


class ResumenPrediccionPayload(BaseModel):
    """Fila del historico paginado (`GET /api/v1/predictions`)."""

    model_config = ConfigDict(populate_by_name=True)

    prediction_id: str = Field(alias="predictionId")
    created_at: str = Field(alias="createdAt")
    comuna: str
    region: str
    p_fatal: float = Field(alias="pFatal")
    victimas_graves_pred: float = Field(alias="victimasGravesPred")
    nivel_alerta: NivelAlerta = Field(alias="nivelAlerta")
    model_version: str = Field(alias="modelVersion")
    operator_uid: str = Field(alias="operatorUid")


class ListadoPrediccionesPayload(BaseModel):
    """Pagina del historico con los filtros aplicados."""

    model_config = ConfigDict(populate_by_name=True)

    items: list[ResumenPrediccionPayload]
    total: int
    limit: int
    offset: int


class PrediccionDetallePayload(BaseModel):
    """Detalle completo de una prediccion, incluida su trazabilidad."""

    model_config = ConfigDict(populate_by_name=True)

    prediction_id: str = Field(alias="predictionId")
    created_at: str = Field(alias="createdAt")
    operator_uid: str = Field(alias="operatorUid")
    input: dict[str, object]
    input_hash: str = Field(alias="inputHash")
    features: dict[str, float | str]
    output: dict[str, object]
    recomendacion: RecomendacionPayload
    model_version: str = Field(alias="modelVersion")
    artifact_version: str = Field(alias="artifactVersion")
    code_version: str = Field(alias="codeVersion")
    latency_ms: int = Field(alias="latencyMs")
    mode: ModoPrediccion
    siniestro_ref: str | None = Field(default=None, alias="siniestroRef")
    trace_id: str = Field(alias="traceId")


class ArtefactoEstadoPayload(BaseModel):
    """Estado de verificación de un artefacto publicado."""

    model_config = ConfigDict(populate_by_name=True)

    nombre: str
    sha256: str
    bytes: int
    verificado: bool


class EstadoSaludPayload(BaseModel):
    """Respuesta de `GET /api/v1/health`."""

    model_config = ConfigDict(populate_by_name=True)

    estado: str
    model_version: str | None = Field(default=None, alias="modelVersion")
    artifact_version: str | None = Field(default=None, alias="artifactVersion")
    integridad_verificada: bool = Field(default=False, alias="integridadVerificada")
    modo_degradado_habilitado: bool = Field(default=False, alias="modoDegradadoHabilitado")
    ultima_verificacion: str = Field(alias="ultimaVerificacion")
    artefactos: list[ArtefactoEstadoPayload]
    mensaje: str | None = None


class ModeloActualPayload(BaseModel):
    """Respuesta de `GET /api/v1/models/current`."""

    model_config = ConfigDict(populate_by_name=True)

    model_version: str = Field(alias="modelVersion")
    artifact_version: str = Field(alias="artifactVersion")
    caracteristicas: list[str]
    artefactos: list[ArtefactoEstadoPayload]


class ConfirmacionEntregaPayload(BaseModel):
    """Respuesta de `POST /api/v1/predictions/{id}/dispatch-sheet/ack`."""

    model_config = ConfigDict(populate_by_name=True)

    prediction_id: str = Field(alias="predictionId")
    recibido: bool
    ack_at: str = Field(alias="ackAt")
    ack_by: str = Field(alias="ackBy")
