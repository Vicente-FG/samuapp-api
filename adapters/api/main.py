"""Aplicación FastAPI de SAMUAPP: adaptador HTTP del nucleo de dominio.

Endpoints (contrato de §4.9 del documento de Etapa 3):

| Endpoint                                        | Metodo | Rol                        |
| ----------------------------------------------- | ------ | -------------------------- |
| `/api/v1/health`                                | GET    | interno                    |
| `/api/v1/predictions`                           | POST   | operador, supervisor, admin |
| `/api/v1/predictions`                           | GET    | supervisor, admin          |
| `/api/v1/predictions/{id}`                      | GET    | propietario, supervisor, admin |
| `/api/v1/predictions/{id}/dispatch-sheet`       | GET    | propietario, supervisor    |
| `/api/v1/predictions/{id}/dispatch-sheet/ack`   | POST   | operador, supervisor       |
| `/api/v1/models/current`                        | GET    | todos                      |

Los errores siguen el sobre `{"error": {"code", "message", "detalles"}}` con
los códigos `unauthenticated`, `permission-denied`, `invalid-argument`,
`failed-precondition`, `resource-exhausted`, `deadline-exceeded` e `internal`.
"""

from __future__ import annotations

import html
import logging
import sqlite3
import threading
import time
import uuid
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as TimeoutFuturo
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.exceptions import HTTPException as StarletteHTTPException

from adapters.api.auth import (
    Rol,
    TokenInvalido,
    Usuario,
    VerificadorNoDisponible,
    VerificadorToken,
    crear_verificador,
    nivel_de_rol,
)
from adapters.api.sesion import calcular_expiracion, crear_token_dev, semilla_usuarios_demo
from adapters.api.settings import Settings, configurar_logging, contexto_trace_id
from adapters.api.store import PredictionStore, RegistroPrediccion, hash_payload
from adapters.api.usuarios import UsuariosStore
from core.contracts import (
    ConfirmacionEntregaPayload,
    DispatchRecommendation,
    EstadoSaludPayload,
    ListadoPrediccionesPayload,
    ModeloActualPayload,
    ModoPrediccion,
    PayloadPrediccion,
    PrediccionDetallePayload,
    PrediccionPayload,
    RecomendacionPayload,
    ResumenPrediccionPayload,
    RespuestaSesion,
    Siniestro,
    SolicitudRegistro,
    SolicitudSesion,
)
from core.features.schema import validar_siniestro
from core.models.bundle import ErrorDeArtefactos, ModelBundle
from core.models.registry import ModelRegistry, ModelRegistryLocal
from core.rules.dispatch_rules import (
    estimar_riesgo_degradado,
    recomendacion_degradada,
    recomendacion_modelo,
)

__all__ = ["ErrorApi", "crear_app"]

logger = logging.getLogger(__name__)

#: Esquema de extraccion del token Bearer. `auto_error=False` para responder
#: con el sobre de error del contrato y no con el 401 por defecto de FastAPI.
esquema_bearer = HTTPBearer(auto_error=False, description="Token de identidad")

#: Códigos de error del contrato y su equivalencia HTTP.
CODIGOS_ERROR: dict[str, int] = {
    "unauthenticated": 401,
    "permission-denied": 403,
    "invalid-argument": 400,
    "failed-precondition": 412,
    "resource-exhausted": 429,
    "deadline-exceeded": 504,
    "internal": 500,
    # Extensiones documentadas: recurso inexistente y verificador no disponible.
    "not-found": 404,
    "service-unavailable": 503,
}

#: Alfabeto del identificador de prediccion (sin caracteres ambiguos).
ALFABETO_PREDICCION = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"

#: Largo del identificador de prediccion, sin el prefijo.
LARGO_PREDICCION = 8

#: Redondeos publicados en el contrato de la API.
DECIMALES_PROBABILIDAD = 4
DECIMALES_VICTIMAS = 2


class ErrorApi(Exception):
    """Error del contrato de la API, con su código y su estado HTTP."""

    def __init__(
        self,
        codigo: str,
        mensaje: str,
        detalles: Any = None,
        http_status: int | None = None,
    ) -> None:
        super().__init__(mensaje)
        self.codigo = codigo
        self.mensaje = mensaje
        self.detalles = detalles
        self.http_status = http_status or CODIGOS_ERROR.get(codigo, 500)

    def a_sobre(self) -> dict[str, Any]:
        """Construye el sobre de error del contrato."""
        return {
            "error": {
                "code": self.codigo,
                "message": self.mensaje,
                "detalles": self.detalles,
            }
        }


def _generar_prediction_id() -> str:
    """Genera un identificador de prediccion con el formato `pr_XXXXXXXX`."""
    import secrets

    sufijo = "".join(secrets.choice(ALFABETO_PREDICCION) for _ in range(LARGO_PREDICCION))
    return f"pr_{sufijo}"


def _formatear_clave_error(local: tuple[Any, ...]) -> str:
    """Convierte la ubicacion de un error de validación en clave por campo."""
    partes = [parte for parte in local if parte != "body"]
    if partes and partes[0] == "data":
        partes = partes[1:]
    clave = ""
    for parte in partes:
        if isinstance(parte, int):
            clave += f"[{parte}]"
        elif clave:
            clave += f".{parte}"
        else:
            clave = str(parte)
    return clave or "payload"


def _a_detalle_recomendacion(recomendacion: DispatchRecommendation) -> dict[str, Any]:
    """Proyecta la recomendacion del nucleo al bloque publico del contrato."""
    return {
        "ambulancias": recomendacion.ambulancias,
        "paramedico": recomendacion.paramedico,
        "helicoptero": recomendacion.helicoptero,
        "alertaHospital": recomendacion.alerta_hospital,
        "fundamento": recomendacion.fundamento,
    }


def _generar_hoja_despacho(registro: dict[str, Any]) -> str:
    """Genera la hoja de despacho en HTML imprimible."""
    salida = registro.get("output", {})
    recomendacion = registro.get("recomendacion", {})
    entrada = registro.get("input", {}).get("data", {})
    involucrados = entrada.get("involucrados", [])

    filas_involucrados = "".join(
        "<tr>"
        f"<td>{html.escape(str(inv.get('sexo', '')))}</td>"
        f"<td>{html.escape(str(inv.get('edad', '')))}</td>"
        f"<td>{html.escape(str(inv.get('tipo', '')))}</td>"
        "</tr>"
        for inv in involucrados
    )

    def celda(etiqueta: str, valor: Any) -> str:
        return (
            f"<div class='celda'><dt>{html.escape(etiqueta)}</dt>"
            f"<dd>{html.escape(str(valor))}</dd></div>"
        )

    vehiculos = ", ".join(str(inv.get("tipo", "")) for inv in involucrados)

    def fila(etiqueta: str, valor: Any) -> str:
        """Fila de la tabla de recomendacion de la hoja de despacho."""
        return (
            f"<tr><th>{html.escape(etiqueta)}</th>"
            f"<td>{html.escape(str(valor))}</td></tr>"
        )

    def si_no(activo: Any) -> str:
        """Traduce un booleano a la etiqueta de la hoja de despacho."""
        return "Si" if activo else "No"

    filas_recomendacion = "".join(
        [
            fila("Ambulancias", recomendacion.get("ambulancias", "")),
            fila("Paramedico", si_no(recomendacion.get("paramedico"))),
            fila("Helicoptero", si_no(recomendacion.get("helicoptero"))),
            fila("Alerta hospital", si_no(recomendacion.get("alertaHospital"))),
            fila("Fundamento", recomendacion.get("fundamento", "")),
        ]
    )
    p_fatal = salida.get("pFatal")
    victimas = salida.get("victimasGravesPred")
    p_fatal_texto = "-"
    if p_fatal is not None:
        p_fatal_texto = f"{float(p_fatal) * 100:.1f} %".replace(".", ",")
    victimas_texto = "-"
    if victimas is not None:
        victimas_texto = f"{float(victimas):.2f}".replace(".", ",")

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Hoja de despacho {html.escape(registro.get('prediction_id', ''))}</title>
<style>
  body {{ font-family: "Segoe UI", Arial, sans-serif; margin: 24px; color: #1c2733; }}
  h1 {{ font-size: 20px; margin: 0 0 4px; }}
  .sub {{ color: #5a6b7b; margin: 0 0 18px; font-size: 13px; }}
  .rejilla {{ display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 18px; }}
  .celda {{ border: 1px solid #cfd8e3; border-radius: 6px; padding: 8px 10px; min-width: 150px; }}
  .celda dt {{ font-size: 11px; text-transform: uppercase; color: #5a6b7b; }}
  .celda dd {{ margin: 2px 0 0; font-size: 15px; font-weight: 600; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 18px; }}
  th, td {{ border: 1px solid #cfd8e3; padding: 6px 8px; text-align: left; font-size: 13px; }}
  .recom {{ border: 2px solid #b3401f; border-radius: 8px;
           padding: 12px 14px; margin-bottom: 18px; }}
  .recom h2 {{ margin: 0 0 8px; font-size: 15px; color: #b3401f; }}
  .firmas {{ display: flex; gap: 24px; margin-top: 40px; font-size: 13px; }}
  .firma {{ flex: 1; border-top: 1px solid #1c2733; padding-top: 6px; }}
  @media print {{ .no-imprimir {{ display: none; }} body {{ margin: 10mm; }} }}
</style>
</head>
<body>
<h1>Hoja de despacho &mdash; SAMUAPP</h1>
<p class="sub">Sistema Predictivo de la Siniestralidad Vial en Chile &middot;
documento de apoyo al despachador</p>

<div class="rejilla">
  {celda("Identificador", registro.get("prediction_id", ""))}
  {celda("Fecha y hora", f"{entrada.get('fecha', '')} {entrada.get('hora', '')}")}
  {celda("Comuna", entrada.get("comuna", ""))}
  {celda("Region", entrada.get("region", ""))}
  {celda("Caracteristica urbano", "Si" if entrada.get("urbano") else "No")}
  {celda("Emitida", registro.get("created_at", ""))}
  {celda("Operador", registro.get("operator_uid", ""))}
  {celda("Referencia siniestro", registro.get("siniestro_ref") or "-")}
</div>

<div class="rejilla">
  {celda("Probabilidad de fatalidad", p_fatal_texto)}
  {celda("Victimas graves estimadas", victimas_texto)}
  {celda("Nivel de alerta", salida.get("nivelAlerta", ""))}
  {celda("Version del modelo", registro.get("model_version", ""))}
</div>

<h2 style="font-size:15px;">Involucrados</h2>
<table>
  <thead><tr><th>Sexo</th><th>Edad</th><th>Tipo de vehiculo</th></tr></thead>
  <tbody>{filas_involucrados}</tbody>
</table>
<p style="font-size:13px;">Vehiculos involucrados: {html.escape(vehiculos)}</p>

<div class="recom">
  <h2>Recomendacion de despacho</h2>
  <table>
    {filas_recomendacion}
  </table>
</div>

<p style="font-size:12px;color:#5a6b7b;">Modo: {html.escape(str(registro.get("mode", "")))} &middot;
version de artefactos: {html.escape(str(registro.get("artifact_version", "")))} &middot;
version de codigo: {html.escape(str(registro.get("code_version", "")))} &middot;
traza: {html.escape(str(registro.get("trace_id", "")))}</p>

<div class="firmas">
  <div class="firma">Firma del despachador</div>
  <div class="firma">Recepcion en el hospital</div>
</div>
<button class="no-imprimir" onclick="window.print()" style="margin-top:20px;">Imprimir</button>
</body>
</html>
"""


def _usuario_actual(
    request: Request,
    credenciales: HTTPAuthorizationCredentials | None = Depends(esquema_bearer),
) -> Usuario:
    """Autentica la peticion a partir del token Bearer."""
    if credenciales is None or not credenciales.credentials:
        raise ErrorApi(
            "unauthenticated",
            "Falta el token de autenticacion: use la cabecera 'Authorization: Bearer <token>'.",
        )
    verificador: VerificadorToken = request.app.state.verificador
    try:
        return verificador.verificar(credenciales.credentials)
    except TokenInvalido as exc:
        raise ErrorApi("unauthenticated", "El token de autenticacion no es valido.") from exc
    except VerificadorNoDisponible as exc:
        raise ErrorApi("service-unavailable", str(exc)) from exc


def _requiere_rol(*roles: Rol):  # noqa: ANN202 - devuelve una dependencia de FastAPI
    """Construye una dependencia que exige un rol mínimo."""

    def dependencia(usuario: Usuario = Depends(_usuario_actual)) -> Usuario:
        minimo = min(nivel_de_rol(rol) for rol in roles)
        if nivel_de_rol(usuario.rol) < minimo:
            raise ErrorApi(
                "permission-denied",
                f"El rol '{usuario.rol.value}' no tiene permiso para esta operacion.",
                {"rolRequerido": sorted(rol.value for rol in roles)},
            )
        return usuario

    return dependencia


def crear_app(settings: Settings | None = None) -> FastAPI:
    """Construye la aplicación FastAPI.

    Los artefactos del modelo se cargan **una sola vez** en el arranque
    (lifespan) y se mantienen en memoria.
    """
    configuracion = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Carga los artefactos, abre el repositorio y prepara los recursos."""
        configurar_logging(configuracion.log_level)
        logger.info(
            "Iniciando SAMUAPP",
            extra={
                "evento": "arranque",
                "code_version": configuracion.version_codigo,
                "modelos_dir": str(configuracion.modelos_dir),
                "auth_mode": configuracion.auth_mode,
                "modo_degradado": configuracion.modo_degradado,
            },
        )

        almacen = PredictionStore(configuracion.db_path)
        almacen.crear_tablas()

        almacen_usuarios = UsuariosStore(configuracion.db_path)
        almacen_usuarios.crear_tablas()
        if configuracion.auth_mode == "dev":
            semilla_usuarios_demo(almacen_usuarios)

        registro_modelos: ModelRegistry = ModelRegistryLocal(configuracion.modelos_dir)
        verificador = crear_verificador(
            configuracion.auth_mode,
            project_id=configuracion.firebase_project_id,
            check_revocado=configuracion.firebase_check_revocado,
        )

        app.state.settings = configuracion
        app.state.store = almacen
        app.state.usuarios = almacen_usuarios
        app.state.registry = registro_modelos
        app.state.verificador = verificador
        app.state.bundle = None
        app.state.error_artefactos = None
        app.state.semaforo_inferencia = threading.Semaphore(
            configuracion.max_concurrencia
        )
        app.state.ejecutor = ThreadPoolExecutor(
            max_workers=configuracion.max_concurrencia,
            thread_name_prefix="samuapp-inferencia",
        )

        try:
            app.state.bundle = registro_modelos.leer_actual()
        except ErrorDeArtefactos as exc:
            app.state.error_artefactos = str(exc)
            logger.error(
                "Artefactos del modelo no disponibles: el servicio operara en modo "
                "degradado si esta habilitado, o respondera failed-precondition",
                extra={"evento": "artefactos_no_disponibles", "detalle": str(exc)},
            )

        yield

        app.state.ejecutor.shutdown(wait=False, cancel_futures=True)
        logger.info("SAMUAPP detenido", extra={"evento": "apagado"})

    app = FastAPI(
        title="SAMUAPP API",
        version=configuracion.version_codigo,
        description=(
            "Sistema Predictivo de la Siniestralidad Vial en Chile. "
            "Backend REST del proyecto integrador MSI607."
        ),
        lifespan=lifespan,
    )

    origenes = configuracion.origenes_cors
    if origenes:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origenes,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # --- middlewares y manejadores de error --------------------------------

    @app.middleware("http")
    async def _middleware_traza(request: Request, call_next):  # noqa: ANN202
        """Asigna un `trace_id` por peticion y lo devuelve en la cabecera."""
        trace_id = str(uuid.uuid4())
        token = contexto_trace_id.set(trace_id)
        inicio = time.perf_counter()
        try:
            respuesta = await call_next(request)
        finally:
            contexto_trace_id.reset(token)
        duracion_ms = round((time.perf_counter() - inicio) * 1000, 2)
        respuesta.headers["X-Trace-Id"] = trace_id
        logger.info(
            "peticion_atendida",
            extra={
                "evento": "http_request",
                "http_metodo": request.method,
                "http_ruta": request.url.path,
                "http_status": respuesta.status_code,
                "duracion_ms": duracion_ms,
            },
        )
        return respuesta

    @app.exception_handler(ErrorApi)
    async def _manejador_error_api(request: Request, exc: ErrorApi) -> JSONResponse:
        """Traduce un `ErrorApi` al sobre de error del contrato."""
        logger.warning(
            "Error de API",
            extra={
                "evento": "error_api",
                "detalle": exc.mensaje,
                "http_ruta": request.url.path,
            },
        )
        return JSONResponse(
            status_code=exc.http_status,
            content=exc.a_sobre(),
            headers={"X-Trace-Id": contexto_trace_id.get()},
        )

    @app.exception_handler(RequestValidationError)
    async def _manejador_validacion(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Traduce un error de validación de Pydantic a `invalid-argument`."""
        errores: dict[str, str] = {}
        for error in exc.errors():
            clave = _formatear_clave_error(tuple(error.get("loc", ())))
            mensaje = str(error.get("msg", "Valor invalido"))
            if clave in errores:
                errores[clave] = f"{errores[clave]}; {mensaje}"
            else:
                errores[clave] = mensaje
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "code": "invalid-argument",
                    "message": "El payload no cumple el esquema ni los rangos esperados.",
                    "detalles": errores,
                }
            },
            headers={"X-Trace-Id": contexto_trace_id.get()},
        )

    @app.exception_handler(StarletteHTTPException)
    async def _manejador_http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        """Traduce las excepciones HTTP de Starlette al sobre del contrato."""
        codigo = {401: "unauthenticated", 403: "permission-denied"}.get(
            exc.status_code, "internal" if exc.status_code >= 500 else "invalid-argument"
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": codigo,
                    "message": str(exc.detail),
                    "detalles": None,
                }
            },
            headers={"X-Trace-Id": contexto_trace_id.get()},
        )

    @app.exception_handler(Exception)
    async def _manejador_interno(request: Request, exc: Exception) -> JSONResponse:
        """Traduce cualquier error no controlado a `internal`."""
        logger.error(
            "Error no controlado",
            extra={"evento": "error_interno", "http_ruta": request.url.path},
            exc_info=exc,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal",
                    "message": "Ocurrio un error interno no controlado.",
                    "detalles": None,
                }
            },
            headers={"X-Trace-Id": contexto_trace_id.get()},
        )

    # --- servicios internos -------------------------------------------------

    def _bundle_disponible() -> ModelBundle | None:
        """Devuelve el bundle cargado, o `None` si los artefactos fallaron."""
        return getattr(app.state, "bundle", None)

    def _inferir(
        bundle: ModelBundle, siniestro: Siniestro
    ) -> tuple[float, float, dict[str, Any]]:
        """Ejecuta la inferencia sobre el bundle cargado."""
        caracteristicas = bundle.transformer.transformar(siniestro)
        p_fatal, victimas_graves = bundle.predecir(caracteristicas)
        return p_fatal, victimas_graves, dict(caracteristicas)

    def _ejecutar_inferencia(
        request: Request, siniestro: Siniestro
    ) -> tuple[float, float, dict[str, Any]]:
        """Infiere con control de concurrencia (429) y de tiempo (504)."""
        bundle = _bundle_disponible()
        if bundle is None:
            raise ErrorApi(
                "failed-precondition",
                "Los artefactos del modelo no estan disponibles.",
                {"detalle": getattr(app.state, "error_artefactos", None)},
            )

        semaforo = app.state.semaforo_inferencia
        if not semaforo.acquire(blocking=False):
            raise ErrorApi(
                "resource-exhausted",
                "El servicio esta saturado; reintente en unos segundos.",
            )
        try:
            futuro = app.state.ejecutor.submit(_inferir, bundle, siniestro)
            try:
                return futuro.result(timeout=configuracion.timeout_segundos)
            except TimeoutFuturo as exc:
                futuro.cancel()
                raise ErrorApi(
                    "deadline-exceeded",
                    "La inferencia excedio el tiempo maximo permitido.",
                    {"timeout_ms": configuracion.timeout_ms},
                ) from exc
        finally:
            semaforo.release()

    # --- rutas --------------------------------------------------------------

    @app.post(
        "/api/v1/auth/sesion",
        response_model=RespuestaSesion,
        status_code=200,
        tags=["autenticacion"],
    )
    def iniciar_sesion(payload: SolicitudSesion, request: Request) -> RespuestaSesion:
        """Inicia sesión y devuelve un token de identidad.

        Solo existe en el modo de desarrollo: en producción el token lo emite
        Firebase Authentication y la SPA lo obtiene con el SDK de Firebase.
        El mensaje de error es generico (CA-01.5): no revela si el correo esta
        registrado.
        """
        if configuracion.auth_mode != "dev":
            raise ErrorApi(
                "failed-precondition",
                "El inicio de sesion esta delegado a Firebase Authentication: la SPA "
                "debe obtener el ID token con el SDK de Firebase.",
            )

        almacen: UsuariosStore = app.state.usuarios
        fila = almacen.buscar_por_correo(payload.correo)
        if fila is None or not almacen.verificar_contrasena(
            str(fila["uid"]), payload.contrasena
        ):
            logger.warning(
                "Intento de inicio de sesion fallido",
                extra={"evento": "sesion_fallida", "correo": payload.correo},
            )
            raise ErrorApi("unauthenticated", "Las credenciales no son validas.")

        uid = str(fila["uid"])
        rol = Rol(str(fila["rol"]))
        return RespuestaSesion(
            token=crear_token_dev(uid, rol),
            uid=uid,
            rol=rol,
            correo=str(fila["correo"]),
            nombre=str(fila["nombre"]),
            expiraEn=calcular_expiracion(configuracion.sesion_ttl_minutos),
            modo="dev",
        )

    @app.post(
        "/api/v1/auth/registro",
        response_model=RespuestaSesion,
        status_code=200,
        tags=["autenticacion"],
    )
    def registrar_usuario(payload: SolicitudRegistro, request: Request) -> RespuestaSesion:
        """Registra un operador nuevo y devuelve su token de identidad.

        La longitud minima de la contraseña (12 caracteres) la impone el
        esquema antes de llegar aqui (CA-01.3). Si el correo ya existe, la
        respuesta es la misma que ante cualquier otro fallo (CA-01.2): no se
        revela la existencia de la cuenta.
        """
        if configuracion.auth_mode != "dev":
            raise ErrorApi(
                "failed-precondition",
                "El registro esta delegado a Firebase Authentication: la SPA debe "
                "usar el SDK de Firebase para crear la cuenta.",
            )

        almacen: UsuariosStore = app.state.usuarios
        mensaje_generico = "No fue posible completar el registro con los datos ingresados."
        if almacen.buscar_por_correo(payload.correo) is not None:
            raise ErrorApi("invalid-argument", mensaje_generico)

        try:
            uid = almacen.crear(
                payload.nombre,
                payload.correo,
                payload.contrasena,
                payload.rol.value,
            )
        except sqlite3.IntegrityError:
            raise ErrorApi("invalid-argument", mensaje_generico) from None

        logger.info(
            "Operador registrado",
            extra={"evento": "registro_usuario", "rol": payload.rol.value, "uid": uid},
        )
        return RespuestaSesion(
            token=crear_token_dev(uid, payload.rol),
            uid=uid,
            rol=payload.rol,
            correo=payload.correo,
            nombre=payload.nombre,
            expiraEn=calcular_expiracion(configuracion.sesion_ttl_minutos),
            modo="dev",
        )

    @app.get("/api/v1/health", response_model=EstadoSaludPayload, tags=["operacion"])
    def health(request: Request) -> EstadoSaludPayload:
        """Estado del servicio, de los artefactos y de sus versions."""
        bundle = _bundle_disponible()
        if configuracion.health_requiere_token:
            _usuario_actual(request)

        if bundle is not None:
            estado = "ok"
            mensaje = None
            artefactos = [
                {
                    "nombre": artefacto.nombre,
                    "sha256": artefacto.sha256,
                    "bytes": artefacto.bytes,
                    "verificado": artefacto.verificado,
                }
                for artefacto in bundle.artefactos
            ]
            model_version = bundle.model_version
            artifact_version = bundle.artifact_version
            integridad = bundle.integridad_verificada
        else:
            estado = "degradado"
            mensaje = getattr(app.state, "error_artefactos", None)
            artefactos = []
            model_version = None
            artifact_version = None
            integridad = False

        return EstadoSaludPayload(
            estado=estado,
            model_version=model_version,
            artifact_version=artifact_version,
            integridad_verificada=integridad,
            modo_degradado_habilitado=configuracion.modo_degradado,
            ultima_verificacion=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            artefactos=artefactos,
            mensaje=mensaje,
        )

    @app.get("/api/v1/models/current", response_model=ModeloActualPayload, tags=["modelo"])
    def modelos_actuales(request: Request) -> ModeloActualPayload:
        """Versión vigente del modelo y de las tablas de referencia."""
        bundle = _bundle_disponible()
        if bundle is None:
            raise ErrorApi(
                "failed-precondition",
                "Los artefactos del modelo no estan disponibles.",
                {"detalle": getattr(app.state, "error_artefactos", None)},
            )
        return ModeloActualPayload(
            model_version=bundle.model_version,
            artifact_version=bundle.artifact_version,
            caracteristicas=list(bundle.transformer.orden_columnas),
            artefactos=[
                {
                    "nombre": artefacto.nombre,
                    "sha256": artefacto.sha256,
                    "bytes": artefacto.bytes,
                    "verificado": artefacto.verificado,
                }
                for artefacto in bundle.artefactos
            ],
        )

    @app.post(
        "/api/v1/predictions",
        response_model=PrediccionPayload,
        status_code=200,
        tags=["predicciones"],
    )
    def crear_prediccion(
        payload: PayloadPrediccion,
        request: Request,
        usuario: Usuario = Depends(_usuario_actual),
    ) -> PrediccionPayload:
        """Registra un siniestro y devuelve la prediccion de riesgo."""
        inicio = time.perf_counter()

        errores = validar_siniestro(payload.data)
        if errores:
            raise ErrorApi(
                "invalid-argument",
                "El payload no cumple el esquema ni los rangos esperados.",
                errores,
            )

        bundle = _bundle_disponible()

        if bundle is None:
            if not configuracion.modo_degradado:
                raise ErrorApi(
                    "failed-precondition",
                    "Los artefactos del modelo no estan disponibles.",
                    {"detalle": getattr(app.state, "error_artefactos", None)},
                )
            p_fatal, victimas_graves = estimar_riesgo_degradado(payload.data)
            recomendacion = recomendacion_degradada(payload.data)
            modo = ModoPrediccion.DEGRADADO
            model_version = "degradado"
            artifact_version = "degradado"
            caracteristicas: dict[str, Any] = {}
        else:
            p_fatal, victimas_graves, caracteristicas = _ejecutar_inferencia(
                request, payload.data
            )
            recomendacion = recomendacion_modelo(p_fatal, victimas_graves)
            modo = ModoPrediccion.MODELO
            model_version = bundle.model_version
            artifact_version = bundle.artifact_version

        latency_ms = int(round((time.perf_counter() - inicio) * 1000))
        prediction_id = _generar_prediction_id()
        trace_id = contexto_trace_id.get()

        p_fatal_publicado = round(float(p_fatal), DECIMALES_PROBABILIDAD)
        victimas_publicadas = round(float(victimas_graves), DECIMALES_VICTIMAS)

        respuesta = PrediccionPayload(
            prediction_id=prediction_id,
            p_fatal=p_fatal_publicado,
            victimas_graves_pred=victimas_publicadas,
            nivel_alerta=recomendacion.nivel_alerta,
            recomendacion=RecomendacionPayload(
                ambulancias=recomendacion.ambulancias,
                paramedico=recomendacion.paramedico,
                helicoptero=recomendacion.helicoptero,
                alerta_hospital=recomendacion.alerta_hospital,
                fundamento=recomendacion.fundamento,
            ),
            model_version=model_version,
            artifact_version=artifact_version,
            latency_ms=latency_ms,
            mode=modo,
        )

        _persistir(
            request,
            usuario=usuario,
            payload=payload,
            prediction_id=prediction_id,
            respuesta=respuesta,
            recomendacion=recomendacion,
            caracteristicas=caracteristicas,
            model_version=model_version,
            artifact_version=artifact_version,
            latency_ms=latency_ms,
            modo=modo,
            trace_id=trace_id,
        )

        logger.info(
            "Prediccion generada",
            extra={
                "evento": "prediccion",
                "prediction_id": prediction_id,
                "model_version": model_version,
                "operador_uid": usuario.uid,
                "comuna": payload.data.comuna,
                "region": payload.data.region,
                "duracion_ms": latency_ms,
            },
        )
        return respuesta

    def _persistir(
        request: Request,
        usuario: Usuario,
        payload: PayloadPrediccion,
        prediction_id: str,
        respuesta: PrediccionPayload,
        recomendacion: DispatchRecommendation,
        caracteristicas: dict[str, Any],
        model_version: str,
        artifact_version: str,
        latency_ms: int,
        modo: ModoPrediccion,
        trace_id: str,
    ) -> None:
        """Persiste la prediccion en el repositorio."""
        entrada = payload.model_dump(mode="json", by_alias=True)
        almacen: PredictionStore = request.app.state.store
        almacen.guardar(
            RegistroPrediccion(
                prediction_id=prediction_id,
                operator_uid=usuario.uid,
                created_at=time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z",
                input=entrada,
                input_hash=hash_payload(entrada),
                features=caracteristicas,
                output={
                    "pFatal": respuesta.p_fatal,
                    "victimasGravesPred": respuesta.victimas_graves_pred,
                    "nivelAlerta": respuesta.nivel_alerta.value,
                },
                recomendacion=_a_detalle_recomendacion(recomendacion),
                model_version=model_version,
                artifact_version=artifact_version,
                code_version=configuracion.version_codigo,
                latency_ms=latency_ms,
                mode=modo.value,
                siniestro_ref=payload.siniestro_ref,
                trace_id=trace_id,
                comuna=payload.data.comuna,
                region=payload.data.region,
            )
        )

    @app.get(
        "/api/v1/predictions",
        response_model=ListadoPrediccionesPayload,
        tags=["predicciones"],
    )
    def listar_predicciones(
        request: Request,
        operator_uid: str | None = Query(default=None, alias="operatorUid"),
        comuna: str | None = Query(default=None),
        fecha_desde: str | None = Query(default=None, alias="fechaDesde"),
        fecha_hasta: str | None = Query(default=None, alias="fechaHasta"),
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
        usuario: Usuario = Depends(_requiere_rol(Rol.SUPERVISOR, Rol.ADMIN)),
    ) -> ListadoPrediccionesPayload:
        """Lista el historico global con filtros y paginacion."""
        almacen: PredictionStore = request.app.state.store
        pagina = almacen.listar(
            operator_uid=operator_uid,
            comuna=comuna,
            fecha_desde=fecha_desde,
            fecha_hasta=fecha_hasta,
            limit=limit,
            offset=offset,
        )
        return ListadoPrediccionesPayload(
            items=[ResumenPrediccionPayload(**item) for item in pagina["items"]],
            total=pagina["total"],
            limit=pagina["limit"],
            offset=pagina["offset"],
        )

    @app.get(
        "/api/v1/predictions/{prediction_id}",
        response_model=PrediccionDetallePayload,
        tags=["predicciones"],
    )
    def obtener_prediccion(
        prediction_id: str,
        request: Request,
        usuario: Usuario = Depends(_usuario_actual),
    ) -> PrediccionDetallePayload:
        """Detalle de una prediccion: el owner, el supervisor y el admin."""
        almacen: PredictionStore = request.app.state.store
        registro = almacen.obtener(prediction_id)
        if registro is None:
            raise ErrorApi("not-found", f"No existe la prediccion {prediction_id}.")
        _exigir_permiso_de_lectura(usuario, registro)
        return PrediccionDetallePayload(
            prediction_id=registro["prediction_id"],
            created_at=registro["created_at"],
            operator_uid=registro["operator_uid"],
            input=registro["input"],
            input_hash=registro["input_hash"],
            features=registro["features"],
            output=registro["output"],
            recomendacion=RecomendacionPayload(**registro["recomendacion"]),
            model_version=registro["model_version"],
            artifact_version=registro["artifact_version"],
            code_version=registro["code_version"],
            latency_ms=registro["latency_ms"],
            mode=ModoPrediccion(registro["mode"]),
            siniestro_ref=registro["siniestro_ref"],
            trace_id=registro["trace_id"],
        )

    @app.get(
        "/api/v1/predictions/{prediction_id}/dispatch-sheet",
        response_class=HTMLResponse,
        tags=["predicciones"],
    )
    def hoja_de_despacho(
        prediction_id: str,
        request: Request,
        usuario: Usuario = Depends(_usuario_actual),
    ) -> HTMLResponse:
        """Hoja de despacho en HTML imprimible."""
        almacen: PredictionStore = request.app.state.store
        registro = almacen.obtener(prediction_id)
        if registro is None:
            raise ErrorApi("not-found", f"No existe la prediccion {prediction_id}.")
        _exigir_permiso_de_lectura(usuario, registro)
        return HTMLResponse(content=_generar_hoja_despacho(registro))

    @app.post(
        "/api/v1/predictions/{prediction_id}/dispatch-sheet/ack",
        response_model=ConfirmacionEntregaPayload,
        tags=["predicciones"],
    )
    def confirmar_entrega(
        prediction_id: str,
        request: Request,
        usuario: Usuario = Depends(_usuario_actual),
    ) -> ConfirmacionEntregaPayload:
        """Registra la confirmacion de entrega de la hoja de despacho."""
        almacen: PredictionStore = request.app.state.store
        registro = almacen.obtener(prediction_id)
        if registro is None:
            raise ErrorApi("not-found", f"No existe la prediccion {prediction_id}.")
        if nivel_de_rol(usuario.rol) < nivel_de_rol(Rol.SUPERVISOR):
            if registro["operator_uid"] != usuario.uid:
                raise ErrorApi(
                    "permission-denied",
                    "Solo el propietario de la prediccion o un supervisor puede "
                    "confirmar la entrega.",
                )
        confirmacion = almacen.registrar_entrega(prediction_id, usuario.uid)
        logger.info(
            "Entrega de la hoja de despacho confirmada",
            extra={
                "evento": "ack_entrega",
                "prediction_id": prediction_id,
                "operador_uid": usuario.uid,
            },
        )
        return ConfirmacionEntregaPayload(**confirmacion)

    return app


def _exigir_permiso_de_lectura(usuario: Usuario, registro: dict[str, Any]) -> None:
    """Aplica la regla de lectura: el owner, el supervisor o el admin."""
    if nivel_de_rol(usuario.rol) >= nivel_de_rol(Rol.SUPERVISOR):
        return
    if registro.get("operator_uid") == usuario.uid:
        return
    raise ErrorApi(
        "permission-denied",
        "Solo puede consultar sus propias predicciones.",
    )


#: Instancia usada por uvicorn: `uvicorn adapters.api.main:app`.
app = crear_app()
