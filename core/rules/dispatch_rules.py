"""Motor de reglas de despacho: RN-01..RN-08 y modo degradado.

Reglas de negocio (documento de Etapa 3, §3.4):

* **RN-01** `ambulancias = max(1, floor(victimas_graves_pred * 1,5))`.
* **RN-02** `helicoptero = victimas_graves_pred > 1,0`.
* **RN-03** `alerta_hospital = victimas_graves_pred > 0,5`.
* **RN-04** `P(fatal) < 10 %` y `VG < 0,3` -> despacho estándar (1 ambulancia).
* **RN-05** `10-30 %` y `VG 0,3-1,0` -> despacho reforzado (1 ambulancia + paramédico).
* **RN-06** `30-60 %` y `VG 1,0-3,0` -> alerta hospitalaria, preparar equipo quirúrgico.
* **RN-07** `> 60 %` y `VG > 3,0` -> despachar helicoptero + multiple ambulancias.
* **RN-08** El mínimo de ambulancias es siempre 1.

**Reconciliacion documentada de RN-02.** La tabla de reglas y el ejemplo del
contrato de la API se contradicen: RN-02 habilita el helicoptero con
`VG > 1,0`, mientras que el ejemplo contractual de `P(fatal) 41,2 %` y
`VG 2,08` declara `helicoptero: false`. Se adopta el criterio de RN-07, que
reserva el helicoptero para la banda máxima (`P(fatal) > 60 %` y `VG > 3,0`),
porque es el único que resulta coherente con el ejemplo publicado en §4.9 del
documento y con la SPA, que nunca envia helicoptero en la banda intermedia.
Queda anotado en el README para la revision del equipo.

**Banda efectiva.** La banda de riesgo se calcula de forma independiente para
la probabilidad de fatalidad y para las victimas graves, y se toma la más
severa de las dos: ante unubinamiento, el despacho es conservador.
"""

from __future__ import annotations

import math

from core.contracts import DispatchRecommendation, NivelAlerta, Siniestro
from core.features.schema import EDAD_MAYOR_DEGRADADO
from core.features.taxonomy import TIPOS_PESADOS

__all__ = [
    "FACTOR_AMBULANCIAS",
    "recomendacion_degradada",
    "recomendacion_modelo",
    "estimar_riesgo_degradado",
    "puntaje_degradado",
]

# --- RN-01, RN-08 ------------------------------------------------------------

#: Factor de ambulancias por victima grave estimada (RN-01).
FACTOR_AMBULANCIAS = 1.5

#: Mínimo absoluto de ambulancias en todo despacho (RN-08).
AMBULANCIAS_MINIMAS = 1

#: Ambulancias mínimas por banda de riesgo (0 estándar, 1 reforzado,
#: 2 alerta hospitalaria, 3 alerta máxima). El piso de la banda 2 reproduce
#: el ejemplo contractual del documento (3 ambulancias con VG 2,08).
AMBULANCIAS_MINIMAS_POR_BANDA: tuple[int, ...] = (1, 1, 3, 3)

# --- RN-04, RN-05, RN-06, RN-07 ----------------------------------------------

#: Umbrales superior (excluido) de la banda de probabilidad de fatalidad.
UMBRAL_P_FATAL_ESTANDAR = 0.10
UMBRAL_P_FATAL_REFORZADO = 0.30
UMBRAL_P_FATAL_ALERTA = 0.60

#: Umbrales de la banda de victimas graves estimadas.
UMBRAL_VG_ESTANDAR = 0.3
UMBRAL_VG_REFORZADO = 1.0
UMBRAL_VG_ALERTA = 3.0

#: Umbral de victimas graves que activa la alerta hospitalaria (RN-03).
UMBRAL_ALERTA_HOSPITAL = 0.5

#: Etiqueta de cada banda de probabilidad de fatalidad.
ETIQUETAS_BANDA_P_FATAL: tuple[str, ...] = ("< 10 %", "10–30 %", "30–60 %", "> 60 %")

#: Etiqueta de cada banda de victimas graves.
ETIQUETAS_BANDA_VG: tuple[str, ...] = ("< 0,3", "0,3–1,0", "1,0–3,0", "> 3,0")

# --- Modo degradado ----------------------------------------------------------

#: Peso del ámbito rural en el puntaje degradado.
PESO_RURAL = 2
#: Peso de la franja horaria de madrugada (02:00 a 06:59).
PESO_HORARIO_MADRUGADA = 2
#: Peso de la franja horaria de evening (18:00 a 21:59).
PESO_HORARIO_EVENIDA = 1
#: Peso de la presencia de un vehículo pesado.
PESO_VEHICULO_PESADO = 2
#: Peso de la presencia de un involvedor de 75 anos o más.
PESO_ADULTO_MAYOR = 1

#: Rango horario de madrugada considerado en el puntaje degradado.
HORA_MADRUGADA_INICIO = 2
HORA_MADRUGADA_FIN = 6
#: Rango horario de evening considerado en el puntaje degradado.
HORA_EVENIDA_INICIO = 18
HORA_EVENIDA_FIN = 21

#: Puntaje a partir del cual el modo degradado exige alerta hospitalaria.
UMBRAL_ALERTA_HOSPITAL_DEGRADADO = 5
#: Puntaje a partir del cual el modo degradado exige despacho reforzado.
UMBRAL_REFORZADO_DEGRADADO = 3

# --- Heurística de riesgo del modo degradado --------------------------------

#: Probabilidad de fatalidad base de la heurística degradada.
P_FATAL_BASE_DEGRADADO = 0.03
#: Incremento de probabilidad de fatalidad por punto del puntaje degradado.
P_FATAL_POR_PUNTAJE = 0.055
#: Techo de la probabilidad de fatalidad en modo degradado.
P_FATAL_MAXIMO_DEGRADADO = 0.60
#: Victimas graves base de la heurística degradada.
VG_BASE_DEGRADADO = 0.15
#: Incremento de victimas graves por punto del puntaje degradado.
VG_POR_PUNTAJE = 0.32
#: Techo de victimas graves en modo degradado.
VG_MAXIMO_DEGRADADO = 3.0


def _formatear_decimal(valor: float, decimales: int = 1) -> str:
    """Formatea un número con coma decimal, como se usa en Chile."""
    return f"{valor:.{decimales}f}".replace(".", ",")


def _banda_p_fatal(p_fatal: float) -> int:
    """Banda de riesgo según la probabilidad de fatalidad (0 a 3)."""
    if p_fatal < UMBRAL_P_FATAL_ESTANDAR:
        return 0
    if p_fatal < UMBRAL_P_FATAL_REFORZADO:
        return 1
    if p_fatal < UMBRAL_P_FATAL_ALERTA:
        return 2
    return 3


def _banda_vg(victimas_graves: float) -> int:
    """Banda de riesgo según las victimas graves estimadas (0 a 3)."""
    if victimas_graves < UMBRAL_VG_ESTANDAR:
        return 0
    if victimas_graves < UMBRAL_VG_REFORZADO:
        return 1
    if victimas_graves <= UMBRAL_VG_ALERTA:
        return 2
    return 3


def _nivel_alerta(banda: int) -> NivelAlerta:
    """Nivel de alerta asociado a una banda de riesgo."""
    if banda == 0:
        return NivelAlerta.ESTANDAR
    if banda == 1:
        return NivelAlerta.REFORZADO
    return NivelAlerta.ALERTA_HOSPITAL


def recomendacion_modelo(p_fatal: float, victimas_graves: float) -> DispatchRecommendation:
    """Recomendacion de despacho a partir de la prediccion del modelo.

    Aplica RN-01..RN-08: la banda efectiva de riesgo gobierna el nivel de
    alerta, el paramédico y el helicoptero; el número de ambulancias combina
    el piso de la banda con la fórmula RN-01, con el mínimo absoluto de RN-08.
    """
    p_fatal = max(0.0, min(1.0, float(p_fatal)))
    victimas_graves = max(0.0, float(victimas_graves))

    banda_p = _banda_p_fatal(p_fatal)
    banda_vg = _banda_vg(victimas_graves)
    banda = max(banda_p, banda_vg)

    ambulancias = max(
        AMBULANCIAS_MINIMAS_POR_BANDA[banda],
        math.floor(victimas_graves * FACTOR_AMBULANCIAS),
    )
    ambulancias = max(ambulancias, AMBULANCIAS_MINIMAS)

    paramedico = banda >= 1
    helicoptero = banda >= 3
    # RN-03 (`VG > 0,5`) mas la banda efectiva: el nivel de alerta y la
    # alerta hospitalaria nunca se contradicen.
    alerta_hospital = banda >= 2 or victimas_graves > UMBRAL_ALERTA_HOSPITAL

    fundamento = (
        f"P(fatal) {_formatear_decimal(p_fatal * 100)} % "
        f"({ETIQUETAS_BANDA_P_FATAL[banda_p]}) y "
        f"{_formatear_decimal(victimas_graves)} víctimas graves estimadas "
        f"({ETIQUETAS_BANDA_VG[banda_vg]})."
    )

    return DispatchRecommendation(
        ambulancias=ambulancias,
        paramedico=paramedico,
        helicoptero=helicoptero,
        alerta_hospital=alerta_hospital,
        fundamento=fundamento,
        nivel_alerta=_nivel_alerta(banda),
    )


def puntaje_degradado(siniestro: Siniestro) -> int:
    """Puntaje de riesgo heurístico del modo degradado.

    Combina geografía (rural), franja horaria, tipo de vehículo y edad del
    involvedor. Es el mismo puntaje que ordena las tres recommendations
    degradadas y que alimenta la heurística de riesgo.
    """
    puntaje = 0
    if not siniestro.urbano:
        puntaje += PESO_RURAL
    if HORA_MADRUGADA_INICIO <= siniestro.hora <= HORA_MADRUGADA_FIN:
        puntaje += PESO_HORARIO_MADRUGADA
    if HORA_EVENIDA_INICIO <= siniestro.hora <= HORA_EVENIDA_FIN:
        puntaje += PESO_HORARIO_EVENIDA
    if any(inv.tipo in TIPOS_PESADOS for inv in siniestro.involucrados):
        puntaje += PESO_VEHICULO_PESADO
    if any(inv.edad >= EDAD_MAYOR_DEGRADADO for inv in siniestro.involucrados):
        puntaje += PESO_ADULTO_MAYOR
    return puntaje


def recomendacion_degradada(siniestro: Siniestro) -> DispatchRecommendation:
    """Recomendacion de despacho sin modelo (modo degradado).

    Es un piso de seguridad, no un sustituto del modelo: no usa ninguna
    prediccion learned y por eso no debe presentarse como resultado del
    sistema predictivo.
    """
    puntaje = puntaje_degradado(siniestro)

    if puntaje >= UMBRAL_ALERTA_HOSPITAL_DEGRADADO:
        return DispatchRecommendation(
            ambulancias=3,
            paramedico=True,
            helicoptero=False,
            alerta_hospital=True,
            nivel_alerta=NivelAlerta.ALERTA_HOSPITAL,
            fundamento=(
                "Modo degradado: heurística por geografía, franja horaria "
                "y tipo de vehículo."
            ),
        )
    if puntaje >= UMBRAL_REFORZADO_DEGRADADO:
        return DispatchRecommendation(
            ambulancias=2,
            paramedico=True,
            helicoptero=False,
            alerta_hospital=False,
            nivel_alerta=NivelAlerta.REFORZADO,
            fundamento="Modo degradado: heurística con puntaje de riesgo intermedio.",
        )
    return DispatchRecommendation(
        ambulancias=1,
        paramedico=False,
        helicoptero=False,
        alerta_hospital=False,
        nivel_alerta=NivelAlerta.ESTANDAR,
        fundamento="Modo degradado: sin factores de riesgo identificados.",
    )


def estimar_riesgo_degradado(siniestro: Siniestro) -> tuple[float, float]:
    """Estima riesgo y victimas graves sin modelo (modo degradado).

    Es una heurística **conservadora**: crece de forma lineal con el puntaje
    degradado y se topa para no sobre-dimensionar el despacho. Sirve para que
    la SPA siga mostrando magnitudes plausibles cuando no hay artefactos
    disponibles; nunca debe interpretarse como una prediccion del modelo.
    """
    puntaje = puntaje_degradado(siniestro)
    p_fatal = min(
        P_FATAL_MAXIMO_DEGRADADO, P_FATAL_BASE_DEGRADADO + P_FATAL_POR_PUNTAJE * puntaje
    )
    victimas_graves = min(
        VG_MAXIMO_DEGRADADO, VG_BASE_DEGRADADO + VG_POR_PUNTAJE * puntaje
    )
    return p_fatal, victimas_graves
