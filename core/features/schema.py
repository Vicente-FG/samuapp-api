"""Validación de entrada del siniestro: rangos y dominios.

La validación vive en el nucleo y no en el adaptador HTTP: es una regla de
negocio y la SPA la replica en cliente (ver `web/src/domain/validacion.ts`),
pero el backend la vuelve a aplicar porque la validación de cliente es una
necesidad de UX, no una garantia de integridad.

Devuelve los errores **por campo**, que es lo que el contrato de la API exige
para que el operador pueda corregir el formulario sin adivinar.
"""

from __future__ import annotations

from core.contracts import Siniestro

__all__ = [
    "EDAD_MAXIMA",
    "EDAD_MAYOR",
    "EDAD_MENOR",
    "EDAD_MINIMA",
    "HORA_MAXIMA",
    "HORA_MINIMA",
    "LARGO_MAXIMO_TEXTO",
    "MAXIMO_INVOLUCRADOS",
    "MINIMO_INVOLUCRADOS",
    "validar_siniestro",
]

#: Rango valido de la hora del día.
HORA_MINIMA = 0
HORA_MAXIMA = 23

#: Rango valido de la edad de un involvedor.
EDAD_MINIMA = 0
EDAD_MAXIMA = 99

#: Número de involucrados admitidos por siniestro.
MINIMO_INVOLUCRADOS = 1
MAXIMO_INVOLUCRADOS = 10

#: Largo máximo de los textos libres de identificacion (comuna y región).
LARGO_MAXIMO_TEXTO = 120

#: Umbrales de edad que activan las características de severidad.
EDAD_MENOR = 18

#: Umbral de `Tiene_mayor`.
#:
#: El modelo publicado se entrenó con `edad >= 60` (ver
#: `notebooks/probar_modelos.py` del repositorio de ciencia de datos): usar 60
#: es la única forma de mantener la paridad con ese entrenamiento. El valor 75
#: que aparece en el documento de Etapa 3 se conserva únicamente para la
#: heurística de despacho degradada, que no usa el modelo.
EDAD_MAYOR = 60

#: Umbral de edad de la heurística de despacho degradada (piso de seguridad).
EDAD_MAYOR_DEGRADADO = 75


def _clave_involucrado(indice: int) -> str:
    """Clave de error de un involvedor concreto: `involucrados[0].edad`."""
    return f"involucrados[{indice}]"


def validar_siniestro(siniestro: Siniestro) -> dict[str, str]:
    """Valida rangos y dominios de un siniestro.

    Devuelve un diccionario `campo -> mensaje` con todos los errores
    encontrados (no se detiene en el primero) y un diccionario vacío cuando
    el siniestro cumple el mínimo duro y los rangos permitidos.
    """
    errores: dict[str, str] = {}

    if not HORA_MINIMA <= siniestro.hora <= HORA_MAXIMA:
        errores["hora"] = (
            f"La hora debe estar entre {HORA_MINIMA} y {HORA_MAXIMA}."
        )

    if not siniestro.comuna:
        errores["comuna"] = "La comuna es obligatoria."
    elif len(siniestro.comuna) > LARGO_MAXIMO_TEXTO:
        errores["comuna"] = f"La comuna no puede exceder {LARGO_MAXIMO_TEXTO} caracteres."

    if not siniestro.region:
        errores["region"] = "La region es obligatoria."
    elif len(siniestro.region) > LARGO_MAXIMO_TEXTO:
        errores["region"] = f"La region no puede exceder {LARGO_MAXIMO_TEXTO} caracteres."

    total = len(siniestro.involucrados)
    if total < MINIMO_INVOLUCRADOS:
        errores["involucrados"] = (
            f"Debe registrar al menos {MINIMO_INVOLUCRADOS} involucrado."
        )
    elif total > MAXIMO_INVOLUCRADOS:
        errores["involucrados"] = (
            f"No se pueden registrar mas de {MAXIMO_INVOLUCRADOS} involucrados."
        )

    for indice, involucrado in enumerate(siniestro.involucrados):
        if not EDAD_MINIMA <= involucrado.edad <= EDAD_MAXIMA:
            errores[f"{_clave_involucrado(indice)}.edad"] = (
                f"La edad debe estar entre {EDAD_MINIMA} y {EDAD_MAXIMA}."
            )

    return errores
