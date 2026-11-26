"""Motor de reglas de despacho: RN-01..RN-08 y modo degradado."""

from core.rules.dispatch_rules import (
    recomendacion_degradada,
    recomendacion_modelo,
)

__all__ = ["recomendacion_degradada", "recomendacion_modelo"]
