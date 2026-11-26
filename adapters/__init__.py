"""Adaptadores externos del nucleo de dominio de SAMUAPP.

El nucleo (`core/`) no conoce FastAPI, Firebase ni SQLite. Este paquete
contiene los adaptadores que conectan el dominio con el mundo exterior:

* `adapters.api` -> API REST (FastAPI), autenticación, configuración y
  persistencia en SQLite.
"""

__all__ = ["api"]
