"""Nucleo de dominio de SAMUAPP (arquitectura hexagonal).

Este paquete contiene la lógica de negocio pura del sistema predictivo de
siniestralidad vial: derivacion de las 40 características, target encoding
bayesiano, reglas de despacho RN-01..RN-08 y versionado de artefactos.

Regla de dependencia (obligatoria): `core/` NO importa FastAPI, Firebase,
SQLAlchemy ni ningun proveedor de nube. Los adaptadores externos viven en
`adapters/`. Las librerias de datos (numpy, pandas) se usan únicamente como
formato de lectura de artefactos, nunca como acoplamiento de infraestructura.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
