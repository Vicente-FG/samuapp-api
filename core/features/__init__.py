"""Características del modelo: taxonomia, target encoding, validación y pipeline.

Este subpaquete es la fuente de verdad de las 40 características que consume el
modelo LightGBM. Su paridad con el entrenamiento es un requisito crítico
(RNF-04): cualquier cambio aquí debe acompanarse de un reentrenamiento o de la
actualizacion del artefacto `features_xgb.joblib`.

Este módulo no reexporta simbolos a propósito: `core.contracts` importa
`core.features.taxonomy`, de modo que un reexport desde aquí volveria ciclico.
"""
