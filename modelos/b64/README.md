# Artefactos del modelo en base64

Los `.joblib` y el `.parquet` son binarios y no pueden versionarse en un
repositorio git sin Git LFS. Para que el servicio pueda desplegarse con los
artefactos reales sin depender de un canal binario externo, se guardan aquí
codificados en base64, en un archivo por parte.

El paso de construcción del despliegue ejecuta
`python -m scripts.desempaquetar_modelos`, que concatena las partes de cada
artefacto, las decodifica en `modelos/` y verifica su SHA-256 contra
`modelos/MANIFEST.json`. Si el checksum no coincide, el despliegue falla de
forma explícita en lugar de servir predicciones silenciosamente corruptas.

El ciclo inverso, para regenerar las partes tras cambiar un artefacto, es:

```bash
python -m scripts.codificar_modelos --directorio modelos --tamano-parte 120000
```

| Artefacto | Binario | Partes |
|---|---|---|
| `encodings.parquet` | 11.500 B | 1 |
| `features_xgb.joblib` | 639 B | 1 |
| `lgb_es_fatal_1_26M.joblib` | 675.700 B | 8 |
| `lgb_victimas_graves_1_26M.joblib` | 565.680 B | 7 |

Los checksums esperados están en `modelos/MANIFEST.json`:

| Artefacto | SHA-256 |
|---|---|
| `lgb_es_fatal_1_26M.joblib` | `a4de29ff4ac86a8942518a007bb38d7e88a945bdf394e6c7ba130c76f4d40b7e` |
| `lgb_victimas_graves_1_26M.joblib` | `520af81825dc3faea885ca5f709b15401c28e6483709fdc151494d0f0d4e7290` |
| `features_xgb.joblib` | `16f0225ba3f78da9e68624ef6fb51842c0c7dd21589680617760bbfafdfd58f7` |
| `encodings.parquet` | `443c45e122e1804733994d77250c3ea90da2c9631c23db79eed8fab61e5a5932` |
| `tasa_global.json` | `36c36f01b4badecee22f7d6cc23f3bec3dbc64793a512d19bf804d48b974b8e0` |
