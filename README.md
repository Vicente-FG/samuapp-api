# SAMUAPP - Backend REST

Backend REST del proyecto **SAMUAPP** ("Sistema Predictivo de la Siniestralidad
Vial en Chile"), integrador MSI607 de UNAB. Reemplaza a las Cloud Functions de
Firebase y se despliega en **Render** como web service Python nativo (sin
Docker).

El servicio registra un siniestro y devuelve:

* `pFatal`: probabilidad de fatalidad del siniestro.
* `victimasGravesPred`: numero de victimas graves estimadas.
* `nivelAlerta`: nivel de alerta del despacho (`ESTANDAR`, `REFORZADO`,
  `ALERTA_HOSPITAL`).
* `recomendacion`: ambulancias, paramedico, helicoptero, alerta hospital y el
  fundamento de la decision.

> **Estado: integrado con el modelo real.** Los artefactos publicados por el
> equipo de ciencia de datos (repositorio `PRYE INTEGRADOR CIENCIA DATOS`) ya
> estan en `modelos/` y el backend opera con ellos. La paridad del pipeline se
> verifico contra el feature engineering del entrenamiento: ver seccion 9.

---

## 1. Arquitectura

Arquitectura **hexagonal**. El nucleo de dominio (`core/`) no importa FastAPI,
Firebase, SQLAlchemy ni ningun proveedor de nube; los adaptadores externos
(`adapters/`) dependen del nucleo, nunca al reves.

```
backend-staging/
|-- requirements.txt              # dependencias con versiones fijadas
|-- pyproject.toml                # configuracion de pytest y de ruff
|-- render.yaml                   # blueprint del web service en Render
|-- .env.example                  # plantilla de configuracion (sin secretos)
|-- core/                         # nucleo de dominio (sin infraestructura)
|   |-- contracts.py              # DTOs de entrada y salida (pydantic v2)
|   |-- features/
|   |   |-- pipeline.py           # FeatureTransformer -> 40 caracteristicas
|   |   |-- encodings.py          # TargetEncodingBayesiano (ALPHA = 50.0)
|   |   |-- taxonomy.py           # 18 tipos de vehiculo, sexos, franjas
|   |   `-- schema.py             # validacion de entrada (rangos y dominios)
|   |-- models/
|   |   |-- bundle.py             # ModelBundle: carga y verificacion SHA-256
|   |   `-- registry.py           # interfaz ModelRegistry + implementacion local
|   `-- rules/
|       `-- dispatch_rules.py     # RN-01..RN-08 + modo degradado
|-- adapters/
|   `-- api/
|       |-- main.py               # aplicacion FastAPI y rutas
|       |-- auth.py               # verificacion de token (Bearer) y roles
|       |-- settings.py           # configuracion y logging JSON
|       `-- store.py              # repositorio de predicciones (SQLite)
|-- scripts/
|   |-- entrenar_modelo_demo.py   # artefactos de DEMOSTRACION (en modelos-demo/)
|   |-- exportar_encodings.py     # tabla de target encoding desde el dataset
|   `-- generar_manifiesto.py    # MANIFEST.json con los SHA-256
|-- modelos/                      # artefactos REALES publicados
|-- tests/
    |-- unit/                     # pipeline, encodings, reglas y store
    `-- contract/                 # contrato de la API con TestClient
```

Regla de dependencia verificable: `core/` solo usa `numpy` y `pandas` como
formato de lectura de artefactos; no importa `fastapi`, `firebase_admin` ni
`sqlalchemy` en ningun modulo.

---

## 2. Puesta en marcha

```bash
# 1. Entorno virtual con Python 3.12
python -m venv .venv

# 2. Dependencias
.\.venv\Scripts\python.exe -m pip install -r requirements.txt   # Windows
python -m pip install -r requirements.txt                        # Linux/macOS

# 3. Pruebas
python -m pytest -q

# 4. Servidor (los artefactos reales ya estan en ./modelos/)
set SAMUAPP_AUTH_MODE=dev
python -m uvicorn adapters.api.main:app --host 0.0.0.0 --port 8000
```

Prueba manual con el token de desarrollo:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/predictions \
  -H "Authorization: Bearer dev:operador:operador1" \
  -H "Content-Type: application/json" \
  -d '{"data":{"fecha":"2026-11-20","hora":3,"comuna":"Colchane","region":"Tarapacá","urbano":false,"involucrados":[{"sexo":"M","edad":34,"tipo":"Camioneta"},{"sexo":"M","edad":28,"tipo":"Camión"},{"sexo":"F","edad":61,"tipo":"Auto"}]},"notas":"Volcamiento en ruta"}'
```

Los artefactos de demostracion (para probar el flujo sin el modelo real) se
generan en un directorio aparte, **nunca** en `modelos/`:

```bash
python -m scripts.entrenar_modelo_demo            # escribe en ./modelos-demo/
```

---

## 3. Configuracion

Todas las variables se leen con prefijo `SAMUAPP_` (ver `.env.example`).

| Variable | Default | Uso |
| --- | --- | --- |
| `SAMUAPP_MODELOS_DIR` | `./modelos/` | Directorio de los artefactos del modelo. |
| `SAMUAPP_DB_PATH` | `./data/predicciones.db` | Ruta del archivo SQLite del historico. |
| `SAMUAPP_AUTH_MODE` | `firebase` | `firebase` (produccion) o `dev` (pruebas locales). |
| `SAMUAPP_MODO_DEGRADADO` | `false` | Habilita la respuesta heuristica sin artefactos. |
| `SAMUAPP_TIMEOUT_MS` | `2000` | Timeout maximo de la inferencia. |
| `SAMUAPP_VERSION_CODIGO` | `0.1.0` | Version de codigo persistida con cada prediccion. |
| `SAMUAPP_LOG_LEVEL` | `INFO` | Nivel de los logs JSON. |
| `SAMUAPP_CORS_ORIGENES` | *(vacio)* | Origenes CORS separados por coma. Vacio: solo mismo origen. |
| `SAMUAPP_FIREBASE_PROJECT_ID` | *(vacio)* | Project ID de Firebase (opcional). |
| `SAMUAPP_FIREBASE_CHECK_REVOCADO` | `true` | Verifica la revocacion del token de Firebase. |
| `SAMUAPP_MAX_CONCURRENCIA` | `32` | Inferencias concurrentes antes de responder 429. |
| `SAMUAPP_HEALTH_REQUIERE_TOKEN` | `false` | Exige token valido en `/api/v1/health`. |

`SAMUAPP_AUTH_MODE` es **`firebase` por defecto**: el modo `dev` solo debe
usarse en desarrollo local y en las pruebas.

---

## 4. Contrato de la API

### 4.1 `POST /api/v1/predictions`

Request:

```json
{
  "data": {
    "fecha": "2026-11-20",
    "hora": 3,
    "comuna": "Colchane",
    "region": "Tarapacá",
    "urbano": false,
    "involucrados": [
      { "sexo": "M", "edad": 34, "tipo": "Camioneta" },
      { "sexo": "M", "edad": 28, "tipo": "Camión" },
      { "sexo": "F", "edad": 61, "tipo": "Auto" }
    ]
  },
  "notas": "Volcamiento en ruta"
}
```

* `notas` es opcional y no interviene en el modelo (es trazabilidad).
* `siniestroRef` es opcional y tambien es trazabilidad.
* `data.involucrados` admite entre **1 y 10** elementos. Cada elemento exige
  `sexo` (`M`, `F`, `ND`), `edad` (0-99) y `tipo` (uno de los 18).
* `tipo` se normaliza: `"Camión"` y `"Camion"` son el mismo tipo.

Response 200:

```json
{
  "predictionId": "pr_8fK2mQ",
  "pFatal": 0.412,
  "victimasGravesPred": 2.08,
  "nivelAlerta": "ALERTA_HOSPITAL",
  "recomendacion": {
    "ambulancias": 3,
    "paramedico": true,
    "helicoptero": false,
    "alertaHospital": true,
    "fundamento": "P(fatal) 41,2 % (30–60 %) y 2,1 víctimas graves estimadas (1,0–3,0)."
  },
  "modelVersion": "lgb_es_fatal_1_26M@2026-08-22",
  "artifactVersion": "encodings@2026-08-22",
  "latencyMs": 187,
  "mode": "modelo"
}
```

`mode` es `modelo` o `degradado`.

### 4.2 Resto de endpoints

| Endpoint | Metodo | Rol |
| --- | --- | --- |
| `/api/v1/health` | GET | interno |
| `/api/v1/predictions` | GET | supervisor, admin (listado paginado) |
| `/api/v1/predictions/{id}` | GET | propietario, supervisor, admin |
| `/api/v1/predictions/{id}/dispatch-sheet` | GET | propietario, supervisor (HTML imprimible) |
| `/api/v1/predictions/{id}/dispatch-sheet/ack` | POST | operador, supervisor |
| `/api/v1/models/current` | GET | todos |

El listado acepta los filtros `operatorUid`, `comuna`, `fechaDesde`,
`fechaHasta`, `limit` y `offset`, y responde `{"items", "total", "limit",
"offset"}`.

### 4.3 Errores

Todos los errores responden con el sobre `{"error": {"code", "message",
"detalles"}}`. Para `invalid-argument`, `detalles` es el diccionario de errores
**por campo** (por ejemplo `{"hora": "La hora debe estar entre 0 y 23."}`).

| code | HTTP | Significado |
| --- | --- | --- |
| `unauthenticated` | 401 | Token ausente, vencido o invalido |
| `permission-denied` | 403 | Rol sin permiso para la operacion |
| `invalid-argument` | 400 | Payload que no cumple esquema o rangos |
| `failed-precondition` | 412 | Artefactos del modelo no disponibles o corruptos |
| `resource-exhausted` | 429 | Limite de instancias alcanzado |
| `deadline-exceeded` | 504 | La inferencia excedio el tiempo maximo |
| `internal` | 500 | Error no controlado |

Dos codigos adicionales, fuera de la tabla del documento, para situaciones que
la tabla no cubre: `not-found` (404, recurso inexistente) y
`service-unavailable` (503, el verificador de tokens no esta disponible porque
`firebase-admin` no esta instalado).

---

## 5. Las 40 caracteristicas

El pipeline calcula un **superconjunto de 43 caracteristicas** y entrega
exactamente las que pide el artefacto `features_xgb.joblib`, en su orden.

**Orden por defecto (40)** cuando no hay artefacto publicado:

| Grupo | n | Caracteristicas |
| --- | --- | --- |
| Temporales | 6 | `Anio`, `Mes_num`, `Dia_semana`, `Hora_num`, `Fin_de_semana`, `Franja_horaria` |
| Composicion | 8 | `Cant_Involucrados_real`, `N_hombres`, `N_mujeres`, `Prop_hombres`, `Prop_mujeres`, `Mixto_genero`, `Tiene_hombres`, `Tiene_mujeres` |
| Tipos de vehiculo | 15 | `Tiene_Auto`, `Tiene_Camioneta`, `Tiene_Furgon`, `Tiene_Jeep`, `Tiene_Moto`, `Tiene_Motoneta`, `Tiene_Bici`, `Tiene_Bus`, `Tiene_Camion`, `Tiene_Tracto`, `Tiene_Taxi`, `Tiene_Minibus`, `Tiene_Ambulancia`, `Tiene_Fuga`, `Tiene_Tractor` |
| Geograficas | 4 | `Urbano_bin`, `Urbano_target_enc`, `Comuna_target_enc`, `Region_target_enc` |
| Edad y severidad | 7 | `Edad_promedio`, `Edad_min`, `Edad_max`, `Tiene_menor`, `Tiene_mayor`, `Diversidad_vehiculos`, `Diversidad_vehiculos_distintos` |

**Orden real publicado** (`modelos/features_xgb.joblib`, 40 columnas): usa los
**18** indicadores de tipo de vehiculo (agrega `Tiene_Maquinaria`,
`Tiene_TraccionAnimal` y `Tiene_Otros`), **no** usa `Franja_horaria` y **no**
usa `Tiene_hombres` ni `Tiene_mujeres`. El pipeline los calcula todos y
selecciona los que pide el artefacto.

Reglas relevantes:

* `Dia_semana` usa `weekday()` de Python: 0 = lunes, 6 = domingo.
* `Fin_de_semana` es 1 para sabado y domingo.
* `Franja_horaria` categoriza la hora: `madrugada` 0-5, `mañana` 6-11,
  `tarde` 12-17, `noche` 18-23. El pipeline expone la **etiqueta** y el modelo
  consumiria el **codigo** 0..3 (`CODIGOS_FRANJA_HORARIA`). El modelo real no
  usa esta caracteristica.
* Los sexo `ND` no cuentan ni para hombres ni para mujeres.
* `Tiene_mayor` se activa con **edad >= 60**, que es el umbral con el que se
  entreno el modelo publicado (ver seccion 9). El valor 75 del documento de
  Etapa 3 se conserva solo para la heuristica degradada.
* `Diversidad_vehiculos` cuenta los tipos **con repeticion**;
  `Diversidad_vehiculos_distintos` cuenta los tipos distintos.

---

## 6. Target encoding bayesiano

```
ALPHA = 50.0
tasa_te(c) = (n_c * tasa_c + ALPHA * tasa_global) / (n_c + ALPHA)
```

La busqueda de la clave es:

1. `("urbano", True)` si el siniestro es urbano.
2. `("urbano", False)` si es rural: es la clave con la que se entreno el modelo
   publicado (la tasa agregada de los siniestros rurales).
3. Si la tabla no publica la clave binaria: `("comuna", comuna)`.
4. Si tampoco: `("region", region)`.
5. Si tampoco: la tasa global, con `n_c = 0` y un **warning** en el log.

`Comuna_target_enc` usa `("comuna", comuna)` con respaldo en `("region",
region)`; `Region_target_enc` usa `("region", region)`. La comparacion de
comunas y regiones ignora tildes y mayusculas, de modo que `Tarapacá` (como lo
envia la SPA) y `Tarapaca` (como esta en el dataset) resuelven a la misma
entrada.

En produccion **no se recalcula**: se lee del artefacto congelado.

---

## 7. Reglas de despacho

| Regla | Efecto |
| --- | --- |
| RN-01 | `ambulancias = max(1, floor(VG * 1,5))` |
| RN-02 | `helicoptero = VG > 1,0` |
| RN-03 | `alerta_hospital = VG > 0,5` |
| RN-04 | `P(fatal) < 10 %` y `VG < 0,3` -> estandar (1 ambulancia) |
| RN-05 | `10-30 %` y `VG 0,3-1,0` | reforzado (1 ambulancia + paramedico) |
| RN-06 | `30-60 %` y `VG 1,0-3,0` -> alerta hospitalaria |
| RN-07 | `> 60 %` y `VG > 3,0` -> helicoptero + multiple ambulancias |
| RN-08 | El minimo de ambulancias es siempre 1 |

La banda efectiva de riesgo se calcula de forma independiente para la
probabilidad de fatalidad y para las victimas graves, y se toma **la mas
severa**: ante un cubinamiento el despacho es conservador. La banda fija el
nivel de alerta, el paramedico y el helicoptero; el numero de ambulancias
combina el piso de la banda con la formula RN-01, con el minimo de RN-08.

> **Reconciliacion de RN-02 (pendiente de revision).** La tabla de reglas del
> documento y el ejemplo del contrato se contradicen: RN-02 habilita el
> helicoptero con `VG > 1,0`, mientras que el ejemplo publicado
> (`P(fatal) 41,2 %`, `VG 2,08`) declara `helicoptero: false`. Se adopta el
> criterio de RN-07, que reserva el helicoptero para la banda maxima, porque
> es el unico coherente con el ejemplo de §4.9 y con la SPA, que nunca envia
> helicoptero en la banda intermedia. Hay que decidirlo con el equipo.

### Modo degradado

Si no hay artefactos disponibles, `POST /api/v1/predictions` responde
`failed-precondition` (412), **salvo** que `SAMUAPP_MODO_DEGRADADO=true`, en
cuyo caso responde 200 con `mode: "degradado"`.

La recomendacion degradada es un piso de seguridad, no un sustituto del modelo.
Su puntaje suma: rural (+2), madrugada 02:00-06:59 (+2), evening 18:00-21:59
(+1), vehiculo pesado (+2) e involvedor de 75 años o mas (+1). Con puntaje
`>= 5` se exige alerta hospitalaria (3 ambulancias), con `>= 3` despacho
reforzado (2 ambulancias) y en el resto despacho estandar (1 ambulancia).

`TIPOS_PESADOS = {Camion, Tracto, Bus, Minibus, Camioneta, Maquinaria,
Tractor}`. La decision esta documentada en `core/features/taxonomy.py`:
`Camioneta` y `Tractor` se incluyen por su masa en rutas rurales, y `Fuga` se
excluye porque no describe masa del vehiculo.

En modo degradado, `pFatal` y `victimasGravesPred` salen de una heuristica
lineal conservadora con topes (`0,60` y `3,0`), **no** del modelo.

---

## 8. Artefactos del modelo

Todos los artefactos viven en `SAMUAPP_MODELOS_DIR` (por defecto `./modelos/`)
con estos **nombres exactos**:

| Archivo | Contenido |
| --- | --- |
| `lgb_es_fatal_1_26M.joblib` | Clasificador de `pFatal` (`LGBMClassifier`). |
| `lgb_victimas_graves_1_26M.joblib` | Regresor de `victimasGravesPred` (`LGBMRegressor`). |
| `features_xgb.joblib` | Orden real de las 40 columnas de caracteristicas. |
| `encodings.parquet` | Tabla congelada de target encoding. |
| `tasa_global.json` | `{"tasa_global": <float>}` (opcionalmente `fecha` y `artifact_version`). |
| `MANIFEST.json` | Checksums SHA-256 y tamanos de cada artefacto. |

### Como viajan los binarios en el repositorio

`.joblib` y `.parquet` son binarios y el repositorio no usa Git LFS. Para que el
servicio se despliegue con los artefactos reales sin depender de un canal
binario externo, el repositorio guarda ademas cada artefacto **codificado en
base64 y dividido en partes** de hasta 120 KB:

```
modelos/b64/lgb_es_fatal_1_26M.joblib.b64.001
modelos/b64/lgb_es_fatal_1_26M.joblib.b64.002
...
```

El paso de construccion del despliegue ejecuta
`python -m scripts.desempaquetar_modelos`, que concatena las partes, decodifica
el base64, escribe el binario en `modelos/` y **verifica su SHA-256 contra
`MANIFEST.json`**. Si el checksum no coincide, el despliegue falla de forma
explícita en lugar de servir predicciones silenciosamente corruptas.

El ciclo inverso, para volver a generar las partes tras cambiar un artefacto, es:

```bash
python -m scripts.codificar_modelos --directorio modelos --tamano-parte 120000
```

### `features_xgb.joblib`

Debe contener una **lista de 40 cadenas** con los nombres exactos de las
caracteristicas, en el orden real del entrenamiento. Se acepta tambien un
diccionario con la clave `columnas` o `features`. Si el artefacto no existe, el
servicio usa el orden por defecto y lo deja registrado en el log.

### `encodings.parquet`

Un parquet con las columnas:

| Columna | Tipo | Descripcion |
| --- | --- | --- |
| `clave` | `str` | `"<tipo>\|<valor>"`, por ejemplo `"comuna\|Colchane"`, `"region\|Tarapaca"`, `"urbano\|True"`. |
| `n_c` | `int` | Numero de siniestros de la categoria. |
| `tasa_c` | `float` | Tasa observada de la categoria (`n_fatales / n_c`). |

Como alternativa se acepta el formato descompuesto, con las columnas `tipo`,
`valor`, `n_c` y `tasa_c`.

### `MANIFEST.json`

```json
{
  "model_version": "lgb_es_fatal_1_26M@2026-08-22",
  "artifact_version": "encodings@2026-08-22",
  "artefactos": {
    "lgb_es_fatal_1_26M.joblib": { "sha256": "a4de29ff...", "bytes": 675700 },
    "lgb_victimas_graves_1_26M.joblib": { "sha256": "520af818...", "bytes": 565680 },
    "features_xgb.joblib": { "sha256": "16f0225b...", "bytes": 639 },
    "encodings.parquet": { "sha256": "443c45e1...", "bytes": 11500 },
    "tasa_global.json": { "sha256": "2573fe7f...", "bytes": 95 }
  }
}
```

* `model_version` es opcional: si falta, se usa el nombre del archivo
  clasificador (`lgb_es_fatal_1_26M`).
* `artifact_version` es opcional: si falta, se usa `encodings@<fecha>` leida de
  `tasa_global.json`.
* Al arrancar se recalcula el SHA-256 de cada artefacto declarado. Si alguno
  no coincide, el servicio **falla de forma explicita** con
  `failed-precondition` (412) en lugar de servir predicciones corruptas.
* Si el MANIFEST no existe, se permite la carga pero se registra el warning
  "integridad no verificada".

### Como se generan

```bash
# Tabla de target encoding desde el dataset de entrenamiento (1.261.124 siniestros)
python -m scripts.exportar_encodings \
    --dataset "<ruta>/PRYE INTEGRADOR CIENCIA DATOS/data/processed/df_train.parquet" \
    --destino ./modelos

# MANIFEST.json con los SHA-256 de los artefactos
python -m scripts.generar_manifiesto --destino ./modelos \
    --model-version "lgb_es_fatal_1_26M@2026-08-22" \
    --artifact-version "encodings@2026-08-22"
```

---

## 9. Integracion con el modelo real (estado y paridad)

**Lo que ya esta resuelto.**

* Los tres artefactos del modelo real estan en `modelos/` y el servicio los
  carga y verifica en el arranque (`integridadVerificada: true`).
* `encodings.parquet` y `tasa_global.json` se reconstruyeron **desde el
  dataset de entrenamiento** (`data/processed/df_train.parquet`, 1.261.124
  siniestros) replicando la formula del notebook: 345 comunas, 16 regiones y
  las dos claves binarias de urbanidad, con `tasa_global = 0,0184217`.
* `MANIFEST.json` esta generado con los SHA-256 reales.

**Verificacion de paridad.** Se compararon las 40 caracteristicas y las
predicciones del backend contra el feature engineering del notebook
(`notebooks/probar_modelos.py`) sobre 2.090 siniestros reales:

| Subconjunto | Resultado |
| --- | --- |
| 90,0 % de las filas (1.880) con representacion 1:1 entre el dato crudo y la taxonomia del dominio | las 40 caracteristicas coinciden y las predicciones son **identicas** (`pFatal` y `victimas_graves`: diferencia maxima **0**) |
| 10 % restante | difieren solo por artefactos del preprocesamiento del entrenamiento (ver abajo) |

Los patrones temporales, de composicion, de edad y geograficos coinciden
exactamente. Las unicas diferencias, todas explicadas, son:

| Columna | Causa | Alcance |
| --- | --- | --- |
| `Tiene_Bus`, `Tiene_Taxi`, `Tiene_Tracto`, `Tiene_Otros` | El entrenamiento buscaba **subcadenas** en el texto crudo de la ficha: `Minibus` contiene `bus`, `Bus / Taxibus` contiene `taxi`, y los tipos sin palabra clave (por ejemplo `Casa Rodante`) no activaban ninguna columna mientras que el dominio los clasifica como `Otro`. | 208 de 2.090 filas |
| `Diversidad_vehiculos_distintos` | El entrenamiento contaba cadenas crudas distintas (`Camioneta` y `Camioneta 100% Electrica` son dos) y el dominio cuenta tipos distintos. | 2 de 2.090 filas |
| `Comuna_target_enc`, `Urbano_target_enc` | El entrenamiento guardaba las tasas en `float32` y el backend en `float64`. | 3e-09, sin efecto en las predicciones |

Como la SPA envia **tipos normalizados** (el catalogo de 18 valores), el
comportamiento del backend es el semanticamente correcto y coincide con el
entrenamiento en todas las categorias que el entrenamiento reconocia.

**Lo que queda por decidir con el equipo.** Ver seccion 12.

---

## 10. Autenticacion, seguridad y observabilidad

* Cabecera `Authorization: Bearer <token>`. El `uid` se toma **siempre** del
  token, nunca del cuerpo de la peticion.
* `VerificadorToken` desacopla el proveedor. `VerificadorFirebase` valida el ID
  token con `firebase-admin` (import perezoso: si la dependencia opcional no
  esta instalada, el servicio arranca y las rutas protegidas responden 503).
  `VerificadorDeDesarrollo` acepta `dev:<rol>:<uid>` y solo se habilita con
  `SAMUAPP_AUTH_MODE=dev`.
* Roles jerarquicos: `operador` < `supervisor` < `admin`.
* Los tokens se **enmascaran** en el log: solo se conservan los ultimos cuatro
  caracteres.
* Logs JSON por linea con, al menos, `nivel`, `evento`, `mensaje`,
  `timestamp`, `trace_id` y `duracion_ms`. El `trace_id` se genera por peticion
  (UUID4) y se devuelve en la cabecera `X-Trace-Id`.
* Ningun secreto en el repositorio: `.env` esta en `.gitignore` y
  `.env.example` trae los placeholders.

---

## 11. Despliegue en Render

`render.yaml` define el web service:

```yaml
buildCommand: pip install -r requirements.txt && python -m scripts.desempaquetar_modelos --directorio modelos
startCommand: uvicorn adapters.api.main:app --host 0.0.0.0 --port $PORT
healthCheckPath: /api/v1/health
```

Pasos sugeridos:

1. Publicar esta carpeta en un repositorio git (por ejemplo `Vicente-FG/samuapp-api`).
2. Subir los artefactos reales a `./modelos/` (se versionan, segun la
   suposicion S1 del diseno) y sus partes en `./modelos/b64/`, o regenerar las
   partes con `scripts.codificar_modelos` para que el despliegue los reconstruya
   automaticamente.
3. Crear el servicio desde el blueprint de Render y configurar las variables de
   entorno (en especial `SAMUAPP_MODELOS_DIR`, `SAMUAPP_DB_PATH` y
   `SAMUAPP_CORS_ORIGENES`).
4. `firebase-admin` es una dependencia **opcional**: si se usa
   `SAMUAPP_AUTH_MODE=firebase`, instalarla en el runtime
   (`pip install firebase-admin`) o declararla en `requirements.txt` al
   confirmar la decision.

El historial de predicciones se guarda en SQLite. En Render, el disco local es
efimero: para un historico persistente hay que montar un disco persistente o
apuntar `SAMUAPP_DB_PATH` a un volumen. La variable de entorno queda lista para
cambiarse.

La SPA (`../web/`) se publica como sitio estatico; su `render.yaml` y el
fallback de `public/_redirects` resuelven el enrutado del lado del cliente.

---

## 12. Supuestos que conviene revisar

1. **Helicoptero (RN-02).** Se aplica el criterio de RN-07, no el literal de
   RN-02, para no contradecir el ejemplo del contrato. Ver seccion 7.
2. **`Tiene_mayor`: 60 y no 75.** El modelo publicado se entreno con
   `edad >= 60`; usar 75 romperia la paridad. El 75 del documento de Etapa 3
   se conserva solo para la heuristica degradada. Hay que confirmar que el
   documento se actualice.
3. **Clave binaria de urbanidad.** `Urbano_target_enc` usa `("urbano", 0|1)`
   (como el entrenamiento) y solo recurre a `("comuna", ...)` si la tabla no
   publica la clave binaria. El documento de Etapa 3 describe la regla de la
   comuna; conviene unificar el criterio.
4. **Indicadores de vehiculo.** El pipeline calcula 18 indicadores (los del
   modelo real); el orden por defecto del documento usa 15. Conviene confirmar
   que la tabla del documento se actualiza a 18.
5. **`Franja_horaria`.** Se conserva en el pipeline (etiqueta + codigo 0..3),
   pero el modelo real no la usa. Si el equipo reentrena, hay que fijar la
   convencion antes.
6. **Bandas de riesgo.** La banda efectiva es el maximo entre la banda de
   `pFatal` y la de victimas graves. El documento no define el cubinamiento.
7. **Ambulancias por banda.** La banda "alerta hospitalaria" tiene un piso de
   3 ambulancias (RN-01 puro daria 1 con `VG = 1,2`). Es lo que reproduce el
   ejemplo del contrato.
8. **`TIPOS_PESADOS`.** Se sigue el conjunto del enunciado; la SPA usa uno
   distinto (incluye `Fuga` y excluye `Camioneta`).
9. **Campos de cada involvedor.** `sexo`, `edad` y `tipo` son obligatorios en
   todos los involucrados. La SPA ya filtra los slots incompletos antes de
   enviar, de modo que la completitud documental de los slots 09-35 no llega al
   backend.
10. **Formato de `clave` en `encodings.parquet`.** Se usa `"tipo|valor"`. Si el
    equipo entrega otro formato, hay que ajustar `core/features/encodings.py`.
11. **`/api/v1/health` es publico** por defecto (no expone datos de usuarios) y
    se puede cerrar con `SAMUAPP_HEALTH_REQUIERE_TOKEN=true`.
12. **Codigos de error extra.** `not-found` (404) y `service-unavailable` (503)
    no estan en la tabla del documento.
13. **Persistencia.** SQLite en disco local: suficiente para la POC y para un
    unico servicio; con varias instancias hay que migrar a un almacen
    compartido.
14. **`lightgbm` es una dependencia obligatoria** para cargar los modelos
    reales (son `LGBMClassifier` y `LGBMRegressor`); el modelo de demostracion
    usa `scikit-learn`.
