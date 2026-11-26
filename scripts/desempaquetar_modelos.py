"""Desempaqueta los artefactos del modelo codificados en base64.

Los `.joblib` y los `.parquet` son binarios: no pueden escribirse en un
repositorio git sin Git LFS. Para que el servicio se despliegue con los
artefactos reales sin depender de un canal binario externo, el repositorio
guarda cada artefacto codificado en base64 y dividido en partes de hasta
120 KB (`modelos/b64/<artefacto>.b64.001`, `.002`, ...).

Este script:

1. concatena las partes de cada artefacto, en orden;
2. decodifica el base64 y escribe el archivo binario en `modelos/`;
3. verifica el SHA-256 de cada artefacto contra `modelos/MANIFEST.json` y
   falla de forma explícita si no coincide.

La verificación es la misma que hace el `Model Bundle` al arrancar: un
artefacto corrupto produce `failed-precondition` (412), nunca una predicción
silenciosamente errónea.

Uso:
    python -m scripts.desempaquetar_modelos
    python -m scripts.desempaquetar_modelos --directorio modelos
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from pathlib import Path

PREFIJO_PARTES = '.b64.'
SUFIJO_PARTE = '.b64'


def _agrupar_partes(directorio: Path) -> dict[str, list[Path]]:
    """Agrupa los archivos de partes por artefacto, ordenados por indice."""
    grupos: dict[str, list[Path]] = {}
    for ruta in sorted(directorio.glob('*' + SUFIJO_PARTE + '.*')):
        nombre = ruta.name
        if PREFIJO_PARTES not in nombre:
            continue
        base, _, resto = nombre.partition(PREFIJO_PARTES)
        if not resto.isdigit():
            continue
        grupos.setdefault(base, []).append(ruta)
    for base in grupos:
        grupos[base].sort(key=lambda p: int(p.name.partition(PREFIJO_PARTES)[2]))
    return grupos


def _unir(grupos: dict[str, list[Path]]) -> dict[str, str]:
    """Devuelve el base64 completo de cada artefacto."""
    resultado: dict[str, str] = {}
    for base, partes in grupos.items():
        morceaux = []
        for parte in partes:
            trozos = parte.read_text(encoding='ascii').split()
            morceaux.append(''.join(trozos))
        resultado[base] = ''.join(morceaux)
    return resultado


def _sha256(ruta: Path) -> str:
    digest = hashlib.sha256()
    with ruta.open('rb') as fh:
        for bloque in iter(lambda: fh.read(1024 * 1024), b''):
            digest.update(bloque)
    return digest.hexdigest()


def desempaquetar(directorio: Path, verificar: bool = True) -> int:
    origen = directorio / 'b64'
    if not origen.is_dir():
        print('No hay directorio de partes base64 en %s: nada que hacer.' % origen)
        return 0

    manifiesto = directorio / 'MANIFEST.json'
    esperado: dict[str, str] = {}
    if manifiesto.is_file():
        datos = json.loads(manifiesto.read_text(encoding='utf-8'))
        for nombre, meta in (datos.get('artefactos') or {}).items():
            esperado[nombre] = meta.get('sha256', '')

    grupos = _agrupar_partes(origen)
    if not grupos:
        print('No hay partes base64 en %s.' % origen)
        return 0

    errores = 0
    for nombre, base64_completo in sorted(_unir(grupos).items()):
        destino = directorio / nombre
        try:
            crudo = base64.b64decode(base64_completo, validate=True)
        except Exception as exc:  # noqa: BLE001 - se reporta con contexto
            print('ERROR %s: base64 invalido (%s)' % (nombre, exc))
            errores += 1
            continue
        destino.write_bytes(crudo)
        real = _sha256(destino)
        if verificar and nombre in esperado and esperado[nombre] != real:
            print('ERROR %s: SHA-256 %s no coincide con el manifiesto %s'
                  % (nombre, real, esperado[nombre]))
            errores += 1
            continue
        print('OK %-36s %9d bytes  sha256=%s' % (nombre, len(crudo), real[:16]))

    if errores:
        print('Se detectaron %d artefacto(s) con problemas.' % errores)
        return errores
    print('Artefactos desempaquetados y verificados en %s' % directorio)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directorio', default='modelos', type=Path,
                        help='directorio de artefactos (por defecto: modelos)')
    parser.add_argument('--sin-verificar', action='store_true',
                        help='omite la comprobacion del SHA-256 contra MANIFEST.json')
    args = parser.parse_args()
    return desempaquetar(args.directorio, verificar=not args.sin_verificar)


if __name__ == '__main__':
    sys.exit(main())
