"""Codifica los artefactos del modelo en base64 y los divide en partes.

Complemento de `scripts/desempaquetar_modelos.py`: este script produce las
partes que viajan en el repositorio, aquel las reconstruye en el despliegue.

Uso:
    python -m scripts.codificar_modelos --directorio modelos --tamano-parte 120000
"""

from __future__ import annotations

import argparse
import base64
import shutil
import sys
from pathlib import Path

EXTENSIONES = {'.joblib', '.parquet'}


def codificar(directorio: Path, tamano: int) -> int:
    origen = directorio / 'b64'
    if origen.exists():
        shutil.rmtree(origen)
    origen.mkdir(parents=True)

    total = 0
    for ruta in sorted(directorio.iterdir()):
        if not ruta.is_file() or ruta.suffix.lower() not in EXTENSIONES:
            continue
        base64_completo = base64.b64encode(ruta.read_bytes()).decode('ascii')
        trozos = [base64_completo[i:i + tamano] for i in range(0, len(base64_completo), tamano)]
        for indice, trozo in enumerate(trozos, start=1):
            destino = origen / ('%s.b64.%03d' % (ruta.name, indice))
            destino.write_text(trozo + '\n', encoding='ascii', newline='\n')
        print('%-36s %9d bytes -> %d parte(s) de %d' % (ruta.name, ruta.stat().st_size, len(trozos), tamano))
        total += len(trozos)
    print('%d parte(s) escritas en %s' % (total, origen))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directorio', default='modelos', type=Path)
    parser.add_argument('--tamano-parte', default=120000, type=int)
    args = parser.parse_args()
    return codificar(args.directorio, args.tamano_parte)


if __name__ == '__main__':
    sys.exit(main())
