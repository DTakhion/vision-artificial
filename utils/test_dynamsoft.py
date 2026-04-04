# utils/test_dynamsoft.py

import argparse
import os
import sys
from pathlib import Path

from dynamsoft_barcode_reader_bundle import (
    CaptureVisionRouter,
    EnumErrorCode,
    EnumPresetTemplate,
    LicenseManager,
)


def load_env_file(env_path: str = ".env") -> None:
    """
    Carga variables desde un archivo .env de forma simple,
    sin depender de librerías externas.

    Formato esperado por línea:
    KEY=VALUE

    Ignora:
    - líneas vacías
    - comentarios con #
    """
    env_file = Path(env_path)
    if not env_file.exists():
        return

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        # No sobreescribe si ya existe en el entorno
        os.environ.setdefault(key, value)


def init_dynamsoft_license() -> None:
    """
    Inicializa la licencia de Dynamsoft usando LICENSE_KEY_DYNAMSOFT
    desde variables de entorno o .env.
    """
    license_key = os.getenv("LICENSE_KEY_DYNAMSOFT")

    if not license_key:
        raise RuntimeError(
            "No se encontró LICENSE_KEY_DYNAMSOFT en el entorno ni en el archivo .env"
        )

    error_code, error_msg = LicenseManager.init_license(license_key)

    if error_code not in (EnumErrorCode.EC_OK, EnumErrorCode.EC_LICENSE_CACHE_USED):
        raise RuntimeError(
            f"Fallo al inicializar licencia Dynamsoft: {error_code} - {error_msg}"
        )


def decode_barcodes(image_path: str) -> dict:
    """
    Decodifica códigos de barras desde una imagen usando Dynamsoft.
    """
    img_path = Path(image_path)

    if not img_path.exists():
        raise FileNotFoundError(f"No existe la imagen: {img_path}")

    cvr = CaptureVisionRouter()
    result = cvr.capture(
        str(img_path),
        EnumPresetTemplate.PT_READ_BARCODES.value,
    )

    if result.get_error_code() != EnumErrorCode.EC_OK:
        return {
            "status": "error",
            "image_path": str(img_path),
            "error_code": int(result.get_error_code()),
            "error_message": result.get_error_string(),
            "items": [],
        }

    barcode_result = result.get_decoded_barcodes_result()

    if barcode_result is None or barcode_result.get_items() == 0:
        return {
            "status": "not_found",
            "image_path": str(img_path),
            "items": [],
        }

    items = []
    for item in barcode_result.get_items():
        items.append(
            {
                "text": item.get_text(),
                "format": item.get_format_string(),
            }
        )

    return {
        "status": "success",
        "image_path": str(img_path),
        "total": len(items),
        "items": items,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prueba simple de lectura de códigos de barras con Dynamsoft."
    )
    parser.add_argument(
        "image_path",
        help="Ruta de la imagen a procesar",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Ruta al archivo .env (por defecto: .env)",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        load_env_file(args.env_file)
        init_dynamsoft_license()
        result = decode_barcodes(args.image_path)

        print("\n=== RESULTADO DYNAMSOFT ===\n")

        if result["status"] == "success":
            print(f"Imagen: {result['image_path']}")
            print(f"Total detectados: {result['total']}")

            for idx, item in enumerate(result["items"], start=1):
                print(f"\nResultado {idx}")
                print(f"Formato: {item['format']}")
                print(f"Texto:   {item['text']}")
        elif result["status"] == "not_found":
            print(f"Imagen: {result['image_path']}")
            print("No se detectaron códigos de barras.")
        else:
            print(f"Imagen: {result['image_path']}")
            print(f"Error code: {result['error_code']}")
            print(f"Error msg:  {result['error_message']}")

        return 0

    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())