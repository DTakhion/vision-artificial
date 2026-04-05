# scripts/test_yolo_barcode.py

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from utils.vision_barcode_yolo import (
    detect_and_decode_with_yolo,
    detect_barcode_rois_yolo,
    draw_yolo_rois,
    crop_rois,
)
from utils.vision_barcode import decode_barcode_1d


def save_crop_variants(crops: list[dict], out_dir: Path, image_stem: str) -> None:
    """
    Guarda automáticamente:
    - crop padded original
    - crop rotado 90
    - crop rotado 270
    """
    crops_dir = out_dir / f"{image_stem}_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    for crop_info in crops:
        roi_index = crop_info["roi_index"]
        crop = crop_info["crop"]

        if crop is None or crop.size == 0:
            continue

        variants = {
            "orig": crop,
            "rot90": cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE),
            "rot270": cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE),
        }

        for variant_name, crop_variant in variants.items():
            out_path = crops_dir / f"{image_stem}_roi_{roi_index}_{variant_name}.jpg"
            cv2.imwrite(str(out_path), crop_variant)

    print(f"Crops guardados en: {crops_dir}")


def print_backend_summary(result: dict) -> None:
    """
    Imprime un resumen simple de los backends usados por los items decodificados.
    """
    items = result.get("items", []) or []
    if not items:
        print("\nNo hubo items decodificados; no hay backend exitoso que resumir.")
        return

    print("\n=== RESUMEN BACKENDS ===\n")
    for i, item in enumerate(items):
        backend = item.get("decoder_backend", "unknown")
        text = item.get("text", "")
        variant = item.get("yolo_crop_variant", "")
        preprocess = item.get("yolo_preprocess", "")
        print(
            f"item_{i}: backend={backend} text={text} "
            f"orientacion={variant} preprocess={preprocess}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Prueba YOLO + decoder de barcode sobre una imagen"
    )
    parser.add_argument(
        "image",
        nargs="?",
        default="data/tests_picking/capture_barcode_test.png",
        help="Ruta de la imagen a probar",
    )
    parser.add_argument(
        "--model",
        default="runs/detect/runs_kuehne_nagel/barcode_v1/weights/best.pt",
        help="Ruta del modelo YOLO entrenado",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold de YOLO",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IOU threshold de YOLO",
    )
    parser.add_argument(
        "--max_det",
        type=int,
        default=10,
        help="Máximo número de detecciones",
    )
    parser.add_argument(
        "--min_size",
        type=int,
        default=40,
        help="Tamaño mínimo de ROI",
    )
    parser.add_argument(
        "--pad_ratio",
        type=float,
        default=0.10,
        help="Padding relativo aplicado a cada ROI",
    )
    parser.add_argument(
        "--decoder_mode",
        default="collect",
        help="Modo para decode_barcode_1d",
    )
    parser.add_argument(
        "--decoder_budget",
        type=int,
        default=800,
        help="Presupuesto de tiempo del decoder por ROI en ms",
    )
    parser.add_argument(
        "--save_vis",
        action="store_true",
        help="Guardar imagen con ROIs detectadas dibujadas",
    )
    parser.add_argument(
        "--save_crops",
        action="store_true",
        help="Guardar crops padded y sus rotaciones",
    )
    parser.add_argument(
        "--out",
        default="results/test_yolo_barcode",
        help="Carpeta de salida",
    )

    args = parser.parse_args()

    image_path = Path(args.image)
    image_stem = image_path.stem

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"No se pudo cargar la imagen: {image_path}")
        return

    # 1) Detección pura de ROIs
    rois = detect_barcode_rois_yolo(
        img_bgr=img,
        model_path=args.model,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        min_size=args.min_size,
    )

    # 1.1) Construcción explícita de crops padded para debug visual
    crops = crop_rois(
        img=img,
        rois=rois,
        pad_ratio=args.pad_ratio,
    )

    # 2) Pipeline detección + decodificación
    result = detect_and_decode_with_yolo(
        img_bgr=img,
        decoder_fn=decode_barcode_1d,
        model_path=args.model,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        min_size=args.min_size,
        pad_ratio=args.pad_ratio,
        decoder_mode=args.decoder_mode,
        decoder_time_budget_ms=args.decoder_budget,
    )

    print("\n=== ROIS DETECTADAS ===\n")
    print(json.dumps(rois, indent=2, ensure_ascii=False))

    print("\n=== RESULTADO DETECCIÓN + DECODIFICACIÓN ===\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    print_backend_summary(result)

    # Guardar JSONs
    rois_json_path = out_dir / f"{image_stem}_rois.json"
    result_json_path = out_dir / f"{image_stem}_result.json"

    with open(rois_json_path, "w", encoding="utf-8") as f:
        json.dump(rois, f, indent=2, ensure_ascii=False)

    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nROIs guardadas en: {rois_json_path}")
    print(f"Resultado guardado en: {result_json_path}")

    # Guardar visualización opcional
    if args.save_vis:
        vis = draw_yolo_rois(img, rois)
        vis_path = out_dir / f"{image_stem}_vis.jpg"
        cv2.imwrite(str(vis_path), vis)
        print(f"Visualización guardada en: {vis_path}")

    # Guardar crops padded y rotaciones
    if args.save_crops:
        save_crop_variants(
            crops=crops,
            out_dir=out_dir,
            image_stem=image_stem,
        )


if __name__ == "__main__":
    main()
