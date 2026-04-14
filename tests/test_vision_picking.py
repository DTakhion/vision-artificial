# tests/test_vision_picking.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime

import cv2

from utils.vision_picking import (
    PickingSheetConfig,
    read_picking_shipping,
    build_picking_debug_images,
    save_debug_images,
)


def _annotate_result(image, result):
    vis = image.copy()

    shipping = result.get("shipping")
    source = result.get("source")
    status = result.get("status")
    score = result.get("score")

    lines = [
        f"status: {status}",
        f"shipping: {shipping}",
        f"source: {source}",
        f"score: {score}",
    ]

    y = 30
    for line in lines:
        cv2.putText(
            vis,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y += 30

    return vis


def main() -> int:
    parser = argparse.ArgumentParser(description="Test Vision Picking")
    parser.add_argument("image", help="Ruta a la imagen de picking")
    args = parser.parse_args()

    image_path = Path(args.image)

    if not image_path.exists():
        print(json.dumps({
            "status": "error",
            "message": f"No existe la imagen: {image_path}",
        }, indent=2))
        return 2

    # base output
    base_out = Path("data/tests_vision_picking")
    base_out.mkdir(parents=True, exist_ok=True)

    # subcarpeta por corrida
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base_out / f"{image_path.stem}_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # cargar imagen
    img = cv2.imread(str(image_path))
    if img is None:
        print(json.dumps({
            "status": "error",
            "message": f"No se pudo cargar la imagen: {image_path}",
        }, indent=2))
        return 2

    # config
    cfg = PickingSheetConfig(
        hybrid_env_file=".env",
        hybrid_model_path="runs/detect/runs_kuehne_nagel/barcode_v1/weights/best.pt",
    )

    # ejecución principal
    result = read_picking_shipping(img, cfg=cfg)

    # guardar summary JSON
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # imagen anotada
    annotated = _annotate_result(img, result)
    annotated_path = out_dir / "annotated.png"
    cv2.imwrite(str(annotated_path), annotated)

    # debug pipeline
    debug_images = build_picking_debug_images(img, result, cfg=cfg)
    debug_paths = save_debug_images(
        debug_images,
        out_dir=str(out_dir / "debug"),
        stem=image_path.stem,
    )

    # output limpio
    printable = {
        "status": result.get("status"),
        "shipping": result.get("shipping"),
        "source": result.get("source"),
        "score": result.get("score"),
        "sheet_found": result.get("sheet_found"),
        "output_dir": str(out_dir),
        "summary_json": str(summary_path),
        "annotated_image": str(annotated_path),
        "debug_images": debug_paths,
    }

    print(json.dumps(printable, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())