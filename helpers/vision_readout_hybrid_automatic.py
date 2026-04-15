# scripts/vision_readout_hybrid_automatic.py
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

from utils.vision_readout_hybrid import (
    annotate_hybrid_result,
    read_barcodes_hybrid,
)


def safe_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_roi_path(event_dir: Path, event_data: Dict[str, Any]) -> Optional[Path]:
    roi_path = event_dir / "roi.jpg"
    if roi_path.exists():
        return roi_path

    burst = event_data.get("burst") or {}
    main_roi_path = burst.get("main_roi_path")
    if main_roi_path:
        candidate = Path(main_roi_path)
        if candidate.exists():
            return candidate

    paths = event_data.get("paths") or {}
    roi_from_event = paths.get("roi")
    if roi_from_event:
        candidate = Path(roi_from_event)
        if candidate.exists():
            return candidate

    return None


def build_readout_payload(
    *,
    image_path: Path,
    result: Dict[str, Any],
    elapsed_ms: int,
) -> Dict[str, Any]:
    items = result.get("items", []) or []

    top_text = items[0].get("text") if items else None
    top_format = items[0].get("format") if items else None

    return {
        "barcode": top_text,
        "qrcode": None,
        "serial": None,
        "status": result.get("status", "unknown"),
        "backend": result.get("backend", "hybrid"),
        "source_image": str(image_path),
        "processed_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_ms": int(elapsed_ms),
        "total": int(result.get("total", 0) or 0),
        "top_format": top_format,
        "items": items,
    }


def should_skip(event_data: Dict[str, Any], force: bool) -> bool:
    if force:
        return False

    readout = event_data.get("readout") or {}
    status = readout.get("status")

    return status not in (None, "", "not_attempted")


def process_event(
    *,
    event_dir: Path,
    env_file: str,
    model_path: str,
    yolo_conf: float,
    yolo_iou: float,
    yolo_max_det: int,
    yolo_min_size: int,
    yolo_pad_ratio: float,
    yolo_decoder_mode: str,
    yolo_decoder_budget: int,
    use_yolo_full_pipeline: bool,
    use_dynamsoft_full_image: bool,
    use_dynamsoft_on_yolo_rois: bool,
    force: bool,
    save_vis: bool,
) -> Dict[str, Any]:
    event_json_path = event_dir / "event.json"

    if not event_json_path.exists():
        return {
            "event_dir": str(event_dir),
            "status": "missing_event_json",
            "processed": False,
        }

    event_data = load_json(event_json_path)

    if should_skip(event_data, force=force):
        return {
            "event_dir": str(event_dir),
            "event_id": event_data.get("event_id"),
            "status": "skipped_already_processed",
            "processed": False,
            "readout_status": (event_data.get("readout") or {}).get("status"),
        }

    roi_path = resolve_roi_path(event_dir, event_data)
    if roi_path is None:
        event_data["readout"] = {
            "barcode": None,
            "qrcode": None,
            "serial": None,
            "status": "missing_roi",
            "backend": "hybrid",
            "source_image": None,
            "processed_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_ms": 0,
            "total": 0,
            "items": [],
        }
        safe_write_json(event_json_path, event_data)

        return {
            "event_dir": str(event_dir),
            "event_id": event_data.get("event_id"),
            "status": "missing_roi",
            "processed": True,
        }

    img = cv2.imread(str(roi_path))
    if img is None:
        event_data["readout"] = {
            "barcode": None,
            "qrcode": None,
            "serial": None,
            "status": "invalid_roi_image",
            "backend": "hybrid",
            "source_image": str(roi_path),
            "processed_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_ms": 0,
            "total": 0,
            "items": [],
        }
        safe_write_json(event_json_path, event_data)

        return {
            "event_dir": str(event_dir),
            "event_id": event_data.get("event_id"),
            "status": "invalid_roi_image",
            "processed": True,
        }

    t0 = time.time()

    result = read_barcodes_hybrid(
        img_bgr=img,
        env_file=env_file,
        model_path=model_path,
        yolo_conf=yolo_conf,
        yolo_iou=yolo_iou,
        yolo_max_det=yolo_max_det,
        yolo_min_size=yolo_min_size,
        yolo_pad_ratio=yolo_pad_ratio,
        yolo_decoder_mode=yolo_decoder_mode,
        yolo_decoder_budget_ms=yolo_decoder_budget,
        use_yolo_full_pipeline=use_yolo_full_pipeline,
        use_dynamsoft_full_image=use_dynamsoft_full_image,
        use_dynamsoft_on_yolo_rois=use_dynamsoft_on_yolo_rois,
    )

    elapsed_ms = int((time.time() - t0) * 1000)

    event_data["readout"] = build_readout_payload(
        image_path=roi_path,
        result=result,
        elapsed_ms=elapsed_ms,
    )

    # Guardamos debug completo aparte para no ensuciar demasiado event.json
    debug_path = event_dir / "readout_hybrid.json"
    safe_write_json(debug_path, result)

    if save_vis:
        vis = annotate_hybrid_result(img, result)
        vis_path = event_dir / "readout_hybrid_vis.jpg"
        cv2.imwrite(str(vis_path), vis)
        event_data["readout"]["vis_path"] = str(vis_path)

    event_data["readout"]["result_json_path"] = str(debug_path)
    safe_write_json(event_json_path, event_data)

    return {
        "event_dir": str(event_dir),
        "event_id": event_data.get("event_id"),
        "status": result.get("status"),
        "processed": True,
        "elapsed_ms": elapsed_ms,
        "total": result.get("total", 0),
        "barcode": event_data["readout"].get("barcode"),
        "roi_path": str(roi_path),
        "result_json_path": str(debug_path),
    }


def discover_event_dirs(events_dir: Path) -> List[Path]:
    return sorted(
        [
            p for p in events_dir.iterdir()
            if p.is_dir() and p.name.startswith("event_")
        ],
        key=lambda p: p.name,
    )


def default_output_path(events_dir: Path) -> Path:
    # events_dir = .../frames_YYYYMMDD_HHMMSS/events
    frame_dir = events_dir.parent
    return frame_dir / "readout_session_hybrid.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Procesa automáticamente todos los eventos de una sesión usando vision_readout_hybrid."
    )
    parser.add_argument("--events-dir", required=True, help="Ruta a .../frames_xxx/events")
    parser.add_argument("--env-file", default=".env", help="Ruta al .env")
    parser.add_argument(
        "--model",
        default="runs/detect/runs_kuehne_nagel/barcode_v1/weights/best.pt",
        help="Ruta al modelo YOLO",
    )

    parser.add_argument("--yolo-conf", type=float, default=0.10)
    parser.add_argument("--yolo-iou", type=float, default=0.45)
    parser.add_argument("--yolo-max-det", type=int, default=10)
    parser.add_argument("--yolo-min-size", type=int, default=40)
    parser.add_argument("--yolo-pad-ratio", type=float, default=0.25)
    parser.add_argument("--yolo-decoder-mode", default="collect_plus")
    parser.add_argument("--yolo-decoder-budget", type=int, default=5000)

    parser.add_argument("--no-yolo-pipeline", action="store_true")
    parser.add_argument("--no-dynamsoft-full", action="store_true")
    parser.add_argument("--no-dynamsoft-rois", action="store_true")

    parser.add_argument("--force", action="store_true", help="Reprocesa aunque el evento ya tenga readout")
    parser.add_argument("--save-vis", action="store_true", help="Guarda imagen anotada por evento")
    parser.add_argument("--out-json", default=None, help="Ruta del resumen agregado de la sesión")

    args = parser.parse_args()

    events_dir = Path(args.events_dir)
    if not events_dir.exists() or not events_dir.is_dir():
        print(f"[ERROR] No existe events_dir: {events_dir}")
        return 1

    event_dirs = discover_event_dirs(events_dir)
    if not event_dirs:
        print(f"[WARN] No se encontraron event_* en: {events_dir}")
        return 0

    print(f"[INFO] events_dir: {events_dir}")
    print(f"[INFO] total eventos detectados: {len(event_dirs)}")

    results: List[Dict[str, Any]] = []
    processed = 0
    skipped = 0

    for event_dir in event_dirs:
        try:
            row = process_event(
                event_dir=event_dir,
                env_file=args.env_file,
                model_path=args.model,
                yolo_conf=args.yolo_conf,
                yolo_iou=args.yolo_iou,
                yolo_max_det=args.yolo_max_det,
                yolo_min_size=args.yolo_min_size,
                yolo_pad_ratio=args.yolo_pad_ratio,
                yolo_decoder_mode=args.yolo_decoder_mode,
                yolo_decoder_budget=args.yolo_decoder_budget,
                use_yolo_full_pipeline=not args.no_yolo_pipeline,
                use_dynamsoft_full_image=not args.no_dynamsoft_full,
                use_dynamsoft_on_yolo_rois=not args.no_dynamsoft_rois,
                force=args.force,
                save_vis=args.save_vis,
            )
            results.append(row)

            if row.get("processed"):
                processed += 1
            else:
                skipped += 1

            print(
                f"[INFO] {event_dir.name} | status={row.get('status')} | "
                f"barcode={row.get('barcode')} | processed={row.get('processed')}"
            )

        except Exception as exc:
            row = {
                "event_dir": str(event_dir),
                "status": "exception",
                "processed": False,
                "error": str(exc),
            }
            results.append(row)
            skipped += 1
            print(f"[ERROR] {event_dir.name}: {exc}")

    out_json = Path(args.out_json) if args.out_json else default_output_path(events_dir)

    payload = {
        "events_dir": str(events_dir),
        "processed_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_events": len(event_dirs),
        "processed_events": processed,
        "skipped_events": skipped,
        "config": {
            "env_file": args.env_file,
            "model": args.model,
            "yolo_conf": args.yolo_conf,
            "yolo_iou": args.yolo_iou,
            "yolo_max_det": args.yolo_max_det,
            "yolo_min_size": args.yolo_min_size,
            "yolo_pad_ratio": args.yolo_pad_ratio,
            "yolo_decoder_mode": args.yolo_decoder_mode,
            "yolo_decoder_budget": args.yolo_decoder_budget,
            "use_yolo_full_pipeline": not args.no_yolo_pipeline,
            "use_dynamsoft_full_image": not args.no_dynamsoft_full,
            "use_dynamsoft_on_yolo_rois": not args.no_dynamsoft_rois,
            "force": args.force,
            "save_vis": args.save_vis,
        },
        "results": results,
    }

    safe_write_json(out_json, payload)
    print(f"[DONE] Resumen guardado en: {out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# python scripts/vision_readout_hybrid_automatic.py \
#   --events-dir data/captures/opencv/frames_20260401_165708/events \
#   --save-vis

# python scripts/vision_readout_hybrid_automatic.py \
#   --events-dir data/captures/opencv/frames_20260401_165708/events \
#   --force \
#   --save-vis