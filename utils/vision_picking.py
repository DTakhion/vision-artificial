# utils/vision_picking.py

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from utils.vision_readout_hybrid import read_barcodes_hybrid


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
@dataclass
class PickingSheetConfig:
    # Validación shipping / orden
    shipping_min_digits: int = 8
    shipping_max_digits: int = 14
    prefer_shipping_len: int = 10

    # Heurística espacial:
    # si el motor retorna bbox consistente, favorece códigos más a la derecha
    prefer_rightmost_numeric: bool = True
    rightmost_weight: float = 1.25

    # Motor hybrid
    dynamsoft_env_file: str = ".env"
    dynamsoft_allowed_formats: Optional[List[str]] = None
    hybrid_model_path: str = "models/barcode_kn_v2.pt"
    hybrid_yolo_conf: float = 0.10
    hybrid_yolo_iou: float = 0.45
    hybrid_yolo_max_det: int = 10
    hybrid_yolo_min_size: int = 20
    hybrid_yolo_pad_ratio: float = 0.25


# ---------------------------------------------------------------------
# Helpers básicos
# ---------------------------------------------------------------------
def _ensure_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _put_label(
    img: np.ndarray,
    text: str,
    org: Tuple[int, int] = (20, 30),
    color=(0, 255, 0),
) -> np.ndarray:
    vis = img.copy()
    cv2.putText(
        vis,
        text[:180],
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )
    return vis


def _draw_bbox(
    img: np.ndarray,
    bbox: Any,
    color=(0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    vis = img.copy()
    if not bbox or len(bbox) != 4:
        return vis

    try:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
    except Exception:
        pass

    return vis


def _draw_polygon(
    img: np.ndarray,
    pts: Any,
    color=(255, 0, 0),
    thickness: int = 2,
) -> np.ndarray:
    vis = img.copy()
    if not pts:
        return vis

    try:
        arr = np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)
        if len(arr) >= 4:
            cv2.polylines(vis, [arr], isClosed=True, color=color, thickness=thickness)
    except Exception:
        pass

    return vis


# ---------------------------------------------------------------------
# Validación / scoring shipping
# ---------------------------------------------------------------------
def _extract_numeric_candidates(
    text: str,
    *,
    min_len: int,
    max_len: int,
) -> List[str]:
    nums = re.findall(r"\d+", text or "")
    return [n for n in nums if min_len <= len(n) <= max_len]


def _is_valid_shipping(
    text: str,
    *,
    min_len: int,
    max_len: int,
) -> bool:
    if not text:
        return False
    t = str(text).strip()
    return t.isdigit() and min_len <= len(t) <= max_len


def _bbox_rightness_score(
    bbox: Any,
    region_width: Optional[int],
) -> float:
    """
    Score espacial: mayor si el bbox está más a la derecha.
    Espera bbox xyxy = (x1, y1, x2, y2).
    """
    if not bbox or region_width is None or region_width <= 0:
        return 0.0

    try:
        x1, _, x2, _ = bbox
        cx = (float(x1) + float(x2)) / 2.0
        return max(0.0, min(1.0, cx / float(region_width)))
    except Exception:
        return 0.0


def _score_shipping_candidate(
    candidate: str,
    *,
    prefer_len: int,
    base_score: float = 0.0,
    bbox: Any = None,
    region_width: Optional[int] = None,
    prefer_rightmost_numeric: bool = True,
    rightmost_weight: float = 1.25,
) -> float:
    score = float(base_score)

    if candidate.isdigit():
        score += 5.0

    score += min(len(candidate), 18) * 0.25

    if len(candidate) == prefer_len:
        score += 2.0
    else:
        score -= min(abs(len(candidate) - prefer_len), 4) * 0.35

    if prefer_rightmost_numeric:
        score += _bbox_rightness_score(bbox, region_width) * float(rightmost_weight)

    return score


def _select_best_shipping_candidate(
    items: List[Dict[str, Any]],
    *,
    min_len: int,
    max_len: int,
    prefer_len: int,
    region_width: Optional[int] = None,
    prefer_rightmost_numeric: bool = True,
    rightmost_weight: float = 1.25,
) -> Optional[Dict[str, Any]]:
    best = None
    best_score = -1e9

    for item in items:
        text = str(item.get("text", "")).strip()
        nums = _extract_numeric_candidates(text, min_len=min_len, max_len=max_len)

        for n in nums:
            score = _score_shipping_candidate(
                n,
                prefer_len=prefer_len,
                base_score=float(item.get("score", 0.0)),
                bbox=item.get("bbox"),
                region_width=region_width,
                prefer_rightmost_numeric=prefer_rightmost_numeric,
                rightmost_weight=rightmost_weight,
            )
            cand = dict(item)
            cand["shipping"] = n
            cand["shipping_score"] = score

            if score > best_score:
                best_score = score
                best = cand

    return best


# ---------------------------------------------------------------------
# Lectura barcode con motor hybrid
# ---------------------------------------------------------------------
def _run_shipping_barcode_readout(
    img_bgr: np.ndarray,
    *,
    cfg: PickingSheetConfig,
    region_label: str,
) -> Dict[str, Any]:
    if img_bgr is None or img_bgr.size == 0:
        return {
            "status": "invalid_image",
            "best": None,
            "candidates": [],
            "raw_result": None,
        }

    allowed_formats = None
    if cfg.dynamsoft_allowed_formats:
        allowed_formats = {
            str(x).strip().upper()
            for x in cfg.dynamsoft_allowed_formats
            if str(x).strip()
        }

    raw_result = read_barcodes_hybrid(
        img_bgr=img_bgr,
        env_file=cfg.dynamsoft_env_file,
        model_path=cfg.hybrid_model_path,
        yolo_conf=cfg.hybrid_yolo_conf,
        yolo_iou=cfg.hybrid_yolo_iou,
        yolo_max_det=cfg.hybrid_yolo_max_det,
        yolo_min_size=cfg.hybrid_yolo_min_size,
        yolo_pad_ratio=cfg.hybrid_yolo_pad_ratio,
        use_dynamsoft_full_image=True,
        use_dynamsoft_on_yolo_rois=True,
        allowed_formats=allowed_formats,
    )

    region_h, region_w = img_bgr.shape[:2]
    items: List[Dict[str, Any]] = []

    for it in raw_result.get("items", []) or []:
        text = str(it.get("text", "")).strip()
        fmt = str(it.get("format", "") or "").strip().upper()

        if allowed_formats and fmt not in allowed_formats:
            continue

        numeric_candidates = _extract_numeric_candidates(
            text,
            min_len=cfg.shipping_min_digits,
            max_len=cfg.shipping_max_digits,
        )
        if not numeric_candidates:
            continue

        item_score = 1.0

        source = str(it.get("source", "") or "").strip()
        if source == "roi_barcode":
            item_score += 0.75
        elif source == "roi_label":
            item_score += 0.35
        elif source == "full_image":
            item_score += 0.15

        if str(it.get("roi_variant", "") or "").strip() == "orig":
            item_score += 0.20

        items.append(
            {
                "source_type": "barcode",
                "region_label": region_label,
                "text": text,
                "format": it.get("format"),
                "bbox": it.get("bbox"),
                "position": it.get("position"),
                "variant": it.get("roi_variant"),
                "score": item_score,
                "readout_source": it.get("source"),
                "backend": it.get("backend"),
                "yolo_roi_index": it.get("yolo_roi_index"),
                "yolo_conf": it.get("yolo_conf"),
                "yolo_bbox_xyxy_original": it.get("yolo_bbox_xyxy_original"),
                "yolo_bbox_xyxy_padded": it.get("yolo_bbox_xyxy_padded"),
                "raw_item": it,
            }
        )

    best = _select_best_shipping_candidate(
        items,
        min_len=cfg.shipping_min_digits,
        max_len=cfg.shipping_max_digits,
        prefer_len=cfg.prefer_shipping_len,
        region_width=region_w,
        prefer_rightmost_numeric=cfg.prefer_rightmost_numeric,
        rightmost_weight=cfg.rightmost_weight,
    )

    return {
        "status": "success" if best else "not_found",
        "best": best,
        "candidates": items,
        "raw_result": raw_result,
        "region_shape": [region_h, region_w],
    }


def _resolve_shipping_from_region(
    img_bgr: np.ndarray,
    *,
    cfg: PickingSheetConfig,
    label: str,
) -> Dict[str, Any]:
    barcode = _run_shipping_barcode_readout(
        img_bgr,
        cfg=cfg,
        region_label=label,
    )

    best_barcode = barcode.get("best")
    final = None

    if best_barcode and _is_valid_shipping(
        best_barcode.get("shipping", ""),
        min_len=cfg.shipping_min_digits,
        max_len=cfg.shipping_max_digits,
    ):
        final = best_barcode

    return {
        "label": label,
        "barcode": barcode,
        "best": final,
        "source": "barcode" if final else None,
    }


# ---------------------------------------------------------------------
# Orquestador principal
# ---------------------------------------------------------------------
def read_picking_shipping(
    img_bgr: np.ndarray,
    *,
    cfg: Optional[PickingSheetConfig] = None,
) -> Dict[str, Any]:
    cfg = cfg or PickingSheetConfig()

    if img_bgr is None or not isinstance(img_bgr, np.ndarray) or img_bgr.size == 0:
        return {
            "status": "invalid_image",
            "shipping": None,
        }

    img_bgr = _ensure_bgr(img_bgr)

    # Estrategia actual:
    # full image -> motor hybrid (full-image + ROI barcode/label si aplica)
    full_region = _resolve_shipping_from_region(
        img_bgr,
        cfg=cfg,
        label="full_image",
    )

    best = full_region.get("best")
    final_source = best.get("source_type") if best else None

    return {
        "status": "success" if best else "not_found",
        "shipping": best.get("shipping") if best else None,
        "source": final_source,
        "score": float(best.get("shipping_score", 0.0)) if best else None,
        "sheet_found": False,
        "rectified": {
            "quad": None,
            "warped_shape": list(img_bgr.shape[:2]),
        },
        "debug": {
            "used_full_image_only": True,
            "sheet_detection": {},
            "used_fallback_full_image": False,
            "hybrid_status": ((full_region.get("barcode") or {}).get("status")),
            "hybrid_total": _safe_int(((full_region.get("barcode") or {}).get("raw_result") or {}).get("total"), default=0),
        },
        "regions": {
            "full_image_result": full_region,
        },
        "config": asdict(cfg),
    }


# ---------------------------------------------------------------------
# Debug visual
# ---------------------------------------------------------------------
def build_picking_debug_images(
    img_bgr: np.ndarray,
    result: Dict[str, Any],
    *,
    cfg: Optional[PickingSheetConfig] = None,
) -> Dict[str, np.ndarray]:
    cfg = cfg or PickingSheetConfig()
    out: Dict[str, np.ndarray] = {}

    if img_bgr is None or img_bgr.size == 0:
        return out

    out["input"] = img_bgr.copy()

    label = f"shipping={result.get('shipping')} source={result.get('source')}"
    labeled = _put_label(img_bgr, label)
    out["input_labeled"] = labeled
    
    full_result = (
        result.get("regions", {})
        .get("full_image_result", {})
        .get("best")
    ) or {}
    
    bbox = full_result.get("bbox")
    position = full_result.get("position")

    # full_result = (
    #     result.get("regions", {})
    #     .get("full_image_result", {})
    #     .get("best", {})
    # )

    # bbox = full_result.get("bbox")
    # position = full_result.get("position")

    detected = labeled.copy()
    detected = _draw_bbox(detected, bbox, color=(0, 255, 0), thickness=2)
    detected = _draw_polygon(detected, position, color=(255, 0, 0), thickness=2)

    out["input_detected"] = detected

    # Overlay adicional con el annotated del hybrid si existe info útil
    raw_result = (
        ((result.get("regions") or {})
         .get("full_image_result") or {})
        .get("barcode") or {}
    ).get("raw_result") or {}
    
    # raw_result = (
    #     result.get("regions", {})
    #     .get("full_image_result", {})
    #     .get("barcode", {})
    #     .get("raw_result", {})
    # )

    try:
        from utils.vision_readout_hybrid import annotate_hybrid_result

        if isinstance(raw_result, dict) and raw_result:
            out["input_hybrid"] = annotate_hybrid_result(img_bgr, raw_result)
    except Exception:
        pass

    return out


def save_debug_images(
    images: Dict[str, np.ndarray],
    out_dir: str,
    stem: str = "picking",
) -> List[str]:
    out_paths: List[str] = []
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)

    for name, img in images.items():
        if img is None or img.size == 0:
            continue
        file_path = p / f"{stem}_{name}.png"
        ok = cv2.imwrite(str(file_path), img)
        if ok:
            out_paths.append(str(file_path))

    return out_paths


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def _default_json_out(image_path: str) -> str:
    p = Path(image_path)
    return f"results/picking/json/{p.stem}_picking.json"


def _default_debug_dir(image_path: str) -> str:
    return "results/picking/debug"


def _build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Lee shipping/orden desde hoja de picking usando el motor hybrid."
    )
    parser.add_argument("image_path", help="Ruta de imagen")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--model", default="models/barcode_kn_v2.pt")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--save-json", action="store_true")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--save-debug", action="store_true")
    parser.add_argument("--debug-dir", default=None)
    return parser


def _safe_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(round(value))
    s = str(value).strip()
    if not s:
        return default
    try:
        return int(float(s))
    except Exception:
        return default


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    if img is None:
        print(json.dumps({
            "status": "error",
            "message": f"No se pudo cargar la imagen: {args.image_path}",
        }, ensure_ascii=False, indent=2))
        return 2

    cfg = PickingSheetConfig(
        dynamsoft_env_file=args.env_file,
        hybrid_model_path=args.model,
    )

    result = read_picking_shipping(img, cfg=cfg)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("\n=== RESULTADO PICKING ===\n")
        print(f"Status:   {result.get('status')}")
        print(f"Shipping: {result.get('shipping')}")
        print(f"Source:   {result.get('source')}")
        print(f"Score:    {result.get('score')}")
        print(f"Sheet:    {result.get('sheet_found')}")

    if args.save_json:
        out_path = args.json_out or _default_json_out(args.image_path)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nJSON guardado en: {out_path}")

    if args.save_debug:
        debug_images = build_picking_debug_images(img, result, cfg=cfg)
        debug_dir = args.debug_dir or _default_debug_dir(args.image_path)
        saved = save_debug_images(
            debug_images,
            debug_dir,
            stem=Path(args.image_path).stem,
        )
        print(f"Debug guardado en: {debug_dir}")
        for s in saved:
            print(f" - {s}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())