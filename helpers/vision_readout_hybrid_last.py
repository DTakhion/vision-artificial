# utils/vision_readout_hybrid_last.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

# from utils.vision_barcode_yolo import (
#     detect_and_decode_with_yolo,
#     detect_barcode_rois_yolo,
#     crop_rois,
# )

from utils.vision_barcode_yolo_v2 import (
    detect_and_decode_with_yolo,
    detect_barcode_rois_yolo,
    crop_rois,
)

from utils.vision_barcode import decode_barcode_1d
from utils.vision_barcode_dynamsoft import decode_barcode_dynamsoft


# ------------------------------------------------------------------
# Configuración por defecto
# ------------------------------------------------------------------
DEFAULT_ALLOWED_FORMATS = {"EAN_13", "CODE_128", "ITF"}


# ------------------------------------------------------------------
# Helpers generales
# ------------------------------------------------------------------
def _to_list_bbox(bbox: Any) -> List[int] | None:
    if bbox is None:
        return None
    return [int(v) for v in bbox]


def _basic_text_validation(text: str) -> bool:
    if not text:
        return False

    text = str(text).strip()
    if len(text) < 4:
        return False

    allowed = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-._/ ")
    cleaned = "".join(ch for ch in text.upper() if ch in allowed)

    if len(cleaned) < max(4, int(len(text) * 0.6)):
        return False

    return True


def _score_item(item: Dict[str, Any]) -> float:
    """
    Score heurístico para elegir la mejor evidencia por texto.
    """
    score = 0.0

    text = str(item.get("text", "")).strip()
    score += min(len(text), 24) * 0.1
    score += sum(ch.isdigit() for ch in text) * 0.05

    source = item.get("source", "")
    backend = item.get("backend", "")

    if source == "full_image":
        score += 5.0
    elif source == "roi":
        score += 4.0
    elif source == "yolo_pipeline":
        score += 3.0

    if backend == "hybrid":
        score += 1.5

    yolo_conf = float(item.get("yolo_conf", 0.0) or 0.0)
    score += yolo_conf * 2.0

    return score


# def _dedupe_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#     """
#     Deduplica por texto, conservando la mejor evidencia.
#     """
#     best_by_text: Dict[str, Dict[str, Any]] = {}

#     for item in items:
#         text = str(item.get("text", "")).strip()
#         if not _basic_text_validation(text):
#             continue

#         current_score = _score_item(item)
#         item["_score"] = current_score

#         prev = best_by_text.get(text)
#         if prev is None or current_score > float(prev.get("_score", -1.0)):
#             best_by_text[text] = item

#     out = list(best_by_text.values())
#     out.sort(key=lambda x: float(x.get("_score", -1.0)), reverse=True)

#     for item in out:
#         item.pop("_score", None)

#     return out

def _bbox_iou(b1: List[int] | None, b2: List[int] | None) -> float:
    if not b1 or not b2 or len(b1) != 4 or len(b2) != 4:
        return 0.0

    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
    area2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])

    union = area1 + area2 - inter
    if union <= 0:
        return 0.0

    return inter / union


def _bbox_center_distance(b1: List[int] | None, b2: List[int] | None) -> float:
    if not b1 or not b2 or len(b1) != 4 or len(b2) != 4:
        return float("inf")

    cx1 = (b1[0] + b1[2]) / 2.0
    cy1 = (b1[1] + b1[3]) / 2.0
    cx2 = (b2[0] + b2[2]) / 2.0
    cy2 = (b2[1] + b2[3]) / 2.0

    return float(((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5)


def _dedupe_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplica sólo si el texto es igual Y la evidencia espacial indica
    que probablemente es el mismo objeto.
    """
    kept: List[Dict[str, Any]] = []

    for item in items:
        text = str(item.get("text", "")).strip()
        if not _basic_text_validation(text):
            continue

        current_score = _score_item(item)
        item["_score"] = current_score
        bbox = item.get("bbox")

        duplicate_idx = None

        for idx, prev in enumerate(kept):
            prev_text = str(prev.get("text", "")).strip()
            if prev_text != text:
                continue

            prev_bbox = prev.get("bbox")
            iou = _bbox_iou(bbox, prev_bbox)
            dist = _bbox_center_distance(bbox, prev_bbox)

            same_object = (iou >= 0.5) or (dist <= 35.0)

            if same_object:
                duplicate_idx = idx
                break

        if duplicate_idx is None:
            kept.append(item)
        else:
            prev = kept[duplicate_idx]
            if current_score > float(prev.get("_score", -1.0)):
                kept[duplicate_idx] = item

    kept.sort(key=lambda x: float(x.get("_score", -1.0)), reverse=True)

    for item in kept:
        item.pop("_score", None)

    return kept


def _normalize_format_name(fmt: Any) -> str:
    return str(fmt or "").strip().upper()


def _filter_items_by_allowed_formats(
    items: List[Dict[str, Any]],
    allowed_formats: set[str] | None,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Filtra items por formatos permitidos.
    Si allowed_formats es None o vacío, no filtra.
    """
    if not allowed_formats:
        return items, []

    allowed_norm = {_normalize_format_name(x) for x in allowed_formats}

    kept: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    for item in items:
        fmt = _normalize_format_name(item.get("format"))
        if fmt in allowed_norm:
            kept.append(item)
        else:
            rejected.append(item)

    return kept, rejected


def _normalize_dynamsoft_item(
    item: Dict[str, Any],
    source: str,
    bbox_override: List[int] | None = None,
    extra_meta: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "text": item.get("text"),
        "format": item.get("format"),
        "backend": "hybrid",
        "source": source,
    }

    bbox = (
        bbox_override if bbox_override is not None else _to_list_bbox(item.get("bbox"))
    )
    if bbox:
        out["bbox"] = bbox

    if item.get("position") is not None:
        out["position"] = item.get("position")

    if extra_meta:
        out.update(extra_meta)

    return out


def _normalize_yolo_item(item: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "text": item.get("text"),
        "format": item.get("format"),
        "backend": "hybrid",
        "source": "yolo_pipeline",
        "bbox": _to_list_bbox(item.get("yolo_bbox_xyxy_padded")),
        "yolo_conf": item.get("yolo_conf"),
        "yolo_roi_index": item.get("yolo_roi_index"),
        "yolo_crop_variant": item.get("yolo_crop_variant"),
        "yolo_preprocess": item.get("yolo_preprocess"),
    }
    return out


def _build_crop_variants(crop: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    variants: List[Tuple[str, np.ndarray]] = [("orig", crop)]

    try:
        variants.append(("rot90", cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)))
    except Exception:
        pass

    try:
        variants.append(("rot270", cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)))
    except Exception:
        pass

    out: List[Tuple[str, np.ndarray]] = []
    seen = set()

    for name, img in variants:
        if name in seen:
            continue
        if img is None or img.size == 0:
            continue
        seen.add(name)
        out.append((name, img))

    return out


# ------------------------------------------------------------------
# Pipeline híbrido
# ------------------------------------------------------------------
def read_barcodes_hybrid(
    img_bgr: np.ndarray,
    *,
    env_file: str = ".env",
    model_path: str = "runs/detect/runs_kuehne_nagel/barcode_v1/weights/best.pt",
    yolo_conf: float = 0.10,
    yolo_iou: float = 0.45,
    yolo_max_det: int = 10,
    yolo_min_size: int = 20, #40
    yolo_pad_ratio: float = 0.25,
    yolo_decoder_mode: str = "collect_plus",
    yolo_decoder_budget_ms: int = 5000,
    use_yolo_full_pipeline: bool = True,
    use_dynamsoft_full_image: bool = True,
    use_dynamsoft_on_yolo_rois: bool = True,
    allowed_formats: set[str] | None = None,
) -> Dict[str, Any]:
    if img_bgr is None or not isinstance(img_bgr, np.ndarray) or img_bgr.size == 0:
        return {
            "status": "invalid_image",
            "items": [],
            "total": 0,
        }

    if allowed_formats is None:
        allowed_formats = set(DEFAULT_ALLOWED_FORMATS)

    collected_items: List[Dict[str, Any]] = []
    debug: Dict[str, Any] = {
        "dynamsoft_full_image": None,
        "yolo_rois": [],
        "yolo_pipeline": None,
        "dynamsoft_yolo_rois": [],
        "allowed_formats": sorted(list(allowed_formats)) if allowed_formats else [],
        "filtered_out": [],
    }

    # --------------------------------------------------------------
    # 1) Dynamsoft en imagen completa
    # --------------------------------------------------------------
    if use_dynamsoft_full_image:
        dyn_full = decode_barcode_dynamsoft(
            img_bgr,
            env_file=env_file,
        )
        debug["dynamsoft_full_image"] = dyn_full

        for item in dyn_full.get("items", []) or []:
            normalized = _normalize_dynamsoft_item(
                item=item,
                source="full_image",
            )
            collected_items.append(normalized)

    # --------------------------------------------------------------
    # 2) YOLO detección pura para ROIs
    # --------------------------------------------------------------
    rois = detect_barcode_rois_yolo(
        img_bgr=img_bgr,
        model_path=model_path,
        conf=yolo_conf,
        iou=yolo_iou,
        max_det=yolo_max_det,
        min_size=yolo_min_size,
    )
    debug["yolo_rois"] = rois

    crops = crop_rois(
        img=img_bgr,
        rois=rois,
        pad_ratio=yolo_pad_ratio,
    )

    # --------------------------------------------------------------
    # 3) Pipeline YOLO existente (por si rescata algo)
    # --------------------------------------------------------------
    if use_yolo_full_pipeline:
        yolo_result = detect_and_decode_with_yolo(
            img_bgr=img_bgr,
            decoder_fn=decode_barcode_1d,
            model_path=model_path,
            conf=yolo_conf,
            iou=yolo_iou,
            max_det=yolo_max_det,
            min_size=yolo_min_size,
            pad_ratio=yolo_pad_ratio,
            decoder_mode=yolo_decoder_mode,
            decoder_time_budget_ms=yolo_decoder_budget_ms,
        )
        debug["yolo_pipeline"] = yolo_result

        for item in yolo_result.get("items", []) or []:
            normalized = _normalize_yolo_item(item)
            collected_items.append(normalized)

    # --------------------------------------------------------------
    # 4) Dynamsoft sobre cada ROI detectado por YOLO
    # --------------------------------------------------------------
    if use_dynamsoft_on_yolo_rois:
        for crop_info in crops:
            roi_index = crop_info["roi_index"]
            roi_bbox_padded = _to_list_bbox(crop_info.get("bbox_xyxy_padded"))
            roi_bbox_original = _to_list_bbox(crop_info.get("bbox_xyxy_original"))
            roi_conf = float(crop_info.get("conf", 0.0) or 0.0)

            crop = crop_info["crop"]
            variants = _build_crop_variants(crop)

            roi_debug = {
                "roi_index": roi_index,
                "bbox_xyxy_original": roi_bbox_original,
                "bbox_xyxy_padded": roi_bbox_padded,
                "conf": roi_conf,
                "variants": [],
            }

            for variant_name, crop_variant in variants:
                dyn_roi = decode_barcode_dynamsoft(
                    crop_variant,
                    env_file=env_file,
                )

                roi_debug["variants"].append(
                    {
                        "variant": variant_name,
                        "result": dyn_roi,
                    }
                )

                for item in dyn_roi.get("items", []) or []:
                    normalized = _normalize_dynamsoft_item(
                        item=item,
                        source="roi",
                        bbox_override=roi_bbox_padded,
                        extra_meta={
                            "yolo_roi_index": roi_index,
                            "yolo_conf": roi_conf,
                            "yolo_bbox_xyxy_original": roi_bbox_original,
                            "yolo_bbox_xyxy_padded": roi_bbox_padded,
                            "roi_variant": variant_name,
                        },
                    )
                    collected_items.append(normalized)

            debug["dynamsoft_yolo_rois"].append(roi_debug)

    filtered_items, filtered_out = _filter_items_by_allowed_formats(
        collected_items,
        allowed_formats=allowed_formats,
    )
    debug["filtered_out"] = filtered_out

    final_items = _dedupe_items(filtered_items)

    return {
        "status": "success" if final_items else "not_found",
        "backend": "hybrid",
        "total": len(final_items),
        "items": final_items,
        "debug": debug,
    }


# ------------------------------------------------------------------
# Visualización
# ------------------------------------------------------------------
def annotate_hybrid_result(
    image: np.ndarray,
    result: Dict[str, Any],
) -> np.ndarray:
    if image is None:
        raise ValueError("La imagen es None")

    vis = image.copy()
    items = result.get("items", []) or []

    for idx, item in enumerate(items, start=1):
        bbox = item.get("bbox")
        text = str(item.get("text", "")).strip()
        fmt = item.get("format", "")
        source = item.get("source", "")
        backend = item.get("backend", "")

        label = f"{idx}: {fmt} | {text} | {source} | {backend}"

        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                vis,
                label[:140],
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                vis,
                label[:140],
                (20, 30 + (idx - 1) * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    return vis


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Readout híbrido de barcodes: YOLO + Dynamsoft"
    )
    parser.add_argument("image_path", help="Ruta de imagen a procesar")
    parser.add_argument("--env-file", default=".env", help="Ruta al archivo .env")
    parser.add_argument(
        "--model",
        default="runs/detect/runs_kuehne_nagel/barcode_v1/weights/best.pt",
        help="Ruta al modelo YOLO",
    )
    parser.add_argument("--yolo-conf", type=float, default=0.10)
    parser.add_argument("--yolo-iou", type=float, default=0.45)
    parser.add_argument("--yolo-max-det", type=int, default=10)
    parser.add_argument("--yolo-min-size", type=int, default=20) #40
    parser.add_argument("--yolo-pad-ratio", type=float, default=0.25)
    parser.add_argument("--yolo-decoder-mode", default="collect_plus")
    parser.add_argument("--yolo-decoder-budget", type=int, default=5000)

    parser.add_argument("--no-yolo-pipeline", action="store_true")
    parser.add_argument("--no-dynamsoft-full", action="store_true")
    parser.add_argument("--no-dynamsoft-rois", action="store_true")

    parser.add_argument(
        "--allowed-formats",
        nargs="+",
        default=sorted(list(DEFAULT_ALLOWED_FORMATS)),
        help="Formatos permitidos. Ej: --allowed-formats EAN_13 CODE_128 ITF",
    )

    parser.add_argument("--json", action="store_true", help="Imprime JSON completo")
    parser.add_argument("--save-json", action="store_true", help="Guarda JSON")
    parser.add_argument("--json-out", default=None, help="Ruta salida JSON")
    parser.add_argument("--save-vis", action="store_true", help="Guarda visualización")
    parser.add_argument("--vis-out", default=None, help="Ruta salida imagen anotada")

    return parser


def _default_json_out(image_path: str) -> str:
    p = Path(image_path)
    return str(p.with_name(f"{p.stem}_hybrid.json"))


def _default_vis_out(image_path: str) -> str:
    p = Path(image_path)
    return str(p.with_name(f"{p.stem}_hybrid_vis{p.suffix}"))


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        img = cv2.imread(args.image_path)
        if img is None:
            raise RuntimeError(f"No se pudo cargar la imagen: {args.image_path}")

        allowed_formats = {_normalize_format_name(x) for x in (args.allowed_formats or [])}

        result = read_barcodes_hybrid(
            img_bgr=img,
            env_file=args.env_file,
            model_path=args.model,
            yolo_conf=args.yolo_conf,
            yolo_iou=args.yolo_iou,
            yolo_max_det=args.yolo_max_det,
            yolo_min_size=args.yolo_min_size,
            yolo_pad_ratio=args.yolo_pad_ratio,
            yolo_decoder_mode=args.yolo_decoder_mode,
            yolo_decoder_budget_ms=args.yolo_decoder_budget,
            use_yolo_full_pipeline=not args.no_yolo_pipeline,
            use_dynamsoft_full_image=not args.no_dynamsoft_full,
            use_dynamsoft_on_yolo_rois=not args.no_dynamsoft_rois,
            allowed_formats=allowed_formats,
        )

        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print("\n=== RESULTADO HYBRID ===\n")
            print(f"Status: {result.get('status')}")
            print(f"Backend: {result.get('backend')}")
            print(f"Total: {result.get('total', 0)}")

            for idx, item in enumerate(result.get("items", []), start=1):
                print(f"\nItem {idx}")
                print(f"Texto:   {item.get('text')}")
                print(f"Formato: {item.get('format')}")
                print(f"Source:  {item.get('source')}")
                print(f"Backend: {item.get('backend')}")
                if item.get("bbox"):
                    print(f"BBox:    {item.get('bbox')}")

        if args.save_json:
            json_out = args.json_out or _default_json_out(args.image_path)
            Path(json_out).parent.mkdir(parents=True, exist_ok=True)
            with open(json_out, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\nJSON guardado en: {json_out}")

        if args.save_vis:
            vis = annotate_hybrid_result(img, result)
            vis_out = args.vis_out or _default_vis_out(args.image_path)
            Path(vis_out).parent.mkdir(parents=True, exist_ok=True)
            ok = cv2.imwrite(vis_out, vis)
            if not ok:
                raise RuntimeError(f"No se pudo guardar visualización en: {vis_out}")
            print(f"Visualización guardada en: {vis_out}")

        return 0

    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
