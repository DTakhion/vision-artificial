# utils/vision_barcode_plus.py
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from utils.vision_barcode import decode_barcode_1d


# -------------------------------------------------------------
# Basic image helpers
# -------------------------------------------------------------
def _zoom(img: np.ndarray, factor: float) -> np.ndarray:
    if factor <= 1.0:
        return img
    return cv2.resize(
        img,
        None,
        fx=factor,
        fy=factor,
        interpolation=cv2.INTER_CUBIC,
    )


def _rotate_image(img: np.ndarray, name: str) -> np.ndarray:
    if name == "rot0":
        return img
    if name == "rot90":
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if name == "rot180":
        return cv2.rotate(img, cv2.ROTATE_180)
    if name == "rot270":
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


# -------------------------------------------------------------
# Strategic crops
# -------------------------------------------------------------
def _clip_box(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    w: int,
    h: int,
) -> Tuple[int, int, int, int]:
    x0 = max(0, min(x0, w - 1))
    y0 = max(0, min(y0, h - 1))
    x1 = max(x0 + 1, min(x1, w))
    y1 = max(y0 + 1, min(y1, h))
    return x0, y0, x1, y1


def _crop(
    img: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> np.ndarray:
    return img[y0:y1, x0:x1].copy()


def _generate_strategic_crops(img: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """
    Conjunto pequeño pero útil de crops:
    - bandas horizontales con overlap
    - bandas verticales con overlap
    - centro
    - cuadrantes amplios
    """
    h, w = img.shape[:2]
    out: List[Tuple[str, np.ndarray]] = []

    bands_h = [
        ("top_band", 0.00, 0.00, 1.00, 0.45),
        ("mid_band", 0.00, 0.22, 1.00, 0.78),
        ("bottom_band", 0.00, 0.55, 1.00, 1.00),
    ]

    bands_v = [
        ("left_band", 0.00, 0.00, 0.45, 1.00),
        ("center_band", 0.22, 0.00, 0.78, 1.00),
        ("right_band", 0.55, 0.00, 1.00, 1.00),
    ]

    center = [("center", 0.15, 0.12, 0.85, 0.88)]

    quads = [
        ("q1", 0.00, 0.00, 0.62, 0.62),
        ("q2", 0.38, 0.00, 1.00, 0.62),
        ("q3", 0.00, 0.38, 0.62, 1.00),
        ("q4", 0.38, 0.38, 1.00, 1.00),
    ]

    for name, fx0, fy0, fx1, fy1 in bands_h + bands_v + center + quads:
        x0 = int(round(w * fx0))
        y0 = int(round(h * fy0))
        x1 = int(round(w * fx1))
        y1 = int(round(h * fy1))
        x0, y0, x1, y1 = _clip_box(x0, y0, x1, y1, w, h)
        out.append((name, _crop(img, x0, y0, x1, y1)))

    return out


# -------------------------------------------------------------
# Micro crops for horizontal barcodes (yellow box rescue)
# -------------------------------------------------------------
def _generate_barcode_micro_crops(img: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """
    Micro-crops orientados a rescatar códigos horizontales pequeños
    en la zona media/baja de la imagen.
    Especialmente útil para la caja amarilla.
    """
    h, w = img.shape[:2]
    out: List[Tuple[str, np.ndarray]] = []

    # Banda media-baja donde aparecen varios productos con barcode horizontal
    y0 = int(h * 0.43)
    y1 = int(h * 0.86)
    y0, y1 = max(0, y0), min(h, y1)
    if y1 <= y0:
        return out

    band = img[y0:y1, :]

    # Ventanas verticales solapadas
    step = max(1, int(w * 0.16))
    win_w = max(1, int(w * 0.34))

    x = 0
    idx = 0
    while x < w:
        x0 = x
        x1 = min(w, x + win_w)
        if x1 - x0 >= max(40, int(w * 0.16)):
            crop = band[:, x0:x1].copy()
            out.append((f"micro_{idx}", crop))
        if x + win_w >= w:
            break
        x += step
        idx += 1

    # Crop dirigido hacia la caja amarilla (zona inferior-derecha)
    x0 = int(w * 0.48)
    y0 = int(h * 0.48)
    x1 = int(w * 0.92)
    y1 = int(h * 0.86)
    x0, y0, x1, y1 = _clip_box(x0, y0, x1, y1, w, h)
    out.append(("yellow_box_focus", _crop(img, x0, y0, x1, y1)))

    return out


# -------------------------------------------------------------
# Barcode validation helpers
# -------------------------------------------------------------
def _normalize_text(txt: Any) -> Optional[str]:
    if txt is None:
        return None
    try:
        s = "".join(str(txt).strip().split())
        return s if s else None
    except Exception:
        return None


def _ean13_checksum_digit(first12: str) -> Optional[int]:
    if len(first12) != 12 or not first12.isdigit():
        return None

    total = 0
    for i, ch in enumerate(first12):
        d = int(ch)
        if (i + 1) % 2 == 0:
            total += 3 * d
        else:
            total += d

    return (10 - (total % 10)) % 10


def _is_valid_ean13(text: str) -> bool:
    if len(text) != 13 or not text.isdigit():
        return False
    check = _ean13_checksum_digit(text[:12])
    if check is None:
        return False
    return check == int(text[-1])


def _looks_like_ean13(item: Dict[str, Any]) -> bool:
    txt = _normalize_text(item.get("text"))
    if not txt:
        return False

    fmt = str(item.get("format") or "").upper().replace("-", "")
    if fmt == "EAN13":
        return True

    return len(txt) == 13 and txt.isdigit()


def _passes_barcode_validation(item: Dict[str, Any]) -> bool:
    txt = _normalize_text(item.get("text"))
    if not txt:
        return False

    if _looks_like_ean13(item):
        return _is_valid_ean13(txt)

    return True


# -------------------------------------------------------------
# Scoring / confidence
# -------------------------------------------------------------
def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _compute_plus_score(item: Dict[str, Any]) -> float:
    """
    Score complementario (no reemplaza el score base).
    Usa:
    - score del motor base
    - validación checksum
    - backend
    - source/source_plus
    """
    score = _safe_float(item.get("score"), 0.0)
    backend = str(item.get("backend") or "").lower()
    source = str(item.get("source") or "").lower()
    source_plus = str(item.get("source_plus") or "").lower()

    if _passes_barcode_validation(item):
        score += 1.25
    else:
        score -= 2.0

    if backend == "zxingcpp":
        score += 0.35
    elif backend == "pyzbar":
        score += 0.15

    if source == "roi":
        score += 0.15
    elif source == "tile":
        score += 0.10

    if source_plus in ("center", "center_band", "yellow_box_focus"):
        score += 0.10

    return score


def _dedup_by_text_keep_best(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}

    for item in items:
        txt = _normalize_text(item.get("text"))
        if not txt:
            continue

        x = dict(item)
        x["text"] = txt
        x["plus_score"] = _compute_plus_score(x)

        prev = best.get(txt)
        if prev is None or _safe_float(x.get("plus_score")) > _safe_float(prev.get("plus_score")):
            best[txt] = x

    out = list(best.values())
    out.sort(key=lambda z: _safe_float(z.get("plus_score")), reverse=True)
    return out


# -------------------------------------------------------------
# Classification
# -------------------------------------------------------------
def _classify_items(items: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    confirmed: List[Dict[str, Any]] = []
    suspect: List[Dict[str, Any]] = []

    for item in items:
        txt = _normalize_text(item.get("text"))
        if not txt:
            continue

        is_valid = _passes_barcode_validation(item)
        plus_score = _safe_float(item.get("plus_score"), 0.0)

        x = dict(item)
        x["validation"] = {
            "ean13_valid": _is_valid_ean13(txt) if _looks_like_ean13(item) else None,
            "passes_basic_validation": is_valid,
        }

        if _looks_like_ean13(item):
            if is_valid:
                confirmed.append(x)
            else:
                suspect.append(x)
        else:
            if plus_score >= 3.0:
                confirmed.append(x)
            else:
                suspect.append(x)

    confirmed.sort(key=lambda z: _safe_float(z.get("plus_score")), reverse=True)
    suspect.sort(key=lambda z: _safe_float(z.get("plus_score")), reverse=True)
    return confirmed, suspect


# -------------------------------------------------------------
# Summary
# -------------------------------------------------------------
def _build_summary(
    confirmed_items: List[Dict[str, Any]],
    suspect_items: List[Dict[str, Any]],
    total_candidates_before_filter: int,
) -> Dict[str, Any]:
    confirmed_texts = sorted({_normalize_text(x.get("text")) for x in confirmed_items if _normalize_text(x.get("text"))})
    suspect_texts = sorted({_normalize_text(x.get("text")) for x in suspect_items if _normalize_text(x.get("text"))})

    return {
        "total_candidates_before_filter": int(total_candidates_before_filter),
        "total_confirmed_items": len(confirmed_items),
        "total_confirmed_texts": len(confirmed_texts),
        "confirmed_texts": confirmed_texts,
        "total_suspect_items": len(suspect_items),
        "total_suspect_texts": len(suspect_texts),
        "suspect_texts": suspect_texts,
    }


# -------------------------------------------------------------
# Single pass helper
# -------------------------------------------------------------
def _run_collect_pass(
    img: np.ndarray,
    *,
    pass_name: str,
    budget_ms: int,
    variants: Optional[str] = None,
    roi_upscale: float = 4.0,
) -> List[Dict[str, Any]]:
    res = decode_barcode_1d(
        img,
        mode="collect",
        time_budget_ms=max(80, int(budget_ms)),
        variants=variants,
        roi_upscale=roi_upscale,
    )

    items = res.get("items", []) or []
    out: List[Dict[str, Any]] = []

    for item in items:
        x = dict(item)
        x["source_plus"] = pass_name
        out.append(x)

    return out


# -------------------------------------------------------------
# Main PLUS pipeline
# -------------------------------------------------------------
def decode_barcode_1d_plus(
    img_bgr: np.ndarray,
    *,
    budget_ms: int = 5000,
    variants: Optional[str] = None,
    roi_upscale: float = 4.0,
    include_rotations: bool = True,
    include_zoom_rescue: bool = True,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    aggregate: List[Dict[str, Any]] = []

    try:
        # ---------------------------------------------------------
        # PASS 1 — full image baseline
        # ---------------------------------------------------------
        base_budget = max(250, int(budget_ms * 0.28))
        aggregate.extend(
            _run_collect_pass(
                img_bgr,
                pass_name="full_base",
                budget_ms=base_budget,
                variants=variants,
                roi_upscale=roi_upscale,
            )
        )

        # ---------------------------------------------------------
        # PASS 2 — strategic crops
        # ---------------------------------------------------------
        crops = _generate_strategic_crops(img_bgr)
        per_crop_budget = 240

        for cname, crop in crops:
            if (time.perf_counter() - t0) * 1000 > budget_ms:
                break

            aggregate.extend(
                _run_collect_pass(
                    crop,
                    pass_name=cname,
                    budget_ms=per_crop_budget,
                    variants=variants,
                    roi_upscale=roi_upscale,
                )
            )

        # ---------------------------------------------------------
        # PASS 2B — micro crops barcode rescue (yellow box)
        # ---------------------------------------------------------
        micro_crops = _generate_barcode_micro_crops(img_bgr)

        for cname, crop in micro_crops:
            if (time.perf_counter() - t0) * 1000 > budget_ms:
                break

            crop_zoom = _zoom(crop, 1.8)

            aggregate.extend(
                _run_collect_pass(
                    crop_zoom,
                    pass_name=cname,
                    budget_ms=220,
                    variants=variants,
                    roi_upscale=roi_upscale,
                )
            )

        # ---------------------------------------------------------
        # PASS 3 — full image rotations
        # ---------------------------------------------------------
        if include_rotations:
            for rname in ("rot90", "rot180", "rot270"):
                if (time.perf_counter() - t0) * 1000 > budget_ms:
                    break

                rimg = _rotate_image(img_bgr, rname)
                aggregate.extend(
                    _run_collect_pass(
                        rimg,
                        pass_name=rname,
                        budget_ms=260,
                        variants=variants,
                        roi_upscale=roi_upscale,
                    )
                )

        # ---------------------------------------------------------
        # PASS 4 — zoom rescue on center-ish crops
        # ---------------------------------------------------------
        if include_zoom_rescue:
            rescue_targets = [
                (
                    "center_zoom2",
                    img_bgr[
                        int(img_bgr.shape[0] * 0.12):int(img_bgr.shape[0] * 0.88),
                        int(img_bgr.shape[1] * 0.15):int(img_bgr.shape[1] * 0.85),
                    ],
                ),
                ("left_zoom2", img_bgr[:, :int(img_bgr.shape[1] * 0.55)]),
                ("right_zoom2", img_bgr[:, int(img_bgr.shape[1] * 0.45):]),
            ]

            for zname, crop in rescue_targets:
                if (time.perf_counter() - t0) * 1000 > budget_ms:
                    break

                zimg = _zoom(crop, 2.0)
                aggregate.extend(
                    _run_collect_pass(
                        zimg,
                        pass_name=zname,
                        budget_ms=260,
                        variants=variants,
                        roi_upscale=roi_upscale,
                    )
                )

        # ---------------------------------------------------------
        # FINAL FILTERING
        # ---------------------------------------------------------
        total_candidates_before_filter = len(aggregate)
        deduped = _dedup_by_text_keep_best(aggregate)
        confirmed_items, suspect_items = _classify_items(deduped)

        elapsed = int((time.perf_counter() - t0) * 1000)

        return {
            "status": "success" if (confirmed_items or suspect_items) else "not_found",
            "mode": "collect_plus",
            "confirmed_items": confirmed_items,
            "suspect_items": suspect_items,
            "items": confirmed_items,
            "elapsed_ms": elapsed,
            "summary": _build_summary(
                confirmed_items=confirmed_items,
                suspect_items=suspect_items,
                total_candidates_before_filter=total_candidates_before_filter,
            ),
        }

    except Exception as e:
        elapsed = int((time.perf_counter() - t0) * 1000)
        return {
            "status": "error",
            "mode": "collect_plus",
            "confirmed_items": [],
            "suspect_items": [],
            "items": [],
            "elapsed_ms": elapsed,
            "error": repr(e),
            "summary": _build_summary([], [], 0),
        }


# -------------------------------------------------------------
# CLI
# -------------------------------------------------------------
def _cli() -> int:
    import argparse

    p = argparse.ArgumentParser(
        description="Enhanced barcode collect wrapper with strategic crops, rotations, zoom rescue and EAN-13 validation."
    )
    p.add_argument("image_path", type=str, help="Path to image")
    p.add_argument("--budget", type=int, default=5000, help="Total budget in ms")
    p.add_argument(
        "--variants",
        type=str,
        default=None,
        help='Variants for underlying decoder, e.g. "all" or "gray,sharp,bw_x2"',
    )
    p.add_argument("--roi_upscale", type=float, default=4.0, help="ROI upscale passed to underlying decoder")
    p.add_argument("--no_rotations", action="store_true", help="Disable full-image rescue rotations")
    p.add_argument("--no_zoom_rescue", action="store_true", help="Disable zoom rescue stage")

    args = p.parse_args()

    img = cv2.imread(args.image_path)
    if img is None:
        print({"status": "error", "error": "Could not load image", "path": args.image_path})
        return 2

    res = decode_barcode_1d_plus(
        img,
        budget_ms=int(args.budget),
        variants=args.variants,
        roi_upscale=float(args.roi_upscale),
        include_rotations=(not args.no_rotations),
        include_zoom_rescue=(not args.no_zoom_rescue),
    )
    print(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())