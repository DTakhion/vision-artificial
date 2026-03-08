# utils/vision_barcode.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, Tuple, Set
import time

import cv2
import numpy as np

from utils.vision_preprocess import preprocess_variants


# Variantes recomendadas (ordenadas por probabilidad de éxito + costo)
DEFAULT_VARIANTS_BARCODE: List[str] = [
    "gray",
    "sharp",
    "bilateral",
    "bilateral_sharp",
    "bilateral_x2",
    "sharp_x2",
    "morph_close",
    "morph_close_x2",
    "bw",
    "bw_x2",
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _resolve_variants(
    variants: Optional[Union[List[str], str]],
    available: List[str],
) -> List[str]:
    """
    variants:
      - None -> DEFAULT_VARIANTS_BARCODE filtrado por disponibles
      - "all" -> todos los disponibles
      - list[str] -> usa esos nombres (en ese orden), filtrando por disponibles
      - "gray,sharp,bw" -> comma-list
      - "gray" -> single
    """
    if variants is None:
        return [v for v in DEFAULT_VARIANTS_BARCODE if v in available]
    if isinstance(variants, str):
        if variants.lower() == "all":
            return available
        if "," in variants:
            req = [x.strip() for x in variants.split(",") if x.strip()]
            return [v for v in req if v in available]
        return [variants] if variants in available else []
    return [v for v in variants if v in available]


def _resize_gray(gray: np.ndarray, factor: float) -> np.ndarray:
    if factor <= 1.0:
        return gray
    return cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)


def _backend_available() -> Dict[str, bool]:
    avail = {"zxingcpp": False, "pyzbar": False, "opencv_barcode": False}

    try:
        import zxingcpp  # type: ignore  # noqa: F401
        avail["zxingcpp"] = True
    except Exception:
        pass

    try:
        from pyzbar.pyzbar import decode as _d  # type: ignore  # noqa: F401
        avail["pyzbar"] = True
    except Exception:
        pass

    try:
        _ = _get_opencv_barcode_detector()
        avail["opencv_barcode"] = _ is not None
    except Exception:
        avail["opencv_barcode"] = False

    return avail


def _get_opencv_barcode_detector() -> Any:
    if hasattr(cv2, "barcode") and hasattr(cv2.barcode, "BarcodeDetector"):
        return cv2.barcode.BarcodeDetector()
    if hasattr(cv2, "barcode_BarcodeDetector"):
        return cv2.barcode_BarcodeDetector()
    return None


def _serialize_zxing_position(pos: Any) -> Any:
    """
    Convierte zxingcpp.Position a una forma serializable (lista de puntos).
    """
    if pos is None:
        return None

    corners: List[Tuple[float, float]] = []
    for name in ("top_left", "top_right", "bottom_right", "bottom_left"):
        if hasattr(pos, name):
            p = getattr(pos, name, None)
            if p is not None and hasattr(p, "x") and hasattr(p, "y"):
                corners.append((float(p.x), float(p.y)))
    if corners:
        return corners

    try:
        pts: List[Tuple[float, float]] = []
        for p in pos:
            if hasattr(p, "x") and hasattr(p, "y"):
                pts.append((float(p.x), float(p.y)))
        return pts or str(pos)
    except Exception:
        return str(pos)


def _safe_str(x: Any) -> Optional[str]:
    if x is None:
        return None
    try:
        s = str(x).strip()
        return s if s else None
    except Exception:
        return None


def _normalize_format(fmt: Any) -> Optional[str]:
    s = _safe_str(fmt)
    if not s:
        return None
    return s.upper()


def _normalize_text(txt: Any) -> Optional[str]:
    s = _safe_str(txt)
    if not s:
        return None
    return s.strip()


def _normalize_text_key(txt: Any) -> Optional[str]:
    """
    Clave de texto para deduplicación por código "real".
    No usa format como parte de la clave.
    """
    s = _normalize_text(txt)
    if not s:
        return None
    s = "".join(s.split())
    return s or None


def _build_collect_summary(
    items: List[Dict[str, Any]],
    *,
    from_rois: bool,
    from_tiles: bool,
    from_full_image: bool,
) -> Dict[str, Any]:
    unique_texts: List[str] = []
    seen: Set[str] = set()

    for item in items:
        key = _normalize_text_key(item.get("text"))
        if not key or key in seen:
            continue
        seen.add(key)
        unique_texts.append(key)

    unique_texts.sort()

    return {
        "total_unique": len(items),  # compatibilidad hacia atrás
        "total_unique_items": len(items),  # dedup actual por (text, format)
        "total_unique_texts": len(unique_texts),  # dedup por texto normalizado
        "unique_texts": unique_texts,
        "from_rois": from_rois,
        "from_tiles": from_tiles,
        "from_full_image": from_full_image,
    }


def _points_to_list(points: Any) -> Any:
    if points is None:
        return None
    try:
        arr = np.asarray(points)
        return arr.tolist()
    except Exception:
        return points


def _normalize_rect(rect: Any) -> Any:
    if rect is None:
        return None
    try:
        if hasattr(rect, "left") and hasattr(rect, "top") and hasattr(rect, "width") and hasattr(rect, "height"):
            return (int(rect.left), int(rect.top), int(rect.width), int(rect.height))
        if isinstance(rect, (tuple, list)) and len(rect) == 4:
            return tuple(int(v) for v in rect)
    except Exception:
        pass
    return rect


def _score_item(item: Dict[str, Any]) -> float:
    """
    Heurística simple de score/ranking.
    No pretende ser "confianza real", solo priorización.
    """
    score = 0.0

    backend = _safe_str(item.get("backend"))
    if backend == "zxingcpp":
        score += 3.0
    elif backend == "pyzbar":
        score += 2.0
    elif backend == "opencv_barcode":
        score += 1.0

    txt = _normalize_text(item.get("text"))
    if txt:
        score += min(len(txt), 32) * 0.03

    fmt = _normalize_format(item.get("format"))
    if fmt:
        score += 0.4

    source = _safe_str(item.get("source"))
    if source == "roi":
        score += 0.8
    elif source == "tile":
        score += 0.55
    elif source == "full":
        score += 0.3

    roi_score = item.get("roi_score")
    try:
        if roi_score is not None:
            score += min(float(roi_score) / 1_000_000.0, 2.0)
    except Exception:
        pass

    points = item.get("points") or item.get("position")
    if points is not None:
        score += 0.25

    rect = item.get("rect")
    if rect is not None:
        score += 0.15

    candidate = _safe_str(item.get("candidate"))
    if candidate:
        if "x4" in candidate:
            score += 0.18
        elif "x3" in candidate:
            score += 0.12
        if "rot90" in candidate or "rot270" in candidate:
            score += 0.05

    return float(score)


def _dedup_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicación por (text, format).
    Conserva el mejor item según score.
    """
    best: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for item in items:
        txt = _normalize_text(item.get("text"))
        if not txt:
            continue

        fmt = _normalize_format(item.get("format")) or "UNKNOWN"
        key = (txt, fmt)

        candidate = dict(item)
        candidate["text"] = txt
        candidate["format"] = None if fmt == "UNKNOWN" else fmt
        candidate["score"] = _score_item(candidate)

        prev = best.get(key)
        if prev is None or float(candidate["score"]) > float(prev.get("score", -1.0)):
            best[key] = candidate

    out = list(best.values())
    out.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return out


def _merge_items(
    aggregate: List[Dict[str, Any]],
    new_items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    return _dedup_items(aggregate + new_items)


def _annotate_items(
    items: List[Dict[str, Any]],
    *,
    variant: str,
    candidate: str,
    source: str,
    roi_index: Optional[int] = None,
    roi_bbox: Optional[Tuple[int, int, int, int]] = None,
    roi_score: Optional[float] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in items:
        x = dict(item)
        x["variant"] = variant
        x["candidate"] = candidate
        x["source"] = source
        if roi_index is not None:
            x["roi_index"] = roi_index
        if roi_bbox is not None:
            x["roi_bbox"] = roi_bbox
        if roi_score is not None:
            x["roi_score"] = roi_score
        out.append(x)
    return out


# ---------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------
def _try_pyzbar(gray: np.ndarray) -> List[Dict[str, Any]]:
    try:
        from pyzbar.pyzbar import decode  # type: ignore
    except Exception:
        return []

    out: List[Dict[str, Any]] = []
    for d in decode(gray):
        try:
            data = d.data.decode("utf-8", errors="replace")
        except Exception:
            data = str(getattr(d, "data", b""))

        rect = getattr(d, "rect", None)
        polygon = getattr(d, "polygon", None)
        points = None
        if polygon is not None:
            try:
                points = [(int(p.x), int(p.y)) for p in polygon]
            except Exception:
                points = None

        out.append(
            {
                "text": data,
                "format": getattr(d, "type", None),
                "rect": _normalize_rect(rect),
                "points": points,
                "backend": "pyzbar",
            }
        )
    return out


def _try_zxingcpp(gray: np.ndarray) -> List[Dict[str, Any]]:
    try:
        import zxingcpp  # type: ignore
    except Exception:
        return []

    try:
        res = zxingcpp.read_barcodes(gray)
    except Exception:
        return []

    out: List[Dict[str, Any]] = []
    for r in res:
        txt = getattr(r, "text", None)
        if not txt:
            continue

        pos = None
        if hasattr(r, "position"):
            try:
                pos = _serialize_zxing_position(getattr(r, "position", None))
            except Exception:
                pos = None

        out.append(
            {
                "text": txt,
                "format": str(getattr(r, "format", None)),
                "position": pos,
                "backend": "zxingcpp",
            }
        )
    return out


def _try_opencv_barcode(gray: np.ndarray) -> List[Dict[str, Any]]:
    det = _get_opencv_barcode_detector()
    if det is None:
        return []

    out: List[Dict[str, Any]] = []

    try:
        r = det.detectAndDecode(gray)
        if isinstance(r, tuple) and len(r) >= 2:
            ok = bool(r[0]) if isinstance(r[0], (bool, np.bool_)) else True
            decoded_info = r[1] if len(r) > 1 else []
            decoded_type = r[2] if len(r) > 2 else None
            points = r[3] if len(r) > 3 else None

            if ok and decoded_info:
                if isinstance(decoded_info, str):
                    decoded_list = [decoded_info] if decoded_info else []
                else:
                    decoded_list = [x for x in decoded_info if x]

                for i, txt in enumerate(decoded_list):
                    pts_i = None
                    if points is not None:
                        try:
                            p = np.asarray(points)
                            if p.ndim == 3 and i < p.shape[0]:
                                pts_i = p[i].tolist()
                            elif p.ndim == 2:
                                pts_i = p.tolist()
                        except Exception:
                            pts_i = None

                    fmt_i = None
                    if isinstance(decoded_type, (list, tuple)) and i < len(decoded_type):
                        fmt_i = decoded_type[i]
                    else:
                        fmt_i = decoded_type

                    out.append(
                        {
                            "text": txt,
                            "format": fmt_i,
                            "points": pts_i,
                            "backend": "opencv_barcode",
                        }
                    )
                if out:
                    return out
    except Exception:
        pass

    try:
        rr = det.detect(gray)
        ok = False
        points = None
        if isinstance(rr, tuple):
            if len(rr) >= 2:
                ok = bool(rr[0])
                points = rr[1]
        else:
            ok = bool(rr)

        if ok and points is not None:
            try:
                dec = det.decode(gray, points)
                if isinstance(dec, tuple) and len(dec) >= 2:
                    decoded_info = dec[0]
                    decoded_type = dec[1] if len(dec) > 1 else None

                    if isinstance(decoded_info, str):
                        decoded_list = [decoded_info] if decoded_info else []
                    else:
                        decoded_list = [x for x in decoded_info if x]

                    p = np.asarray(points)

                    for i, txt in enumerate(decoded_list):
                        pts_i = None
                        try:
                            if p.ndim == 3 and i < p.shape[0]:
                                pts_i = p[i].tolist()
                            elif p.ndim == 2:
                                pts_i = p.tolist()
                        except Exception:
                            pts_i = None

                        fmt_i = None
                        if isinstance(decoded_type, (list, tuple)) and i < len(decoded_type):
                            fmt_i = decoded_type[i]
                        else:
                            fmt_i = decoded_type

                        out.append(
                            {
                                "text": txt,
                                "format": fmt_i,
                                "points": pts_i,
                                "backend": "opencv_barcode",
                            }
                        )
            except Exception:
                pass
    except Exception:
        pass

    return out


# ---------------------------------------------------------------------
# Extra candidates (blur rescue)
# ---------------------------------------------------------------------
def _unsharp(g: np.ndarray, amount: float = 1.2) -> np.ndarray:
    blur = cv2.GaussianBlur(g, (0, 0), sigmaX=1.2)
    return cv2.addWeighted(g, 1.0 + amount, blur, -amount, 0)


def _otsu(g: np.ndarray) -> np.ndarray:
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw


def _adapt(g: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)


def _extra_candidates(gray: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """
    Candidatos fuertes para rescatar blur / barcode pequeño.
    Mantiene orden “probable”.
    """
    cands: List[Tuple[str, np.ndarray]] = []

    g = gray
    g_x3 = _resize_gray(g, 3.0)
    g_x4 = _resize_gray(g, 4.0)

    s = _unsharp(g, 1.2)
    s_x3 = _resize_gray(s, 3.0)
    s_x4 = _resize_gray(s, 4.0)

    cands.extend(
        [
            ("sharp_x4_local", s_x4),
            ("gray_x4_local", g_x4),
            ("sharp_x3_local", s_x3),
            ("gray_x3_local", g_x3),
            ("sharp_local", s),
            ("gray_local", g),
        ]
    )

    try:
        cands.extend(
            [
                ("otsu_gray_x4_local", _otsu(g_x4)),
                ("adapt_gray_x4_local", _adapt(g_x4)),
                ("otsu_gray_local", _otsu(g)),
                ("adapt_gray_local", _adapt(g)),
            ]
        )
    except Exception:
        pass

    return cands


def _rot90(gray: np.ndarray, k: int) -> np.ndarray:
    k = k % 4
    if k == 0:
        return gray
    if k == 1:
        return cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
    if k == 2:
        return cv2.rotate(gray, cv2.ROTATE_180)
    return cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)


def _augment_with_rotations(
    candidates: List[Tuple[str, np.ndarray]],
    *,
    enable_rotations: bool,
    only_for: Optional[Set[str]] = None,
) -> List[Tuple[str, np.ndarray]]:
    if not enable_rotations:
        return candidates

    out: List[Tuple[str, np.ndarray]] = []
    for cname, gray in candidates:
        out.append((cname, gray))

        if only_for is not None and cname not in only_for:
            continue

        out.append((f"{cname}_rot90", _rot90(gray, 1)))
        out.append((f"{cname}_rot180", _rot90(gray, 2)))
        out.append((f"{cname}_rot270", _rot90(gray, 3)))
    return out


# ---------------------------------------------------------------------
# ROI rescue (barcode region detection)
# ---------------------------------------------------------------------
def _find_barcode_rois(
    img_bgr: np.ndarray,
    max_rois: int = 3,
) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
    """
    Heurística 1D:
      - gradiente en X (barras verticales)
      - threshold + morph close
      - contornos grandes y con buen aspecto

    Ajuste quirúrgico:
      - ahora acepta regiones alargadas horizontales o verticales
      - mantiene sesgo a ROIs útiles, pero no excluye códigos de pie
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gx = cv2.convertScaleAbs(gx)
    gx = cv2.GaussianBlur(gx, (0, 0), 2.0)

    _, bw = cv2.threshold(gx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 7))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=2)
    bw = cv2.erode(bw, None, iterations=1)
    bw = cv2.dilate(bw, None, iterations=2)

    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    out: List[Tuple[np.ndarray, Dict[str, Any]]] = []

    for c in cnts[:20]:
        x, y, ww, hh = cv2.boundingRect(c)
        area = float(ww * hh)
        if area < 0.006 * (w * h):
            continue

        aspect = ww / float(hh + 1e-6)
        aspect_inv = hh / float(ww + 1e-6)
        elongated = max(aspect, aspect_inv)

        if elongated < 2.2:
            continue

        pad = int(0.08 * max(ww, hh))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(w, x + ww + pad)
        y1 = min(h, y + hh + pad)
        if x1 <= x0 or y1 <= y0:
            continue

        roi = img_bgr[y0:y1, x0:x1].copy()
        score = area * elongated
        meta = {"bbox": (x0, y0, x1 - x0, y1 - y0), "score": score}
        out.append((roi, meta))

        if len(out) >= max_rois:
            break

    out.sort(key=lambda t: float(t[1].get("score", 0.0)), reverse=True)
    return out


def _generate_multibarcode_tiles(
    img_bgr: np.ndarray,
    max_tiles: int = 5,
) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
    """
    Tiles amplios y superpuestos para rescatar códigos pequeños/lejanos
    sin caer en sliding-window caro.
    """
    h, w = img_bgr.shape[:2]
    out: List[Tuple[np.ndarray, Dict[str, Any]]] = []

    windows = [
        ("center", 0.10, 0.10, 0.80, 0.80),
        ("tl", 0.00, 0.00, 0.65, 0.65),
        ("tr", 0.35, 0.00, 0.65, 0.65),
        ("bl", 0.00, 0.35, 0.65, 0.65),
        ("br", 0.35, 0.35, 0.65, 0.65),
    ][: max(1, max_tiles)]

    for name, fx, fy, fw, fh in windows:
        x0 = int(round(w * fx))
        y0 = int(round(h * fy))
        x1 = int(round(w * (fx + fw)))
        y1 = int(round(h * (fy + fh)))

        x0 = max(0, min(x0, w - 1))
        y0 = max(0, min(y0, h - 1))
        x1 = max(x0 + 1, min(x1, w))
        y1 = max(y0 + 1, min(y1, h))

        crop = img_bgr[y0:y1, x0:x1].copy()
        out.append((crop, {"name": name, "bbox": (x0, y0, x1 - x0, y1 - y0)}))

    return out


# ---------------------------------------------------------------------
# Core decoding helpers
# ---------------------------------------------------------------------
def _run_backend(be: str, gray: np.ndarray) -> List[Dict[str, Any]]:
    if be == "zxingcpp":
        return _try_zxingcpp(gray)
    if be == "pyzbar":
        return _try_pyzbar(gray)
    return _try_opencv_barcode(gray)


def _prepare_variants(
    img_bgr: np.ndarray,
    variants: Optional[Union[List[str], str]],
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    ims = preprocess_variants(img_bgr)
    available = list(ims.keys())
    to_try = _resolve_variants(variants, available)
    if not to_try:
        to_try = [v for v in DEFAULT_VARIANTS_BARCODE if v in available] or available
    return ims, to_try


def _decode_barcode_core_fast(
    img_bgr: np.ndarray,
    *,
    t0: float,
    time_budget_ms: int,
    variants: Optional[Union[List[str], str]],
    backend_order: List[str],
    source: str,
    roi_index: Optional[int] = None,
    roi_bbox: Optional[Tuple[int, int, int, int]] = None,
    roi_score: Optional[float] = None,
) -> Dict[str, Any]:
    tried: List[str] = []
    stage_t0 = time.perf_counter()

    ims, to_try = _prepare_variants(img_bgr, variants)

    for vname in to_try:
        if (time.perf_counter() - stage_t0) * 1000 > time_budget_ms:
            break

        base = ims.get(vname)
        if base is None:
            continue

        candidates: List[Tuple[str, np.ndarray]] = [(vname, base)]
        candidates.extend(_extra_candidates(base))

        for cname, gray in candidates:
            if (time.perf_counter() - stage_t0) * 1000 > time_budget_ms:
                break

            for be in backend_order:
                if (time.perf_counter() - stage_t0) * 1000 > time_budget_ms:
                    break

                tried.append(f"{source}:{cname}:{be}")
                items = _run_backend(be, gray)

                if items:
                    items = _annotate_items(
                        items,
                        variant=vname,
                        candidate=cname,
                        source=source,
                        roi_index=roi_index,
                        roi_bbox=roi_bbox,
                        roi_score=roi_score,
                    )
                    items = _dedup_items(items)
                    elapsed = int((time.perf_counter() - t0) * 1000)
                    return {
                        "status": "success",
                        "items": items,
                        "backend": be,
                        "elapsed_ms": elapsed,
                        "variant": vname,
                        "candidate": cname,
                        "source": source,
                    }

    elapsed = int((time.perf_counter() - t0) * 1000)
    return {
        "status": "not_found",
        "items": [],
        "backend": None,
        "elapsed_ms": elapsed,
        "variant": None,
        "candidate": None,
        "source": source,
        "tried": tried,
    }


def _decode_barcode_core_collect(
    img_bgr: np.ndarray,
    *,
    t0: float,
    time_budget_ms: int,
    variants: Optional[Union[List[str], str]],
    backend_order: List[str],
    source: str,
    roi_index: Optional[int] = None,
    roi_bbox: Optional[Tuple[int, int, int, int]] = None,
    roi_score: Optional[float] = None,
    max_collect_items: int = 64,
    enable_rotations: bool = False,
) -> Dict[str, Any]:
    tried: List[str] = []
    aggregate: List[Dict[str, Any]] = []
    stage_t0 = time.perf_counter()

    ims, to_try = _prepare_variants(img_bgr, variants)

    strong_rotation_candidates = {
        "sharp_x4_local",
        "gray_x4_local",
        "sharp_x3_local",
        "gray_x3_local",
    }

    for vname in to_try:
        if (time.perf_counter() - stage_t0) * 1000 > time_budget_ms:
            break

        base = ims.get(vname)
        if base is None:
            continue

        candidates: List[Tuple[str, np.ndarray]] = [(vname, base)]
        candidates.extend(_extra_candidates(base))

        candidates = _augment_with_rotations(
            candidates,
            enable_rotations=enable_rotations,
            only_for=strong_rotation_candidates,
        )

        for cname, gray in candidates:
            if (time.perf_counter() - stage_t0) * 1000 > time_budget_ms:
                break

            for be in backend_order:
                if (time.perf_counter() - stage_t0) * 1000 > time_budget_ms:
                    break

                tried.append(f"{source}:{cname}:{be}")
                items = _run_backend(be, gray)

                if items:
                    items = _annotate_items(
                        items,
                        variant=vname,
                        candidate=cname,
                        source=source,
                        roi_index=roi_index,
                        roi_bbox=roi_bbox,
                        roi_score=roi_score,
                    )
                    aggregate = _merge_items(aggregate, items)

                    if len(aggregate) >= max_collect_items:
                        elapsed = int((time.perf_counter() - t0) * 1000)
                        return {
                            "status": "success",
                            "items": aggregate,
                            "backend": None,
                            "elapsed_ms": elapsed,
                            "variant": None,
                            "candidate": None,
                            "source": source,
                            "tried": tried,
                        }

    elapsed = int((time.perf_counter() - t0) * 1000)
    return {
        "status": "success" if aggregate else "not_found",
        "items": aggregate,
        "backend": None,
        "elapsed_ms": elapsed,
        "variant": None,
        "candidate": None,
        "source": source,
        "tried": tried,
    }


# ---------------------------------------------------------------------
# Public API - mode 1 (fast)
# ---------------------------------------------------------------------
def decode_barcode_1d_fast(
    img_bgr: np.ndarray,
    *,
    time_budget_ms: int = 180,
    variants: Optional[Union[List[str], str]] = None,
    prefer: str = "zxingcpp",
    enable_fallback: bool = True,
    enable_roi_rescue: bool = True,
    max_rois: int = 3,
    roi_upscale: float = 3.0,
) -> Dict[str, Any]:
    t0 = time.perf_counter()

    try:
        avail = _backend_available()

        backend_order: List[str] = []
        if prefer in ("zxingcpp", "pyzbar", "opencv_barcode"):
            backend_order.append(prefer)

        if enable_fallback:
            for be in ("zxingcpp", "pyzbar", "opencv_barcode"):
                if be not in backend_order:
                    backend_order.append(be)

        backend_order = [be for be in backend_order if avail.get(be, False)]

        if not backend_order:
            elapsed = int((time.perf_counter() - t0) * 1000)
            return {
                "status": "not_available",
                "mode": "fast",
                "items": [],
                "backend": None,
                "elapsed_ms": elapsed,
                "variant": None,
            }

        if enable_roi_rescue:
            roi_budget = min(260, max(120, time_budget_ms // 2))
            rois = _find_barcode_rois(img_bgr, max_rois=max_rois)

            for idx, (roi_bgr, meta) in enumerate(rois):
                if (time.perf_counter() - t0) * 1000 > time_budget_ms:
                    break

                if roi_upscale > 1.0:
                    roi_bgr = cv2.resize(
                        roi_bgr,
                        None,
                        fx=roi_upscale,
                        fy=roi_upscale,
                        interpolation=cv2.INTER_CUBIC,
                    )

                res_roi = _decode_barcode_core_fast(
                    roi_bgr,
                    t0=t0,
                    time_budget_ms=min(time_budget_ms, roi_budget),
                    variants=variants,
                    backend_order=backend_order,
                    source="roi",
                    roi_index=idx,
                    roi_bbox=meta.get("bbox"),
                    roi_score=meta.get("score"),
                )
                if res_roi.get("status") == "success":
                    res_roi["mode"] = "fast"
                    res_roi["note"] = "roi_rescue"
                    return res_roi

        res = _decode_barcode_core_fast(
            img_bgr,
            t0=t0,
            time_budget_ms=time_budget_ms,
            variants=variants,
            backend_order=backend_order,
            source="full",
        )
        res["mode"] = "fast"
        return res

    except Exception as e:
        elapsed = int((time.perf_counter() - t0) * 1000)
        return {
            "status": "error",
            "mode": "fast",
            "items": [],
            "backend": None,
            "elapsed_ms": elapsed,
            "error": repr(e),
            "variant": None,
            "tried": [],
        }


# ---------------------------------------------------------------------
# Public API - mode 2 (collect / aggregate)
# ---------------------------------------------------------------------
def decode_barcode_1d_collect(
    img_bgr: np.ndarray,
    *,
    time_budget_ms: int = 420,
    variants: Optional[Union[List[str], str]] = None,
    prefer: str = "zxingcpp",
    enable_fallback: bool = True,
    enable_roi_rescue: bool = True,
    max_rois: int = 6,
    roi_upscale: float = 3.0,
    include_full_image: bool = True,
    max_collect_items: int = 64,
    enable_tile_sweep: bool = True,
    enable_collect_rotations: bool = True,
    roi_stage_ratio: float = 0.45,
    max_tiles: int = 5,
) -> Dict[str, Any]:
    t0 = time.perf_counter()

    try:
        avail = _backend_available()

        backend_order: List[str] = []
        if prefer in ("zxingcpp", "pyzbar", "opencv_barcode"):
            backend_order.append(prefer)

        if enable_fallback:
            for be in ("zxingcpp", "pyzbar", "opencv_barcode"):
                if be not in backend_order:
                    backend_order.append(be)

        backend_order = [be for be in backend_order if avail.get(be, False)]

        if not backend_order:
            elapsed = int((time.perf_counter() - t0) * 1000)
            return {
                "status": "not_available",
                "mode": "collect",
                "items": [],
                "backend": None,
                "elapsed_ms": elapsed,
                "variant": None,
            }

        aggregate: List[Dict[str, Any]] = []
        tried: List[str] = []
        saw_roi = False
        saw_tile = False
        saw_full = False

        if enable_roi_rescue:
            rois = _find_barcode_rois(img_bgr, max_rois=max_rois)

            roi_stage_ratio = max(0.10, min(float(roi_stage_ratio), 0.90))
            roi_budget_total = int(time_budget_ms * roi_stage_ratio) if include_full_image else time_budget_ms
            roi_budget_total = max(150, roi_budget_total)

            roi_stage_t0 = time.perf_counter()

            for idx, (roi_bgr, meta) in enumerate(rois):
                if (time.perf_counter() - t0) * 1000 > time_budget_ms:
                    break

                elapsed_total = (time.perf_counter() - t0) * 1000
                elapsed_roi_stage = (time.perf_counter() - roi_stage_t0) * 1000

                remain_total = max(1, int(time_budget_ms - elapsed_total))
                remain_roi_stage = max(1, int(roi_budget_total - elapsed_roi_stage))

                if remain_total <= 0 or remain_roi_stage <= 0:
                    break

                per_roi_budget = max(90, min(220, remain_roi_stage // max(1, (len(rois) - idx))))

                if roi_upscale > 1.0:
                    roi_bgr = cv2.resize(
                        roi_bgr,
                        None,
                        fx=roi_upscale,
                        fy=roi_upscale,
                        interpolation=cv2.INTER_CUBIC,
                    )

                res_roi = _decode_barcode_core_collect(
                    roi_bgr,
                    t0=t0,
                    time_budget_ms=min(remain_total, per_roi_budget),
                    variants=variants,
                    backend_order=backend_order,
                    source="roi",
                    roi_index=idx,
                    roi_bbox=meta.get("bbox"),
                    roi_score=meta.get("score"),
                    max_collect_items=max_collect_items,
                    enable_rotations=enable_collect_rotations,
                )

                tried.extend(res_roi.get("tried", []))
                aggregate = _merge_items(aggregate, res_roi.get("items", []))
                if res_roi.get("items"):
                    saw_roi = True

                if len(aggregate) >= max_collect_items:
                    elapsed = int((time.perf_counter() - t0) * 1000)
                    return {
                        "status": "success",
                        "mode": "collect",
                        "items": aggregate,
                        "backend": None,
                        "elapsed_ms": elapsed,
                        "variant": None,
                        "candidate": None,
                        "tried": tried,
                        "summary": _build_collect_summary(
                            aggregate,
                            from_rois=saw_roi,
                            from_tiles=saw_tile,
                            from_full_image=saw_full,
                        ),
                    }

        if enable_tile_sweep and include_full_image and (time.perf_counter() - t0) * 1000 <= time_budget_ms:
            tiles = _generate_multibarcode_tiles(img_bgr, max_tiles=max_tiles)

            for tidx, (tile_bgr, tmeta) in enumerate(tiles):
                if (time.perf_counter() - t0) * 1000 > time_budget_ms:
                    break

                remain_total = max(1, int(time_budget_ms - (time.perf_counter() - t0) * 1000))
                per_tile_budget = max(90, min(180, remain_total // max(1, (len(tiles) - tidx + 1))))

                res_tile = _decode_barcode_core_collect(
                    tile_bgr,
                    t0=t0,
                    time_budget_ms=per_tile_budget,
                    variants=variants,
                    backend_order=backend_order,
                    source="tile",
                    roi_index=tidx,
                    roi_bbox=tmeta.get("bbox"),
                    roi_score=None,
                    max_collect_items=max_collect_items,
                    enable_rotations=enable_collect_rotations,
                )

                tried.extend(res_tile.get("tried", []))
                aggregate = _merge_items(aggregate, res_tile.get("items", []))
                if res_tile.get("items"):
                    saw_tile = True

                if len(aggregate) >= max_collect_items:
                    break

        if include_full_image and (time.perf_counter() - t0) * 1000 <= time_budget_ms:
            remain_total = max(1, int(time_budget_ms - (time.perf_counter() - t0) * 1000))

            res_full = _decode_barcode_core_collect(
                img_bgr,
                t0=t0,
                time_budget_ms=remain_total,
                variants=variants,
                backend_order=backend_order,
                source="full",
                max_collect_items=max_collect_items,
                enable_rotations=enable_collect_rotations,
            )
            tried.extend(res_full.get("tried", []))
            aggregate = _merge_items(aggregate, res_full.get("items", []))
            if res_full.get("items"):
                saw_full = True

        elapsed = int((time.perf_counter() - t0) * 1000)
        return {
            "status": "success" if aggregate else "not_found",
            "mode": "collect",
            "items": aggregate,
            "backend": None,
            "elapsed_ms": elapsed,
            "variant": None,
            "candidate": None,
            "tried": tried,
            "summary": _build_collect_summary(
                aggregate,
                from_rois=saw_roi,
                from_tiles=saw_tile,
                from_full_image=saw_full,
            ),
        }

    except Exception as e:
        elapsed = int((time.perf_counter() - t0) * 1000)
        return {
            "status": "error",
            "mode": "collect",
            "items": [],
            "backend": None,
            "elapsed_ms": elapsed,
            "error": repr(e),
            "variant": None,
            "tried": [],
        }


# ---------------------------------------------------------------------
# Unified public entrypoint
# ---------------------------------------------------------------------
def decode_barcode_1d(
    img_bgr: np.ndarray,
    *,
    mode: str = "fast",  # "fast" | "collect"
    time_budget_ms: Optional[int] = None,
    variants: Optional[Union[List[str], str]] = None,
    prefer: str = "zxingcpp",
    enable_fallback: bool = True,
    enable_roi_rescue: bool = True,
    max_rois: Optional[int] = None,
    roi_upscale: float = 3.0,
    include_full_image: bool = True,
    max_collect_items: int = 64,
    enable_tile_sweep: bool = True,
    enable_collect_rotations: bool = True,
    roi_stage_ratio: float = 0.45,
    max_tiles: int = 5,
) -> Dict[str, Any]:
    """
    Unified barcode decoder.

    mode:
      - "fast": prioriza latencia, retorna primer éxito útil
      - "collect": barrido/agregación, junta múltiples lecturas únicas
    """
    mode = (mode or "fast").strip().lower()

    if mode == "collect":
        return decode_barcode_1d_collect(
            img_bgr,
            time_budget_ms=420 if time_budget_ms is None else int(time_budget_ms),
            variants=variants,
            prefer=prefer,
            enable_fallback=enable_fallback,
            enable_roi_rescue=enable_roi_rescue,
            max_rois=6 if max_rois is None else int(max_rois),
            roi_upscale=roi_upscale,
            include_full_image=include_full_image,
            max_collect_items=max_collect_items,
            enable_tile_sweep=enable_tile_sweep,
            enable_collect_rotations=enable_collect_rotations,
            roi_stage_ratio=roi_stage_ratio,
            max_tiles=max_tiles,
        )

    return decode_barcode_1d_fast(
        img_bgr,
        time_budget_ms=180 if time_budget_ms is None else int(time_budget_ms),
        variants=variants,
        prefer=prefer,
        enable_fallback=enable_fallback,
        enable_roi_rescue=enable_roi_rescue,
        max_rois=3 if max_rois is None else int(max_rois),
        roi_upscale=roi_upscale,
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def _cli() -> int:
    import argparse

    p = argparse.ArgumentParser(
        description=(
            "Decode barcode(s) from an image using preprocess variants + ROI rescue + "
            "optional tile sweep + multiple backends (zxingcpp/pyzbar/opencv-barcode). "
            "Supports fast and collect modes."
        )
    )
    p.add_argument("image_path", type=str, help="Path to image file")
    p.add_argument(
        "--mode",
        type=str,
        default="fast",
        choices=["fast", "collect"],
        help="Decode mode: fast (default) or collect",
    )
    p.add_argument(
        "--budget",
        type=int,
        default=None,
        help="Time budget in ms. Defaults: fast=180, collect=420",
    )
    p.add_argument(
        "--variants",
        type=str,
        default="default",
        help='Variants: "default" | "all" | comma-list e.g. "gray,sharp,bw_x2"',
    )
    p.add_argument(
        "--prefer",
        type=str,
        default="zxingcpp",
        choices=["zxingcpp", "pyzbar", "opencv_barcode"],
        help="Preferred backend (default: zxingcpp)",
    )
    p.add_argument("--no_fallback", action="store_true", help="Disable fallback to other backends.")
    p.add_argument("--no_roi", action="store_true", help="Disable ROI rescue stage.")
    p.add_argument("--max_rois", type=int, default=None, help="Max ROIs to inspect")
    p.add_argument("--roi_upscale", type=float, default=3.0, help="ROI upscale factor (default: 3.0)")
    p.add_argument(
        "--no_full_image",
        action="store_true",
        help="In collect mode, disable full-image sweep after ROI/tile stages.",
    )
    p.add_argument(
        "--max_collect_items",
        type=int,
        default=64,
        help="Collect mode: soft limit for unique decoded items.",
    )
    p.add_argument(
        "--no_tile_sweep",
        action="store_true",
        help="Disable tile sweep in collect mode.",
    )
    p.add_argument(
        "--no_collect_rotations",
        action="store_true",
        help="Disable rotated candidates in collect mode.",
    )
    p.add_argument(
        "--roi_stage_ratio",
        type=float,
        default=0.45,
        help="Fraction of collect budget reserved for ROI stage (default: 0.45).",
    )
    p.add_argument(
        "--max_tiles",
        type=int,
        default=5,
        help="Max tiles in tile sweep (default: 5).",
    )
    args = p.parse_args()

    img = cv2.imread(args.image_path)
    if img is None:
        print({"status": "error", "error": "Could not load image", "path": args.image_path})
        return 2

    v: Optional[Union[List[str], str]] = None if args.variants == "default" else args.variants

    result = decode_barcode_1d(
        img,
        mode=args.mode,
        time_budget_ms=args.budget,
        variants=v,
        prefer=args.prefer,
        enable_fallback=(not args.no_fallback),
        enable_roi_rescue=(not args.no_roi),
        max_rois=args.max_rois,
        roi_upscale=args.roi_upscale,
        include_full_image=(not args.no_full_image),
        max_collect_items=args.max_collect_items,
        enable_tile_sweep=(not args.no_tile_sweep),
        enable_collect_rotations=(not args.no_collect_rotations),
        roi_stage_ratio=float(args.roi_stage_ratio),
        max_tiles=int(args.max_tiles),
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())