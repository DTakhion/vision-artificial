# utils/vision_hybrid.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Set, Union

import time

import cv2
import numpy as np
from ultralytics import YOLO

from utils.vision_preprocess import preprocess_variants


# ------------------------------------------------------------------
# YOLO model
# ------------------------------------------------------------------
DEFAULT_MODEL_PATH = "runs/detect/runs_kuehne_nagel/barcode_v1/weights/best.pt"
_MODEL_CACHE: Dict[str, YOLO] = {}


def load_yolo_model(model_path: str = DEFAULT_MODEL_PATH) -> YOLO:
    if model_path not in _MODEL_CACHE:
        _MODEL_CACHE[model_path] = YOLO(model_path)
    return _MODEL_CACHE[model_path]


# ------------------------------------------------------------------
# Variants / formats
# ------------------------------------------------------------------
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

ALLOWED_1D_FORMATS: Set[str] = {
    "EAN-13",
    "EAN13",
    "EAN-8",
    "EAN8",
    "UPC-A",
    "UPCA",
    "UPC-E",
    "UPCE",
    "CODE-128",
    "CODE128",
    "CODE-39",
    "CODE39",
    "ITF",
    "ITF-14",
    "CODABAR",
    "RSS-14",
    "RSS_EXPANDED",
}

BLOCKED_2D_FORMATS: Set[str] = {
    "QR CODE",
    "QRCODE",
    "DATA MATRIX",
    "DATAMATRIX",
    "AZTEC",
    "PDF417",
    "MAXICODE",
}


# ------------------------------------------------------------------
# Basic helpers
# ------------------------------------------------------------------
def _safe_str(x: Any) -> Optional[str]:
    if x is None:
        return None
    try:
        s = str(x).strip()
        return s if s else None
    except Exception:
        return None


def _normalize_text(txt: Any) -> Optional[str]:
    s = _safe_str(txt)
    return s.strip() if s else None


def _normalize_format(fmt: Any) -> Optional[str]:
    s = _safe_str(fmt)
    return s.upper() if s else None


def _normalize_text_key(txt: Any) -> Optional[str]:
    s = _normalize_text(txt)
    if not s:
        return None
    s = "".join(s.split())
    return s or None


def _is_1d_format(fmt: Any) -> bool:
    s = _normalize_format(fmt)
    if not s:
        return False
    if s in BLOCKED_2D_FORMATS:
        return False
    return s in ALLOWED_1D_FORMATS


def _filter_1d_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in items:
        if not _normalize_text(item.get("text")):
            continue
        fmt = item.get("format")
        # si el backend no entrega formato fiable, dejamos pasar sólo si parece barcode
        if fmt is not None and not _is_1d_format(fmt):
            continue
        out.append(item)
    return out


def _clip_bbox_xyxy(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    img_w: int,
    img_h: int,
) -> Tuple[int, int, int, int]:
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w))
    y2 = max(0, min(y2, img_h))
    return x1, y1, x2, y2


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _ensure_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _resize_gray(gray: np.ndarray, factor: float) -> np.ndarray:
    if factor <= 1.0:
        return gray
    return cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)


def _light_sharpen(gray: np.ndarray) -> np.ndarray:
    kernel = np.array(
        [
            [0, -0.5, 0],
            [-0.5, 3.0, -0.5],
            [0, -0.5, 0],
        ],
        dtype=np.float32,
    )
    sharp = cv2.filter2D(gray, -1, kernel)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def _unsharp(g: np.ndarray, amount: float = 1.2) -> np.ndarray:
    blur = cv2.GaussianBlur(g, (0, 0), sigmaX=1.2)
    return cv2.addWeighted(g, 1.0 + amount, blur, -amount, 0)


def _otsu(g: np.ndarray) -> np.ndarray:
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw


def _adapt(g: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
    )


def _normalize_contrast(gray: np.ndarray) -> np.ndarray:
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)
    except Exception:
        return cv2.equalizeHist(gray)


def _basic_text_validation(text: str) -> bool:
    if not text:
        return False
    text = text.strip()
    if len(text) < 4:
        return False

    allowed = set("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-._/ ")
    cleaned = "".join(ch for ch in text.upper() if ch in allowed)

    if len(cleaned) < max(4, int(len(text) * 0.6)):
        return False

    num_digits = sum(ch.isdigit() for ch in text)
    if num_digits == 0 and len(text) < 6:
        return False

    return True


# ------------------------------------------------------------------
# Rotation / deskew
# ------------------------------------------------------------------
def _rotate_image_bound(
    img: np.ndarray,
    angle_deg: float,
    border_value: int | tuple[int, int, int] = 255,
) -> np.ndarray:
    if img is None or img.size == 0:
        return img

    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)

    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    M[0, 2] += (new_w / 2.0) - center[0]
    M[1, 2] += (new_h / 2.0) - center[1]

    return cv2.warpAffine(
        img,
        M,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def _estimate_barcode_angle(gray: np.ndarray) -> float:
    if gray is None or gray.size == 0:
        return 0.0

    try:
        work = _normalize_contrast(gray)
        grad_x = cv2.Sobel(work, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(work, cv2.CV_32F, 0, 1, ksize=3)

        magnitude = cv2.magnitude(grad_x, grad_y)
        angle = cv2.phase(grad_x, grad_y, angleInDegrees=True)

        thr = np.percentile(magnitude, 80)
        mask = magnitude > thr

        if np.count_nonzero(mask) < 30:
            return 0.0

        angles = angle[mask]
        candidate_angles = []

        for a in angles:
            aa = ((float(a) + 90.0) % 180.0) - 90.0
            bar_angle = aa - 90.0
            if bar_angle < -90.0:
                bar_angle += 180.0
            if bar_angle >= 90.0:
                bar_angle -= 180.0
            candidate_angles.append(bar_angle)

        candidate_angles = np.asarray(candidate_angles, dtype=np.float32)
        median_angle = float(np.median(candidate_angles))

        if median_angle < -45.0:
            median_angle += 90.0
        elif median_angle > 45.0:
            median_angle -= 90.0

        return median_angle
    except Exception:
        return 0.0


def _compute_projection_score(gray: np.ndarray) -> float:
    if gray is None or gray.size == 0:
        return -1.0

    try:
        if gray.ndim != 2:
            gray = _to_gray(gray)

        work = _normalize_contrast(gray)
        proj = work.mean(axis=0).astype(np.float32)

        if proj.size < 4:
            return -1.0

        diff = np.abs(np.diff(proj))
        return float(diff.mean() + diff.std())
    except Exception:
        return -1.0


def _autodeskew_barcode(
    img: np.ndarray,
    coarse_limit: float = 18.0,
    coarse_step: float = 3.0,
    fine_step: float = 0.75,
) -> List[Tuple[str, np.ndarray]]:
    if img is None or img.size == 0:
        return []

    gray = _to_gray(img)
    est = _estimate_barcode_angle(gray)

    variants: List[Tuple[str, np.ndarray]] = []

    if abs(est) > 0.3:
        rotated_est = _rotate_image_bound(img, -est, border_value=255)
        variants.append((f"deskew_est_{est:.2f}", rotated_est))

    center_angle = max(-coarse_limit, min(coarse_limit, est))
    candidates = np.arange(
        center_angle - coarse_step,
        center_angle + coarse_step + 1e-9,
        fine_step,
        dtype=np.float32,
    )

    scored: List[Tuple[float, float, np.ndarray]] = []

    for a in candidates:
        rotated = _rotate_image_bound(img, -float(a), border_value=255)
        score = _compute_projection_score(_to_gray(rotated))
        scored.append((score, float(a), rotated))

    scored.sort(key=lambda x: x[0], reverse=True)

    for idx, (_, a, rotated) in enumerate(scored[:3]):
        variants.append((f"deskew_top{idx+1}_{a:.2f}", rotated))

    return variants


def _rot90(gray: np.ndarray, k: int) -> np.ndarray:
    k = k % 4
    if k == 0:
        return gray
    if k == 1:
        return cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
    if k == 2:
        return cv2.rotate(gray, cv2.ROTATE_180)
    return cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)


# ------------------------------------------------------------------
# Variant preparation
# ------------------------------------------------------------------
def _resolve_variants(
    variants: Optional[Union[List[str], str]],
    available: List[str],
) -> List[str]:
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


def _extra_candidates(gray: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    cands: List[Tuple[str, np.ndarray]] = []

    g = gray
    g_x2 = _resize_gray(g, 2.0)
    g_x3 = _resize_gray(g, 3.0)
    g_x4 = _resize_gray(g, 4.0)

    s = _unsharp(g, 1.2)
    s_x2 = _resize_gray(s, 2.0)
    s_x3 = _resize_gray(s, 3.0)
    s_x4 = _resize_gray(s, 4.0)

    cands.extend(
        [
            ("gray_local", g),
            ("gray_x2_local", g_x2),
            ("gray_x3_local", g_x3),
            ("gray_x4_local", g_x4),
            ("sharp_local", s),
            ("sharp_x2_local", s_x2),
            ("sharp_x3_local", s_x3),
            ("sharp_x4_local", s_x4),
        ]
    )

    try:
        cands.extend(
            [
                ("otsu_gray_x2_local", _otsu(g_x2)),
                ("adapt_gray_x2_local", _adapt(g_x2)),
                ("otsu_gray_local", _otsu(g)),
                ("adapt_gray_local", _adapt(g)),
            ]
        )
    except Exception:
        pass

    return cands


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


# ------------------------------------------------------------------
# Backend availability
# ------------------------------------------------------------------
def _get_opencv_barcode_detector() -> Any:
    if hasattr(cv2, "barcode") and hasattr(cv2.barcode, "BarcodeDetector"):
        return cv2.barcode.BarcodeDetector()
    if hasattr(cv2, "barcode_BarcodeDetector"):
        return cv2.barcode_BarcodeDetector()
    return None


def _backend_available() -> Dict[str, bool]:
    avail = {"zxingcpp": False, "pyzbar": False, "opencv_barcode": False}

    try:
        import zxingcpp  # type: ignore  # noqa
        avail["zxingcpp"] = True
    except Exception:
        pass

    try:
        from pyzbar.pyzbar import decode as _d  # type: ignore  # noqa
        avail["pyzbar"] = True
    except Exception:
        pass

    try:
        det = _get_opencv_barcode_detector()
        avail["opencv_barcode"] = det is not None
    except Exception:
        avail["opencv_barcode"] = False

    return avail


# ------------------------------------------------------------------
# Backends
# ------------------------------------------------------------------
def _normalize_rect(rect: Any) -> Any:
    if rect is None:
        return None
    try:
        if (
            hasattr(rect, "left")
            and hasattr(rect, "top")
            and hasattr(rect, "width")
            and hasattr(rect, "height")
        ):
            return (int(rect.left), int(rect.top), int(rect.width), int(rect.height))
        if isinstance(rect, (tuple, list)) and len(rect) == 4:
            return tuple(int(v) for v in rect)
    except Exception:
        pass
    return rect


def _serialize_zxing_position(pos: Any) -> Any:
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
        img = _ensure_bgr(gray)

        if hasattr(det, "detectAndDecodeWithType"):
            decoded_info, decoded_type, points = det.detectAndDecodeWithType(img)
            if decoded_info:
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
        r = det.detectAndDecode(_ensure_bgr(gray))
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

    return out


def _run_backend(be: str, gray: np.ndarray) -> List[Dict[str, Any]]:
    if be == "zxingcpp":
        items = _try_zxingcpp(gray)
    elif be == "pyzbar":
        items = _try_pyzbar(gray)
    else:
        items = _try_opencv_barcode(gray)

    items = _filter_1d_items(items)

    valid: List[Dict[str, Any]] = []
    for item in items:
        txt = _normalize_text(item.get("text"))
        if txt and _basic_text_validation(txt):
            valid.append(item)

    return valid


# ------------------------------------------------------------------
# Scoring / dedup
# ------------------------------------------------------------------
def _score_item(item: Dict[str, Any]) -> float:
    score = 0.0

    backend = _safe_str(item.get("backend"))
    if backend == "zxingcpp":
        score += 2.9
    elif backend == "pyzbar":
        score += 2.7
    elif backend == "opencv_barcode":
        score += 1.0

    txt = _normalize_text(item.get("text"))
    if txt:
        score += min(len(txt), 32) * 0.03
        score += sum(ch.isdigit() for ch in txt) * 0.04

    fmt = _normalize_format(item.get("format"))
    if fmt:
        score += 0.35

    source = _safe_str(item.get("source"))
    if source == "yolo_subroi":
        score += 1.35
    elif source == "yolo_roi":
        score += 1.15
    elif source == "subroi":
        score += 1.0
    elif source == "roi":
        score += 0.8
    elif source == "tile":
        score += 0.5
    elif source == "full":
        score += 0.25

    if item.get("yolo_conf") is not None:
        try:
            score += float(item["yolo_conf"]) * 2.0
        except Exception:
            pass

    roi_score = item.get("roi_score")
    try:
        if roi_score is not None:
            score += min(float(roi_score) / 1_000_000.0, 2.0)
    except Exception:
        pass

    candidate = _safe_str(item.get("candidate"))
    if candidate:
        if "x4" in candidate:
            score += 0.18
        elif "x3" in candidate:
            score += 0.12
        elif "x2" in candidate:
            score += 0.08
        if "rot90" in candidate or "rot270" in candidate or "deskew" in candidate:
            score += 0.06

    return float(score)


def _dedup_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
    yolo_conf: Optional[float] = None,
    yolo_bbox_xyxy_original: Optional[Tuple[int, int, int, int]] = None,
    yolo_bbox_xyxy_padded: Optional[Tuple[int, int, int, int]] = None,
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
        if yolo_conf is not None:
            x["yolo_conf"] = yolo_conf
        if yolo_bbox_xyxy_original is not None:
            x["yolo_bbox_xyxy_original"] = yolo_bbox_xyxy_original
        if yolo_bbox_xyxy_padded is not None:
            x["yolo_bbox_xyxy_padded"] = yolo_bbox_xyxy_padded

        out.append(x)

    return out


# ------------------------------------------------------------------
# ROI rescue from classic pipeline
# ------------------------------------------------------------------
def _find_label_rois(
    img_bgr: np.ndarray,
    max_rois: int = 8,
) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, (0, 0, 150), (180, 85, 255))

    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k1, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k1, iterations=1)
    mask = cv2.dilate(mask, k2, iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []

    out: List[Tuple[np.ndarray, Dict[str, Any]]] = []

    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:40]:
        x, y, ww, hh = cv2.boundingRect(c)
        area = float(ww * hh)

        if area < 0.0025 * (w * h):
            continue
        if area > 0.30 * (w * h):
            continue

        aspect = ww / float(hh + 1e-6)
        aspect_inv = hh / float(ww + 1e-6)
        elongated = max(aspect, aspect_inv)
        if elongated < 1.15:
            continue

        border_touch = (
            int(x <= 2)
            + int(y <= 2)
            + int(x + ww >= w - 2)
            + int(y + hh >= h - 2)
        )

        pad = int(0.12 * max(ww, hh))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(w, x + ww + pad)
        y1 = min(h, y + hh + pad)

        if x1 <= x0 or y1 <= y0:
            continue

        roi = img_bgr[y0:y1, x0:x1].copy()
        score = area * elongated * (1.0 if border_touch == 0 else 0.75)

        out.append(
            (
                roi,
                {
                    "bbox": (x0, y0, x1 - x0, y1 - y0),
                    "score": score,
                    "kind": "label",
                },
            )
        )

        if len(out) >= max_rois:
            break

    out.sort(key=lambda t: float(t[1].get("score", 0.0)), reverse=True)
    return out


def _find_sub_barcode_rois(
    roi_bgr: np.ndarray,
    max_rois: int = 4,
) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
    h0, w0 = roi_bgr.shape[:2]
    if h0 < 40 or w0 < 40:
        return []

    out: List[Tuple[np.ndarray, Dict[str, Any]]] = []
    seen: Set[Tuple[str, int, int, int, int]] = set()

    oriented_inputs: List[Tuple[str, np.ndarray]] = [
        ("orig", roi_bgr),
        ("rot90", cv2.rotate(roi_bgr, cv2.ROTATE_90_CLOCKWISE)),
    ]

    for orient_name, img in oriented_inputs:
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = _normalize_contrast(gray)

        gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
        ax = cv2.convertScaleAbs(gx)
        ax = cv2.GaussianBlur(ax, (0, 0), 1.2)

        _, bw = cv2.threshold(ax, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5))
        k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k_close, iterations=2)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k_open, iterations=1)
        bw = cv2.dilate(bw, None, iterations=1)

        cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue

        for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:20]:
            x, y, ww, hh = cv2.boundingRect(c)
            area = float(ww * hh)

            if area < 0.015 * (w * h):
                continue
            if area > 0.85 * (w * h):
                continue

            aspect = ww / float(hh + 1e-6)
            aspect_inv = hh / float(ww + 1e-6)
            elongated = max(aspect, aspect_inv)
            if elongated < 1.35:
                continue

            contour_area = float(cv2.contourArea(c))
            extent = contour_area / float(area + 1e-6)
            if extent < 0.18:
                continue

            pad = int(0.08 * max(ww, hh))
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(w, x + ww + pad)
            y1 = min(h, y + hh + pad)
            if x1 <= x0 or y1 <= y0:
                continue

            key = (orient_name, x0, y0, x1, y1)
            if key in seen:
                continue
            seen.add(key)

            crop = img[y0:y1, x0:x1].copy()
            score = area * elongated * (1.15 if orient_name == "rot90" else 1.0)

            out.append(
                (
                    crop,
                    {
                        "bbox": (x0, y0, x1 - x0, y1 - y0),
                        "score": score,
                        "kind": "sub_barcode",
                        "orientation": orient_name,
                    },
                )
            )

            if len(out) >= max_rois:
                break

        if len(out) >= max_rois:
            break

    out.sort(key=lambda t: float(t[1].get("score", 0.0)), reverse=True)
    return out[:max_rois]


def _find_barcode_rois(
    img_bgr: np.ndarray,
    max_rois: int = 6,
) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
    h, w = img_bgr.shape[:2]
    out: List[Tuple[np.ndarray, Dict[str, Any]]] = []
    seen: Set[Tuple[int, int, int, int]] = set()

    def _add_roi(roi: np.ndarray, meta: Dict[str, Any]) -> None:
        bbox = tuple(int(v) for v in meta.get("bbox", (0, 0, 0, 0)))
        if bbox in seen:
            return
        seen.add(bbox)
        out.append((roi, meta))

    label_rois = _find_label_rois(img_bgr, max_rois=max(4, max_rois))
    for roi, meta in label_rois:
        _add_roi(roi, meta)
        if len(out) >= max_rois:
            out.sort(key=lambda t: float(t[1].get("score", 0.0)), reverse=True)
            return out[:max_rois]

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    g = cv2.addWeighted(cv2.convertScaleAbs(gx), 0.75, cv2.convertScaleAbs(gy), 0.25, 0)
    g = cv2.GaussianBlur(g, (0, 0), 2.0)

    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 9))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=2)
    bw = cv2.erode(bw, None, iterations=1)
    bw = cv2.dilate(bw, None, iterations=2)

    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:30]:
            x, y, ww, hh = cv2.boundingRect(c)
            area = float(ww * hh)

            if area < 0.0025 * (w * h):
                continue

            aspect = ww / float(hh + 1e-6)
            aspect_inv = hh / float(ww + 1e-6)
            elongated = max(aspect, aspect_inv)
            if elongated < 1.8:
                continue

            pad = int(0.10 * max(ww, hh))
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(w, x + ww + pad)
            y1 = min(h, y + hh + pad)
            if x1 <= x0 or y1 <= y0:
                continue

            roi = img_bgr[y0:y1, x0:x1].copy()
            score = area * elongated

            _add_roi(
                roi,
                {
                    "bbox": (x0, y0, x1 - x0, y1 - y0),
                    "score": score,
                    "kind": "barcode_grad",
                },
            )

            if len(out) >= max_rois:
                break

    out.sort(key=lambda t: float(t[1].get("score", 0.0)), reverse=True)
    return out[:max_rois]


def _generate_multibarcode_tiles(
    img_bgr: np.ndarray,
    max_tiles: int = 5,
) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
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


# ------------------------------------------------------------------
# YOLO ROI stage
# ------------------------------------------------------------------
def detect_barcode_rois_yolo(
    img_bgr: np.ndarray,
    model_path: str = DEFAULT_MODEL_PATH,
    conf: float = 0.20,
    iou: float = 0.45,
    max_det: int = 12,
    min_size: int = 36,
) -> List[Dict[str, Any]]:
    if img_bgr is None or not isinstance(img_bgr, np.ndarray) or img_bgr.size == 0:
        return []

    model = load_yolo_model(model_path)
    img_h, img_w = img_bgr.shape[:2]

    results = model.predict(
        source=img_bgr,
        conf=conf,
        iou=iou,
        max_det=max_det,
        verbose=False,
    )

    rois: List[Dict[str, Any]] = []

    for r in results:
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            continue

        for b in boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            x1, y1, x2, y2 = _clip_bbox_xyxy(x1, y1, x2, y2, img_w, img_h)

            w = x2 - x1
            h = y2 - y1

            if w < min_size or h < min_size:
                continue

            cls = int(b.cls[0]) if b.cls is not None else -1
            conf_score = float(b.conf[0]) if b.conf is not None else 0.0

            rois.append(
                {
                    "bbox_xyxy": (x1, y1, x2, y2),
                    "bbox": (x1, y1, w, h),
                    "conf": conf_score,
                    "cls": cls,
                    "area": w * h,
                }
            )

    rois.sort(key=lambda r: (r["conf"], r["area"]), reverse=True)
    return rois


def crop_rois_multi_pad(
    img: np.ndarray,
    rois: List[Dict[str, Any]],
    pad_ratios: Tuple[float, ...] = (0.08, 0.15, 0.28),
) -> List[Dict[str, Any]]:
    if img is None or img.size == 0:
        return []

    img_h, img_w = img.shape[:2]
    out: List[Dict[str, Any]] = []

    for i, r in enumerate(rois):
        x, y, ww, hh = r["bbox"]

        for pad_ratio in pad_ratios:
            pad_x = int(max(4, round(ww * pad_ratio)))
            pad_y = int(max(4, round(hh * (pad_ratio * 0.85))))

            x0 = max(0, x - pad_x)
            y0 = max(0, y - pad_y)
            x1 = min(img_w, x + ww + pad_x)
            y1 = min(img_h, y + hh + pad_y)

            crop = img[y0:y1, x0:x1].copy()
            if crop.size == 0:
                continue

            out.append(
                {
                    "crop": crop,
                    "roi_index": i,
                    "pad_ratio": float(pad_ratio),
                    "bbox_xyxy_original": r["bbox_xyxy"],
                    "bbox_xyxy_padded": (x0, y0, x1, y1),
                    "bbox": r["bbox"],
                    "conf": r["conf"],
                    "cls": r["cls"],
                    "roi_score": float(r.get("area", 0.0)),
                }
            )

    return out


# ------------------------------------------------------------------
# Decode core on a given image
# ------------------------------------------------------------------
def _decode_candidates_collect(
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
    yolo_conf: Optional[float] = None,
    yolo_bbox_xyxy_original: Optional[Tuple[int, int, int, int]] = None,
    yolo_bbox_xyxy_padded: Optional[Tuple[int, int, int, int]] = None,
    enable_rotations: bool = True,
    add_deskew: bool = True,
    max_collect_items: int = 64,
) -> Dict[str, Any]:
    tried: List[str] = []
    aggregate: List[Dict[str, Any]] = []
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

        strong_rotation_candidates = {
            vname,
            "gray",
            "sharp",
            "bilateral",
            "bilateral_sharp",
            "bw",
            "bw_x2",
            "sharp_x4_local",
            "gray_x4_local",
            "sharp_x3_local",
            "gray_x3_local",
            "sharp_x2_local",
            "gray_x2_local",
            "sharp_local",
            "gray_local",
        }

        candidates = _augment_with_rotations(
            candidates,
            enable_rotations=enable_rotations,
            only_for=strong_rotation_candidates,
        )

        if add_deskew:
            deskewed: List[Tuple[str, np.ndarray]] = []
            for cname, cand in candidates:
                deskewed.append((cname, cand))
                if cand is None or cand.size == 0:
                    continue
                if cand.shape[0] < 30 or cand.shape[1] < 30:
                    continue
                try:
                    ds = _autodeskew_barcode(_ensure_bgr(cand))
                    for ds_name, ds_img in ds[:2]:
                        deskewed.append((f"{cname}_{ds_name}", _to_gray(ds_img)))
                except Exception:
                    pass
            candidates = deskewed

        # dedup simple by candidate name
        clean_candidates: List[Tuple[str, np.ndarray]] = []
        seen_names: Set[str] = set()
        for cname, cand in candidates:
            if cname in seen_names:
                continue
            if cand is None or cand.size == 0:
                continue
            seen_names.add(cname)
            clean_candidates.append((cname, cand))
        candidates = clean_candidates

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
                        yolo_conf=yolo_conf,
                        yolo_bbox_xyxy_original=yolo_bbox_xyxy_original,
                        yolo_bbox_xyxy_padded=yolo_bbox_xyxy_padded,
                    )
                    aggregate = _merge_items(aggregate, items)

                    if len(aggregate) >= max_collect_items:
                        return {
                            "status": "success",
                            "items": aggregate,
                            "tried": tried,
                            "elapsed_ms": int((time.perf_counter() - t0) * 1000),
                        }

    return {
        "status": "success" if aggregate else "not_found",
        "items": aggregate,
        "tried": tried,
        "elapsed_ms": int((time.perf_counter() - t0) * 1000),
    }


# ------------------------------------------------------------------
# Hybrid stages
# ------------------------------------------------------------------
def _decode_yolo_stage(
    img_bgr: np.ndarray,
    *,
    t0: float,
    total_budget_ms: int,
    variants: Optional[Union[List[str], str]],
    backend_order: List[str],
    model_path: str,
    yolo_conf: float,
    yolo_iou: float,
    yolo_max_det: int,
    yolo_min_size: int,
    yolo_pad_ratios: Tuple[float, ...],
    max_collect_items: int,
) -> Dict[str, Any]:
    aggregate: List[Dict[str, Any]] = []
    tried: List[str] = []

    rois = detect_barcode_rois_yolo(
        img_bgr=img_bgr,
        model_path=model_path,
        conf=yolo_conf,
        iou=yolo_iou,
        max_det=yolo_max_det,
        min_size=yolo_min_size,
    )

    if not rois:
        return {
            "status": "not_found",
            "items": [],
            "tried": [],
            "rois": [],
            "elapsed_ms": int((time.perf_counter() - t0) * 1000),
        }

    crops = crop_rois_multi_pad(img_bgr, rois=rois, pad_ratios=yolo_pad_ratios)

    per_crop_budget = max(120, min(550, total_budget_ms // max(1, len(crops))))

    for crop_info in crops:
        if (time.perf_counter() - t0) * 1000 > total_budget_ms:
            break

        crop = crop_info["crop"]
        roi_index = int(crop_info["roi_index"])

        # subrois internas primero
        sub_rois = _find_sub_barcode_rois(crop, max_rois=4)

        for sub_bgr, smeta in sub_rois:
            if (time.perf_counter() - t0) * 1000 > total_budget_ms:
                break

            sub_budget = max(120, min(320, per_crop_budget // max(1, len(sub_rois) + 1)))

            res_sub = _decode_candidates_collect(
                sub_bgr,
                t0=t0,
                time_budget_ms=sub_budget,
                variants=variants,
                backend_order=backend_order,
                source="yolo_subroi",
                roi_index=roi_index,
                roi_bbox=crop_info["bbox"],
                roi_score=max(float(crop_info["roi_score"]), float(smeta.get("score", 0.0))),
                yolo_conf=float(crop_info["conf"]),
                yolo_bbox_xyxy_original=crop_info["bbox_xyxy_original"],
                yolo_bbox_xyxy_padded=crop_info["bbox_xyxy_padded"],
                enable_rotations=True,
                add_deskew=True,
                max_collect_items=max_collect_items,
            )
            tried.extend(res_sub.get("tried", []))
            aggregate = _merge_items(aggregate, res_sub.get("items", []))

            if len(aggregate) >= max_collect_items:
                return {
                    "status": "success",
                    "items": aggregate,
                    "tried": tried,
                    "rois": rois,
                    "elapsed_ms": int((time.perf_counter() - t0) * 1000),
                }

        remain = max(120, min(per_crop_budget, int(total_budget_ms - (time.perf_counter() - t0) * 1000)))

        res_roi = _decode_candidates_collect(
            crop,
            t0=t0,
            time_budget_ms=remain,
            variants=variants,
            backend_order=backend_order,
            source="yolo_roi",
            roi_index=roi_index,
            roi_bbox=crop_info["bbox"],
            roi_score=float(crop_info["roi_score"]),
            yolo_conf=float(crop_info["conf"]),
            yolo_bbox_xyxy_original=crop_info["bbox_xyxy_original"],
            yolo_bbox_xyxy_padded=crop_info["bbox_xyxy_padded"],
            enable_rotations=True,
            add_deskew=True,
            max_collect_items=max_collect_items,
        )
        tried.extend(res_roi.get("tried", []))
        aggregate = _merge_items(aggregate, res_roi.get("items", []))

        if len(aggregate) >= max_collect_items:
            break

    return {
        "status": "success" if aggregate else "not_found",
        "items": aggregate,
        "tried": tried,
        "rois": rois,
        "elapsed_ms": int((time.perf_counter() - t0) * 1000),
    }


def _decode_classic_roi_stage(
    img_bgr: np.ndarray,
    *,
    t0: float,
    total_budget_ms: int,
    variants: Optional[Union[List[str], str]],
    backend_order: List[str],
    max_rois: int,
    roi_upscale: float,
    max_collect_items: int,
) -> Dict[str, Any]:
    aggregate: List[Dict[str, Any]] = []
    tried: List[str] = []

    rois = _find_barcode_rois(img_bgr, max_rois=max_rois)
    if not rois:
        return {
            "status": "not_found",
            "items": [],
            "tried": [],
            "elapsed_ms": int((time.perf_counter() - t0) * 1000),
        }

    per_roi_budget = max(160, min(520, total_budget_ms // max(1, len(rois))))

    for idx, (roi_bgr, meta) in enumerate(rois):
        if (time.perf_counter() - t0) * 1000 > total_budget_ms:
            break

        work_roi = roi_bgr
        if roi_upscale > 1.0:
            work_roi = cv2.resize(
                work_roi,
                None,
                fx=roi_upscale,
                fy=roi_upscale,
                interpolation=cv2.INTER_CUBIC,
            )

        sub_rois = _find_sub_barcode_rois(work_roi, max_rois=4)

        for sub_bgr, smeta in sub_rois:
            if (time.perf_counter() - t0) * 1000 > total_budget_ms:
                break

            sub_budget = max(120, min(320, per_roi_budget // max(1, len(sub_rois) + 1)))

            res_sub = _decode_candidates_collect(
                sub_bgr,
                t0=t0,
                time_budget_ms=sub_budget,
                variants=variants,
                backend_order=backend_order,
                source="subroi",
                roi_index=idx,
                roi_bbox=meta.get("bbox"),
                roi_score=max(float(meta.get("score", 0.0)), float(smeta.get("score", 0.0))),
                enable_rotations=True,
                add_deskew=True,
                max_collect_items=max_collect_items,
            )
            tried.extend(res_sub.get("tried", []))
            aggregate = _merge_items(aggregate, res_sub.get("items", []))

            if len(aggregate) >= max_collect_items:
                return {
                    "status": "success",
                    "items": aggregate,
                    "tried": tried,
                    "elapsed_ms": int((time.perf_counter() - t0) * 1000),
                }

        remain = max(120, min(per_roi_budget, int(total_budget_ms - (time.perf_counter() - t0) * 1000)))

        res_roi = _decode_candidates_collect(
            work_roi,
            t0=t0,
            time_budget_ms=remain,
            variants=variants,
            backend_order=backend_order,
            source="roi",
            roi_index=idx,
            roi_bbox=meta.get("bbox"),
            roi_score=float(meta.get("score", 0.0)),
            enable_rotations=True,
            add_deskew=True,
            max_collect_items=max_collect_items,
        )
        tried.extend(res_roi.get("tried", []))
        aggregate = _merge_items(aggregate, res_roi.get("items", []))

        if len(aggregate) >= max_collect_items:
            break

    return {
        "status": "success" if aggregate else "not_found",
        "items": aggregate,
        "tried": tried,
        "elapsed_ms": int((time.perf_counter() - t0) * 1000),
    }


def _decode_tile_stage(
    img_bgr: np.ndarray,
    *,
    t0: float,
    total_budget_ms: int,
    variants: Optional[Union[List[str], str]],
    backend_order: List[str],
    max_tiles: int,
    max_collect_items: int,
) -> Dict[str, Any]:
    aggregate: List[Dict[str, Any]] = []
    tried: List[str] = []

    tiles = _generate_multibarcode_tiles(img_bgr, max_tiles=max_tiles)
    if not tiles:
        return {
            "status": "not_found",
            "items": [],
            "tried": [],
            "elapsed_ms": int((time.perf_counter() - t0) * 1000),
        }

    per_tile_budget = max(120, min(260, total_budget_ms // max(1, len(tiles))))

    for idx, (tile_bgr, tmeta) in enumerate(tiles):
        if (time.perf_counter() - t0) * 1000 > total_budget_ms:
            break

        res_tile = _decode_candidates_collect(
            tile_bgr,
            t0=t0,
            time_budget_ms=per_tile_budget,
            variants=variants,
            backend_order=backend_order,
            source="tile",
            roi_index=idx,
            roi_bbox=tmeta.get("bbox"),
            roi_score=None,
            enable_rotations=True,
            add_deskew=True,
            max_collect_items=max_collect_items,
        )
        tried.extend(res_tile.get("tried", []))
        aggregate = _merge_items(aggregate, res_tile.get("items", []))

        if len(aggregate) >= max_collect_items:
            break

    return {
        "status": "success" if aggregate else "not_found",
        "items": aggregate,
        "tried": tried,
        "elapsed_ms": int((time.perf_counter() - t0) * 1000),
    }


def _decode_full_stage(
    img_bgr: np.ndarray,
    *,
    t0: float,
    total_budget_ms: int,
    variants: Optional[Union[List[str], str]],
    backend_order: List[str],
    max_collect_items: int,
) -> Dict[str, Any]:
    return _decode_candidates_collect(
        img_bgr,
        t0=t0,
        time_budget_ms=total_budget_ms,
        variants=variants,
        backend_order=backend_order,
        source="full",
        enable_rotations=True,
        add_deskew=True,
        max_collect_items=max_collect_items,
    )


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------
def decode_barcode_hybrid(
    img_bgr: np.ndarray,
    *,
    time_budget_ms: int = 6000,
    variants: Optional[Union[List[str], str]] = None,
    prefer: str = "zxingcpp",
    enable_fallback: bool = True,
    model_path: str = DEFAULT_MODEL_PATH,
    yolo_first: bool = True,
    yolo_conf: float = 0.10,
    yolo_iou: float = 0.45,
    yolo_max_det: int = 12,
    yolo_min_size: int = 36,
    yolo_pad_ratios: Tuple[float, ...] = (0.08, 0.15, 0.28),
    enable_classic_roi_rescue: bool = True,
    classic_max_rois: int = 6,
    classic_roi_upscale: float = 2.5,
    enable_tile_sweep: bool = True,
    max_tiles: int = 5,
    include_full_image: bool = True,
    max_collect_items: int = 64,
    yolo_stage_ratio: float = 0.45,
    classic_stage_ratio: float = 0.30,
    tile_stage_ratio: float = 0.15,
) -> Dict[str, Any]:
    t0 = time.perf_counter()

    if img_bgr is None or not isinstance(img_bgr, np.ndarray) or img_bgr.size == 0:
        return {
            "status": "invalid_image",
            "items": [],
            "elapsed_ms": 0,
        }

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
            return {
                "status": "not_available",
                "items": [],
                "elapsed_ms": int((time.perf_counter() - t0) * 1000),
            }

        aggregate: List[Dict[str, Any]] = []
        tried: List[str] = []
        yolo_rois_meta: List[Dict[str, Any]] = []

        total_budget = max(500, int(time_budget_ms))
        yolo_budget = int(total_budget * max(0.0, min(yolo_stage_ratio, 0.8)))
        classic_budget = int(total_budget * max(0.0, min(classic_stage_ratio, 0.8)))
        tile_budget = int(total_budget * max(0.0, min(tile_stage_ratio, 0.8)))

        remain_for_full = max(
            120,
            total_budget - yolo_budget - classic_budget - tile_budget,
        )

        if yolo_first:
            res_yolo = _decode_yolo_stage(
                img_bgr,
                t0=t0,
                total_budget_ms=max(200, yolo_budget),
                variants=variants,
                backend_order=backend_order,
                model_path=model_path,
                yolo_conf=yolo_conf,
                yolo_iou=yolo_iou,
                yolo_max_det=yolo_max_det,
                yolo_min_size=yolo_min_size,
                yolo_pad_ratios=yolo_pad_ratios,
                max_collect_items=max_collect_items,
            )
            aggregate = _merge_items(aggregate, res_yolo.get("items", []))
            tried.extend(res_yolo.get("tried", []))
            yolo_rois_meta = res_yolo.get("rois", []) or []

        if enable_classic_roi_rescue and (time.perf_counter() - t0) * 1000 < total_budget:
            res_classic = _decode_classic_roi_stage(
                img_bgr,
                t0=t0,
                total_budget_ms=max(200, classic_budget),
                variants=variants,
                backend_order=backend_order,
                max_rois=classic_max_rois,
                roi_upscale=classic_roi_upscale,
                max_collect_items=max_collect_items,
            )
            aggregate = _merge_items(aggregate, res_classic.get("items", []))
            tried.extend(res_classic.get("tried", []))

        if enable_tile_sweep and (time.perf_counter() - t0) * 1000 < total_budget:
            res_tile = _decode_tile_stage(
                img_bgr,
                t0=t0,
                total_budget_ms=max(150, tile_budget),
                variants=variants,
                backend_order=backend_order,
                max_tiles=max_tiles,
                max_collect_items=max_collect_items,
            )
            aggregate = _merge_items(aggregate, res_tile.get("items", []))
            tried.extend(res_tile.get("tried", []))

        if include_full_image and (time.perf_counter() - t0) * 1000 < total_budget:
            remain = max(120, int(total_budget - (time.perf_counter() - t0) * 1000))
            remain = max(remain, remain_for_full)

            res_full = _decode_full_stage(
                img_bgr,
                t0=t0,
                total_budget_ms=remain,
                variants=variants,
                backend_order=backend_order,
                max_collect_items=max_collect_items,
            )
            aggregate = _merge_items(aggregate, res_full.get("items", []))
            tried.extend(res_full.get("tried", []))

        elapsed = int((time.perf_counter() - t0) * 1000)

        unique_texts: List[str] = []
        seen: Set[str] = set()
        for item in aggregate:
            key = _normalize_text_key(item.get("text"))
            if not key or key in seen:
                continue
            seen.add(key)
            unique_texts.append(key)
        unique_texts.sort()

        return {
            "status": "success" if aggregate else "not_found",
            "mode": "hybrid_collect",
            "items": aggregate,
            "elapsed_ms": elapsed,
            "tried_total": len(tried),
            "summary": {
                "total_unique_items": len(aggregate),
                "total_unique_texts": len(unique_texts),
                "unique_texts": unique_texts,
                "yolo_rois_detected": len(yolo_rois_meta),
                "backends_used": backend_order,
            },
            "yolo_rois": yolo_rois_meta,
        }

    except Exception as e:
        return {
            "status": "error",
            "items": [],
            "elapsed_ms": int((time.perf_counter() - t0) * 1000),
            "error": repr(e),
        }


# ------------------------------------------------------------------
# Visualization helper
# ------------------------------------------------------------------
def draw_detected_items(
    img_bgr: np.ndarray,
    items: List[Dict[str, Any]],
) -> np.ndarray:
    vis = img_bgr.copy()

    for i, item in enumerate(items):
        txt = _normalize_text(item.get("text")) or f"item_{i}"
        bbox = item.get("yolo_bbox_xyxy_original") or item.get("roi_bbox")

        if isinstance(bbox, (tuple, list)) and len(bbox) == 4:
            x, y, w_or_x2, h_or_y2 = bbox

            # detect whether xyxy or xywh
            if item.get("yolo_bbox_xyxy_original") is not None:
                x1, y1, x2, y2 = map(int, bbox)
            else:
                x1, y1, ww, hh = map(int, bbox)
                x2, y2 = x1 + ww, y1 + hh

            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                vis,
                txt[:32],
                (x1, max(18, y1 - 6)),
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
def _cli() -> int:
    import argparse
    import json

    p = argparse.ArgumentParser(
        description=(
            "Hybrid barcode decoder: YOLO ROIs + sub-ROIs + classic ROI rescue + "
            "tile sweep + multi-backend decoding."
        )
    )
    p.add_argument("image_path", type=str, help="Path to image file")
    p.add_argument("--budget", type=int, default=6000, help="Global time budget in ms")
    p.add_argument("--variants", type=str, default="default", help='Variants: "default" | "all" | comma-list')
    p.add_argument(
        "--prefer",
        type=str,
        default="zxingcpp",
        choices=["zxingcpp", "pyzbar", "opencv_barcode"],
        help="Preferred backend",
    )
    p.add_argument("--no_fallback", action="store_true", help="Disable fallback backends")
    p.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="YOLO model path")
    p.add_argument("--yolo_conf", type=float, default=0.10, help="YOLO confidence threshold")
    p.add_argument("--yolo_iou", type=float, default=0.45, help="YOLO IoU threshold")
    p.add_argument("--yolo_max_det", type=int, default=12, help="YOLO max detections")
    p.add_argument("--yolo_min_size", type=int, default=36, help="YOLO minimum bbox side")
    p.add_argument("--no_classic_roi", action="store_true", help="Disable classic ROI rescue")
    p.add_argument("--classic_max_rois", type=int, default=6, help="Classic max ROIs")
    p.add_argument("--classic_roi_upscale", type=float, default=2.5, help="Classic ROI upscale")
    p.add_argument("--no_tile_sweep", action="store_true", help="Disable tile sweep")
    p.add_argument("--max_tiles", type=int, default=5, help="Max tiles")
    p.add_argument("--no_full_image", action="store_true", help="Disable full image stage")
    p.add_argument("--max_collect_items", type=int, default=64, help="Soft cap of collected items")
    p.add_argument("--save_vis", action="store_true", help="Save visualization image")

    args = p.parse_args()

    img = cv2.imread(args.image_path)
    if img is None:
        print(json.dumps({"status": "error", "error": "Could not load image", "path": args.image_path}, indent=2))
        return 2

    v: Optional[Union[List[str], str]] = None if args.variants == "default" else args.variants

    result = decode_barcode_hybrid(
        img,
        time_budget_ms=int(args.budget),
        variants=v,
        prefer=args.prefer,
        enable_fallback=(not args.no_fallback),
        model_path=args.model_path,
        yolo_conf=float(args.yolo_conf),
        yolo_iou=float(args.yolo_iou),
        yolo_max_det=int(args.yolo_max_det),
        yolo_min_size=int(args.yolo_min_size),
        enable_classic_roi_rescue=(not args.no_classic_roi),
        classic_max_rois=int(args.classic_max_rois),
        classic_roi_upscale=float(args.classic_roi_upscale),
        enable_tile_sweep=(not args.no_tile_sweep),
        max_tiles=int(args.max_tiles),
        include_full_image=(not args.no_full_image),
        max_collect_items=int(args.max_collect_items),
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.save_vis and result.get("items"):
        import os
        from pathlib import Path

        out_dir = Path("results/test_hybrid_barcode")
        out_dir.mkdir(parents=True, exist_ok=True)

        vis = draw_detected_items(img, result["items"])
        stem = Path(args.image_path).stem
        out_path = out_dir / f"{stem}_hybrid_vis.jpg"
        cv2.imwrite(str(out_path), vis)
        print(f"\nVisualización guardada en: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())

# python -m utils.vision_hybrid data/tests_picking/capture_barcode_test.png \
#   --budget 10000 \
#   --variants all \
#   --yolo_conf 0.05 \
#   --yolo_max_det 20 \
#   --classic_max_rois 8 \
#   --classic_roi_upscale 3.0 \
#   --max_tiles 5 \
#   --save_vis