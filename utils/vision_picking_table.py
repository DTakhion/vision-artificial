# utils/vision_picking_table.py
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from utils.vision_preprocess import preprocess_variants, PreprocessConfig


# ============================================================
# Config
# ============================================================
@dataclass
class PickingTableConfig:
    resize_max_side: int = 2400

    # Búsqueda vertical de la tabla
    table_search_y0: float = 0.40
    table_search_y1: float = 0.98

    # Detección de la franja azul del encabezado de tabla
    blue_header_min_width_ratio: float = 0.50
    blue_header_min_height_px: int = 8
    blue_header_max_height_px: int = 90

    # OCR / parseo
    tesseract_psm_table: int = 6
    min_code_len: int = 4
    max_code_len: int = 24
    min_qty: int = 1

    # Filas / celdas
    row_merge_y_ratio: float = 0.020
    column_merge_gap_px: int = 18

    # OCR multipass
    enable_table_multipass: bool = True
    max_table_passes: int = 6

    # Normalización geométrica
    enable_auto_rotate: bool = True
    enable_deskew: bool = True
    deskew_max_angle_deg: float = 8.0
    orientation_try_angles: Tuple[int, ...] = (0, 90, 180, 270)

    # Salida
    save_json: bool = True
    output_dir: str = "data/picking_table"
    save_debug: bool = False
    debug_dir: Optional[str] = None


HEADER_CONCEPTS: Dict[str, set[str]] = {
    "ruta": {"ruta"},
    "orden": {"orden"},
    "kits": {"kits", "kit"},
    "articulo": {"articulo", "articulos", "art"},
    "art_ref": {"ref", "artref", "artref1"},
    "descripcion": {"descripcion", "descripcian", "descripci6n"},
    "unidades": {"unidades", "unidad"},
    "obs": {"obs", "observacion", "observaciones"},
}

COMMON_NON_ITEM_TOKENS = {
    "RUTA",
    "ORDEN",
    "KITS",
    "KITS?",
    "ARTICULO",
    "ART",
    "REF",
    "DESCRIPCION",
    "UNIDADES",
    "OBS",
    "TOTALES",
    "TOTAL",
    "CLIENTE",
    "DIRECCION",
    "EXPORTACION",
    "REGIONES",
    "CORTE",
    "TAREAS",
    "TIPO",
    "GRUA",
    "PISO",
    "TRANSPORTE",
    "REVISAR",
    "HORA",
    "ENTREGA",
    "WAVE",
    "MO",
    "NM",
}


# ============================================================
# Helpers generales
# ============================================================
def _clean_text(txt: str) -> str:
    txt = (txt or "").replace("\r", "\n")
    txt = txt.replace("—", "-").replace("–", "-")
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{2,}", "\n", txt)
    return txt.strip()


def _resize_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0:
        return img
    h, w = img.shape[:2]
    s = max(h, w)
    if s <= max_side:
        return img
    scale = max_side / float(s)
    nw = int(round(w * scale))
    nh = int(round(h * scale))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def _clip_bbox(x: int, y: int, w: int, h: int, W: int, H: int) -> Tuple[int, int, int, int]:
    x = max(0, min(W, x))
    y = max(0, min(H, y))
    w = max(0, min(W - x, w))
    h = max(0, min(H - y, h))
    return x, y, w, h


def _crop(img: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = bbox
    return img[y:y + h, x:x + w].copy()


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _timestamp_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _configure_tesseract_from_env() -> None:
    try:
        import pytesseract  # type: ignore
    except Exception:
        return

    cmd = os.getenv("TESSERACT_CMD")
    if cmd:
        try:
            pytesseract.pytesseract.tesseract_cmd = cmd
        except Exception:
            pass


def _tesseract_available() -> bool:
    try:
        import pytesseract  # type: ignore  # noqa: F401
        return True
    except Exception:
        return False


def _normalize_alnum_upper(txt: str) -> str:
    txt = (txt or "").upper()
    txt = txt.replace(" ", "")
    txt = txt.replace(".", "")
    txt = txt.replace(",", "")
    txt = txt.replace("(", "")
    txt = txt.replace(")", "")
    txt = txt.replace(":", "")
    txt = txt.replace(";", "")
    return txt


def _tokenize_compact(txt: str) -> List[str]:
    txt = _clean_text(txt)
    if not txt:
        return []
    return [t for t in txt.split() if t]


def _normalize_header_token(txt: str) -> str:
    s = (txt or "").lower().strip()
    s = s.replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
    s = s.replace("0", "o")
    s = s.replace("1", "i")
    s = s.replace("5", "s")
    s = s.replace("?", "")
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def _is_header_like_row(row_text: str) -> bool:
    toks = _tokenize_compact(row_text)
    norm_toks = {_normalize_header_token(t) for t in toks if t}

    matched_concepts = set()

    for concept, variants in HEADER_CONCEPTS.items():
        for tok in norm_toks:
            if tok in variants:
                matched_concepts.add(concept)
                break

    hits = len(matched_concepts)
    if hits >= 3:
        return True

    joined = " ".join(sorted(norm_toks))
    if "ruta" in joined and "orden" in joined:
        return True
    if "descripcion" in joined and "unidades" in joined:
        return True

    return False


def _rotate_image_90(img: np.ndarray, angle: int) -> np.ndarray:
    a = angle % 360
    if a == 0:
        return img.copy()
    if a == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if a == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if a == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img.copy()


# ============================================================
# OCR helpers
# ============================================================
def _pick_preprocessed_roi(
    roi_bgr: np.ndarray,
    *,
    prefer_text: bool = False,
) -> Tuple[str, np.ndarray]:
    cfg = PreprocessConfig(
        resize_max_side=2200,
        clahe=True,
        denoise=False,
        bilateral=True,
        sharpen=True,
        morph_close=True,
        binarize=True,
        upscale=True,
        upscale_factors=(2.0, 3.0),
    )
    ims = preprocess_variants(roi_bgr, cfg=cfg) or {}

    if prefer_text:
        order = [
            "bw_x3", "bw_x2", "bw",
            "sharp_x3", "sharp_x2", "sharp",
            "gray_x3", "gray_x2", "gray",
            "bilateral_sharp_x2", "bilateral_sharp", "bilateral"
        ]
    else:
        order = [
            "gray_x3", "gray_x2", "gray",
            "sharp_x3", "sharp_x2", "sharp",
            "bw_x3", "bw_x2", "bw",
            "bilateral_sharp_x2", "bilateral_sharp", "bilateral"
        ]

    for k in order:
        if k in ims and ims[k] is not None:
            return k, ims[k]

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    return "gray_fallback", gray


def _build_manual_variants(roi_bgr: np.ndarray) -> Dict[str, np.ndarray]:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    variants: Dict[str, np.ndarray] = {"gray_manual": gray}

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    variants["clahe_manual"] = clahe

    sharp = cv2.GaussianBlur(clahe, (0, 0), 1.2)
    sharp = cv2.addWeighted(clahe, 1.7, sharp, -0.7, 0)
    variants["sharp_manual"] = sharp

    otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    variants["otsu_manual"] = otsu

    adapt = cv2.adaptiveThreshold(
        sharp,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        12,
    )
    variants["adaptive_manual"] = adapt

    for name in list(variants.keys()):
        img = variants[name]
        h, w = img.shape[:2]
        up2 = cv2.resize(img, (max(1, int(w * 2.0)), max(1, int(h * 2.0))), interpolation=cv2.INTER_CUBIC)
        variants[f"{name}_x2"] = up2

    return variants


def _rescale_ocr_boxes(
    items: List[Dict[str, Any]],
    *,
    src_shape: Tuple[int, int],
    dst_shape: Tuple[int, int],
) -> List[Dict[str, Any]]:
    src_h, src_w = src_shape[:2]
    dst_h, dst_w = dst_shape[:2]

    if src_h <= 0 or src_w <= 0 or dst_h <= 0 or dst_w <= 0:
        return items

    sx = dst_w / float(src_w)
    sy = dst_h / float(src_h)

    out: List[Dict[str, Any]] = []
    for it in items:
        bbox = it.get("bbox")
        if not bbox:
            out.append(it)
            continue

        x, y, w, h = bbox
        nx = int(round(x * sx))
        ny = int(round(y * sy))
        nw = int(round(w * sx))
        nh = int(round(h * sy))

        nx = max(0, min(dst_w - 1, nx)) if dst_w > 0 else 0
        ny = max(0, min(dst_h - 1, ny)) if dst_h > 0 else 0
        nw = max(1, min(dst_w - nx, nw))
        nh = max(1, min(dst_h - ny, nh))

        new_it = dict(it)
        new_it["bbox"] = (nx, ny, nw, nh)
        out.append(new_it)

    return out


def _ocr_text(
    gray: np.ndarray,
    *,
    psm: int = 6,
    whitelist: Optional[str] = None,
) -> str:
    try:
        import pytesseract  # type: ignore
    except Exception:
        return ""

    config = f"--oem 1 --psm {psm}"
    if whitelist:
        config += f' -c tessedit_char_whitelist="{whitelist}"'

    try:
        txt = pytesseract.image_to_string(gray, config=config)
        return _clean_text(txt)
    except Exception:
        return ""


def _ocr_lines_from_data(
    gray: np.ndarray,
    *,
    psm: int = 6,
    whitelist: Optional[str] = None,
) -> List[Dict[str, Any]]:
    try:
        import pytesseract  # type: ignore
    except Exception:
        return []

    config = f"--oem 1 --psm {psm}"
    if whitelist:
        config += f' -c tessedit_char_whitelist="{whitelist}"'

    try:
        data = pytesseract.image_to_data(gray, config=config, output_type=pytesseract.Output.DICT)
    except Exception:
        return []

    n = len(data.get("text", []))
    groups: Dict[Tuple[int, int, int], Dict[str, Any]] = {}

    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue

        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = -1.0

        if conf < 0:
            continue

        key = (
            int(data["block_num"][i]),
            int(data["par_num"][i]),
            int(data["line_num"][i]),
        )

        left = int(data["left"][i])
        top = int(data["top"][i])
        width = int(data["width"][i])
        height = int(data["height"][i])

        if key not in groups:
            groups[key] = {
                "words": [txt],
                "conf": [conf],
                "left": left,
                "top": top,
                "right": left + width,
                "bottom": top + height,
            }
        else:
            g = groups[key]
            g["words"].append(txt)
            g["conf"].append(conf)
            g["left"] = min(g["left"], left)
            g["top"] = min(g["top"], top)
            g["right"] = max(g["right"], left + width)
            g["bottom"] = max(g["bottom"], top + height)

    out: List[Dict[str, Any]] = []
    for g in groups.values():
        txt = _clean_text(" ".join(g["words"]))
        if not txt:
            continue
        confs = g["conf"] or []
        avg_conf = float(sum(confs) / len(confs)) if confs else None
        out.append(
            {
                "text": txt,
                "confidence": avg_conf,
                "bbox": (
                    int(g["left"]),
                    int(g["top"]),
                    int(g["right"] - g["left"]),
                    int(g["bottom"] - g["top"]),
                ),
            }
        )

    out.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))
    return out


def _ocr_words_from_data(
    gray: np.ndarray,
    *,
    psm: int = 6,
    whitelist: Optional[str] = None,
) -> List[Dict[str, Any]]:
    try:
        import pytesseract  # type: ignore
    except Exception:
        return []

    config = f"--oem 1 --psm {psm}"
    if whitelist:
        config += f' -c tessedit_char_whitelist="{whitelist}"'

    try:
        data = pytesseract.image_to_data(gray, config=config, output_type=pytesseract.Output.DICT)
    except Exception:
        return []

    n = len(data.get("text", []))
    out: List[Dict[str, Any]] = []

    for i in range(n):
        txt = _clean_text(data["text"][i] or "")
        if not txt:
            continue

        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = -1.0

        if conf < 0:
            continue

        left = int(data["left"][i])
        top = int(data["top"][i])
        width = int(data["width"][i])
        height = int(data["height"][i])

        out.append(
            {
                "text": txt,
                "confidence": conf,
                "bbox": (left, top, width, height),
            }
        )

    out.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))
    return out


def _ocr_table_candidates(
    table_roi: np.ndarray,
    cfg: PickingTableConfig,
) -> List[Dict[str, Any]]:
    whitelist = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_/.:()&? '

    roi_h, roi_w = table_roi.shape[:2]
    candidates: List[Dict[str, Any]] = []

    main_variant_name, main_gray = _pick_preprocessed_roi(table_roi, prefer_text=False)

    main_words = _ocr_words_from_data(main_gray, psm=cfg.tesseract_psm_table, whitelist=whitelist)
    main_lines = _ocr_lines_from_data(main_gray, psm=cfg.tesseract_psm_table, whitelist=whitelist)

    main_words = _rescale_ocr_boxes(main_words, src_shape=main_gray.shape[:2], dst_shape=(roi_h, roi_w))
    main_lines = _rescale_ocr_boxes(main_lines, src_shape=main_gray.shape[:2], dst_shape=(roi_h, roi_w))

    candidates.append(
        {
            "name": main_variant_name,
            "gray": main_gray,
            "words": main_words,
            "lines": main_lines,
        }
    )

    if cfg.enable_table_multipass:
        manual = _build_manual_variants(table_roi)
        preferred = [
            "gray_manual_x2",
            "sharp_manual_x2",
            "adaptive_manual_x2",
            "otsu_manual_x2",
            "clahe_manual_x2",
        ]
        count = 0
        for name in preferred:
            if name not in manual:
                continue

            gray = manual[name]
            words = _ocr_words_from_data(gray, psm=cfg.tesseract_psm_table, whitelist=whitelist)
            lines = _ocr_lines_from_data(gray, psm=cfg.tesseract_psm_table, whitelist=whitelist)

            words = _rescale_ocr_boxes(words, src_shape=gray.shape[:2], dst_shape=(roi_h, roi_w))
            lines = _rescale_ocr_boxes(lines, src_shape=gray.shape[:2], dst_shape=(roi_h, roi_w))

            candidates.append(
                {
                    "name": name,
                    "gray": gray,
                    "words": words,
                    "lines": lines,
                }
            )
            count += 1
            if count >= max(0, cfg.max_table_passes - 1):
                break

    return candidates


# ============================================================
# Normalización geométrica
# ============================================================
def _estimate_skew_angle(gray: np.ndarray, max_angle_deg: float = 8.0) -> float:
    h, w = gray.shape[:2]
    if h == 0 or w == 0:
        return 0.0

    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)

    lines = cv2.HoughLinesP(
        bw,
        rho=1,
        theta=np.pi / 180.0,
        threshold=120,
        minLineLength=max(80, int(w * 0.18)),
        maxLineGap=15,
    )

    if lines is None:
        return 0.0

    angles: List[float] = []
    for ln in lines[:, 0, :]:
        x1, y1, x2, y2 = ln
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) < 1:
            continue
        ang = np.degrees(np.arctan2(dy, dx))
        if -max_angle_deg <= ang <= max_angle_deg:
            angles.append(float(ang))

    if not angles:
        return 0.0

    return float(np.median(angles))


def _rotate_arbitrary(img: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    return cv2.warpAffine(
        img,
        M,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _deskew_if_needed(img_bgr: np.ndarray, cfg: PickingTableConfig) -> Tuple[np.ndarray, float]:
    if not cfg.enable_deskew:
        return img_bgr, 0.0

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    ang = _estimate_skew_angle(gray, max_angle_deg=cfg.deskew_max_angle_deg)

    if abs(ang) < 0.35:
        return img_bgr, 0.0

    corrected = _rotate_arbitrary(img_bgr, ang)
    return corrected, float(ang)


# ============================================================
# Detección de tabla
# ============================================================
def _find_blue_header_bbox(
    img_bgr: np.ndarray,
    cfg: PickingTableConfig,
) -> Optional[Tuple[int, int, int, int]]:
    H, W = img_bgr.shape[:2]
    y0 = int(round(H * cfg.table_search_y0))
    y1 = int(round(H * cfg.table_search_y1))
    search = img_bgr[y0:y1, :].copy()
    if search.size == 0:
        return None

    hsv = cv2.cvtColor(search, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([85, 35, 30], dtype=np.uint8)
    upper_blue = np.array([140, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    candidates: List[Tuple[float, Tuple[int, int, int, int]]] = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)

        if w < int(W * cfg.blue_header_min_width_ratio):
            continue
        if h < cfg.blue_header_min_height_px or h > cfg.blue_header_max_height_px:
            continue

        area = w * h
        width_score = w * 8.0
        y_score = -abs((y + y0) - int(H * 0.62)) * 0.7
        height_score = -abs(h - 28) * 8.0
        score = area + width_score + y_score + height_score

        candidates.append((score, (x, y + y0, w, h)))

    if not candidates:
        return None

    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


def _detect_table_bbox(
    img_bgr: np.ndarray,
    cfg: PickingTableConfig,
) -> Optional[Tuple[int, int, int, int]]:
    H, W = img_bgr.shape[:2]

    blue_bbox = _find_blue_header_bbox(img_bgr, cfg)
    if blue_bbox is None:
        return None

    bx, by, bw, bh = blue_bbox

    x0 = max(0, bx - int(0.006 * W))
    x1 = min(W, bx + bw + int(0.006 * W))

    y0 = max(0, by + int(0.85 * bh))
    y1 = min(H, int(H * 0.965))

    return _clip_bbox(x0, y0, x1 - x0, y1 - y0, W, H)


def _fallback_table_bbox(
    img_bgr: np.ndarray,
    cfg: PickingTableConfig,
) -> Tuple[int, int, int, int]:
    H, W = img_bgr.shape[:2]
    y0 = int(round(H * cfg.table_search_y0))
    y1 = int(round(H * cfg.table_search_y1))
    return _clip_bbox(0, y0, W, max(1, y1 - y0), W, H)


def _table_presence_score(img_bgr: np.ndarray, cfg: PickingTableConfig) -> Tuple[float, Dict[str, Any]]:
    H, W = img_bgr.shape[:2]
    blue_bbox = _find_blue_header_bbox(img_bgr, cfg)

    blue_score = 0.0
    if blue_bbox is not None:
        bx, by, bw, bh = blue_bbox
        blue_score += 4.0
        blue_score += min(4.0, bw / max(1.0, W * 0.70))
        blue_center_y = by + bh / 2.0
        ideal_y = H * 0.64
        blue_score += max(0.0, 2.0 - abs(blue_center_y - ideal_y) / max(1.0, H * 0.18))

    landscape_score = 1.0 if W >= H else -1.5
    total = blue_score + landscape_score

    return total, {
        "score_total": total,
        "score_blue": blue_score,
        "score_landscape": landscape_score,
        "blue_bbox": blue_bbox,
    }


def _auto_orient_image(
    img_bgr: np.ndarray,
    cfg: PickingTableConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if not cfg.enable_auto_rotate:
        return img_bgr, {"selected_angle": 0, "candidates": []}

    candidates = []
    best_score = -1e18
    best_img = img_bgr
    best_angle = 0
    best_details: Dict[str, Any] = {}

    for angle in cfg.orientation_try_angles:
        cand = _rotate_image_90(img_bgr, angle)
        score, details = _table_presence_score(cand, cfg)
        candidates.append({"angle": angle, "score": score, **details})

        if score > best_score:
            best_score = score
            best_img = cand
            best_angle = angle
            best_details = details

    return best_img, {
        "selected_angle": best_angle,
        "best_score": best_score,
        "best_details": best_details,
        "candidates": candidates,
    }


# ============================================================
# Parseo de productos
# ============================================================
def _looks_like_route(token: str) -> bool:
    s = _normalize_alnum_upper(token)
    return bool(re.fullmatch(r"SCL\d{6}-?\d{2}", s))


def _looks_like_order(token: str) -> bool:
    s = _normalize_alnum_upper(token)
    return bool(re.fullmatch(r"\d{8,12}", s))


def _looks_like_item_code(token: str, cfg: PickingTableConfig) -> bool:
    s = _normalize_alnum_upper(token)
    if len(s) < cfg.min_code_len or len(s) > cfg.max_code_len:
        return False
    if _looks_like_route(s):
        return False
    if _looks_like_order(s):
        return False
    if s in COMMON_NON_ITEM_TOKENS:
        return False
    if not re.fullmatch(r"[A-Z0-9_-]+", s):
        return False

    has_letters = bool(re.search(r"[A-Z]", s))
    has_digits = bool(re.search(r"\d", s))

    # Palabras OCR basura puramente alfabéticas
    if re.fullmatch(r"[A-Z]{1,3}", s):
        return False

    # Casos típicos alfanuméricos
    if has_letters and has_digits:
        return True

    # Casos puramente alfabéticos razonables como EZAROTE
    if re.fullmatch(r"[A-Z]{5,16}", s):
        return True

    return False


def _fix_route_ocr(txt: str) -> str:
    s = _normalize_alnum_upper(txt)
    s = s.replace("SCI", "SCL")
    s = s.replace("SCT", "SCL")
    s = s.replace("SC1", "SCL")
    s = s.replace("5CL", "SCL")
    return s


def _fix_item_code_ocr(txt: str) -> str:
    s = _normalize_alnum_upper(txt)
    if not s:
        return s

    s = s.strip("_-")
    s = re.sub(r"_{2,}", "_", s)

    s = re.sub(r"^E29F", "EZ9F", s)
    s = re.sub(r"^EZGF", "EZ9F", s)
    s = re.sub(r"^EZ9FS", "EZ9F5", s)

    s = s.replace("TMO", "TM0")
    s = s.replace("CAO", "CA0")

    s = re.sub(r"^(C\d{2}F3)1M(\d{3})$", r"\1TM\2", s)
    s = re.sub(r"^(C\d{2}F3)[1IL]M(\d{3})$", r"\1TM\2", s)

    s = re.sub(r"[^A-Z0-9_-]", "", s)
    return s


def _extract_qty_from_text(txt: str, cfg: PickingTableConfig) -> Optional[int]:
    nums = re.findall(r"\b\d+\b", txt or "")
    if not nums:
        return None
    try:
        val = int(nums[-1])
        if val >= cfg.min_qty:
            return val
    except Exception:
        return None
    return None


def _cluster_bbox(cluster: List[Dict[str, Any]]) -> Optional[Tuple[int, int, int, int]]:
    boxes = [ln.get("bbox") for ln in cluster if ln.get("bbox") is not None]
    if not boxes:
        return None
    xs = [b[0] for b in boxes]
    ys = [b[1] for b in boxes]
    x2s = [b[0] + b[2] for b in boxes]
    y2s = [b[1] + b[3] for b in boxes]
    return (min(xs), min(ys), max(x2s) - min(xs), max(y2s) - min(ys))


def _cluster_confidence(cluster: List[Dict[str, Any]]) -> Optional[float]:
    confs = [ln.get("confidence") for ln in cluster if ln.get("confidence") is not None]
    if not confs:
        return None
    return float(sum(confs) / len(confs))


def _filter_table_noise(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for wd in words:
        txt = _clean_text(wd.get("text") or "")
        if not txt:
            continue

        low = txt.lower()
        if low in {
            "ruta", "orden", "kits?", "kits", "articulo", "art", "ref",
            "descripcion", "unidades", "obs", "totales", "total"
        }:
            continue

        if "totales" in low:
            continue

        out.append(wd)
    return out


def _row_clusters_from_words(
    words: List[Dict[str, Any]],
    *,
    table_h: int,
    cfg: PickingTableConfig,
) -> List[List[Dict[str, Any]]]:
    filtered = _filter_table_noise(words)
    if not filtered:
        return []

    filtered.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))

    merge_thr = max(10, int(round(table_h * cfg.row_merge_y_ratio)))
    clusters: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = [filtered[0]]
    current_center = filtered[0]["bbox"][1] + filtered[0]["bbox"][3] / 2.0

    for wd in filtered[1:]:
        y = wd["bbox"][1] + wd["bbox"][3] / 2.0
        if abs(y - current_center) <= merge_thr:
            current.append(wd)
            ys = [it["bbox"][1] + it["bbox"][3] / 2.0 for it in current]
            current_center = float(sum(ys) / len(ys))
        else:
            clusters.append(current)
            current = [wd]
            current_center = y

    if current:
        clusters.append(current)

    return clusters


def _merge_words_horizontally(
    words: List[Dict[str, Any]],
    *,
    max_gap_px: int,
) -> List[Dict[str, Any]]:
    if not words:
        return []

    words = sorted(words, key=lambda x: x["bbox"][0])
    merged: List[Dict[str, Any]] = []

    cur = {
        "text": _clean_text(words[0]["text"]),
        "confidence": words[0].get("confidence"),
        "bbox": words[0]["bbox"],
    }

    for wd in words[1:]:
        txt = _clean_text(wd["text"])
        if not txt:
            continue

        x, y, w, h = wd["bbox"]
        cx, cy, cw, ch = cur["bbox"]
        cur_right = cx + cw
        gap = x - cur_right

        same_band = abs((y + h / 2.0) - (cy + ch / 2.0)) <= max(h, ch) * 0.7

        if same_band and gap <= max_gap_px:
            cur["text"] = _clean_text(f'{cur["text"]} {txt}')
            conf_a = cur.get("confidence")
            conf_b = wd.get("confidence")
            vals = [v for v in [conf_a, conf_b] if isinstance(v, (int, float))]
            cur["confidence"] = float(sum(vals) / len(vals)) if vals else None

            nx = min(cx, x)
            ny = min(cy, y)
            nr = max(cx + cw, x + w)
            nb = max(cy + ch, y + h)
            cur["bbox"] = (nx, ny, nr - nx, nb - ny)
        else:
            merged.append(cur)
            cur = {
                "text": txt,
                "confidence": wd.get("confidence"),
                "bbox": wd["bbox"],
            }

    merged.append(cur)
    return merged


def _cluster_to_cells(
    cluster: List[Dict[str, Any]],
    cfg: PickingTableConfig,
) -> List[Dict[str, Any]]:
    merged = _merge_words_horizontally(cluster, max_gap_px=cfg.column_merge_gap_px)
    merged.sort(key=lambda x: x["bbox"][0])
    return merged


def _parse_row_from_cells(
    cells: List[Dict[str, Any]],
    cfg: PickingTableConfig,
) -> Optional[Dict[str, Any]]:
    if not cells:
        return None

    texts = [_clean_text(c["text"]) for c in cells if _clean_text(c["text"])]
    if not texts:
        return None

    row_text = _clean_text(" ".join(texts)).strip()
    low = row_text.lower()

    if "totales" in low:
        return None
    if _is_header_like_row(row_text):
        return None

    qty = None
    qty_idx = None
    for i in range(len(cells) - 1, max(-1, len(cells) - 4), -1):
        q = _extract_qty_from_text(cells[i]["text"], cfg)
        if q is not None:
            qty = q
            qty_idx = i
            break

    if qty is None or qty_idx is None:
        return None

    before = cells[:qty_idx]
    if len(before) < 2:
        return None

    ruta = None
    ruta_idx = None
    for i, c in enumerate(before[:3]):
        tok = _fix_route_ocr(c["text"])
        if _looks_like_route(tok):
            ruta = tok
            ruta_idx = i
            break

    orden = None
    orden_idx = None
    start_for_order = (ruta_idx + 1) if ruta_idx is not None else 0
    for i in range(start_for_order, min(len(before), start_for_order + 4)):
        tok = _normalize_alnum_upper(before[i]["text"])
        if _looks_like_order(tok):
            orden = tok
            orden_idx = i
            break

    codigo_item = None
    codigo_idx = None
    start_for_code = (orden_idx + 1) if orden_idx is not None else 0
    for i in range(start_for_code, len(before)):
        tok = _fix_item_code_ocr(before[i]["text"])
        if _looks_like_item_code(tok, cfg):
            codigo_item = tok
            codigo_idx = i
            break

    if codigo_item is None:
        for i, c in enumerate(before):
            tok = _fix_item_code_ocr(c["text"])
            if _looks_like_item_code(tok, cfg):
                codigo_item = tok
                codigo_idx = i
                break

    if codigo_item is None:
        return None

    desc_cells: List[str] = []
    for i, c in enumerate(before):
        if i in {ruta_idx, orden_idx, codigo_idx}:
            continue
        desc_cells.append(_clean_text(c["text"]))

    descripcion = _clean_text(" ".join([x for x in desc_cells if x])) or None
    if not descripcion or len(descripcion) < 5:
        return None

    desc_tokens = [t for t in descripcion.split() if t]
    if len(desc_tokens) < 1:
        return None

    row_bbox = _cluster_bbox(cells)
    avg_conf = _cluster_confidence(cells)

    return {
        "ruta": ruta,
        "orden": orden,
        "articulo": codigo_item,
        "descripcion": descripcion,
        "unidades": qty,
        "raw_line": row_text,
        "bbox": row_bbox,
        "confidence": avg_conf,
        "cells": [
            {
                "text": _clean_text(c["text"]),
                "bbox": c["bbox"],
                "confidence": c.get("confidence"),
            }
            for c in cells
        ],
    }


def _parse_products_from_words(
    words: List[Dict[str, Any]],
    cfg: PickingTableConfig,
    table_bbox: Tuple[int, int, int, int],
) -> List[Dict[str, Any]]:
    _, _, _, th = table_bbox
    row_clusters = _row_clusters_from_words(words, table_h=th, cfg=cfg)

    items: List[Dict[str, Any]] = []
    seen: set[Tuple[str, int]] = set()

    for cluster in row_clusters:
        cells = _cluster_to_cells(cluster, cfg)
        row = _parse_row_from_cells(cells, cfg)
        if row is None:
            continue

        key = (row["articulo"], int(row["unidades"]))
        if key in seen:
            continue
        seen.add(key)
        items.append(row)

    return items


def _score_products(products: List[Dict[str, Any]]) -> float:
    score = 0.0
    score += len(products) * 8.0

    unique_codes = {p.get("articulo") for p in products if p.get("articulo")}
    score += len(unique_codes) * 2.0

    for p in products:
        code = p.get("articulo") or ""
        desc = p.get("descripcion") or ""
        units = p.get("unidades")
        if code and len(code) >= 4:
            score += 2.0
        if desc and len(desc) >= 5:
            score += 1.0
        if isinstance(units, int) and units > 0:
            score += 1.0
        if p.get("ruta"):
            score += 0.5
        if p.get("orden"):
            score += 0.5

    return score


def _select_best_products_result(
    candidates: List[Dict[str, Any]],
    cfg: PickingTableConfig,
    table_bbox: Tuple[int, int, int, int],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    evaluated: List[Dict[str, Any]] = []

    for cand in candidates:
        words = cand.get("words") or []
        lines = cand.get("lines") or []

        products_words = _parse_products_from_words(words, cfg, table_bbox) if words else []
        products_lines = _parse_products_from_words(lines, cfg, table_bbox) if lines else []

        score_words = _score_products(products_words)
        score_lines = _score_products(products_lines)

        if score_lines > score_words:
            best_products = products_lines
            mode = "lines"
            best_score = score_lines
        else:
            best_products = products_words
            mode = "words"
            best_score = score_words

        evaluated.append(
            {
                "variant": cand.get("name"),
                "mode": mode,
                "score": best_score,
                "products_count": len(best_products),
                "products": best_products,
            }
        )

    if not evaluated:
        return [], {"selected": None, "candidates": []}

    evaluated.sort(key=lambda x: x["score"], reverse=True)
    best = evaluated[0]

    return best["products"], {
        "selected": {
            "variant": best["variant"],
            "mode": best["mode"],
            "score": best["score"],
            "products_count": best["products_count"],
        },
        "candidates": [
            {
                "variant": e["variant"],
                "mode": e["mode"],
                "score": e["score"],
                "products_count": e["products_count"],
            }
            for e in evaluated
        ],
    }


# ============================================================
# Guardado
# ============================================================
def _save_result_json(
    result: Dict[str, Any],
    *,
    cfg: PickingTableConfig,
    source_name: Optional[str] = None,
) -> str:
    out_dir = _ensure_dir(cfg.output_dir)
    ts = _timestamp_now()

    safe_source = None
    if source_name:
        safe_source = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(source_name).stem).strip("_")
        if not safe_source:
            safe_source = "picking_table"

    filename = f"{safe_source}_{ts}.json" if safe_source else f"picking_table_{ts}.json"
    out_path = out_dir / filename

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return str(out_path)


def build_table_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    products = result.get("products") or []
    total_units = 0

    summary_products: List[Dict[str, Any]] = []
    for p in products:
        units = p.get("unidades") if isinstance(p.get("unidades"), int) else 0
        total_units += units
        summary_products.append(
            {
                "ruta": p.get("ruta"),
                "orden": p.get("orden"),
                "articulo": p.get("articulo"),
                "descripcion": p.get("descripcion"),
                "unidades": units,
            }
        )

    return {
        "document_type": "picking_sheet_table_only",
        "status": result.get("status"),
        "source_name": result.get("source_name"),
        "timestamp": result.get("timestamp"),
        "products": summary_products,
        "totals": {
            "lineas": len(summary_products),
            "unidades_totales": total_units,
        },
    }


# ============================================================
# API principal
# ============================================================
def extract_picking_table(
    img_bgr: np.ndarray,
    *,
    cfg: Optional[PickingTableConfig] = None,
    source_name: Optional[str] = None,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    cfg = cfg or PickingTableConfig()

    _configure_tesseract_from_env()

    if img_bgr is None or not isinstance(img_bgr, np.ndarray) or img_bgr.size == 0:
        return {
            "status": "error",
            "error": "invalid_image",
            "elapsed_ms": 0,
        }

    if not _tesseract_available():
        return {
            "status": "not_available",
            "error": "tesseract_not_available",
            "elapsed_ms": 0,
        }

    img0 = _resize_max_side(img_bgr, cfg.resize_max_side)

    oriented_img, orientation_info = _auto_orient_image(img0, cfg)
    final_img, deskew_angle = _deskew_if_needed(oriented_img, cfg)
    img = final_img

    H, W = img.shape[:2]

    table_bbox = _detect_table_bbox(img, cfg)
    if table_bbox is None:
        table_bbox = _fallback_table_bbox(img, cfg)
    table_roi = _crop(img, table_bbox)

    table_candidates = _ocr_table_candidates(table_roi, cfg)
    products, table_eval = _select_best_products_result(table_candidates, cfg, table_bbox)

    table_variant = table_candidates[0]["name"] if table_candidates else None
    table_gray = table_candidates[0]["gray"] if table_candidates else cv2.cvtColor(table_roi, cv2.COLOR_BGR2GRAY)
    table_words = table_candidates[0]["words"] if table_candidates else []
    table_lines = table_candidates[0]["lines"] if table_candidates else []

    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    result: Dict[str, Any] = {
        "status": "success",
        "elapsed_ms": elapsed_ms,
        "source_name": source_name,
        "timestamp": datetime.now().isoformat(),
        "image_size": {"w": W, "h": H},
        "products": products,
        "counts": {
            "products_detected": len(products),
        },
        "rois": {
            "table": {"bbox": table_bbox, "variant": table_variant},
        },
        "normalization": {
            "selected_rotation_deg": orientation_info.get("selected_angle"),
            "deskew_angle_deg": deskew_angle,
        },
        "raw": {
            "table_words": table_words,
            "table_lines": table_lines,
            "table_text_block": _ocr_text(
                table_gray,
                psm=cfg.tesseract_psm_table,
                whitelist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_/.:() ",
            ),
            "table_eval": table_eval,
            "orientation_candidates": orientation_info.get("candidates", []),
        },
        "config": asdict(cfg),
        "summary": None,
        "saved_json_path": None,
    }

    if cfg.save_debug:
        debug_dir = cfg.debug_dir or os.path.join(cfg.output_dir, f"debug_{_timestamp_now()}")
        dbg = _ensure_dir(debug_dir)

        cv2.imwrite(str(dbg / "00_input_normalized.png"), img)
        cv2.imwrite(str(dbg / "10_table_roi.png"), table_roi)
        cv2.imwrite(str(dbg / "20_table_gray.png"), table_gray)

        dbg_img = img.copy()
        tx, ty, tw, th = table_bbox
        cv2.rectangle(dbg_img, (tx, ty), (tx + tw, ty + th), (0, 255, 0), 2)

        blue_bbox = _find_blue_header_bbox(img, cfg)
        if blue_bbox is not None:
            x, y, w, h = blue_bbox
            cv2.rectangle(dbg_img, (x, y), (x + w, y + h), (0, 255, 255), 2)

        dbg_words = table_roi.copy()
        for wd in table_words:
            x, y, w, h = wd["bbox"]
            cv2.rectangle(dbg_words, (x, y), (x + w, y + h), (0, 255, 255), 1)
            txt = _clean_text(wd["text"])[:20]
            if txt:
                cv2.putText(
                    dbg_words,
                    txt,
                    (x, max(12, y - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

        dbg_rows = table_roi.copy()
        for item in products:
            bbox = item.get("bbox")
            if bbox:
                x, y, w, h = bbox
                cv2.rectangle(dbg_rows, (x, y), (x + w, y + h), (0, 255, 0), 2)
            for cell in item.get("cells", []):
                cb = cell.get("bbox")
                if cb:
                    cx, cy, cw, ch = cb
                    cv2.rectangle(dbg_rows, (cx, cy), (cx + cw, cy + ch), (255, 0, 255), 1)

        cv2.imwrite(str(dbg / "30_debug_boxes.png"), dbg_img)
        cv2.imwrite(str(dbg / "31_table_words.png"), dbg_words)
        cv2.imwrite(str(dbg / "32_table_rows.png"), dbg_rows)

        result["debug_dir"] = str(dbg)

    summary = build_table_summary(result)
    result["summary"] = summary

    if cfg.save_json:
        result["saved_json_path"] = _save_result_json(result, cfg=cfg, source_name=source_name)

    return result


def extract_picking_table_from_path(
    image_path: str,
    *,
    cfg: Optional[PickingTableConfig] = None,
) -> Dict[str, Any]:
    img = cv2.imread(image_path)
    if img is None:
        return {
            "status": "error",
            "error": "could_not_load_image",
            "path": image_path,
            "elapsed_ms": 0,
        }
    return extract_picking_table(img, cfg=cfg, source_name=Path(image_path).name)


# ============================================================
# CLI
# ============================================================
def _cli() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Extract only the product table from a picking sheet.")
    parser.add_argument("image_path", type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--output_dir", type=str, default="data/picking_table")
    parser.add_argument("--no_auto_rotate", action="store_true")
    parser.add_argument("--no_deskew", action="store_true")
    args = parser.parse_args()

    cfg = PickingTableConfig(
        save_json=(not args.no_save),
        output_dir=args.output_dir,
        save_debug=args.debug,
        enable_auto_rotate=(not args.no_auto_rotate),
        enable_deskew=(not args.no_deskew),
    )

    res = extract_picking_table_from_path(args.image_path, cfg=cfg)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0 if res.get("status") == "success" else 2


if __name__ == "__main__":
    raise SystemExit(_cli())