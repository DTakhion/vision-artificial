# utils/vision_barcode_yolo.py

from __future__ import annotations

from typing import Any, Callable, Dict, List

import cv2
import numpy as np
from ultralytics import YOLO


# ------------------------------------------------------------------
# Modelo YOLO
# ------------------------------------------------------------------
DEFAULT_MODEL_PATH = "runs/detect/runs_kuehne_nagel/barcode_v1/weights/best.pt"

_MODEL_CACHE: Dict[str, YOLO] = {}


def load_yolo_model(model_path: str = DEFAULT_MODEL_PATH) -> YOLO:
    """
    Carga el modelo YOLO una sola vez por ruta y lo deja cacheado.
    """
    if model_path not in _MODEL_CACHE:
        _MODEL_CACHE[model_path] = YOLO(model_path)
    return _MODEL_CACHE[model_path]


# ------------------------------------------------------------------
# Helpers base
# ------------------------------------------------------------------
def _clip_bbox_xyxy(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int]:
    """
    Ajusta bbox a los límites de la imagen.
    """
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w))
    y2 = max(0, min(y2, img_h))
    return x1, y1, x2, y2


def _to_gray(img: np.ndarray) -> np.ndarray:
    """
    Convierte a escala de grises si corresponde.
    """
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _ensure_bgr(img: np.ndarray) -> np.ndarray:
    """
    Asegura imagen BGR para módulos que lo requieren.
    """
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _safe_resize(
    img: np.ndarray,
    scale: float,
    interpolation: int = cv2.INTER_CUBIC,
) -> np.ndarray:
    """
    Resize robusto evitando dimensiones inválidas.
    """
    h, w = img.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=interpolation)


def _light_sharpen(gray: np.ndarray) -> np.ndarray:
    """
    Sharpen suave. Menos agresivo que uno clásico duro.
    """
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


def _normalize_contrast(gray: np.ndarray) -> np.ndarray:
    """
    Normalización de contraste suave.
    """
    if gray is None or gray.size == 0:
        return gray

    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)
    except Exception:
        return cv2.equalizeHist(gray)


def _basic_text_validation(text: str) -> bool:
    """
    Validación básica de texto decodificado para filtrar basura.
    """
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


def _debug_print(enabled: bool, message: str) -> None:
    """
    Imprime sólo si debug está activado.
    """
    if enabled:
        print(message)


# ------------------------------------------------------------------
# Geometría: orientación y deskew
# ------------------------------------------------------------------
def _rotate_image_bound(
    img: np.ndarray,
    angle_deg: float,
    border_value: int | tuple[int, int, int] = 255,
) -> np.ndarray:
    """
    Rota una imagen preservando contenido (canvas expandido).
    """
    if img is None or img.size == 0:
        return img

    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)

    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    matrix[0, 2] += (new_w / 2.0) - center[0]
    matrix[1, 2] += (new_h / 2.0) - center[1]

    return cv2.warpAffine(
        img,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def _estimate_barcode_angle(gray: np.ndarray) -> float:
    """
    Estima el ángulo dominante del barcode dentro del ROI.

    Retorna un ángulo en grados tal que al rotar la imagen por -ángulo
    el barcode debería quedar más horizontal.
    """
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
    """
    Score para evaluar qué tan "barcode-like" queda una imagen:
    mayor variación de la proyección vertical suele favorecer 1D barcode.
    """
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
) -> List[tuple[str, np.ndarray]]:
    """
    Genera variantes con orientación corregida:
    - estimate
    - barrido fino alrededor de estimate
    """
    if img is None or img.size == 0:
        return []

    gray = _to_gray(img)
    est = _estimate_barcode_angle(gray)

    variants: List[tuple[str, np.ndarray]] = []

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

    scored: List[tuple[float, float, np.ndarray]] = []

    for angle in candidates:
        rotated = _rotate_image_bound(img, -float(angle), border_value=255)
        score = _compute_projection_score(_to_gray(rotated))
        scored.append((score, float(angle), rotated))

    scored.sort(key=lambda x: x[0], reverse=True)

    for idx, (_, angle, rotated) in enumerate(scored[:3]):
        variants.append((f"deskew_top{idx + 1}_{angle:.2f}", rotated))

    return variants


# ------------------------------------------------------------------
# Refinamiento opcional de región
# ------------------------------------------------------------------
def refine_barcode_region(gray: np.ndarray) -> np.ndarray:
    """
    Encuentra una subregión probable de barcode dentro del ROI,
    pero con un enfoque conservador.

    OJO:
    - esta función NO debe reemplazar el crop original;
    - úsala como variante adicional.
    """
    if gray is None or gray.size == 0:
        return gray

    if gray.ndim != 2:
        gray = _to_gray(gray)

    h, w = gray.shape[:2]
    if h < 20 or w < 20:
        return gray

    try:
        work = _normalize_contrast(gray)

        grad_x = cv2.Sobel(work, cv2.CV_32F, 1, 0, ksize=3)
        grad_x = cv2.convertScaleAbs(grad_x)

        blurred = cv2.GaussianBlur(grad_x, (5, 5), 0)

        _, thresh = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return gray

        best = None
        best_score = -1.0

        for contour in contours:
            x, y, ww, hh = cv2.boundingRect(contour)
            area = ww * hh
            if area < 0.02 * (w * h):
                continue

            aspect = ww / max(1.0, float(hh))
            fill = area / max(1.0, float(w * h))
            score = float(area) * (1.0 + min(aspect, 8.0) * 0.15) * (1.0 + fill)

            if score > best_score:
                best_score = score
                best = (x, y, ww, hh)

        if best is None:
            return gray

        x, y, ww, hh = best

        pad_x = int(round(ww * 0.18))
        pad_y = int(round(hh * 0.25))

        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(w, x + ww + pad_x)
        y1 = min(h, y + hh + pad_y)

        refined = gray[y0:y1, x0:x1]
        return refined if refined.size > 0 else gray
    except Exception:
        return gray


# ------------------------------------------------------------------
# Variantes de crop / orientación
# ------------------------------------------------------------------
def _build_crop_variants(crop: np.ndarray) -> List[tuple[str, np.ndarray]]:
    """
    Genera variantes del crop para robustecer orientación.
    """
    if crop is None or crop.size == 0:
        return []

    variants: List[tuple[str, np.ndarray]] = [("orig", crop)]

    try:
        variants.append(("rot90", cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)))
    except Exception:
        pass

    try:
        variants.append(("rot270", cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)))
    except Exception:
        pass

    try:
        variants.extend(_autodeskew_barcode(crop))
    except Exception:
        pass

    for name, img_var in list(variants[:]):
        if name in {"rot90", "rot270"}:
            try:
                deskewed = _autodeskew_barcode(img_var)
                for dname, dimg in deskewed:
                    variants.append((f"{name}_{dname}", dimg))
            except Exception:
                pass

    out: List[tuple[str, np.ndarray]] = []
    seen = set()
    for name, img_var in variants:
        if name in seen:
            continue
        if img_var is None or img_var.size == 0:
            continue
        seen.add(name)
        out.append((name, img_var))

    return out


# ------------------------------------------------------------------
# Preproceso para decodificación
# ------------------------------------------------------------------
def _preprocess_for_barcode(img: np.ndarray) -> List[tuple[str, np.ndarray]]:
    """
    Genera variantes de preproceso orientadas a DECODIFICAR,
    no sólo a detectar.
    """
    if img is None or img.size == 0:
        return []

    variants: List[tuple[str, np.ndarray]] = [("orig", img)]

    gray = _to_gray(img)
    variants.append(("gray", gray))

    gray_norm = _normalize_contrast(gray)
    variants.append(("gray_norm", gray_norm))

    gray_up2 = _safe_resize(gray_norm, 2.0)
    gray_up3 = _safe_resize(gray_norm, 3.0)
    variants.append(("gray_up2", gray_up2))
    variants.append(("gray_up3", gray_up3))

    sharp_up2 = _light_sharpen(gray_up2)
    sharp_up3 = _light_sharpen(gray_up3)
    variants.append(("sharp_up2", sharp_up2))
    variants.append(("sharp_up3", sharp_up3))

    _, otsu_up2 = cv2.threshold(
        sharp_up2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    variants.append(("otsu_up2", otsu_up2))

    adaptive_up2 = cv2.adaptiveThreshold(
        sharp_up2,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        4,
    )
    variants.append(("adaptive_up2", adaptive_up2))

    try:
        refined = refine_barcode_region(gray_norm)
        if refined is not None and refined.size > 0:
            variants.append(("refined_gray", refined))

            refined_up2 = _safe_resize(refined, 2.0)
            variants.append(("refined_up2", refined_up2))

            refined_sharp_up2 = _light_sharpen(refined_up2)
            variants.append(("refined_sharp_up2", refined_sharp_up2))

            _, refined_otsu = cv2.threshold(
                refined_sharp_up2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            variants.append(("refined_otsu_up2", refined_otsu))
    except Exception:
        pass

    out: List[tuple[str, np.ndarray]] = []
    seen = set()
    for name, value in variants:
        if name in seen:
            continue
        if value is None or value.size == 0:
            continue
        seen.add(name)
        out.append((name, value))

    return out


# ------------------------------------------------------------------
# OpenCV Barcode fallback
# ------------------------------------------------------------------
def _opencv_barcode_fallback(img: np.ndarray) -> Dict[str, Any]:
    """
    Fallback usando OpenCV BarcodeDetector.
    """
    try:
        if img is None or img.size == 0:
            return {"status": "invalid_image", "items": []}

        if not hasattr(cv2, "barcode"):
            return {
                "status": "opencv_barcode_unavailable",
                "items": [],
            }

        img_for_detector = _ensure_bgr(img)
        detector = cv2.barcode.BarcodeDetector()
        items: List[Dict[str, Any]] = []

        if hasattr(detector, "detectAndDecodeWithType"):
            decoded_info, decoded_type, points = detector.detectAndDecodeWithType(
                img_for_detector
            )

            if decoded_info:
                if not isinstance(decoded_info, (list, tuple)):
                    decoded_info = [decoded_info]

                if decoded_type is None or not isinstance(decoded_type, (list, tuple)):
                    decoded_type = [decoded_type] * len(decoded_info)

                for idx, text in enumerate(decoded_info):
                    text = (text or "").strip()
                    if not text or not _basic_text_validation(text):
                        continue

                    fmt = decoded_type[idx] if idx < len(decoded_type) else None
                    item: Dict[str, Any] = {
                        "text": text,
                        "format": fmt,
                        "meta": {
                            "backend": "opencv_barcode",
                        },
                    }

                    if points is not None:
                        try:
                            pts = points[idx] if len(points) > idx else points
                            item["meta"]["points"] = np.asarray(pts).tolist()
                        except Exception:
                            pass

                    items.append(item)

                if items:
                    return {
                        "status": "success",
                        "items": items,
                    }

        ok, decoded_info, decoded_type, points = detector.detectAndDecode(
            img_for_detector
        )

        if ok and decoded_info:
            if not isinstance(decoded_info, (list, tuple)):
                decoded_info = [decoded_info]

            if decoded_type is None or not isinstance(decoded_type, (list, tuple)):
                decoded_type = [decoded_type] * len(decoded_info)

            for idx, text in enumerate(decoded_info):
                text = (text or "").strip()
                if not text or not _basic_text_validation(text):
                    continue

                fmt = decoded_type[idx] if idx < len(decoded_type) else None
                item = {
                    "text": text,
                    "format": fmt,
                    "meta": {
                        "backend": "opencv_barcode",
                    },
                }

                if points is not None:
                    try:
                        pts = points[idx] if len(points) > idx else points
                        item["meta"]["points"] = np.asarray(pts).tolist()
                    except Exception:
                        pass

                items.append(item)

            if items:
                return {
                    "status": "success",
                    "items": items,
                }

        ok_detect, points_detect = detector.detect(img_for_detector)
        if ok_detect:
            return {
                "status": "detected_no_decode",
                "items": [],
                "meta": {
                    "backend": "opencv_barcode",
                    "points": np.asarray(points_detect).tolist()
                    if points_detect is not None
                    else None,
                },
            }

        return {
            "status": "no_decode",
            "items": [],
        }

    except Exception as e:
        return {
            "status": "opencv_decoder_error",
            "items": [],
            "error": str(e),
        }


# ------------------------------------------------------------------
# Decoder híbrido
# ------------------------------------------------------------------
def _decode_barcode_hybrid(
    decoder_fn: Callable[..., Dict[str, Any]],
    img: np.ndarray,
    mode: str,
    time_budget_ms: int,
) -> Dict[str, Any]:
    """
    Intenta:
    1) decoder_fn existente
    2) fallback con OpenCV BarcodeDetector
    """
    try:
        result_primary = decoder_fn(
            img,
            mode=mode,
            time_budget_ms=time_budget_ms,
        )
    except Exception as e:
        result_primary = {
            "status": "decoder_error",
            "items": [],
            "error": str(e),
        }

    items_primary = result_primary.get("items", []) or []
    valid_primary = []

    for item in items_primary:
        text = str(item.get("text", "")).strip()
        if _basic_text_validation(text):
            valid_primary.append(item)

    if valid_primary:
        return {
            "status": result_primary.get("status", "success"),
            "items": valid_primary,
            "backend": "primary_decoder",
            "raw_result": result_primary,
        }

    result_fallback = _opencv_barcode_fallback(img)
    items_fallback = result_fallback.get("items", []) or []

    valid_fallback = []
    for item in items_fallback:
        text = str(item.get("text", "")).strip()
        if _basic_text_validation(text):
            valid_fallback.append(item)

    if valid_fallback:
        return {
            "status": result_fallback.get("status", "success"),
            "items": valid_fallback,
            "backend": "opencv_barcode",
            "raw_result": result_fallback,
        }

    return {
        "status": "no_decode",
        "items": [],
        "backend": "none",
        "raw_result": {
            "primary": result_primary,
            "fallback": result_fallback,
        },
    }


# ------------------------------------------------------------------
# Score / deduplicación de candidatos
# ------------------------------------------------------------------
def _score_decoded_item(
    item: Dict[str, Any],
    yolo_conf: float,
    variant_name: str,
    prep_name: str,
    backend: str,
) -> float:
    """
    Score heurístico para priorizar lecturas.
    """
    text = str(item.get("text", "")).strip()
    if not text:
        return -1.0

    score = 0.0
    score += min(len(text), 24) * 0.15
    score += sum(ch.isdigit() for ch in text) * 0.08
    score += float(yolo_conf) * 2.0

    if backend == "primary_decoder":
        score += 1.0
    elif backend == "opencv_barcode":
        score += 0.6

    preferred_preps = {
        "gray",
        "gray_norm",
        "gray_up2",
        "gray_up3",
        "sharp_up2",
        "sharp_up3",
        "refined_up2",
        "refined_sharp_up2",
    }
    if prep_name in preferred_preps:
        score += 0.5

    destructive_preps = {
        "adaptive_up2",
        "refined_otsu_up2",
    }
    if prep_name in destructive_preps:
        score -= 0.15

    if "deskew" in variant_name:
        score += 0.4
    if variant_name in {"orig", "rot90", "rot270"}:
        score += 0.1

    return score


def _dedupe_and_rank_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplica por texto, conservando la mejor evidencia.
    """
    best_by_text: Dict[str, Dict[str, Any]] = {}

    for item in items:
        text = str(item.get("text", "")).strip()
        if not text:
            continue

        prev = best_by_text.get(text)
        if prev is None:
            best_by_text[text] = item
            continue

        prev_score = float(prev.get("_candidate_score", -1.0))
        curr_score = float(item.get("_candidate_score", -1.0))

        if curr_score > prev_score:
            best_by_text[text] = item

    ranked = list(best_by_text.values())
    ranked.sort(
        key=lambda x: (
            float(x.get("_candidate_score", -1.0)),
            float(x.get("yolo_conf", 0.0)),
            len(str(x.get("text", ""))),
        ),
        reverse=True,
    )

    for item in ranked:
        item.pop("_candidate_score", None)

    return ranked


# ------------------------------------------------------------------
# Detección de ROIs
# ------------------------------------------------------------------
def detect_barcode_rois_yolo(
    img_bgr: np.ndarray,
    model_path: str = DEFAULT_MODEL_PATH,
    conf: float = 0.10,
    iou: float = 0.45,
    max_det: int = 10,
    min_size: int = 20,
) -> List[Dict[str, Any]]:
    """
    Detecta ROIs candidatas de códigos de barra usando YOLO.

    Importante:
    - Este método SOLO detecta regiones candidatas.
    - La decodificación ocurre después, en detect_and_decode_with_yolo().
    """
    if img_bgr is None or not isinstance(img_bgr, np.ndarray):
        return []

    if img_bgr.size == 0:
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

    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1, y1, x2, y2 = _clip_bbox_xyxy(x1, y1, x2, y2, img_w, img_h)

            w = x2 - x1
            h = y2 - y1

            if w < min_size or h < min_size:
                continue

            cls = int(box.cls[0]) if box.cls is not None else -1
            conf_score = float(box.conf[0]) if box.conf is not None else 0.0

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


# ------------------------------------------------------------------
# Crop de ROIs
# ------------------------------------------------------------------
def crop_rois(
    img: np.ndarray,
    rois: List[Dict[str, Any]],
    pad_ratio: float = 0.15,
) -> List[Dict[str, Any]]:
    """
    Recorta ROIs con padding y retorna metadata útil.
    """
    if img is None or img.size == 0:
        return []

    img_h, img_w = img.shape[:2]
    out: List[Dict[str, Any]] = []

    for idx, roi in enumerate(rois):
        x, y, ww, hh = roi["bbox"]

        pad_x = int(max(4, round(ww * pad_ratio)))
        pad_y = int(max(4, round(hh * (pad_ratio * 0.75))))

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
                "roi_index": idx,
                "bbox_xyxy_padded": (x0, y0, x1, y1),
                "bbox_xyxy_original": roi["bbox_xyxy"],
                "bbox": roi["bbox"],
                "conf": roi["conf"],
                "cls": roi["cls"],
            }
        )

    return out


# ------------------------------------------------------------------
# Pipeline completo: detectar + decodificar
# ------------------------------------------------------------------
def detect_and_decode_with_yolo(
    img_bgr: np.ndarray,
    decoder_fn: Callable[..., Dict[str, Any]],
    model_path: str = DEFAULT_MODEL_PATH,
    conf: float = 0.25,
    iou: float = 0.45,
    max_det: int = 10,
    min_size: int = 40,
    pad_ratio: float = 0.15,
    decoder_mode: str = "collect",
    decoder_time_budget_ms: int = 800,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Pipeline completo:
    YOLO -> ROI -> variantes geométricas -> variantes suaves de preproceso
    -> decoder existente (con fallback OpenCV) -> ranking
    """
    if img_bgr is None or not isinstance(img_bgr, np.ndarray) or img_bgr.size == 0:
        return {
            "status": "invalid_image",
            "total_rois": 0,
            "items": [],
            "rois": [],
        }

    rois = detect_barcode_rois_yolo(
        img_bgr=img_bgr,
        model_path=model_path,
        conf=conf,
        iou=iou,
        max_det=max_det,
        min_size=min_size,
    )

    if not rois:
        return {
            "status": "no_rois",
            "total_rois": 0,
            "items": [],
            "rois": [],
        }

    crops = crop_rois(
        img=img_bgr,
        rois=rois,
        pad_ratio=pad_ratio,
    )

    candidate_items: List[Dict[str, Any]] = []

    for crop_info in crops:
        crop = crop_info["crop"]
        roi_index = crop_info["roi_index"]

        crop_variants = _build_crop_variants(crop)

        for variant_name, crop_variant in crop_variants:
            _debug_print(
                debug,
                f"[YOLO ROI {roi_index}] probando orientación: {variant_name}",
            )

            preprocessed_list = _preprocess_for_barcode(crop_variant)

            for prep_name, prep_img in preprocessed_list:
                _debug_print(
                    debug,
                    f"[YOLO ROI {roi_index}] orientación={variant_name} preprocess={prep_name}",
                )

                hybrid_result = _decode_barcode_hybrid(
                    decoder_fn=decoder_fn,
                    img=prep_img,
                    mode=decoder_mode,
                    time_budget_ms=decoder_time_budget_ms,
                )

                items = hybrid_result.get("items", []) or []
                if not items:
                    continue

                backend_used = hybrid_result.get("backend", "unknown")

                for decoded_item in items:
                    text = str(decoded_item.get("text", "")).strip()
                    if not _basic_text_validation(text):
                        continue

                    item = dict(decoded_item)
                    item["source"] = f"yolo_roi_{roi_index}"
                    item["yolo_roi_index"] = roi_index
                    item["yolo_bbox_xyxy_original"] = crop_info["bbox_xyxy_original"]
                    item["yolo_bbox_xyxy_padded"] = crop_info["bbox_xyxy_padded"]
                    item["yolo_conf"] = crop_info["conf"]
                    item["yolo_cls"] = crop_info["cls"]
                    item["yolo_crop_variant"] = variant_name
                    item["yolo_preprocess"] = prep_name
                    item["decoder_backend"] = backend_used
                    item["_candidate_score"] = _score_decoded_item(
                        item=item,
                        yolo_conf=float(crop_info["conf"]),
                        variant_name=variant_name,
                        prep_name=prep_name,
                        backend=backend_used,
                    )
                    candidate_items.append(item)

    final_items = _dedupe_and_rank_items(candidate_items)

    return {
        "status": "success" if final_items else "no_decode",
        "total_rois": len(crops),
        "items": final_items,
        "rois": [
            {
                "roi_index": crop_info["roi_index"],
                "bbox_xyxy_original": crop_info["bbox_xyxy_original"],
                "bbox_xyxy_padded": crop_info["bbox_xyxy_padded"],
                "conf": crop_info["conf"],
                "cls": crop_info["cls"],
            }
            for crop_info in crops
        ],
    }


# ------------------------------------------------------------------
# Utilidad opcional para dibujar ROIs
# ------------------------------------------------------------------
def draw_yolo_rois(
    img_bgr: np.ndarray,
    rois: List[Dict[str, Any]],
) -> np.ndarray:
    """
    Dibuja ROIs detectadas sobre la imagen.
    Útil para depuración visual.
    """
    vis = img_bgr.copy()

    for idx, roi in enumerate(rois):
        x1, y1, x2, y2 = roi["bbox_xyxy"]
        conf = roi.get("conf", 0.0)

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis,
            f"roi_{idx} conf={conf:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return vis

def _json_default(obj):
    """
    Serializador seguro para numpy / tuples / Path-like en JSON.
    """
    try:
        import numpy as _np

        if isinstance(obj, (_np.integer,)):
            return int(obj)
        if isinstance(obj, (_np.floating,)):
            return float(obj)
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
    except Exception:
        pass

    if isinstance(obj, tuple):
        return list(obj)

    return str(obj)


def _resolve_decoder_fn() -> Callable[..., Dict[str, Any]] | None:
    """
    Intenta resolver una función de decodificación del proyecto sin acoplarse
    rígidamente a un solo módulo.

    Busca funciones típicas en módulos conocidos del proyecto.
    """
    candidates = [
        ("utils.vision_readout_hybrid", [
            "decode_barcode_hybrid",
            "decode_barcodes_hybrid",
            "decode_with_hybrid",
            "read_barcode_hybrid",
        ]),
        ("utils.vision_barcode_dynamsoft", [
            "decode_barcode_dynamsoft",
            "decode_barcode",
            "decode_barcodes",
            "read_barcodes",
        ]),
    ]

    import importlib

    for module_name, fn_names in candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue

        for fn_name in fn_names:
            fn = getattr(module, fn_name, None)
            if callable(fn):
                print(f"[INFO] Decoder encontrado: {module_name}.{fn_name}")
                return fn

    print("[WARN] No encontré un decoder del proyecto. El CLI seguirá en modo detección-only.")
    return None


def _save_crops_from_rois(
    img_bgr: np.ndarray,
    rois: List[Dict[str, Any]],
    out_dir: Path,
    image_stem: str,
    pad_ratio: float,
) -> List[str]:
    """
    Guarda crops padded de las ROIs detectadas.
    """
    crops_dir = out_dir / f"{image_stem}_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: List[str] = []
    crops = crop_rois(img=img_bgr, rois=rois, pad_ratio=pad_ratio)

    for crop_info in crops:
        roi_index = crop_info["roi_index"]
        conf = float(crop_info.get("conf", 0.0))
        crop = crop_info["crop"]

        crop_path = crops_dir / f"{image_stem}_roi_{roi_index:02d}_conf_{conf:.3f}.jpg"
        ok = cv2.imwrite(str(crop_path), crop)
        if ok:
            saved_paths.append(str(crop_path))
            print(f"[OK] Crop guardado: {crop_path}")
        else:
            print(f"[WARN] No se pudo guardar crop: {crop_path}")

    return saved_paths


def _draw_yolo_rois_with_top_text(
    img_bgr: np.ndarray,
    rois: List[Dict[str, Any]],
    items: List[Dict[str, Any]] | None = None,
) -> np.ndarray:
    """
    Dibuja ROIs y, si existe, el mejor texto asociado por ROI.
    """
    vis = img_bgr.copy()

    top_text_by_roi: Dict[int, str] = {}
    if items:
        for item in items:
            roi_idx = item.get("yolo_roi_index")
            text = str(item.get("text", "")).strip()
            if roi_idx is None or not text:
                continue
            if roi_idx not in top_text_by_roi:
                top_text_by_roi[int(roi_idx)] = text

    for idx, roi in enumerate(rois):
        x1, y1, x2, y2 = roi["bbox_xyxy"]
        conf = float(roi.get("conf", 0.0))

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"roi_{idx} conf={conf:.2f}"
        if idx in top_text_by_roi:
            label += f" | {top_text_by_roi[idx][:32]}"

        cv2.putText(
            vis,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return vis

# ==============================
# CLI DIRECTO
# ==============================

if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    import cv2

    parser = argparse.ArgumentParser(
        description="YOLO Barcode Detection + Decode"
    )

    parser.add_argument("image", type=str, help="Ruta de la imagen")

    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/test_vision_barcode_yolo",
        help="Carpeta de salida",
    )

    parser.add_argument(
        "--save-vis",
        action="store_true",
        help="Guardar imagen anotada con ROIs",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Guardar JSON resultado",
    )
    parser.add_argument(
        "--save-crops",
        action="store_true",
        help="Guardar crops padded de las ROIs detectadas",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Imprimir detalle de variantes geométricas y preprocesos",
    )

    parser.add_argument("--conf", type=float, default=0.10, help="Confianza YOLO")
    parser.add_argument("--iou", type=float, default=0.45, help="IOU YOLO")
    parser.add_argument("--max-det", type=int, default=10, help="Máximo de detecciones")
    parser.add_argument("--min-size", type=int, default=20, help="Tamaño mínimo de ROI")
    parser.add_argument(
        "--pad_ratio",
        type=float,
        default=0.80,
        help="Padding relativo aplicado al crop de la ROI",
    )
    parser.add_argument(
        "--decoder_mode",
        type=str,
        default="collect_plus",
        help="Modo del decoder",
    )
    parser.add_argument(
        "--decoder_budget",
        type=int,
        default=5000,
        help="Presupuesto de tiempo del decoder en ms",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Ruta del modelo YOLO",
    )
    parser.add_argument(
        "--detect-only",
        action="store_true",
        help="Sólo detectar ROIs, sin intentar decode",
    )

    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[ERROR] Imagen no existe: {image_path}")
        raise SystemExit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] ===========================================")
    print("[INFO] vision_barcode_yolo | inicio")
    print(f"[INFO] Imagen         : {image_path}")
    print(f"[INFO] Modelo YOLO    : {args.model_path}")
    print(f"[INFO] Output dir     : {out_dir}")
    print(f"[INFO] conf/iou       : {args.conf} / {args.iou}")
    print(f"[INFO] max_det        : {args.max_det}")
    print(f"[INFO] min_size       : {args.min_size}")
    print(f"[INFO] pad_ratio      : {args.pad_ratio}")
    print(f"[INFO] decoder_mode   : {args.decoder_mode}")
    print(f"[INFO] decoder_budget : {args.decoder_budget} ms")
    print(f"[INFO] debug          : {args.debug}")
    print("[INFO] ===========================================")

    print("[INFO] Cargando imagen...")
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[ERROR] No se pudo leer imagen: {image_path}")
        raise SystemExit(1)

    print(f"[INFO] Imagen cargada: shape={img.shape}")

    print("[INFO] Ejecutando detección YOLO...")
    rois = detect_barcode_rois_yolo(
        img_bgr=img,
        model_path=args.model_path,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        min_size=args.min_size,
    )

    print(f"[INFO] Total ROIs detectadas: {len(rois)}")

    if args.save_crops and rois:
        print("[INFO] Guardando crops de ROIs...")
        _save_crops_from_rois(
            img_bgr=img,
            rois=rois,
            out_dir=out_dir,
            image_stem=image_path.stem,
            pad_ratio=args.pad_ratio,
        )

    final_result: Dict[str, Any]

    if args.detect_only:
        print("[INFO] Modo detect-only activado.")
        final_result = {
            "status": "success" if rois else "no_rois",
            "image": str(image_path),
            "total_rois": len(rois),
            "rois": rois,
            "items": [],
        }
    else:
        decoder_fn = _resolve_decoder_fn()

        if decoder_fn is None:
            final_result = {
                "status": "success" if rois else "no_rois",
                "image": str(image_path),
                "total_rois": len(rois),
                "rois": rois,
                "items": [],
                "warning": "No se encontró decoder_fn. Se ejecutó sólo detección.",
            }
        else:
            print("[INFO] Ejecutando pipeline completo: detect + decode...")
            final_result = detect_and_decode_with_yolo(
                img_bgr=img,
                decoder_fn=decoder_fn,
                model_path=args.model_path,
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                min_size=args.min_size,
                pad_ratio=args.pad_ratio,
                decoder_mode=args.decoder_mode,
                decoder_time_budget_ms=args.decoder_budget,
                debug=args.debug,
            )
            final_result["image"] = str(image_path)

    print("\n=== RESULTADO FINAL ===")
    print(json.dumps(final_result, indent=2, ensure_ascii=False, default=_json_default))

    if args.save_vis:
        print("[INFO] Guardando visualización...")
        vis_items = final_result.get("items", []) or []
        vis = _draw_yolo_rois_with_top_text(img, rois, vis_items)
        vis_path = out_dir / f"{image_path.stem}_yolo_decode_vis.jpg"
        ok = cv2.imwrite(str(vis_path), vis)
        if ok:
            print(f"[OK] Imagen guardada: {vis_path}")
        else:
            print(f"[WARN] No se pudo guardar imagen: {vis_path}")

    if args.save_json:
        print("[INFO] Guardando JSON...")
        json_path = out_dir / f"{image_path.stem}_yolo_decode_result.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(final_result, f, indent=2, ensure_ascii=False, default=_json_default)
        print(f"[OK] JSON guardado: {json_path}")

    print("[INFO] Proceso finalizado.")