# utils/vision_barcode_yolo_v2.py
from __future__ import annotations

from typing import Any, Dict, List

import cv2
import numpy as np
from ultralytics import YOLO


# ------------------------------------------------------------------
# Modelo YOLO v2
# ------------------------------------------------------------------
DEFAULT_MODEL_PATH = "models/barcode_kn_v2.pt"

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
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w))
    y2 = max(0, min(y2, img_h))
    return x1, y1, x2, y2


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


# ------------------------------------------------------------------
# Geometría / utilidades de dedupe espacial
# ------------------------------------------------------------------
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


def _dedupe_rois(
    rois: List[Dict[str, Any]],
    iou_thr: float = 0.65,
    center_dist_thr: float = 30.0,
) -> List[Dict[str, Any]]:
    """
    Deduplica ROIs conservando la de mayor score.
    """
    kept: List[Dict[str, Any]] = []

    for roi in rois:
        bbox = list(roi["bbox_xyxy"])
        duplicate_idx = None

        for idx, prev in enumerate(kept):
            prev_bbox = list(prev["bbox_xyxy"])
            iou = _bbox_iou(bbox, prev_bbox)
            dist = _bbox_center_distance(bbox, prev_bbox)

            if iou >= iou_thr or dist <= center_dist_thr:
                duplicate_idx = idx
                break

        if duplicate_idx is None:
            kept.append(roi)
        else:
            prev = kept[duplicate_idx]
            if float(roi.get("priority_score", -1.0)) > float(prev.get("priority_score", -1.0)):
                kept[duplicate_idx] = roi

    kept.sort(
        key=lambda x: (
            float(x.get("priority_score", 0.0)),
            float(x.get("conf", 0.0)),
            float(x.get("area", 0.0)),
        ),
        reverse=True,
    )
    return kept


# ------------------------------------------------------------------
# Clases y prioridad
# ------------------------------------------------------------------
def _get_class_names(model: YOLO) -> Dict[int, str]:
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {idx: str(name) for idx, name in enumerate(names)}
    return {}


def _normalize_class_name(name: str) -> str:
    return str(name or "").strip().lower()


def _class_priority(class_name: str) -> int:
    """
    Prioridad deseada:
    1) barcode
    2) label
    3) box
    """
    cname = _normalize_class_name(class_name)

    if cname == "barcode":
        return 300
    if cname == "label":
        return 200
    if cname == "box":
        return 100
    return 10


def _class_pad_ratio(class_name: str, default_pad_ratio: float) -> float:
    """
    Padding específico por clase.
    """
    cname = _normalize_class_name(class_name)

    if cname == "barcode":
        return max(0.10, default_pad_ratio)
    if cname == "label":
        return max(0.18, default_pad_ratio)
    if cname == "box":
        return max(0.25, default_pad_ratio)
    return default_pad_ratio


def _is_allowed_candidate_class(class_name: str) -> bool:
    cname = _normalize_class_name(class_name)
    return cname in {"barcode", "label", "box"}


# ------------------------------------------------------------------
# Resumen
# ------------------------------------------------------------------
def _build_summary_by_class(rois: List[Dict[str, Any]]) -> Dict[str, int]:
    summary = {
        "barcode": 0,
        "label": 0,
        "box": 0,
        "other": 0,
    }

    for roi in rois:
        cname = _normalize_class_name(roi.get("class_name", ""))
        if cname in summary:
            summary[cname] += 1
        else:
            summary["other"] += 1

    return summary


# ------------------------------------------------------------------
# Detección de ROIs
# ------------------------------------------------------------------
def detect_candidate_rois_yolo_v2(
    img_bgr: np.ndarray,
    model_path: str = DEFAULT_MODEL_PATH,
    conf: float = 0.25,
    iou: float = 0.45,
    max_det: int = 20,
    min_size: int = 20,
    allowed_classes: List[str] | None = None,
) -> List[Dict[str, Any]]:
    """
    Detecta ROIs candidatas usando el modelo v2 y prioriza:
    barcode > label > box
    """
    if img_bgr is None or not isinstance(img_bgr, np.ndarray):
        return []

    if img_bgr.size == 0:
        return []

    model = load_yolo_model(model_path)
    class_names = _get_class_names(model)

    if allowed_classes is None:
        allowed_classes = ["barcode", "label", "box"]

    allowed_norm = {_normalize_class_name(x) for x in allowed_classes}

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

            class_name = _normalize_class_name(class_names.get(cls, f"class_{cls}"))
            if class_name not in allowed_norm:
                continue

            if not _is_allowed_candidate_class(class_name):
                continue

            priority = _class_priority(class_name)
            priority_score = priority + (conf_score * 10.0)

            rois.append(
                {
                    "bbox_xyxy": (x1, y1, x2, y2),
                    "bbox": (x1, y1, w, h),
                    "conf": conf_score,
                    "cls": cls,
                    "class_name": class_name,
                    "area": w * h,
                    "priority": priority,
                    "priority_score": priority_score,
                }
            )

    rois = _dedupe_rois(rois)
    return rois


def detect_barcode_rois_yolo(
    img_bgr: np.ndarray,
    model_path: str = DEFAULT_MODEL_PATH,
    conf: float = 0.25,
    iou: float = 0.45,
    max_det: int = 20,
    min_size: int = 20,
) -> List[Dict[str, Any]]:
    """
    Compatibilidad con el módulo anterior.
    Devuelve ROIs candidatas priorizadas.
    """
    return detect_candidate_rois_yolo_v2(
        img_bgr=img_bgr,
        model_path=model_path,
        conf=conf,
        iou=iou,
        max_det=max_det,
        min_size=min_size,
        allowed_classes=["barcode", "label", "box"],
    )


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
    El padding se adapta por clase detectada.
    """
    if img is None or img.size == 0:
        return []

    img_h, img_w = img.shape[:2]
    out: List[Dict[str, Any]] = []

    for idx, roi in enumerate(rois):
        x, y, ww, hh = roi["bbox"]
        class_name = _normalize_class_name(roi.get("class_name", ""))
        effective_pad_ratio = _class_pad_ratio(class_name, pad_ratio)

        pad_x = int(max(4, round(ww * effective_pad_ratio)))
        pad_y = int(max(4, round(hh * (effective_pad_ratio * 0.75))))

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
                "class_name": class_name,
                "priority": roi.get("priority"),
                "priority_score": roi.get("priority_score"),
            }
        )

    return out


# ------------------------------------------------------------------
# Pipeline completo: sólo detección
# ------------------------------------------------------------------
def detect_with_yolo_v2(
    img_bgr: np.ndarray,
    model_path: str = DEFAULT_MODEL_PATH,
    conf: float = 0.25,
    iou: float = 0.45,
    max_det: int = 20,
    min_size: int = 20,
    pad_ratio: float = 0.15,
) -> Dict[str, Any]:
    """
    Pipeline v2:
    YOLO v2 -> ROIs priorizadas (barcode > label > box) -> crops
    No intenta decodificar.
    """
    if img_bgr is None or not isinstance(img_bgr, np.ndarray) or img_bgr.size == 0:
        return {
            "status": "invalid_image",
            "total_rois": 0,
            "summary_by_class": _build_summary_by_class([]),
            "items": [],
            "rois": [],
        }

    rois = detect_candidate_rois_yolo_v2(
        img_bgr=img_bgr,
        model_path=model_path,
        conf=conf,
        iou=iou,
        max_det=max_det,
        min_size=min_size,
        allowed_classes=["barcode", "label", "box"],
    )

    if not rois:
        return {
            "status": "no_rois",
            "total_rois": 0,
            "summary_by_class": _build_summary_by_class([]),
            "items": [],
            "rois": [],
        }

    crops = crop_rois(
        img=img_bgr,
        rois=rois,
        pad_ratio=pad_ratio,
    )

    return {
        "status": "success",
        "total_rois": len(crops),
        "summary_by_class": _build_summary_by_class(rois),
        "items": [],
        "rois": [
            {
                "roi_index": crop_info["roi_index"],
                "bbox_xyxy_original": crop_info["bbox_xyxy_original"],
                "bbox_xyxy_padded": crop_info["bbox_xyxy_padded"],
                "conf": crop_info["conf"],
                "cls": crop_info["cls"],
                "class_name": crop_info.get("class_name"),
                "priority": crop_info.get("priority"),
                "priority_score": crop_info.get("priority_score"),
            }
            for crop_info in crops
        ],
    }


def detect_with_yolo(
    img_bgr: np.ndarray,
    model_path: str = DEFAULT_MODEL_PATH,
    conf: float = 0.25,
    iou: float = 0.45,
    max_det: int = 20,
    min_size: int = 20,
    pad_ratio: float = 0.15,
) -> Dict[str, Any]:
    """
    Alias de compatibilidad para detección-only.
    """
    return detect_with_yolo_v2(
        img_bgr=img_bgr,
        model_path=model_path,
        conf=conf,
        iou=iou,
        max_det=max_det,
        min_size=min_size,
        pad_ratio=pad_ratio,
    )


# ------------------------------------------------------------------
# Visualización
# ------------------------------------------------------------------
def draw_yolo_rois(
    img_bgr: np.ndarray,
    rois: List[Dict[str, Any]],
) -> np.ndarray:
    vis = img_bgr.copy()

    for idx, roi in enumerate(rois):
        x1, y1, x2, y2 = roi["bbox_xyxy"]
        conf = float(roi.get("conf", 0.0))
        class_name = roi.get("class_name", "unknown")

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis,
            f"roi_{idx} {class_name} conf={conf:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return vis


def _save_crops_from_rois(
    img_bgr: np.ndarray,
    rois: List[Dict[str, Any]],
    out_dir: str | Any,
    image_stem: str,
    pad_ratio: float,
) -> List[str]:
    from pathlib import Path

    crops_dir = Path(out_dir) / f"{image_stem}_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: List[str] = []
    crops = crop_rois(img=img_bgr, rois=rois, pad_ratio=pad_ratio)

    for crop_info in crops:
        roi_index = crop_info["roi_index"]
        conf = float(crop_info.get("conf", 0.0))
        class_name = str(crop_info.get("class_name", "unknown"))
        crop = crop_info["crop"]

        crop_path = crops_dir / f"{image_stem}_roi_{roi_index:02d}_{class_name}_conf_{conf:.3f}.jpg"
        ok = cv2.imwrite(str(crop_path), crop)
        if ok:
            saved_paths.append(str(crop_path))
            print(f"[OK] Crop guardado: {crop_path}")
        else:
            print(f"[WARN] No se pudo guardar crop: {crop_path}")

    return saved_paths


# ==============================
# CLI DIRECTO
# ==============================
if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="YOLO v2 Detection only (barcode > label > box)"
    )

    parser.add_argument("image", type=str, help="Ruta de la imagen")

    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/test_vision_barcode_yolo_v2",
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

    parser.add_argument("--conf", type=float, default=0.25, help="Confianza YOLO")
    parser.add_argument("--iou", type=float, default=0.45, help="IOU YOLO")
    parser.add_argument("--max-det", type=int, default=20, help="Máximo de detecciones")
    parser.add_argument("--min-size", type=int, default=20, help="Tamaño mínimo de ROI")
    parser.add_argument(
        "--pad-ratio",
        type=float,
        default=0.15,
        help="Padding relativo base aplicado al crop de la ROI",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Ruta del modelo YOLO v2",
    )

    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[ERROR] Imagen no existe: {image_path}")
        raise SystemExit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] ===========================================")
    print("[INFO] vision_barcode_yolo_v2 | detección-only")
    print(f"[INFO] Imagen         : {image_path}")
    print(f"[INFO] Modelo YOLO    : {args.model_path}")
    print(f"[INFO] Output dir     : {out_dir}")
    print(f"[INFO] conf/iou       : {args.conf} / {args.iou}")
    print(f"[INFO] max_det        : {args.max_det}")
    print(f"[INFO] min_size       : {args.min_size}")
    print(f"[INFO] pad_ratio      : {args.pad_ratio}")
    print("[INFO] ===========================================")

    print("[INFO] Cargando imagen...")
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[ERROR] No se pudo leer imagen: {image_path}")
        raise SystemExit(1)

    print(f"[INFO] Imagen cargada: shape={img.shape}")

    print("[INFO] Ejecutando detección YOLO v2...")
    rois = detect_candidate_rois_yolo_v2(
        img_bgr=img,
        model_path=args.model_path,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        min_size=args.min_size,
        allowed_classes=["barcode", "label", "box"],
    )

    print(f"[INFO] Total ROIs detectadas: {len(rois)}")
    print(f"[INFO] Resumen por clase: {_build_summary_by_class(rois)}")

    if args.save_crops and rois:
        print("[INFO] Guardando crops de ROIs...")
        _save_crops_from_rois(
            img_bgr=img,
            rois=rois,
            out_dir=out_dir,
            image_stem=image_path.stem,
            pad_ratio=args.pad_ratio,
        )

    final_result = detect_with_yolo_v2(
        img_bgr=img,
        model_path=args.model_path,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        min_size=args.min_size,
        pad_ratio=args.pad_ratio,
    )
    final_result["image"] = str(image_path)

    print("\n=== RESULTADO FINAL ===")
    print(json.dumps(final_result, indent=2, ensure_ascii=False, default=_json_default))

    if args.save_vis:
        print("[INFO] Guardando visualización...")
        vis = draw_yolo_rois(img, rois)
        vis_path = out_dir / f"{image_path.stem}_yolo_v2_detect_vis.jpg"
        ok = cv2.imwrite(str(vis_path), vis)
        if ok:
            print(f"[OK] Imagen guardada: {vis_path}")
        else:
            print(f"[WARN] No se pudo guardar imagen: {vis_path}")

    if args.save_json:
        print("[INFO] Guardando JSON...")
        json_path = out_dir / f"{image_path.stem}_yolo_v2_detect_result.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(final_result, f, indent=2, ensure_ascii=False, default=_json_default)
        print(f"[OK] JSON guardado: {json_path}")

    print("[INFO] Proceso finalizado.")