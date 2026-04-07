# utils/vision_barcode_dynamsoft.py

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from dynamsoft_barcode_reader_bundle import (
    CaptureVisionRouter,
    EnumErrorCode,
    EnumPresetTemplate,
    LicenseManager,
)

# Estado global simple para no reinicializar licencia innecesariamente
_LICENSE_READY = False
_LICENSE_SOURCE = None


def load_env_file(env_path: str = ".env") -> None:
    """
    Carga variables desde un archivo .env simple, sin dependencias externas.

    Soporta líneas tipo:
        KEY=VALUE

    Ignora:
    - líneas vacías
    - comentarios con #
    """
    env_file = Path(env_path)
    if not env_file.exists():
        return

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        os.environ.setdefault(key, value)


def init_dynamsoft_license(env_file: str = ".env", force: bool = False) -> None:
    """
    Inicializa la licencia de Dynamsoft desde:
    1) variable de entorno LICENSE_KEY_DYNAMSOFT
    2) archivo .env

    Lanza excepción si falla.
    """
    global _LICENSE_READY, _LICENSE_SOURCE

    if _LICENSE_READY and not force:
        return

    load_env_file(env_file)

    license_key = os.getenv("LICENSE_KEY_DYNAMSOFT")
    if not license_key:
        raise RuntimeError(
            "No se encontró LICENSE_KEY_DYNAMSOFT en el entorno ni en el archivo .env"
        )

    error_code, error_msg = LicenseManager.init_license(license_key)

    if error_code not in (EnumErrorCode.EC_OK, EnumErrorCode.EC_LICENSE_CACHE_USED):
        raise RuntimeError(
            f"Fallo al inicializar licencia Dynamsoft: {error_code} - {error_msg}"
        )

    _LICENSE_READY = True
    _LICENSE_SOURCE = env_file


def _normalize_points(location: Any) -> Optional[List[Tuple[float, float]]]:
    """
    Intenta extraer una lista de puntos (x, y) desde el objeto location
    retornado por Dynamsoft.

    Este bloque es tolerante porque la estructura concreta puede variar
    entre versiones.
    """
    if location is None:
        return None

    # Caso 1: iterable de puntos/tuplas
    try:
        pts = []
        for p in location:
            x = None
            y = None

            if hasattr(p, "x") and hasattr(p, "y"):
                x = float(p.x)
                y = float(p.y)
            elif isinstance(p, (tuple, list)) and len(p) >= 2:
                x = float(p[0])
                y = float(p[1])

            if x is not None and y is not None:
                pts.append((x, y))

        if pts:
            return pts
    except Exception:
        pass

    # Caso 2: objeto con points
    try:
        pts_obj = getattr(location, "points", None)
        if pts_obj is not None:
            pts = []
            for p in pts_obj:
                if hasattr(p, "x") and hasattr(p, "y"):
                    pts.append((float(p.x), float(p.y)))
                elif isinstance(p, (tuple, list)) and len(p) >= 2:
                    pts.append((float(p[0]), float(p[1])))
            if pts:
                return pts
    except Exception:
        pass

    return None


def _points_to_bbox(points: Optional[Sequence[Tuple[float, float]]]) -> Optional[Tuple[int, int, int, int]]:
    """
    Convierte lista de puntos en bbox (x1, y1, x2, y2).
    """
    if not points:
        return None

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    return (
        int(round(min(xs))),
        int(round(min(ys))),
        int(round(max(xs))),
        int(round(max(ys))),
    )


def _extract_item_data(item: Any) -> Dict[str, Any]:
    """
    Extrae información estandarizada de un resultado individual de Dynamsoft.
    """
    text = None
    fmt = None
    points = None

    try:
        text = item.get_text()
    except Exception:
        text = None

    try:
        fmt = item.get_format_string()
    except Exception:
        fmt = None

    try:
        location = item.get_location()
        points = _normalize_points(location)
    except Exception:
        points = None

    bbox = _points_to_bbox(points)

    out: Dict[str, Any] = {
        "text": text,
        "format": fmt,
        "backend": "dynamsoft",
    }

    if points:
        out["position"] = points

    if bbox:
        out["bbox"] = bbox

    return out

def _bbox_iou(
    bbox1: Optional[Tuple[int, int, int, int]],
    bbox2: Optional[Tuple[int, int, int, int]],
) -> float:
    """
    Calcula IoU entre dos bboxes xyxy.
    """
    if not bbox1 or not bbox2:
        return 0.0

    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0, bbox1[2] - bbox1[0]) * max(0, bbox1[3] - bbox1[1])
    area2 = max(0, bbox2[2] - bbox2[0]) * max(0, bbox2[3] - bbox2[1])

    union = area1 + area2 - inter_area
    if union <= 0:
        return 0.0

    return inter_area / union


def _bbox_center_distance(
    bbox1: Optional[Tuple[int, int, int, int]],
    bbox2: Optional[Tuple[int, int, int, int]],
) -> float:
    """
    Distancia euclidiana entre centros de dos bboxes xyxy.
    """
    if not bbox1 or not bbox2:
        return float("inf")

    cx1 = (bbox1[0] + bbox1[2]) / 2.0
    cy1 = (bbox1[1] + bbox1[3]) / 2.0
    cx2 = (bbox2[0] + bbox2[2]) / 2.0
    cy2 = (bbox2[1] + bbox2[3]) / 2.0

    return float(((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5)


def _dedupe_items_spatial(
    items: List[Dict[str, Any]],
    iou_threshold: float = 0.5,
    center_distance_threshold: float = 35.0,
) -> List[Dict[str, Any]]:
    """
    Deduplica sólo duplicados espaciales probables:
    - mismo text
    - mismo format
    - bbox muy solapada o muy cercana

    Conserva ambos si el texto es igual pero están en posiciones distintas.
    """
    kept: List[Dict[str, Any]] = []

    for item in items:
        text = item.get("text")
        fmt = item.get("format")
        bbox = item.get("bbox")

        is_duplicate = False

        for prev in kept:
            same_text = prev.get("text") == text
            same_format = prev.get("format") == fmt

            if not (same_text and same_format):
                continue

            prev_bbox = prev.get("bbox")
            iou = _bbox_iou(bbox, prev_bbox)
            dist = _bbox_center_distance(bbox, prev_bbox)

            if iou >= iou_threshold or dist <= center_distance_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(item)

    return kept

# def _dedupe_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#     """
#     Elimina duplicados obvios por (text, format).
#     Conserva el primer resultado.
#     """
#     seen = set()
#     deduped = []

#     for item in items:
#         key = (item.get("text"), item.get("format"))
#         if key in seen:
#             continue
#         seen.add(key)
#         deduped.append(item)

#     return deduped


def _capture_from_path(image_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Ejecuta lectura Dynamsoft a partir de una ruta de imagen.
    """
    img_path = Path(image_path)

    if not img_path.exists():
        raise FileNotFoundError(f"No existe la imagen: {img_path}")

    cvr = CaptureVisionRouter()
    result = cvr.capture(
        str(img_path),
        EnumPresetTemplate.PT_READ_BARCODES.value,
    )

    if result.get_error_code() != EnumErrorCode.EC_OK:
        return {
            "status": "error",
            "image_path": str(img_path),
            "backend": "dynamsoft",
            "error_code": int(result.get_error_code()),
            "error_message": result.get_error_string(),
            "items": [],
            "total": 0,
        }

    barcode_result = result.get_decoded_barcodes_result()

    if barcode_result is None or barcode_result.get_items() == 0:
        return {
            "status": "not_found",
            "image_path": str(img_path),
            "backend": "dynamsoft",
            "items": [],
            "total": 0,
        }

    raw_items = barcode_result.get_items()
    items = [_extract_item_data(item) for item in raw_items]
    items = [item for item in items if item.get("text")]
    items = _dedupe_items_spatial(items)    

    if not items:
        return {
            "status": "not_found",
            "image_path": str(img_path),
            "backend": "dynamsoft",
            "items": [],
            "total": 0,
        }

    return {
        "status": "success",
        "image_path": str(img_path),
        "backend": "dynamsoft",
        "items": items,
        "total": len(items),
    }


def decode_barcode_dynamsoft_from_path(
    image_path: Union[str, Path],
    env_file: str = ".env",
) -> Dict[str, Any]:
    """
    API principal por ruta de archivo.
    """
    init_dynamsoft_license(env_file=env_file)
    return _capture_from_path(image_path)


def decode_barcode_dynamsoft(
    image: np.ndarray,
    env_file: str = ".env",
    temp_suffix: str = ".png",
) -> Dict[str, Any]:
    """
    API principal por imagen en memoria (np.ndarray).

    Guarda temporalmente a disco y luego usa el motor oficial
    sobre esa ruta temporal.
    """
    if image is None:
        raise ValueError("La imagen recibida es None")

    if not isinstance(image, np.ndarray):
        raise TypeError("Se esperaba un np.ndarray")

    if image.size == 0:
        raise ValueError("La imagen está vacía")

    init_dynamsoft_license(env_file=env_file)

    with tempfile.NamedTemporaryFile(suffix=temp_suffix, delete=False) as tmp:
        tmp_path = tmp.name

    try:
        ok = cv2.imwrite(tmp_path, image)
        if not ok:
            raise RuntimeError("No se pudo escribir la imagen temporal para Dynamsoft")

        result = _capture_from_path(tmp_path)
        result["source"] = "ndarray"
        return result

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


def annotate_detections(
    image: np.ndarray,
    result: Dict[str, Any],
) -> np.ndarray:
    """
    Dibuja bbox y texto sobre la imagen.

    Útil para debug/validación visual.
    """
    if image is None:
        raise ValueError("La imagen recibida es None")

    vis = image.copy()
    items = result.get("items", [])

    for idx, item in enumerate(items, start=1):
        bbox = item.get("bbox")
        text = item.get("text", "")
        fmt = item.get("format", "")

        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"{idx}: {fmt} | {text}"
            cv2.putText(
                vis,
                label[:120],
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        else:
            # Si no hay bbox, al menos dejamos texto arriba
            cv2.putText(
                vis,
                f"{idx}: {fmt} | {text}"[:120],
                (20, 30 + (idx - 1) * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    return vis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backend de detección y decodificación de códigos con Dynamsoft."
    )
    parser.add_argument(
        "image_path",
        help="Ruta de la imagen a procesar",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Ruta al archivo .env (default: .env)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Imprime salida completa en JSON",
    )
    parser.add_argument(
        "--save-vis",
        action="store_true",
        help="Guarda imagen anotada con detecciones",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Ruta de salida para imagen anotada. Si no se indica, se genera automáticamente.",
    )
    
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Guarda resultado JSON en archivo",
    )
    
    parser.add_argument(
        "--json-out",
        default=None,
        help="Ruta de salida para JSON",
    )
    return parser


def _default_vis_output_path(image_path: Union[str, Path]) -> str:
    p = Path(image_path)
    return str(p.with_name(f"{p.stem}_dynamsoft_vis{p.suffix}"))


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        result = decode_barcode_dynamsoft_from_path(
            image_path=args.image_path,
            env_file=args.env_file,
        )
        
        if args.save_json:
            json_out_path = args.json_out
            if not json_out_path:
                p = Path(args.image_path)
                json_out_path = str(p.with_name(f"{p.stem}_dynamsoft.json"))
            
            Path(json_out_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(json_out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
                
            print(f"\nJSON guardado en: {json_out_path}")

        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print("\n=== RESULTADO DYNAMSOFT ===\n")
            print(f"Status: {result['status']}")
            print(f"Backend: {result.get('backend', 'dynamsoft')}")
            print(f"Imagen: {result.get('image_path')}")
            print(f"Total detectados: {result.get('total', 0)}")

            if result["status"] == "success":
                for idx, item in enumerate(result["items"], start=1):
                    print(f"\nResultado {idx}")
                    print(f"Formato: {item.get('format')}")
                    print(f"Texto:   {item.get('text')}")
                    if item.get("bbox"):
                        print(f"BBox:    {item.get('bbox')}")
            elif result["status"] == "error":
                print(f"Error code: {result.get('error_code')}")
                print(f"Error msg:  {result.get('error_message')}")

        if args.save_vis:
            img = cv2.imread(args.image_path)
            if img is None:
                raise RuntimeError("No se pudo cargar la imagen para generar visualización")

            vis = annotate_detections(img, result)
            out_path = args.out or _default_vis_output_path(args.image_path)

            ok = cv2.imwrite(out_path, vis)
            if not ok:
                raise RuntimeError(f"No se pudo guardar la visualización en: {out_path}")

            print(f"\nVisualización guardada en: {out_path}")

        return 0

    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())