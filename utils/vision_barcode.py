# utils/vision_barcode.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
import time

import numpy as np
import cv2

from utils.vision_preprocess import preprocess_variants


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
            data = str(d.data)
        out.append(
            {
                "text": data,
                "type": getattr(d, "type", None),
                "rect": getattr(d, "rect", None),
            }
        )
    return out


def _try_zxingcpp(gray: np.ndarray) -> List[Dict[str, Any]]:
    try:
        import zxingcpp  # type: ignore
    except Exception:
        return []

    # zxingcpp expects numpy uint8 gray
    res = zxingcpp.read_barcodes(gray)
    out: List[Dict[str, Any]] = []
    for r in res:
        out.append(
            {
                "text": r.text,
                "format": str(r.format),
                "position": (r.position if hasattr(r, "position") else None),
            }
        )
    return out


def decode_barcode_1d(
    img_bgr: np.ndarray,
    *,
    time_budget_ms: int = 180,
    prefer: str = "zxingcpp",  # or "pyzbar"
    variants: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Returns:
      {
        "status": "success" | "not_found" | "not_available" | "error",
        "items": [ {text,type/format,...}, ... ],
        "backend": str | None,
        "elapsed_ms": int
      }
    """
    t0 = time.perf_counter()
    variants = variants or ["gray", "sharp", "bw"]  # barcode may benefit from bw if enabled

    ims = preprocess_variants(img_bgr, cfg=None)  # cfg default; no bw unless you enable cfg.binarize elsewhere
    # If bw is missing, it's fine; loop will skip.

    backends = [prefer, "pyzbar" if prefer != "pyzbar" else "zxingcpp"]

    try:
        for name in variants:
            if (time.perf_counter() - t0) * 1000 > time_budget_ms:
                break
            gray = ims.get(name)
            if gray is None:
                continue

            for be in backends:
                if (time.perf_counter() - t0) * 1000 > time_budget_ms:
                    break
                if be == "pyzbar":
                    items = _try_pyzbar(gray)
                    backend_used = "pyzbar"
                else:
                    items = _try_zxingcpp(gray)
                    backend_used = "zxingcpp"

                if items:
                    elapsed = int((time.perf_counter() - t0) * 1000)
                    return {
                        "status": "success",
                        "items": items,
                        "backend": backend_used,
                        "elapsed_ms": elapsed,
                        "variant": name,
                    }

        # If no backend is installed at all:
        if not _try_zxingcpp(ims["gray"]) and not _try_pyzbar(ims["gray"]):
            elapsed = int((time.perf_counter() - t0) * 1000)
            return {
                "status": "not_available",
                "items": [],
                "backend": None,
                "elapsed_ms": elapsed,
                "variant": None,
            }

        elapsed = int((time.perf_counter() - t0) * 1000)
        return {
            "status": "not_found",
            "items": [],
            "backend": None,
            "elapsed_ms": elapsed,
            "variant": None,
        }
    except Exception as e:
        elapsed = int((time.perf_counter() - t0) * 1000)
        return {
            "status": "error",
            "items": [],
            "backend": None,
            "elapsed_ms": elapsed,
            "error": repr(e),
            "variant": None,
        }
        
if __name__ == "__main__":
    import sys
    import cv2
    if len(sys.argv) < 2:
        print("Usage: python vision_barcode.py path/to/image.jpg")
        sys.exit(1)

    img = cv2.imread(sys.argv[1])
    if img is None:
        print("Could not load image")
        sys.exit(1)

    result = decode_barcode_1d(img)
    print(result)