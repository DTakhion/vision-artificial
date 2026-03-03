# utils/vision_ocr.py
from __future__ import annotations

from typing import Any, Dict, Optional, List
import time

import cv2
import numpy as np

from utils.vision_preprocess import preprocess_variants, PreprocessConfig


def ocr_serial_best_effort(
    img_bgr: np.ndarray,
    *,
    time_budget_ms: int = 350,
    mode: str = "tesseract",  # "tesseract" | "easyocr"
    aggressive: bool = False,
) -> Dict[str, Any]:
    """
    Intended for SERIAL text (not QR/1D).
    Returns:
      { "status": "success"|"not_found"|"not_available"|"error",
        "text": str|None,
        "backend": str|None,
        "elapsed_ms": int }
    """
    t0 = time.perf_counter()

    # OCR often benefits from binarize in aggressive mode
    cfg = PreprocessConfig(
        resize_max_side=1600 if aggressive else 1280,
        clahe=True,
        denoise=aggressive,
        sharpen=True,
        binarize=aggressive,
    )
    ims = preprocess_variants(img_bgr, cfg=cfg)

    try:
        if mode == "easyocr":
            try:
                import easyocr  # type: ignore
            except Exception:
                elapsed = int((time.perf_counter() - t0) * 1000)
                return {"status": "not_available", "text": None, "backend": None, "elapsed_ms": elapsed}

            reader = easyocr.Reader(["en", "es"], gpu=False)
            # prefer bw if exists
            gray = ims.get("bw") or ims.get("sharp") or ims["gray"]
            if (time.perf_counter() - t0) * 1000 > time_budget_ms:
                elapsed = int((time.perf_counter() - t0) * 1000)
                return {"status": "not_found", "text": None, "backend": "easyocr", "elapsed_ms": elapsed}

            res = reader.readtext(gray, detail=0, paragraph=True)
            txt = " ".join(res).strip() if res else ""
            elapsed = int((time.perf_counter() - t0) * 1000)
            if txt:
                return {"status": "success", "text": txt, "backend": "easyocr", "elapsed_ms": elapsed}
            return {"status": "not_found", "text": None, "backend": "easyocr", "elapsed_ms": elapsed}

        # default: tesseract
        try:
            import pytesseract  # type: ignore
        except Exception:
            elapsed = int((time.perf_counter() - t0) * 1000)
            return {"status": "not_available", "text": None, "backend": None, "elapsed_ms": elapsed}

        gray = ims.get("bw") or ims.get("sharp") or ims["gray"]

        # A conservative config for serial-like strings
        config = "--oem 1 --psm 6"
        txt = pytesseract.image_to_string(gray, config=config)
        txt = (txt or "").strip()
        elapsed = int((time.perf_counter() - t0) * 1000)
        if txt:
            return {"status": "success", "text": txt, "backend": "tesseract", "elapsed_ms": elapsed}
        return {"status": "not_found", "text": None, "backend": "tesseract", "elapsed_ms": elapsed}

    except Exception as e:
        elapsed = int((time.perf_counter() - t0) * 1000)
        return {"status": "error", "text": None, "backend": mode, "elapsed_ms": elapsed, "error": repr(e)}
    
if __name__ == "__main__":
    import sys
    import cv2
    if len(sys.argv) < 2:
        print("Usage: python vision_ocr.py path/to/image.jpg")
        sys.exit(1)

    img = cv2.imread(sys.argv[1])
    if img is None:
        print("Could not load image")
        sys.exit(1)

    result = ocr_serial_best_effort(img, aggressive=True)
    print(result)