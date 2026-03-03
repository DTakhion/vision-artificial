# utils/vision_readout.py
from __future__ import annotations

from typing import Any, Dict, Optional
import time

import numpy as np

from utils.vision_qr import decode_qr_opencv
from utils.vision_barcode import decode_barcode_1d
from utils.vision_ocr import ocr_serial_best_effort


def readout_immediate(
    img_bgr: np.ndarray,
    *,
    time_budget_ms: int = 450,
) -> Dict[str, Any]:
    """
    Fast attempt: QR + 1D barcode. OCR is optional / usually deferred.
    """
    t0 = time.perf_counter()
    out: Dict[str, Any] = {
        "status": "not_found",
        "qr": None,
        "barcode1d": None,
        "serial": None,
        "elapsed_ms": None,
        "needs_retry": False,
    }

    # 1) QR (fast)
    qr = decode_qr_opencv(img_bgr, time_budget_ms=min(150, time_budget_ms))
    out["qr"] = qr

    # 2) Barcode 1D (can be heavier)
    remaining = time_budget_ms - int((time.perf_counter() - t0) * 1000)
    if remaining > 60:
        b1 = decode_barcode_1d(img_bgr, time_budget_ms=min(220, remaining))
    else:
        b1 = {"status": "pending_retry", "items": [], "backend": None, "elapsed_ms": 0}
    out["barcode1d"] = b1

    # Decide success
    success = (qr.get("status") == "success") or (b1.get("status") == "success")
    out["status"] = "success" if success else "not_found"

    # If not found, allow retry later with more aggressive preprocessing + OCR
    if not success:
        out["needs_retry"] = True

    out["elapsed_ms"] = int((time.perf_counter() - t0) * 1000)
    return out


def readout_retry(
    img_bgr: np.ndarray,
    *,
    time_budget_ms: int = 1500,
    enable_ocr: bool = True,
) -> Dict[str, Any]:
    """
    Slower retry: try barcode again + OCR (aggressive).
    """
    t0 = time.perf_counter()

    out: Dict[str, Any] = {
        "status": "not_found",
        "qr": None,
        "barcode1d": None,
        "serial": None,
        "elapsed_ms": None,
    }

    # QR again (sometimes works after exposure settles / slight differences)
    qr = decode_qr_opencv(img_bgr, time_budget_ms=250, variants=["sharp", "gray"])
    out["qr"] = qr

    # Barcode again with more time (and backends)
    b1 = decode_barcode_1d(img_bgr, time_budget_ms=600, prefer="zxingcpp")
    out["barcode1d"] = b1

    # OCR as best-effort
    if enable_ocr:
        remaining = time_budget_ms - int((time.perf_counter() - t0) * 1000)
        if remaining > 250:
            serial = ocr_serial_best_effort(img_bgr, time_budget_ms=min(700, remaining), aggressive=True)
        else:
            serial = {"status": "pending_retry", "text": None, "backend": None, "elapsed_ms": 0}
        out["serial"] = serial
    else:
        out["serial"] = {"status": "disabled", "text": None, "backend": None, "elapsed_ms": 0}

    success = (
        (qr.get("status") == "success")
        or (b1.get("status") == "success")
        or (enable_ocr and out["serial"].get("status") == "success")
    )
    out["status"] = "success" if success else "not_found"
    out["elapsed_ms"] = int((time.perf_counter() - t0) * 1000)
    return out