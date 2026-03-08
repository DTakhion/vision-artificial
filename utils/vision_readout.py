# utils/vision_readout.py
from __future__ import annotations

from typing import Any, Dict, Optional, List, Union
import time

import numpy as np

from utils.vision_qr import decode_qr_opencv
from utils.vision_barcode import decode_barcode_1d
from utils.vision_barcode_plus import decode_barcode_1d_plus
from utils.vision_ocr import ocr_serial_best_effort


def _ms_since(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


def _clamp_budget(x: int, lo: int, hi: int) -> int:
    if hi < lo:
        return max(0, hi)
    return max(lo, min(hi, x))


def _disabled_stage_payload(kind: str) -> Dict[str, Any]:
    return {
        "status": "disabled",
        "kind": kind,
        "elapsed_ms": 0,
    }


def _pending_stage_payload(kind: str) -> Dict[str, Any]:
    return {
        "status": "pending_retry",
        "kind": kind,
        "elapsed_ms": 0,
    }


def _get_barcode_items_for_readout(barcode_res: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not barcode_res or barcode_res.get("status") != "success":
        return []

    mode = str(barcode_res.get("mode") or "").strip().lower()

    if mode == "collect_plus":
        confirmed = barcode_res.get("confirmed_items") or []
        if confirmed:
            return confirmed
        return barcode_res.get("items") or []

    return barcode_res.get("items") or []


def _extract_best_barcode(barcode_res: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not barcode_res or barcode_res.get("status") != "success":
        return None

    items = _get_barcode_items_for_readout(barcode_res)
    if not items:
        return None

    best_item = items[0]
    txt = best_item.get("text")
    if not txt:
        return None

    return {
        "kind": "barcode1d",
        "text": txt,
        "meta": {
            "backend": barcode_res.get("backend") or best_item.get("backend"),
            "variant": best_item.get("variant") or barcode_res.get("variant"),
            "candidate": best_item.get("candidate") or barcode_res.get("candidate"),
            "format": best_item.get("format"),
            "mode": barcode_res.get("mode"),
            "source": best_item.get("source"),
            "source_plus": best_item.get("source_plus"),
            "score": best_item.get("score"),
            "plus_score": best_item.get("plus_score"),
            "roi_index": best_item.get("roi_index"),
            "roi_bbox": best_item.get("roi_bbox"),
            "roi_stage_ratio": barcode_res.get("config", {}).get("roi_stage_ratio")
            if isinstance(barcode_res.get("config"), dict)
            else None,
            "validation": best_item.get("validation"),
            "summary": barcode_res.get("summary"),
        },
    }


def _extract_best_ocr(serial_res: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not serial_res or serial_res.get("status") != "success":
        return None

    txt = serial_res.get("text")
    if not txt:
        return None

    return {
        "kind": "serial",
        "text": txt,
        "meta": {
            "backend": serial_res.get("backend"),
            "variant": serial_res.get("variant"),
            "candidate": serial_res.get("candidate"),
            "mode": serial_res.get("mode"),
            "psm": serial_res.get("psm"),
            "confidence": serial_res.get("confidence"),
            "score": serial_res.get("score"),
            "roi_kind": serial_res.get("roi_kind"),
            "roi_index": serial_res.get("roi_index"),
            "roi_bbox": serial_res.get("roi_bbox"),
            "subkind": serial_res.get("subkind"),
        },
    }


def _extract_best_qr(qr_res: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not qr_res or qr_res.get("status") != "success":
        return None

    txt = qr_res.get("text")
    if not txt:
        return None

    return {
        "kind": "qr",
        "text": txt,
        "meta": {
            "backend": qr_res.get("backend"),
            "variant": qr_res.get("variant"),
            "candidate": qr_res.get("candidate"),
        },
    }


def _pick_best(
    barcode1d: Optional[Dict[str, Any]],
    serial: Optional[Dict[str, Any]] = None,
    qr: Optional[Dict[str, Any]] = None,
    *,
    priority: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Prioridad por defecto alineada al nuevo desafío:
      barcode1d > serial > qr

    priority admite por ejemplo:
      ["barcode1d", "serial", "qr"]
      ["barcode1d", "qr", "serial"]
    """
    priority = priority or ["barcode1d", "serial", "qr"]

    candidates: Dict[str, Optional[Dict[str, Any]]] = {
        "barcode1d": _extract_best_barcode(barcode1d),
        "serial": _extract_best_ocr(serial),
        "qr": _extract_best_qr(qr),
    }

    for kind in priority:
        c = candidates.get(kind)
        if c is not None:
            return c

    return {"kind": None, "text": None, "meta": {}}


def _collect_all_texts(
    barcode1d: Optional[Dict[str, Any]],
    serial: Optional[Dict[str, Any]],
    qr: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Resumen útil para inspección/debug del orquestador.
    """
    barcode_items = []
    barcode_confirmed = []
    barcode_suspect = []
    barcode_summary = None

    if barcode1d and barcode1d.get("status") == "success":
        barcode_summary = barcode1d.get("summary")

        mode = str(barcode1d.get("mode") or "").strip().lower()

        if mode == "collect_plus":
            for it in (barcode1d.get("confirmed_items") or []):
                txt = it.get("text")
                if txt:
                    barcode_confirmed.append(
                        {
                            "text": txt,
                            "format": it.get("format"),
                            "backend": it.get("backend"),
                            "score": it.get("score"),
                            "plus_score": it.get("plus_score"),
                            "source": it.get("source"),
                            "source_plus": it.get("source_plus"),
                            "validation": it.get("validation"),
                        }
                    )

            for it in (barcode1d.get("suspect_items") or []):
                txt = it.get("text")
                if txt:
                    barcode_suspect.append(
                        {
                            "text": txt,
                            "format": it.get("format"),
                            "backend": it.get("backend"),
                            "score": it.get("score"),
                            "plus_score": it.get("plus_score"),
                            "source": it.get("source"),
                            "source_plus": it.get("source_plus"),
                            "validation": it.get("validation"),
                        }
                    )

        else:
            for it in (barcode1d.get("items") or []):
                txt = it.get("text")
                if txt:
                    barcode_items.append(
                        {
                            "text": txt,
                            "format": it.get("format"),
                            "backend": it.get("backend"),
                            "score": it.get("score"),
                            "source": it.get("source"),
                        }
                    )

    serial_obj = None
    if serial and serial.get("status") == "success":
        serial_obj = {
            "text": serial.get("text"),
            "score": serial.get("score"),
            "confidence": serial.get("confidence"),
            "roi_kind": serial.get("roi_kind"),
            "subkind": serial.get("subkind"),
        }

    qr_text = None
    if qr and qr.get("status") == "success":
        qr_text = qr.get("text")

    return {
        "barcode1d": barcode_items,
        "barcode1d_confirmed": barcode_confirmed,
        "barcode1d_suspect": barcode_suspect,
        "barcode1d_summary": barcode_summary,
        "serial": serial_obj,
        "qr": qr_text,
    }


def _run_barcode_stage(
    img_bgr: np.ndarray,
    *,
    barcode_mode: str,
    budget_ms: int,
    barcode_prefer: str,
    barcode_fallback: bool,
    barcode_roi_rescue: bool,
    barcode_max_rois: Optional[int],
    barcode_roi_upscale: float,
    barcode_include_full_image: bool,
    barcode_max_collect_items: int,
    barcode_enable_tile_sweep: bool,
    barcode_enable_collect_rotations: bool,
    barcode_roi_stage_ratio: float,
    barcode_max_tiles: int,
) -> Dict[str, Any]:
    mode = (barcode_mode or "fast").strip().lower()

    if mode == "collect_plus":
        return decode_barcode_1d_plus(
            img_bgr,
            budget_ms=max(80, int(budget_ms)),
            variants="all" if barcode_enable_tile_sweep else None,
            roi_upscale=barcode_roi_upscale,
            include_rotations=barcode_enable_collect_rotations,
            include_zoom_rescue=True,
        )

    return decode_barcode_1d(
        img_bgr,
        mode=mode,
        time_budget_ms=max(80, int(budget_ms)),
        prefer=barcode_prefer,
        enable_fallback=barcode_fallback,
        enable_roi_rescue=barcode_roi_rescue,
        max_rois=barcode_max_rois,
        roi_upscale=barcode_roi_upscale,
        include_full_image=barcode_include_full_image,
        max_collect_items=barcode_max_collect_items,
        enable_tile_sweep=barcode_enable_tile_sweep,
        enable_collect_rotations=barcode_enable_collect_rotations,
        roi_stage_ratio=barcode_roi_stage_ratio,
        max_tiles=barcode_max_tiles,
    )


def readout_immediate(
    img_bgr: np.ndarray,
    *,
    time_budget_ms: int = 450,
    enable_barcode: bool = True,
    enable_ocr: bool = False,
    enable_qr: bool = False,
    barcode_budget_ms: int = 220,
    ocr_budget_ms: int = 140,
    qr_budget_ms: int = 120,
    barcode_mode: str = "fast",  # "fast" | "collect" | "collect_plus"
    barcode_prefer: str = "zxingcpp",
    barcode_fallback: bool = True,
    barcode_roi_rescue: bool = True,
    barcode_max_rois: Optional[int] = None,
    barcode_roi_upscale: float = 3.0,
    barcode_include_full_image: bool = True,
    barcode_max_collect_items: int = 64,
    barcode_enable_tile_sweep: bool = False,
    barcode_enable_collect_rotations: bool = False,
    barcode_roi_stage_ratio: float = 0.45,
    barcode_max_tiles: int = 5,
    ocr_mode: str = "fast",  # "fast" | "collect"
    ocr_no_budget: bool = False,
    ocr_max_tries: int = 60,
    ocr_roi: str = "all",
    ocr_variants: Optional[Union[List[str], str]] = None,
    ocr_aggressive: bool = False,
    ocr_numeric_only: bool = True,
    ocr_min_numeric_len: int = 4,
    ocr_max_rois: int = 4,
    qr_variants: Optional[List[str]] = None,
    priority: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    First pass (low latency), alineado al nuevo desafío:
      1) Barcode
      2) OCR
      3) QR

    Por defecto:
      - Barcode: ON
      - OCR: OFF
      - QR: OFF

    Nota:
      En immediate dejamos tile sweep / rotations OFF por defecto
      para proteger latencia. Se pueden activar por parámetro.
      collect_plus está permitido, pero úsalo con criterio porque cuesta más.
    """
    t0 = time.perf_counter()

    out: Dict[str, Any] = {
        "status": "not_found",
        "best": {"kind": None, "text": None, "meta": {}},
        "barcode1d": None,
        "serial": None,
        "qr": None,
        "all_texts": None,
        "elapsed_ms": None,
        "needs_retry": False,
        "config": {
            "mode": "immediate",
            "enable_barcode": bool(enable_barcode),
            "enable_ocr": bool(enable_ocr),
            "enable_qr": bool(enable_qr),
            "barcode_mode": barcode_mode,
            "ocr_mode": ocr_mode,
            "priority": priority or ["barcode1d", "serial", "qr"],
        },
        "budgets": {
            "total_ms": int(time_budget_ms),
            "barcode_ms": int(barcode_budget_ms),
            "ocr_ms": int(ocr_budget_ms),
            "qr_ms": int(qr_budget_ms),
        },
        "stage_elapsed_ms": {},
    }

    # 1) Barcode
    if enable_barcode:
        remaining = int(time_budget_ms) - _ms_since(t0)
        if remaining > 70:
            b_budget = _clamp_budget(min(int(barcode_budget_ms), remaining), 80, remaining)
            barcode_res = _run_barcode_stage(
                img_bgr,
                barcode_mode=barcode_mode,
                budget_ms=b_budget,
                barcode_prefer=barcode_prefer,
                barcode_fallback=barcode_fallback,
                barcode_roi_rescue=barcode_roi_rescue,
                barcode_max_rois=barcode_max_rois,
                barcode_roi_upscale=barcode_roi_upscale,
                barcode_include_full_image=barcode_include_full_image,
                barcode_max_collect_items=barcode_max_collect_items,
                barcode_enable_tile_sweep=barcode_enable_tile_sweep,
                barcode_enable_collect_rotations=barcode_enable_collect_rotations,
                barcode_roi_stage_ratio=barcode_roi_stage_ratio,
                barcode_max_tiles=barcode_max_tiles,
            )
        else:
            barcode_res = _pending_stage_payload("barcode1d")
    else:
        barcode_res = _disabled_stage_payload("barcode1d")

    out["barcode1d"] = barcode_res
    out["stage_elapsed_ms"]["barcode_ms"] = int(barcode_res.get("elapsed_ms", 0))

    # 2) OCR
    if enable_ocr:
        remaining = int(time_budget_ms) - _ms_since(t0)
        if remaining > 120:
            o_budget = _clamp_budget(min(int(ocr_budget_ms), remaining), 120, remaining)
            serial_res = ocr_serial_best_effort(
                img_bgr,
                time_budget_ms=o_budget,
                no_budget=ocr_no_budget,
                max_tries=ocr_max_tries,
                roi_mode=ocr_roi,
                variants=ocr_variants,
                aggressive=ocr_aggressive,
                numeric_only=ocr_numeric_only,
                min_numeric_len=ocr_min_numeric_len,
                mode=ocr_mode,
                max_rois=ocr_max_rois,
            )
        else:
            serial_res = _pending_stage_payload("serial")
    else:
        serial_res = _disabled_stage_payload("serial")

    out["serial"] = serial_res
    out["stage_elapsed_ms"]["ocr_ms"] = int(serial_res.get("elapsed_ms", 0))

    # 3) QR
    if enable_qr:
        remaining = int(time_budget_ms) - _ms_since(t0)
        if remaining > 60:
            q_budget = _clamp_budget(min(int(qr_budget_ms), remaining), 60, remaining)
            if qr_variants is None:
                qr_res = decode_qr_opencv(img_bgr, time_budget_ms=q_budget)
            else:
                qr_res = decode_qr_opencv(img_bgr, time_budget_ms=q_budget, variants=qr_variants)
        else:
            qr_res = _pending_stage_payload("qr")
    else:
        qr_res = _disabled_stage_payload("qr")

    out["qr"] = qr_res
    out["stage_elapsed_ms"]["qr_ms"] = int(qr_res.get("elapsed_ms", 0))

    best = _pick_best(
        barcode_res if enable_barcode else None,
        serial_res if enable_ocr else None,
        qr_res if enable_qr else None,
        priority=priority,
    )
    out["best"] = best
    out["status"] = "success" if best.get("kind") is not None else "not_found"
    out["needs_retry"] = out["status"] != "success"
    out["all_texts"] = _collect_all_texts(
        barcode_res if enable_barcode else None,
        serial_res if enable_ocr else None,
        qr_res if enable_qr else None,
    )
    out["elapsed_ms"] = _ms_since(t0)
    return out


def readout_retry(
    img_bgr: np.ndarray,
    *,
    time_budget_ms: int = 1500,
    enable_barcode: bool = True,
    enable_ocr: bool = True,
    enable_qr: bool = False,
    barcode_budget_ms: int = 650,
    ocr_budget_ms: int = 700,
    qr_budget_ms: int = 250,
    barcode_mode: str = "collect",  # "fast" | "collect" | "collect_plus"
    barcode_prefer: str = "zxingcpp",
    barcode_fallback: bool = True,
    barcode_roi_rescue: bool = True,
    barcode_max_rois: Optional[int] = None,
    barcode_roi_upscale: float = 3.0,
    barcode_include_full_image: bool = True,
    barcode_max_collect_items: int = 64,
    barcode_enable_tile_sweep: bool = True,
    barcode_enable_collect_rotations: bool = True,
    barcode_roi_stage_ratio: float = 0.45,
    barcode_max_tiles: int = 5,
    ocr_mode: str = "collect",  # retry => más recall
    ocr_no_budget: bool = False,
    ocr_max_tries: int = 120,
    ocr_roi: str = "all",
    ocr_variants: Optional[Union[List[str], str]] = None,
    ocr_aggressive: bool = True,
    ocr_numeric_only: bool = True,
    ocr_min_numeric_len: int = 4,
    ocr_max_rois: int = 4,
    qr_variants: Optional[List[str]] = None,
    priority: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Second pass (high recall), alineado al nuevo desafío:
      1) Barcode
      2) OCR
      3) QR

    Nota:
      En retry dejamos tile sweep / rotations ON por defecto,
      porque aquí priorizamos recall.
      collect_plus es válido cuando quieras más exploración espacial.
    """
    t0 = time.perf_counter()

    out: Dict[str, Any] = {
        "status": "not_found",
        "best": {"kind": None, "text": None, "meta": {}},
        "barcode1d": None,
        "serial": None,
        "qr": None,
        "all_texts": None,
        "elapsed_ms": None,
        "config": {
            "mode": "retry",
            "enable_barcode": bool(enable_barcode),
            "enable_ocr": bool(enable_ocr),
            "enable_qr": bool(enable_qr),
            "barcode_mode": barcode_mode,
            "ocr_mode": ocr_mode,
            "priority": priority or ["barcode1d", "serial", "qr"],
        },
        "budgets": {
            "total_ms": int(time_budget_ms),
            "barcode_ms": int(barcode_budget_ms),
            "ocr_ms": int(ocr_budget_ms),
            "qr_ms": int(qr_budget_ms),
        },
        "stage_elapsed_ms": {},
    }

    # 1) Barcode
    if enable_barcode:
        remaining = int(time_budget_ms) - _ms_since(t0)
        if remaining > 120:
            b_budget = _clamp_budget(min(int(barcode_budget_ms), remaining), 150, remaining)
            barcode_res = _run_barcode_stage(
                img_bgr,
                barcode_mode=barcode_mode,
                budget_ms=b_budget,
                barcode_prefer=barcode_prefer,
                barcode_fallback=barcode_fallback,
                barcode_roi_rescue=barcode_roi_rescue,
                barcode_max_rois=barcode_max_rois,
                barcode_roi_upscale=barcode_roi_upscale,
                barcode_include_full_image=barcode_include_full_image,
                barcode_max_collect_items=barcode_max_collect_items,
                barcode_enable_tile_sweep=barcode_enable_tile_sweep,
                barcode_enable_collect_rotations=barcode_enable_collect_rotations,
                barcode_roi_stage_ratio=barcode_roi_stage_ratio,
                barcode_max_tiles=barcode_max_tiles,
            )
        else:
            barcode_res = _pending_stage_payload("barcode1d")
    else:
        barcode_res = _disabled_stage_payload("barcode1d")

    out["barcode1d"] = barcode_res
    out["stage_elapsed_ms"]["barcode_ms"] = int(barcode_res.get("elapsed_ms", 0))

    # 2) OCR
    if enable_ocr:
        remaining = int(time_budget_ms) - _ms_since(t0)
        if remaining > 220:
            o_budget = _clamp_budget(min(int(ocr_budget_ms), remaining), 220, remaining)
            serial_res = ocr_serial_best_effort(
                img_bgr,
                time_budget_ms=o_budget,
                no_budget=ocr_no_budget,
                max_tries=ocr_max_tries,
                roi_mode=ocr_roi,
                variants=ocr_variants,
                aggressive=ocr_aggressive,
                numeric_only=ocr_numeric_only,
                min_numeric_len=ocr_min_numeric_len,
                mode=ocr_mode,
                max_rois=ocr_max_rois,
            )
        else:
            serial_res = _pending_stage_payload("serial")
    else:
        serial_res = _disabled_stage_payload("serial")

    out["serial"] = serial_res
    out["stage_elapsed_ms"]["ocr_ms"] = int(serial_res.get("elapsed_ms", 0))

    # 3) QR
    if enable_qr:
        remaining = int(time_budget_ms) - _ms_since(t0)
        if remaining > 100:
            q_budget = _clamp_budget(min(int(qr_budget_ms), remaining), 100, remaining)
            if qr_variants is None:
                qr_res = decode_qr_opencv(img_bgr, time_budget_ms=q_budget, variants=["sharp", "gray"])
            else:
                qr_res = decode_qr_opencv(img_bgr, time_budget_ms=q_budget, variants=qr_variants)
        else:
            qr_res = _pending_stage_payload("qr")
    else:
        qr_res = _disabled_stage_payload("qr")

    out["qr"] = qr_res
    out["stage_elapsed_ms"]["qr_ms"] = int(qr_res.get("elapsed_ms", 0))

    best = _pick_best(
        barcode_res if enable_barcode else None,
        serial_res if enable_ocr else None,
        qr_res if enable_qr else None,
        priority=priority,
    )
    out["best"] = best
    out["status"] = "success" if best.get("kind") is not None else "not_found"
    out["all_texts"] = _collect_all_texts(
        barcode_res if enable_barcode else None,
        serial_res if enable_ocr else None,
        qr_res if enable_qr else None,
    )
    out["elapsed_ms"] = _ms_since(t0)
    return out


if __name__ == "__main__":
    import argparse
    import cv2

    parser = argparse.ArgumentParser(description="Vision Readout Orchestrator (Barcode / OCR / QR)")
    parser.add_argument("image", help="path to image")
    parser.add_argument("--mode", choices=["immediate", "retry"], default="immediate")

    # switches
    parser.add_argument("--barcode", dest="enable_barcode", action="store_true", help="enable barcode stage")
    parser.add_argument("--no-barcode", dest="enable_barcode", action="store_false", help="disable barcode stage")
    parser.set_defaults(enable_barcode=True)

    parser.add_argument("--ocr", dest="enable_ocr", action="store_true", help="enable OCR stage")
    parser.add_argument("--no-ocr", dest="enable_ocr", action="store_false", help="disable OCR stage")
    parser.set_defaults(enable_ocr=False)

    parser.add_argument("--qr", dest="enable_qr", action="store_true", help="enable QR stage")
    parser.add_argument("--no-qr", dest="enable_qr", action="store_false", help="disable QR stage")
    parser.set_defaults(enable_qr=False)

    # budgets
    parser.add_argument("--budget", type=int, default=None, help="total orchestration budget in ms")
    parser.add_argument("--barcode_budget", type=int, default=None, help="barcode stage budget in ms")
    parser.add_argument("--ocr_budget", type=int, default=None, help="ocr stage budget in ms")
    parser.add_argument("--qr_budget", type=int, default=None, help="qr stage budget in ms")

    # barcode config
    parser.add_argument(
        "--barcode_mode",
        choices=["fast", "collect", "collect_plus"],
        default=None,
        help="barcode mode",
    )
    parser.add_argument(
        "--barcode_prefer",
        choices=["zxingcpp", "pyzbar", "opencv_barcode"],
        default="zxingcpp",
        help="preferred barcode backend",
    )
    parser.add_argument("--no_barcode_fallback", action="store_true", help="disable barcode backend fallback")
    parser.add_argument("--no_barcode_roi", action="store_true", help="disable barcode ROI rescue")
    parser.add_argument("--barcode_max_rois", type=int, default=None, help="max barcode ROIs")
    parser.add_argument("--barcode_roi_upscale", type=float, default=3.0, help="barcode ROI upscale")
    parser.add_argument("--no_barcode_full_image", action="store_true", help="collect mode: disable full image sweep")
    parser.add_argument("--barcode_max_collect_items", type=int, default=64, help="collect mode unique item cap")
    parser.add_argument("--no_barcode_tile_sweep", action="store_true", help="disable barcode tile sweep")
    parser.add_argument("--no_barcode_collect_rotations", action="store_true", help="disable barcode collect rotations")
    parser.add_argument("--barcode_roi_stage_ratio", type=float, default=0.45, help="barcode ROI stage ratio")
    parser.add_argument("--barcode_max_tiles", type=int, default=5, help="barcode max tiles")

    # OCR config
    parser.add_argument("--ocr_mode", choices=["fast", "collect"], default=None, help="ocr mode")
    parser.add_argument("--ocr_no_budget", action="store_true", help="disable OCR hard budget")
    parser.add_argument("--ocr_max_tries", type=int, default=None, help="OCR max tries")
    parser.add_argument("--ocr_roi", type=str, default="all", help='OCR ROI mode, e.g. "all"')
    parser.add_argument("--ocr_aggressive", action="store_true", help="use aggressive OCR preprocessing")
    parser.add_argument("--ocr_text", action="store_true", help="disable numeric_only OCR mode")
    parser.add_argument("--ocr_min_digits", type=int, default=4, help="minimum numeric OCR length")
    parser.add_argument("--ocr_max_rois", type=int, default=4, help="max OCR ROIs")

    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise RuntimeError(f"Could not read image: {args.image}")

    if args.mode == "immediate":
        res = readout_immediate(
            img,
            time_budget_ms=450 if args.budget is None else int(args.budget),
            enable_barcode=bool(args.enable_barcode),
            enable_ocr=bool(args.enable_ocr),
            enable_qr=bool(args.enable_qr),
            barcode_budget_ms=220 if args.barcode_budget is None else int(args.barcode_budget),
            ocr_budget_ms=140 if args.ocr_budget is None else int(args.ocr_budget),
            qr_budget_ms=120 if args.qr_budget is None else int(args.qr_budget),
            barcode_mode="fast" if args.barcode_mode is None else args.barcode_mode,
            barcode_prefer=args.barcode_prefer,
            barcode_fallback=(not args.no_barcode_fallback),
            barcode_roi_rescue=(not args.no_barcode_roi),
            barcode_max_rois=args.barcode_max_rois,
            barcode_roi_upscale=args.barcode_roi_upscale,
            barcode_include_full_image=(not args.no_barcode_full_image),
            barcode_max_collect_items=args.barcode_max_collect_items,
            barcode_enable_tile_sweep=(not args.no_barcode_tile_sweep),
            barcode_enable_collect_rotations=(not args.no_barcode_collect_rotations),
            barcode_roi_stage_ratio=float(args.barcode_roi_stage_ratio),
            barcode_max_tiles=int(args.barcode_max_tiles),
            ocr_mode="fast" if args.ocr_mode is None else args.ocr_mode,
            ocr_no_budget=bool(args.ocr_no_budget),
            ocr_max_tries=60 if args.ocr_max_tries is None else int(args.ocr_max_tries),
            ocr_roi=args.ocr_roi,
            ocr_aggressive=bool(args.ocr_aggressive),
            ocr_numeric_only=(not args.ocr_text),
            ocr_min_numeric_len=int(args.ocr_min_digits),
            ocr_max_rois=int(args.ocr_max_rois),
        )
    else:
        res = readout_retry(
            img,
            time_budget_ms=1500 if args.budget is None else int(args.budget),
            enable_barcode=bool(args.enable_barcode),
            enable_ocr=bool(args.enable_ocr),
            enable_qr=bool(args.enable_qr),
            barcode_budget_ms=650 if args.barcode_budget is None else int(args.barcode_budget),
            ocr_budget_ms=700 if args.ocr_budget is None else int(args.ocr_budget),
            qr_budget_ms=250 if args.qr_budget is None else int(args.qr_budget),
            barcode_mode="collect" if args.barcode_mode is None else args.barcode_mode,
            barcode_prefer=args.barcode_prefer,
            barcode_fallback=(not args.no_barcode_fallback),
            barcode_roi_rescue=(not args.no_barcode_roi),
            barcode_max_rois=args.barcode_max_rois,
            barcode_roi_upscale=args.barcode_roi_upscale,
            barcode_include_full_image=(not args.no_barcode_full_image),
            barcode_max_collect_items=args.barcode_max_collect_items,
            barcode_enable_tile_sweep=(not args.no_barcode_tile_sweep),
            barcode_enable_collect_rotations=(not args.no_barcode_collect_rotations),
            barcode_roi_stage_ratio=float(args.barcode_roi_stage_ratio),
            barcode_max_tiles=int(args.barcode_max_tiles),
            ocr_mode="collect" if args.ocr_mode is None else args.ocr_mode,
            ocr_no_budget=bool(args.ocr_no_budget),
            ocr_max_tries=120 if args.ocr_max_tries is None else int(args.ocr_max_tries),
            ocr_roi=args.ocr_roi,
            ocr_aggressive=True if not args.ocr_aggressive else bool(args.ocr_aggressive),
            ocr_numeric_only=(not args.ocr_text),
            ocr_min_numeric_len=int(args.ocr_min_digits),
            ocr_max_rois=int(args.ocr_max_rois),
        )

    print(res)