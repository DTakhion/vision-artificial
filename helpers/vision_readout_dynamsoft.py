# # utils/vision_readout_dynamsoft.py
# from __future__ import annotations

# from typing import Any, Dict, Optional, List
# import time
# import json
# from pathlib import Path

# import numpy as np

# from utils.vision_barcode_dynamsoft import decode_barcode_dynamsoft


# def _ms_since(t0: float) -> int:
#     return int((time.perf_counter() - t0) * 1000)


# def _safe_write_json(path_str: str, payload: Dict[str, Any]) -> None:
#     path = Path(path_str)
#     path.parent.mkdir(parents=True, exist_ok=True)
#     tmp = path.with_suffix(path.suffix + ".tmp")
#     with tmp.open("w", encoding="utf-8") as f:
#         json.dump(payload, f, ensure_ascii=False, indent=2)
#     tmp.replace(path)


# def _disabled_stage_payload(kind: str) -> Dict[str, Any]:
#     return {
#         "status": "disabled",
#         "kind": kind,
#         "elapsed_ms": 0,
#     }


# def _norm_barcode(txt: Any) -> str:
#     s = "" if txt is None else str(txt)
#     return "".join(ch for ch in s if ch.isdigit())


# def _dedupe_preserve_order(items: List[str]) -> List[str]:
#     out: List[str] = []
#     seen = set()

#     for raw in items:
#         value = _norm_barcode(raw)
#         if not value:
#             continue
#         if value in seen:
#             continue
#         seen.add(value)
#         out.append(value)

#     return out


# def _extract_detected_barcodes_payload(readout_res: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     Devuelve un payload plano compatible con app.main --detected_barcodes_json:
#     {
#       "detected_barcodes": [...]
#     }
#     """
#     barcode_res = readout_res.get("barcode1d") or {}
#     items: List[str] = []

#     if barcode_res.get("status") == "success":
#         for it in (barcode_res.get("items") or []):
#             txt = _norm_barcode(it.get("text"))
#             if txt:
#                 items.append(txt)

#     best = readout_res.get("best") or {}
#     if isinstance(best, dict) and best.get("kind") == "barcode1d":
#         txt = _norm_barcode(best.get("text"))
#         if txt:
#             items.append(txt)

#     detected_barcodes = _dedupe_preserve_order(items)

#     return {
#         "status": "success",
#         "processed_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
#         "detected_barcodes": detected_barcodes,
#         "counts": {
#             "detected_barcodes": len(detected_barcodes),
#         },
#     }


# def _get_barcode_items_for_readout(barcode_res: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
#     if not barcode_res or barcode_res.get("status") != "success":
#         return []

#     return barcode_res.get("items") or []


# def _extract_best_barcode(barcode_res: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
#     if not barcode_res or barcode_res.get("status") != "success":
#         return None

#     items = _get_barcode_items_for_readout(barcode_res)
#     if not items:
#         return None

#     best_item = items[0]
#     txt = best_item.get("text")
#     if not txt:
#         return None

#     return {
#         "kind": "barcode1d",
#         "text": txt,
#         "meta": {
#             "backend": barcode_res.get("backend") or best_item.get("backend"),
#             "format": best_item.get("format"),
#             "position": best_item.get("position"),
#             "bbox": best_item.get("bbox"),
#             "total": barcode_res.get("total"),
#             "source": barcode_res.get("source"),
#         },
#     }


# def _pick_best(barcode1d: Optional[Dict[str, Any]]) -> Dict[str, Any]:
#     """
#     Orquestador reducido: prioridad única barcode1d.
#     """
#     c = _extract_best_barcode(barcode1d)
#     if c is not None:
#         return c

#     return {"kind": None, "text": None, "meta": {}}


# def _collect_all_texts(barcode1d: Optional[Dict[str, Any]]) -> Dict[str, Any]:
#     """
#     Resumen útil para inspección/debug del orquestador.
#     """
#     barcode_items = []

#     if barcode1d and barcode1d.get("status") == "success":
#         for it in (barcode1d.get("items") or []):
#             txt = it.get("text")
#             if txt:
#                 barcode_items.append(
#                     {
#                         "text": txt,
#                         "format": it.get("format"),
#                         "backend": it.get("backend"),
#                         "bbox": it.get("bbox"),
#                         "position": it.get("position"),
#                     }
#                 )

#     return {
#         "barcode1d": barcode_items,
#     }


# def _run_barcode_stage(
#     img_bgr: np.ndarray,
#     *,
#     env_file: str = ".env",
#     stage_label: str = "barcode1d",
# ) -> Dict[str, Any]:
#     """
#     Ejecuta lectura barcode usando Dynamsoft.
#     """
#     t0 = time.perf_counter()

#     try:
#         barcode_res = decode_barcode_dynamsoft(img_bgr, env_file=env_file)
#     except Exception as exc:
#         return {
#             "status": "error",
#             "kind": stage_label,
#             "backend": "dynamsoft",
#             "items": [],
#             "total": 0,
#             "elapsed_ms": _ms_since(t0),
#             "error_message": str(exc),
#         }

#     if not isinstance(barcode_res, dict):
#         return {
#             "status": "error",
#             "kind": stage_label,
#             "backend": "dynamsoft",
#             "items": [],
#             "total": 0,
#             "elapsed_ms": _ms_since(t0),
#             "error_message": "La respuesta del backend no es un dict válido",
#         }

#     barcode_res["kind"] = stage_label
#     barcode_res["elapsed_ms"] = _ms_since(t0)

#     if "backend" not in barcode_res:
#         barcode_res["backend"] = "dynamsoft"

#     if "items" not in barcode_res or barcode_res["items"] is None:
#         barcode_res["items"] = []

#     if "total" not in barcode_res:
#         barcode_res["total"] = len(barcode_res["items"])

#     return barcode_res


# def readout_immediate(
#     img_bgr: np.ndarray,
#     *,
#     time_budget_ms: int = 450,
#     enable_barcode: bool = True,
#     env_file: str = ".env",
# ) -> Dict[str, Any]:
#     """
#     First pass (low latency), enfocado solo en barcode con Dynamsoft.
#     """
#     t0 = time.perf_counter()

#     out: Dict[str, Any] = {
#         "status": "not_found",
#         "best": {"kind": None, "text": None, "meta": {}},
#         "barcode1d": None,
#         "serial": _disabled_stage_payload("serial"),
#         "qr": _disabled_stage_payload("qr"),
#         "all_texts": None,
#         "elapsed_ms": None,
#         "needs_retry": False,
#         "config": {
#             "mode": "immediate",
#             "enable_barcode": bool(enable_barcode),
#             "enable_ocr": False,
#             "enable_qr": False,
#             "barcode_backend": "dynamsoft",
#             "priority": ["barcode1d"],
#             "env_file": env_file,
#         },
#         "budgets": {
#             "total_ms": int(time_budget_ms),
#             "barcode_ms": int(time_budget_ms),
#             "ocr_ms": 0,
#             "qr_ms": 0,
#         },
#         "stage_elapsed_ms": {},
#     }

#     if enable_barcode:
#         barcode_res = _run_barcode_stage(
#             img_bgr,
#             env_file=env_file,
#             stage_label="barcode1d",
#         )
#     else:
#         barcode_res = _disabled_stage_payload("barcode1d")

#     out["barcode1d"] = barcode_res
#     out["stage_elapsed_ms"]["barcode_ms"] = int(barcode_res.get("elapsed_ms", 0))

#     best = _pick_best(barcode_res if enable_barcode else None)
#     out["best"] = best
#     out["status"] = "success" if best.get("kind") is not None else "not_found"
#     out["needs_retry"] = out["status"] != "success"
#     out["all_texts"] = _collect_all_texts(barcode_res if enable_barcode else None)
#     out["elapsed_ms"] = _ms_since(t0)

#     return out


# def readout_retry(
#     img_bgr: np.ndarray,
#     *,
#     time_budget_ms: int = 1500,
#     enable_barcode: bool = True,
#     env_file: str = ".env",
# ) -> Dict[str, Any]:
#     """
#     Second pass (high recall), pero en esta versión sigue enfocado solo en barcode.
#     Se conserva por compatibilidad de flujo.
#     """
#     t0 = time.perf_counter()

#     out: Dict[str, Any] = {
#         "status": "not_found",
#         "best": {"kind": None, "text": None, "meta": {}},
#         "barcode1d": None,
#         "serial": _disabled_stage_payload("serial"),
#         "qr": _disabled_stage_payload("qr"),
#         "all_texts": None,
#         "elapsed_ms": None,
#         "config": {
#             "mode": "retry",
#             "enable_barcode": bool(enable_barcode),
#             "enable_ocr": False,
#             "enable_qr": False,
#             "barcode_backend": "dynamsoft",
#             "priority": ["barcode1d"],
#             "env_file": env_file,
#         },
#         "budgets": {
#             "total_ms": int(time_budget_ms),
#             "barcode_ms": int(time_budget_ms),
#             "ocr_ms": 0,
#             "qr_ms": 0,
#         },
#         "stage_elapsed_ms": {},
#     }

#     if enable_barcode:
#         barcode_res = _run_barcode_stage(
#             img_bgr,
#             env_file=env_file,
#             stage_label="barcode1d",
#         )
#     else:
#         barcode_res = _disabled_stage_payload("barcode1d")

#     out["barcode1d"] = barcode_res
#     out["stage_elapsed_ms"]["barcode_ms"] = int(barcode_res.get("elapsed_ms", 0))

#     best = _pick_best(barcode_res if enable_barcode else None)
#     out["best"] = best
#     out["status"] = "success" if best.get("kind") is not None else "not_found"
#     out["all_texts"] = _collect_all_texts(barcode_res if enable_barcode else None)
#     out["elapsed_ms"] = _ms_since(t0)

#     return out


# if __name__ == "__main__":
#     import argparse
#     import cv2

#     parser = argparse.ArgumentParser(description="Vision Readout Orchestrator (Barcode only / Dynamsoft)")
#     parser.add_argument("image", help="path to image")
#     parser.add_argument("--mode", choices=["immediate", "retry"], default="immediate")

#     # switches
#     parser.add_argument("--barcode", dest="enable_barcode", action="store_true", help="enable barcode stage")
#     parser.add_argument("--no-barcode", dest="enable_barcode", action="store_false", help="disable barcode stage")
#     parser.set_defaults(enable_barcode=True)

#     # compatibilidad con flujo previo
#     parser.add_argument("--ocr", dest="enable_ocr", action="store_true", help="ignored in this dynamsoft barcode-only version")
#     parser.add_argument("--no-ocr", dest="enable_ocr", action="store_false", help="ignored in this dynamsoft barcode-only version")
#     parser.set_defaults(enable_ocr=False)

#     parser.add_argument("--qr", dest="enable_qr", action="store_true", help="ignored in this dynamsoft barcode-only version")
#     parser.add_argument("--no-qr", dest="enable_qr", action="store_false", help="ignored in this dynamsoft barcode-only version")
#     parser.set_defaults(enable_qr=False)

#     # budgets
#     parser.add_argument("--budget", type=int, default=None, help="total orchestration budget in ms")

#     # dynamsoft config
#     parser.add_argument(
#         "--env-file",
#         type=str,
#         default=".env",
#         help="Ruta al archivo .env con LICENSE_KEY_DYNAMSOFT",
#     )

#     # outputs
#     parser.add_argument(
#         "--out_json",
#         type=str,
#         default=None,
#         help="Ruta para guardar el resultado completo del readout en JSON",
#     )
#     parser.add_argument(
#         "--out_detected_barcodes_json",
#         type=str,
#         default=None,
#         help="Ruta para guardar un JSON plano compatible con app.main: {'detected_barcodes': [...]}",
#     )

#     args = parser.parse_args()

#     img = cv2.imread(args.image)
#     if img is None:
#         raise RuntimeError(f"Could not read image: {args.image}")

#     if args.mode == "immediate":
#         res = readout_immediate(
#             img,
#             time_budget_ms=450 if args.budget is None else int(args.budget),
#             enable_barcode=bool(args.enable_barcode),
#             env_file=args.env_file,
#         )
#     else:
#         res = readout_retry(
#             img,
#             time_budget_ms=1500 if args.budget is None else int(args.budget),
#             enable_barcode=bool(args.enable_barcode),
#             env_file=args.env_file,
#         )

#     if args.out_json:
#         _safe_write_json(args.out_json, res)
#         print(f"[OK] Resultado readout guardado en: {args.out_json}")

#     if args.out_detected_barcodes_json:
#         detected_payload = _extract_detected_barcodes_payload(res)
#         _safe_write_json(args.out_detected_barcodes_json, detected_payload)
#         print(f"[OK] Detected barcodes guardado en: {args.out_detected_barcodes_json}")

#     print(res)
    
# # python -m utils.vision_readout_dynamsoft \
# #   data/tests_picking/capture_barcode_test.png \
# #   --out_json results/readout_dynamsoft.json \
# #   --out_detected_barcodes_json results/detected_barcodes_dynamsoft.json