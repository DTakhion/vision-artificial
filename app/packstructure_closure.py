# # app/packstructure_closure.py
# from __future__ import annotations

# from collections import defaultdict
# from typing import Any, Dict, List, Optional


# def _norm_text(txt: Any) -> str:
#     s = "" if txt is None else str(txt)
#     return s.strip()


# def _norm_code(txt: Any) -> str:
#     s = _norm_text(txt).upper()
#     s = s.replace(" ", "").replace(".", "").replace(",", "")
#     return s


# def _norm_barcode(txt: Any) -> str:
#     s = _norm_text(txt)
#     return "".join(ch for ch in s if ch.isdigit())


# def _safe_int(value: Any, default: int = 0) -> int:
#     if value is None:
#         return default
#     if isinstance(value, bool):
#         return int(value)
#     if isinstance(value, int):
#         return value
#     if isinstance(value, float):
#         return int(round(value))
#     s = str(value).strip()
#     if not s:
#         return default
#     try:
#         return int(float(s))
#     except Exception:
#         return default


# def _safe_float(value: Any, default: float = 0.0) -> float:
#     if value is None:
#         return default
#     if isinstance(value, (int, float)):
#         return float(value)
#     s = str(value).strip().replace(",", ".")
#     if not s:
#         return default
#     try:
#         return float(s)
#     except Exception:
#         return default


# def _pick_first(prod: Dict[str, Any], *keys: str) -> Any:
#     for key in keys:
#         if key in prod and prod.get(key) is not None:
#             return prod.get(key)
#     return None


# def _extract_product_sku(prod: Dict[str, Any]) -> str:
#     return _norm_code(_pick_first(prod, "codigo_item", "sku", "Articulo", "articulo"))


# def _extract_product_description(prod: Dict[str, Any]) -> Optional[str]:
#     return _pick_first(
#         prod,
#         "descripcion_ocr",
#         "descripcion_fillrate",
#         "descripcion_packstructure",
#         "descripcion",
#         "Descripcion",
#     )


# def _extract_expected_units(prod: Dict[str, Any]) -> int:
#     raw = _pick_first(
#         prod,
#         "unidades_ocr",
#         "cant_original_total",
#         "qty_expected",
#         "cantidad_esperada",
#     )
#     return _safe_int(raw, default=0)


# def _extract_pack_match_status(prod: Dict[str, Any]) -> Optional[str]:
#     pack = prod.get("packstructure") or {}
#     return _pick_first(prod, "match_status") or pack.get("match_status")


# def _extract_pack_fields(prod: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     Soporta dos esquemas:

#     1) Antiguo:
#        prod["packstructure"] = {
#          "ean_pack", "ean_in_pack", "ean_cs_pack",
#          "qty_ea_pack", "qty_inn_pack", "qty_cs_pack"
#        }

#     2) Actual summary_fillRate_packStructure:
#        prod = {
#          "ean_ea", "ean_in", "ean_cs",
#          "qty_ea", "qty_inn", "qty_cs"
#        }
#        o dentro de packstructure_full
#     """
#     pack = prod.get("packstructure") or {}
#     pack_full = prod.get("packstructure_full") or {}

#     ean_pack = _pick_first(prod, "ean_ea")
#     if ean_pack is None:
#         ean_pack = _pick_first(pack, "ean_pack")
#     if ean_pack is None:
#         ean_pack = _pick_first(pack_full, "ean", "ean_ea")

#     ean_in_pack = _pick_first(prod, "ean_in")
#     if ean_in_pack is None:
#         ean_in_pack = _pick_first(pack, "ean_in_pack")
#     if ean_in_pack is None:
#         ean_in_pack = _pick_first(pack_full, "ean_in")

#     ean_cs_pack = _pick_first(prod, "ean_cs")
#     if ean_cs_pack is None:
#         ean_cs_pack = _pick_first(pack, "ean_cs_pack")
#     if ean_cs_pack is None:
#         ean_cs_pack = _pick_first(pack_full, "ean_cs")

#     qty_ea_pack = _pick_first(prod, "qty_ea")
#     if qty_ea_pack is None:
#         qty_ea_pack = _pick_first(pack, "qty_ea_pack")
#     if qty_ea_pack is None:
#         qty_ea_pack = _pick_first(pack_full, "qty_ea")

#     qty_inn_pack = _pick_first(prod, "qty_inn")
#     if qty_inn_pack is None:
#         qty_inn_pack = _pick_first(pack, "qty_inn_pack")
#     if qty_inn_pack is None:
#         qty_inn_pack = _pick_first(pack_full, "qty_inn")

#     qty_cs_pack = _pick_first(prod, "qty_cs")
#     if qty_cs_pack is None:
#         qty_cs_pack = _pick_first(pack, "qty_cs_pack")
#     if qty_cs_pack is None:
#         qty_cs_pack = _pick_first(pack_full, "qty_cs")

#     return {
#         "ean_pack": _norm_barcode(ean_pack),
#         "ean_in_pack": _norm_barcode(ean_in_pack),
#         "ean_cs_pack": _norm_barcode(ean_cs_pack),
#         "qty_ea_pack": _safe_int(qty_ea_pack, default=0),
#         "qty_inn_pack": _safe_int(qty_inn_pack, default=0),
#         "qty_cs_pack": _safe_int(qty_cs_pack, default=0),
#     }


# def extract_final_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     Acepta:
#     1) el final_summary directamente
#     2) el payload completo generado por run_picking_match(...)
#     """
#     if not isinstance(payload, dict):
#         return {}

#     nested_1 = payload.get("summary_pickingVision_fillRate_packStructure")
#     if isinstance(nested_1, dict) and nested_1.get("products") is not None:
#         return nested_1

#     nested_2 = payload.get("summary_fillRate_packStructure")
#     if isinstance(nested_2, dict) and nested_2.get("products") is not None:
#         return nested_2

#     if payload.get("products") is not None:
#         return payload

#     return {}


# def _unwrap_readout_payload(readout_payload: Dict[str, Any]) -> Dict[str, Any]:
#     if not isinstance(readout_payload, dict):
#         return {}

#     root = readout_payload.get("result")
#     if isinstance(root, dict):
#         return root

#     return readout_payload


# def collect_detected_items_from_readout(readout_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
#     payload = _unwrap_readout_payload(readout_payload)
#     if not isinstance(payload, dict):
#         return []

#     out: List[Dict[str, Any]] = []
#     seen_keys = set()

#     items = payload.get("items") or []
#     if isinstance(items, list):
#         for item in items:
#             if not isinstance(item, dict):
#                 continue

#             code = _norm_barcode(item.get("text"))
#             if not code:
#                 continue

#             key = (
#                 code,
#                 _norm_text(item.get("source")),
#                 _norm_text(item.get("roi_variant")),
#                 _norm_text(item.get("format")),
#                 str(item.get("bbox")),
#             )
#             if key in seen_keys:
#                 continue

#             seen_keys.add(key)
#             out.append(
#                 {
#                     "barcode": code,
#                     "text": item.get("text"),
#                     "format": item.get("format"),
#                     "backend": item.get("backend"),
#                     "source": item.get("source"),
#                     "bbox": item.get("bbox"),
#                     "position": item.get("position"),
#                     "yolo_roi_index": item.get("yolo_roi_index"),
#                     "yolo_conf": item.get("yolo_conf"),
#                     "yolo_bbox_xyxy_original": item.get("yolo_bbox_xyxy_original"),
#                     "yolo_bbox_xyxy_padded": item.get("yolo_bbox_xyxy_padded"),
#                     "roi_variant": item.get("roi_variant"),
#                 }
#             )

#     barcode1d = payload.get("barcode1d") or {}
#     confirmed_items = barcode1d.get("confirmed_items") or []

#     if isinstance(confirmed_items, list):
#         for item in confirmed_items:
#             if not isinstance(item, dict):
#                 continue

#             code = _norm_barcode(item.get("text"))
#             if not code:
#                 continue

#             key = (
#                 code,
#                 _norm_text(item.get("source")),
#                 _norm_text(item.get("variant")),
#                 _norm_text(item.get("format")),
#                 str(item.get("bbox")),
#             )
#             if key in seen_keys:
#                 continue

#             seen_keys.add(key)
#             out.append(
#                 {
#                     "barcode": code,
#                     "text": item.get("text"),
#                     "format": item.get("format"),
#                     "backend": item.get("backend", "barcode1d"),
#                     "source": item.get("source", "legacy_confirmed_items"),
#                     "bbox": item.get("bbox"),
#                     "position": item.get("position"),
#                     "yolo_roi_index": item.get("yolo_roi_index"),
#                     "yolo_conf": item.get("yolo_conf"),
#                     "yolo_bbox_xyxy_original": item.get("yolo_bbox_xyxy_original"),
#                     "yolo_bbox_xyxy_padded": item.get("yolo_bbox_xyxy_padded"),
#                     "roi_variant": item.get("variant") or item.get("roi_variant"),
#                 }
#             )

#     best = payload.get("best") or {}
#     if isinstance(best, dict):
#         best_kind = _norm_text(best.get("kind")).lower()
#         best_text = _norm_barcode(best.get("text"))

#         if best_kind in {"barcode1d", "hybrid"} and best_text:
#             key = (
#                 best_text,
#                 _norm_text(best.get("source")),
#                 _norm_text(best.get("variant")),
#                 _norm_text(best.get("format")),
#                 str(best.get("bbox")),
#             )
#             if key not in seen_keys:
#                 seen_keys.add(key)
#                 out.append(
#                     {
#                         "barcode": best_text,
#                         "text": best.get("text"),
#                         "format": best.get("format"),
#                         "backend": best.get("backend", best_kind),
#                         "source": best.get("source", "legacy_best"),
#                         "bbox": best.get("bbox"),
#                         "position": best.get("position"),
#                         "yolo_roi_index": best.get("yolo_roi_index"),
#                         "yolo_conf": best.get("yolo_conf"),
#                         "yolo_bbox_xyxy_original": best.get("yolo_bbox_xyxy_original"),
#                         "yolo_bbox_xyxy_padded": best.get("yolo_bbox_xyxy_padded"),
#                         "roi_variant": best.get("variant") or best.get("roi_variant"),
#                     }
#                 )

#     return out


# def collect_detected_barcodes_from_readout(readout_payload: Dict[str, Any]) -> List[str]:
#     items = collect_detected_items_from_readout(readout_payload)
#     return [x["barcode"] for x in items if x.get("barcode")]


# def build_barcode_index_from_summary(summary_payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
#     summary = extract_final_summary(summary_payload)
#     products = summary.get("products") or []

#     idx: Dict[str, Dict[str, Any]] = {}

#     for prod in products:
#         sku = _extract_product_sku(prod)
#         descripcion = _extract_product_description(prod)
#         pack_fields = _extract_pack_fields(prod)

#         levels = [
#             ("EA", pack_fields.get("ean_pack"), pack_fields.get("qty_ea_pack"), "ean_pack"),
#             ("IN", pack_fields.get("ean_in_pack"), pack_fields.get("qty_inn_pack"), "ean_in_pack"),
#             ("CS", pack_fields.get("ean_cs_pack"), pack_fields.get("qty_cs_pack"), "ean_cs_pack"),
#         ]

#         for level, raw_barcode, raw_units, source_field in levels:
#             barcode = _norm_barcode(raw_barcode)
#             units = _safe_int(raw_units, default=0)

#             if not sku:
#                 continue
#             if not barcode:
#                 continue
#             if units <= 0:
#                 continue

#             idx[barcode] = {
#                 "sku": sku,
#                 "descripcion": descripcion,
#                 "level": level,
#                 "units_per_barcode": units,
#                 "source_field": source_field,
#             }

#     return idx


# def build_expected_index_from_summary(summary_payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
#     summary = extract_final_summary(summary_payload)
#     products = summary.get("products") or []

#     idx: Dict[str, Dict[str, Any]] = {}

#     for prod in products:
#         sku = _extract_product_sku(prod)
#         if not sku:
#             continue

#         expected_units = _extract_expected_units(prod)
#         descripcion = _extract_product_description(prod)
#         pack_fields = _extract_pack_fields(prod)
#         pack_match_status = _extract_pack_match_status(prod)

#         idx[sku] = {
#             "sku": sku,
#             "descripcion": descripcion,
#             "expected_units": expected_units,
#             "packstructure_match_status": pack_match_status,
#             "ean_pack": pack_fields.get("ean_pack"),
#             "ean_in_pack": pack_fields.get("ean_in_pack"),
#             "ean_cs_pack": pack_fields.get("ean_cs_pack"),
#             "qty_ea_pack": pack_fields.get("qty_ea_pack"),
#             "qty_inn_pack": pack_fields.get("qty_inn_pack"),
#             "qty_cs_pack": pack_fields.get("qty_cs_pack"),
#         }

#     return idx


# def aggregate_observed_units(
#     detected_items: List[Dict[str, Any]],
#     barcode_index: Dict[str, Dict[str, Any]],
# ) -> Dict[str, Any]:
#     observed_by_sku: Dict[str, Dict[str, Any]] = {}
#     unknown_barcodes_unique: List[str] = []
#     unknown_seen = set()
#     barcode_hits: Dict[str, int] = defaultdict(int)

#     for item in detected_items:
#         if isinstance(item, dict):
#             barcode = _norm_barcode(item.get("barcode") or item.get("text"))
#             raw_item = item
#         else:
#             barcode = _norm_barcode(item)
#             raw_item = {"barcode": barcode}

#         if not barcode:
#             continue

#         barcode_hits[barcode] += 1
#         meta = barcode_index.get(barcode)

#         if not meta:
#             if barcode not in unknown_seen:
#                 unknown_seen.add(barcode)
#                 unknown_barcodes_unique.append(barcode)
#             continue

#         sku = meta["sku"]
#         entry = observed_by_sku.setdefault(
#             sku,
#             {
#                 "sku": sku,
#                 "descripcion": meta.get("descripcion"),
#                 "observed_units": 0,
#                 "observed_barcodes": [],
#                 "breakdown_by_level": {
#                     "EA": {"count": 0, "units": 0},
#                     "IN": {"count": 0, "units": 0},
#                     "CS": {"count": 0, "units": 0},
#                 },
#             },
#         )

#         units_per_barcode = _safe_int(meta.get("units_per_barcode"), default=0)
#         level = meta.get("level") or "UNK"

#         entry["observed_units"] += units_per_barcode
#         entry["observed_barcodes"].append(
#             {
#                 "barcode": barcode,
#                 "level": level,
#                 "units_per_barcode": units_per_barcode,
#                 "format": raw_item.get("format"),
#                 "backend": raw_item.get("backend"),
#                 "source": raw_item.get("source"),
#                 "bbox": raw_item.get("bbox"),
#                 "position": raw_item.get("position"),
#                 "yolo_roi_index": raw_item.get("yolo_roi_index"),
#                 "yolo_conf": raw_item.get("yolo_conf"),
#                 "yolo_bbox_xyxy_original": raw_item.get("yolo_bbox_xyxy_original"),
#                 "yolo_bbox_xyxy_padded": raw_item.get("yolo_bbox_xyxy_padded"),
#                 "roi_variant": raw_item.get("roi_variant"),
#             }
#         )

#         if level not in entry["breakdown_by_level"]:
#             entry["breakdown_by_level"][level] = {"count": 0, "units": 0}

#         entry["breakdown_by_level"][level]["count"] += 1
#         entry["breakdown_by_level"][level]["units"] += units_per_barcode

#     return {
#         "observed_by_sku": observed_by_sku,
#         "unknown_barcodes": unknown_barcodes_unique,
#         "barcode_hits": dict(barcode_hits),
#     }


# def _status_from_diff(expected_units: int, observed_units: int) -> str:
#     if expected_units == observed_units:
#         return "matched"
#     if observed_units == 0 and expected_units > 0:
#         return "missing"
#     if 0 < observed_units < expected_units:
#         return "partial"
#     if observed_units > expected_units:
#         return "excess"
#     return "unknown"


# def compare_expected_vs_observed(
#     summary_payload: Dict[str, Any],
#     observed_payload: Dict[str, Any],
# ) -> Dict[str, Any]:
#     summary = extract_final_summary(summary_payload)
#     expected_idx = build_expected_index_from_summary(summary)
#     observed_by_sku = observed_payload.get("observed_by_sku") or {}
#     unknown_barcodes = observed_payload.get("unknown_barcodes") or []

#     results: List[Dict[str, Any]] = []
#     unexpected_skus: List[Dict[str, Any]] = []

#     total_expected = 0
#     total_observed = 0
#     matched_count = 0
#     partial_count = 0
#     missing_count = 0
#     excess_count = 0

#     for sku, exp in expected_idx.items():
#         obs = observed_by_sku.get(sku, {})
#         expected_units = _safe_int(exp.get("expected_units"), default=0)
#         observed_units = _safe_int(obs.get("observed_units"), default=0)
#         diff_units = observed_units - expected_units
#         status = _status_from_diff(expected_units, observed_units)

#         total_expected += expected_units
#         total_observed += observed_units

#         if status == "matched":
#             matched_count += 1
#         elif status == "partial":
#             partial_count += 1
#         elif status == "missing":
#             missing_count += 1
#         elif status == "excess":
#             excess_count += 1

#         results.append(
#             {
#                 "sku": sku,
#                 "descripcion": exp.get("descripcion"),
#                 "expected_units": expected_units,
#                 "observed_units": observed_units,
#                 "difference_units": diff_units,
#                 "status": status,
#                 "matched": status == "matched",
#                 "packstructure_match_status": exp.get("packstructure_match_status"),
#                 "expected_pack_levels": {
#                     "EA": {
#                         "ean": exp.get("ean_pack"),
#                         "units": exp.get("qty_ea_pack"),
#                     },
#                     "IN": {
#                         "ean": exp.get("ean_in_pack"),
#                         "units": exp.get("qty_inn_pack"),
#                     },
#                     "CS": {
#                         "ean": exp.get("ean_cs_pack"),
#                         "units": exp.get("qty_cs_pack"),
#                     },
#                 },
#                 "observed_breakdown": obs.get(
#                     "breakdown_by_level",
#                     {
#                         "EA": {"count": 0, "units": 0},
#                         "IN": {"count": 0, "units": 0},
#                         "CS": {"count": 0, "units": 0},
#                     },
#                 ),
#                 "observed_barcodes": obs.get("observed_barcodes", []),
#             }
#         )

#     for sku, obs in observed_by_sku.items():
#         if sku in expected_idx:
#             continue
#         unexpected_skus.append(
#             {
#                 "sku": sku,
#                 "descripcion": obs.get("descripcion"),
#                 "observed_units": _safe_int(obs.get("observed_units"), default=0),
#                 "status": "unexpected_sku",
#                 "observed_breakdown": obs.get("breakdown_by_level", {}),
#                 "observed_barcodes": obs.get("observed_barcodes", []),
#             }
#         )

#     route_value = summary.get("route")
#     if route_value is None:
#         metadata = summary.get("metadata") or {}
#         route_value = metadata.get("ruta")

#     if (
#         len(unexpected_skus) == 0
#         and len(unknown_barcodes) == 0
#         and matched_count == len(expected_idx)
#     ):
#         closure_status = "complete_match"
#     elif total_observed == 0:
#         closure_status = "no_detection"
#     else:
#         closure_status = "partial_match"

#     return {
#         "status": "success",
#         "closure_status": closure_status,
#         "route": route_value,
#         "metadata": summary.get("metadata") or {},
#         "products": results,
#         "unexpected_skus": unexpected_skus,
#         "unknown_barcodes": unknown_barcodes,
#         "counts": {
#             "expected_products": len(expected_idx),
#             "matched_products": matched_count,
#             "partial_products": partial_count,
#             "missing_products": missing_count,
#             "excess_products": excess_count,
#             "unexpected_skus": len(unexpected_skus),
#             "unknown_barcodes": len(unknown_barcodes),
#         },
#         "totals": {
#             "expected_units": total_expected,
#             "observed_units": total_observed,
#             "difference_units": total_observed - total_expected,
#         },
#         "flags": {
#             "has_unknown_barcodes": len(unknown_barcodes) > 0,
#             "has_unexpected_skus": len(unexpected_skus) > 0,
#             "all_products_matched": matched_count == len(expected_idx) and len(expected_idx) > 0,
#         },
#     }


# def build_readout_trace(readout_payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
#     payload = _unwrap_readout_payload(readout_payload or {})
#     if not isinstance(payload, dict):
#         return {}

#     detected_items = collect_detected_items_from_readout(payload)
#     debug = payload.get("debug") if isinstance(payload.get("debug"), dict) else {}

#     trace = {
#         "status": payload.get("status"),
#         "backend": payload.get("backend"),
#         "total_reported": _safe_int(payload.get("total"), default=len(detected_items)),
#         "total_items_extracted": len(detected_items),
#         "detected_items": detected_items,
#         "debug_available": bool(debug),
#     }

#     if debug:
#         trace["debug_keys"] = list(debug.keys())
#         trace["yolo_rois_detected"] = len(debug.get("yolo_rois") or [])
#         trace["dynamsoft_full_image_total"] = _safe_int(
#             ((debug.get("dynamsoft_full_image") or {}).get("total")),
#             default=0,
#         )
#         trace["yolo_pipeline_status"] = (debug.get("yolo_pipeline") or {}).get("status")
#         trace["dynamsoft_yolo_rois_count"] = len(debug.get("dynamsoft_yolo_rois") or [])

#     return trace


# def run_packstructure_closure(
#     *,
#     summary_payload: Dict[str, Any],
#     readout_payload: Optional[Dict[str, Any]] = None,
#     detected_barcodes: Optional[List[str]] = None,
#     event_context: Optional[Dict[str, Any]] = None,
# ) -> Dict[str, Any]:
#     """
#     Ejecuta el cierre completo.

#     Puedes pasar:
#     - readout_payload (wrapped_result.json, result interno o salida directa del híbrido)
#     - o detected_barcodes directamente
#     """
#     summary = extract_final_summary(summary_payload)
#     if not summary:
#         return {
#             "status": "error",
#             "reason": "summary_payload_invalido",
#         }

#     readout_trace = build_readout_trace(readout_payload)

#     if detected_barcodes is None:
#         detected_items = collect_detected_items_from_readout(readout_payload or {})
#     else:
#         detected_items = [{"barcode": _norm_barcode(x), "text": x} for x in detected_barcodes]

#     detected_items = [x for x in detected_items if _norm_barcode(x.get("barcode") or x.get("text"))]

#     barcode_index = build_barcode_index_from_summary(summary)
#     observed_payload = aggregate_observed_units(detected_items, barcode_index)
#     comparison = compare_expected_vs_observed(summary, observed_payload)

#     comparison["detected_barcodes"] = [
#         _norm_barcode(x.get("barcode") or x.get("text"))
#         for x in detected_items
#         if _norm_barcode(x.get("barcode") or x.get("text"))
#     ]
#     comparison["detected_items"] = detected_items
#     comparison["barcode_hits"] = observed_payload.get("barcode_hits", {})
#     comparison["barcode_index_size"] = len(barcode_index)
#     comparison["event_context"] = event_context or {}
#     comparison["readout_trace"] = readout_trace

#     return comparison

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, List, Optional


def _norm_text(txt: Any) -> str:
    s = "" if txt is None else str(txt)
    return s.strip()


def _norm_code(txt: Any) -> str:
    s = _norm_text(txt).upper()
    s = s.replace(" ", "").replace(".", "").replace(",", "")
    return s


def _norm_barcode(txt: Any) -> str:
    s = _norm_text(txt)
    return "".join(ch for ch in s if ch.isdigit())


def _norm_shipping(value: Any) -> str:
    """
    Normaliza shipping para comparar:
    - Excel puede traer 824434708
    - Hoja puede traer 0824434708
    Para el flujo KN, usamos 10 dígitos cuando es posible.
    """
    s = _norm_barcode(value)
    if not s:
        return ""
    if len(s) < 10:
        return s.zfill(10)
    return s


def _shipping_variants(value: Any) -> set[str]:
    """
    Permite comparar shipping con y sin cero inicial.
    """
    raw = _norm_barcode(value)
    norm = _norm_shipping(value)
    out = set()
    if raw:
        out.add(raw)
        out.add(raw.lstrip("0") or raw)
    if norm:
        out.add(norm)
        out.add(norm.lstrip("0") or norm)
    return {x for x in out if x}


def _safe_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(round(value))
    s = str(value).strip()
    if not s:
        return default
    try:
        return int(float(s))
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().replace(",", ".")
    if not s:
        return default
    try:
        return float(s)
    except Exception:
        return default


def _pick_first(prod: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in prod and prod.get(key) is not None:
            return prod.get(key)
    return None


def _extract_product_sku(prod: Dict[str, Any]) -> str:
    return _norm_code(_pick_first(prod, "codigo_item", "sku", "Articulo", "articulo"))


def _extract_product_description(prod: Dict[str, Any]) -> Optional[str]:
    return _pick_first(
        prod,
        "descripcion_ocr",
        "descripcion_fillrate",
        "descripcion_packstructure",
        "descripcion",
        "Descripcion",
    )


def _extract_expected_units(prod: Dict[str, Any]) -> int:
    raw = _pick_first(
        prod,
        "unidades_ocr",
        "cant_original_total",
        "qty_expected",
        "cantidad_esperada",
    )
    return _safe_int(raw, default=0)


def _extract_pack_match_status(prod: Dict[str, Any]) -> Optional[str]:
    pack = prod.get("packstructure") or {}
    return _pick_first(prod, "match_status") or pack.get("match_status")


def _extract_product_shipping_values(prod: Dict[str, Any]) -> List[str]:
    values: List[Any] = []

    raw_values = prod.get("shipping_values")
    if isinstance(raw_values, list):
        values.extend(raw_values)
    elif raw_values is not None:
        values.append(raw_values)

    for key in ("shipping", "Shipping", "orden", "Orden", "orden_shipping"):
        if prod.get(key) is not None:
            values.append(prod.get(key))

    for row in prod.get("fillrate_full") or []:
        if isinstance(row, dict):
            if row.get("Shipping") is not None:
                values.append(row.get("Shipping"))
            if row.get("shipping") is not None:
                values.append(row.get("shipping"))

    out: List[str] = []
    seen = set()
    for v in values:
        norm = _norm_shipping(v)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)

    return out


def _product_matches_shipping(prod: Dict[str, Any], target_shipping: Optional[str]) -> bool:
    if not target_shipping:
        return True

    target_variants = _shipping_variants(target_shipping)
    if not target_variants:
        return True

    product_values = _extract_product_shipping_values(prod)
    if not product_values:
        return False

    for value in product_values:
        if _shipping_variants(value) & target_variants:
            return True

    return False


def _extract_pack_fields(prod: Dict[str, Any]) -> Dict[str, Any]:
    """
    Soporta dos esquemas:

    1) Antiguo:
       prod["packstructure"] = {
         "ean_pack", "ean_in_pack", "ean_cs_pack",
         "qty_ea_pack", "qty_inn_pack", "qty_cs_pack"
       }

    2) Actual summary_fillRate_packStructure:
       prod = {
         "ean_ea", "ean_in", "ean_cs",
         "qty_ea", "qty_inn", "qty_cs"
       }
       o dentro de packstructure_full.
    """
    pack = prod.get("packstructure") or {}
    pack_full = prod.get("packstructure_full") or {}

    ean_pack = _pick_first(prod, "ean_ea")
    if ean_pack is None:
        ean_pack = _pick_first(pack, "ean_pack")
    if ean_pack is None:
        ean_pack = _pick_first(pack_full, "ean", "ean_ea")

    ean_in_pack = _pick_first(prod, "ean_in")
    if ean_in_pack is None:
        ean_in_pack = _pick_first(pack, "ean_in_pack")
    if ean_in_pack is None:
        ean_in_pack = _pick_first(pack_full, "ean_in")

    ean_cs_pack = _pick_first(prod, "ean_cs")
    if ean_cs_pack is None:
        ean_cs_pack = _pick_first(pack, "ean_cs_pack")
    if ean_cs_pack is None:
        ean_cs_pack = _pick_first(pack_full, "ean_cs")

    qty_ea_pack = _pick_first(prod, "qty_ea")
    if qty_ea_pack is None:
        qty_ea_pack = _pick_first(pack, "qty_ea_pack")
    if qty_ea_pack is None:
        qty_ea_pack = _pick_first(pack_full, "qty_ea")

    qty_inn_pack = _pick_first(prod, "qty_inn")
    if qty_inn_pack is None:
        qty_inn_pack = _pick_first(pack, "qty_inn_pack")
    if qty_inn_pack is None:
        qty_inn_pack = _pick_first(pack_full, "qty_inn")

    qty_cs_pack = _pick_first(prod, "qty_cs")
    if qty_cs_pack is None:
        qty_cs_pack = _pick_first(pack, "qty_cs_pack")
    if qty_cs_pack is None:
        qty_cs_pack = _pick_first(pack_full, "qty_cs")

    return {
        "ean_pack": _norm_barcode(ean_pack),
        "ean_in_pack": _norm_barcode(ean_in_pack),
        "ean_cs_pack": _norm_barcode(ean_cs_pack),
        "qty_ea_pack": _safe_int(qty_ea_pack, default=0),
        "qty_inn_pack": _safe_int(qty_inn_pack, default=0),
        "qty_cs_pack": _safe_int(qty_cs_pack, default=0),
    }


def extract_final_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Acepta:
    1) el final_summary directamente
    2) el payload completo generado por run_picking_match(...)
    """
    if not isinstance(payload, dict):
        return {}

    nested_1 = payload.get("summary_pickingVision_fillRate_packStructure")
    if isinstance(nested_1, dict) and nested_1.get("products") is not None:
        return nested_1

    nested_2 = payload.get("summary_fillRate_packStructure")
    if isinstance(nested_2, dict) and nested_2.get("products") is not None:
        return nested_2

    if payload.get("products") is not None:
        return payload

    return {}


def filter_summary_by_shipping(
    summary_payload: Dict[str, Any],
    target_shipping: Optional[str],
) -> Dict[str, Any]:
    """
    Devuelve un summary con solo productos del shipping objetivo.
    Si target_shipping viene vacío, devuelve el summary original.
    """
    summary = extract_final_summary(summary_payload)
    if not summary:
        return summary_payload

    if not target_shipping:
        return summary_payload

    products = summary.get("products") or []
    filtered_products = [
        prod for prod in products
        if isinstance(prod, dict) and _product_matches_shipping(prod, target_shipping)
    ]

    filtered_summary = deepcopy(summary)
    filtered_summary["products"] = filtered_products

    original_counts = summary.get("counts") or {}
    filtered_summary["counts"] = {
        **original_counts,
        "products_before_shipping_filter": len(products),
        "products_after_shipping_filter": len(filtered_products),
    }

    filtered_summary["shipping_filter"] = {
        "enabled": True,
        "target_shipping": _norm_shipping(target_shipping),
        "target_shipping_raw": str(target_shipping),
    }

    # Si el payload original venía anidado, conservamos su forma.
    if isinstance(summary_payload, dict) and isinstance(summary_payload.get("summary_fillRate_packStructure"), dict):
        out = deepcopy(summary_payload)
        out["summary_fillRate_packStructure"] = filtered_summary
        return out

    if isinstance(summary_payload, dict) and isinstance(summary_payload.get("summary_pickingVision_fillRate_packStructure"), dict):
        out = deepcopy(summary_payload)
        out["summary_pickingVision_fillRate_packStructure"] = filtered_summary
        return out

    return filtered_summary


def _unwrap_readout_payload(readout_payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(readout_payload, dict):
        return {}

    root = readout_payload.get("result")
    if isinstance(root, dict):
        return root

    return readout_payload


def collect_detected_items_from_readout(readout_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    payload = _unwrap_readout_payload(readout_payload)
    if not isinstance(payload, dict):
        return []

    out: List[Dict[str, Any]] = []
    seen_keys = set()

    items = payload.get("items") or []
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue

            code = _norm_barcode(item.get("text"))
            if not code:
                continue

            key = (
                code,
                _norm_text(item.get("source")),
                _norm_text(item.get("roi_variant")),
                _norm_text(item.get("format")),
                str(item.get("bbox")),
            )
            if key in seen_keys:
                continue

            seen_keys.add(key)
            out.append(
                {
                    "barcode": code,
                    "text": item.get("text"),
                    "format": item.get("format"),
                    "backend": item.get("backend"),
                    "source": item.get("source"),
                    "bbox": item.get("bbox"),
                    "position": item.get("position"),
                    "yolo_roi_index": item.get("yolo_roi_index"),
                    "yolo_conf": item.get("yolo_conf"),
                    "yolo_bbox_xyxy_original": item.get("yolo_bbox_xyxy_original"),
                    "yolo_bbox_xyxy_padded": item.get("yolo_bbox_xyxy_padded"),
                    "roi_variant": item.get("roi_variant"),
                }
            )

    barcode1d = payload.get("barcode1d") or {}
    confirmed_items = barcode1d.get("confirmed_items") or []

    if isinstance(confirmed_items, list):
        for item in confirmed_items:
            if not isinstance(item, dict):
                continue

            code = _norm_barcode(item.get("text"))
            if not code:
                continue

            key = (
                code,
                _norm_text(item.get("source")),
                _norm_text(item.get("variant")),
                _norm_text(item.get("format")),
                str(item.get("bbox")),
            )
            if key in seen_keys:
                continue

            seen_keys.add(key)
            out.append(
                {
                    "barcode": code,
                    "text": item.get("text"),
                    "format": item.get("format"),
                    "backend": item.get("backend", "barcode1d"),
                    "source": item.get("source", "legacy_confirmed_items"),
                    "bbox": item.get("bbox"),
                    "position": item.get("position"),
                    "yolo_roi_index": item.get("yolo_roi_index"),
                    "yolo_conf": item.get("yolo_conf"),
                    "yolo_bbox_xyxy_original": item.get("yolo_bbox_xyxy_original"),
                    "yolo_bbox_xyxy_padded": item.get("yolo_bbox_xyxy_padded"),
                    "roi_variant": item.get("variant") or item.get("roi_variant"),
                }
            )

    best = payload.get("best") or {}
    if isinstance(best, dict):
        best_kind = _norm_text(best.get("kind")).lower()
        best_text = _norm_barcode(best.get("text"))

        if best_kind in {"barcode1d", "hybrid"} and best_text:
            key = (
                best_text,
                _norm_text(best.get("source")),
                _norm_text(best.get("variant")),
                _norm_text(best.get("format")),
                str(best.get("bbox")),
            )
            if key not in seen_keys:
                seen_keys.add(key)
                out.append(
                    {
                        "barcode": best_text,
                        "text": best.get("text"),
                        "format": best.get("format"),
                        "backend": best.get("backend", best_kind),
                        "source": best.get("source", "legacy_best"),
                        "bbox": best.get("bbox"),
                        "position": best.get("position"),
                        "yolo_roi_index": best.get("yolo_roi_index"),
                        "yolo_conf": best.get("yolo_conf"),
                        "yolo_bbox_xyxy_original": best.get("yolo_bbox_xyxy_original"),
                        "yolo_bbox_xyxy_padded": best.get("yolo_bbox_xyxy_padded"),
                        "roi_variant": best.get("variant") or best.get("roi_variant"),
                    }
                )

    return out


def collect_detected_barcodes_from_readout(readout_payload: Dict[str, Any]) -> List[str]:
    items = collect_detected_items_from_readout(readout_payload)
    return [x["barcode"] for x in items if x.get("barcode")]


def build_barcode_index_from_summary(summary_payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    summary = extract_final_summary(summary_payload)
    products = summary.get("products") or []

    idx: Dict[str, Dict[str, Any]] = {}

    for prod in products:
        sku = _extract_product_sku(prod)
        descripcion = _extract_product_description(prod)
        pack_fields = _extract_pack_fields(prod)
        shipping_values = _extract_product_shipping_values(prod)

        levels = [
            ("EA", pack_fields.get("ean_pack"), pack_fields.get("qty_ea_pack"), "ean_pack"),
            ("IN", pack_fields.get("ean_in_pack"), pack_fields.get("qty_inn_pack"), "ean_in_pack"),
            ("CS", pack_fields.get("ean_cs_pack"), pack_fields.get("qty_cs_pack"), "ean_cs_pack"),
        ]

        for level, raw_barcode, raw_units, source_field in levels:
            barcode = _norm_barcode(raw_barcode)
            units = _safe_int(raw_units, default=0)

            if not sku:
                continue
            if not barcode:
                continue
            if units <= 0:
                continue

            idx[barcode] = {
                "sku": sku,
                "descripcion": descripcion,
                "level": level,
                "units_per_barcode": units,
                "source_field": source_field,
                "shipping_values": shipping_values,
            }

    return idx


def build_expected_index_from_summary(summary_payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    summary = extract_final_summary(summary_payload)
    products = summary.get("products") or []

    idx: Dict[str, Dict[str, Any]] = {}

    for prod in products:
        sku = _extract_product_sku(prod)
        if not sku:
            continue

        expected_units = _extract_expected_units(prod)
        descripcion = _extract_product_description(prod)
        pack_fields = _extract_pack_fields(prod)
        pack_match_status = _extract_pack_match_status(prod)
        shipping_values = _extract_product_shipping_values(prod)

        idx[sku] = {
            "sku": sku,
            "descripcion": descripcion,
            "expected_units": expected_units,
            "packstructure_match_status": pack_match_status,
            "ean_pack": pack_fields.get("ean_pack"),
            "ean_in_pack": pack_fields.get("ean_in_pack"),
            "ean_cs_pack": pack_fields.get("ean_cs_pack"),
            "qty_ea_pack": pack_fields.get("qty_ea_pack"),
            "qty_inn_pack": pack_fields.get("qty_inn_pack"),
            "qty_cs_pack": pack_fields.get("qty_cs_pack"),
            "shipping_values": shipping_values,
        }

    return idx


def aggregate_observed_units(
    detected_items: List[Dict[str, Any]],
    barcode_index: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    observed_by_sku: Dict[str, Dict[str, Any]] = {}
    unknown_barcodes_unique: List[str] = []
    unknown_seen = set()
    barcode_hits: Dict[str, int] = defaultdict(int)

    for item in detected_items:
        if isinstance(item, dict):
            barcode = _norm_barcode(item.get("barcode") or item.get("text"))
            raw_item = item
        else:
            barcode = _norm_barcode(item)
            raw_item = {"barcode": barcode}

        if not barcode:
            continue

        barcode_hits[barcode] += 1
        meta = barcode_index.get(barcode)

        if not meta:
            if barcode not in unknown_seen:
                unknown_seen.add(barcode)
                unknown_barcodes_unique.append(barcode)
            continue

        sku = meta["sku"]
        entry = observed_by_sku.setdefault(
            sku,
            {
                "sku": sku,
                "descripcion": meta.get("descripcion"),
                "observed_units": 0,
                "observed_barcodes": [],
                "breakdown_by_level": {
                    "EA": {"count": 0, "units": 0},
                    "IN": {"count": 0, "units": 0},
                    "CS": {"count": 0, "units": 0},
                },
            },
        )

        units_per_barcode = _safe_int(meta.get("units_per_barcode"), default=0)
        level = meta.get("level") or "UNK"

        entry["observed_units"] += units_per_barcode
        entry["observed_barcodes"].append(
            {
                "barcode": barcode,
                "level": level,
                "units_per_barcode": units_per_barcode,
                "format": raw_item.get("format"),
                "backend": raw_item.get("backend"),
                "source": raw_item.get("source"),
                "bbox": raw_item.get("bbox"),
                "position": raw_item.get("position"),
                "yolo_roi_index": raw_item.get("yolo_roi_index"),
                "yolo_conf": raw_item.get("yolo_conf"),
                "yolo_bbox_xyxy_original": raw_item.get("yolo_bbox_xyxy_original"),
                "yolo_bbox_xyxy_padded": raw_item.get("yolo_bbox_xyxy_padded"),
                "roi_variant": raw_item.get("roi_variant"),
                "shipping_values": meta.get("shipping_values") or [],
            }
        )

        if level not in entry["breakdown_by_level"]:
            entry["breakdown_by_level"][level] = {"count": 0, "units": 0}

        entry["breakdown_by_level"][level]["count"] += 1
        entry["breakdown_by_level"][level]["units"] += units_per_barcode

    return {
        "observed_by_sku": observed_by_sku,
        "unknown_barcodes": unknown_barcodes_unique,
        "barcode_hits": dict(barcode_hits),
    }


def classify_detected_barcodes_for_shipping(
    detected_items: List[Dict[str, Any]],
    *,
    global_barcode_index: Dict[str, Dict[str, Any]],
    filtered_barcode_index: Dict[str, Dict[str, Any]],
    target_shipping: Optional[str],
) -> Dict[str, Any]:
    """
    Clasifica detecciones:
    - valid_for_target_shipping: existe en el summary filtrado por shipping.
    - contaminated_barcodes: existe globalmente, pero no pertenece al shipping actual.
    - unknown_barcodes: no existe globalmente.
    """
    valid: List[Dict[str, Any]] = []
    contaminated: List[Dict[str, Any]] = []
    unknown: List[str] = []
    unknown_seen = set()

    for item in detected_items:
        barcode = _norm_barcode(item.get("barcode") or item.get("text"))
        if not barcode:
            continue

        if barcode in filtered_barcode_index:
            valid.append(
                {
                    "barcode": barcode,
                    "sku": filtered_barcode_index[barcode].get("sku"),
                    "level": filtered_barcode_index[barcode].get("level"),
                    "units_per_barcode": filtered_barcode_index[barcode].get("units_per_barcode"),
                    "target_shipping": _norm_shipping(target_shipping),
                }
            )
            continue

        if barcode in global_barcode_index:
            meta = global_barcode_index[barcode]
            contaminated.append(
                {
                    "barcode": barcode,
                    "sku": meta.get("sku"),
                    "descripcion": meta.get("descripcion"),
                    "level": meta.get("level"),
                    "units_per_barcode": meta.get("units_per_barcode"),
                    "shipping_values": meta.get("shipping_values") or [],
                    "target_shipping": _norm_shipping(target_shipping),
                    "reason": "barcode_exists_but_not_in_target_shipping",
                }
            )
            continue

        if barcode not in unknown_seen:
            unknown_seen.add(barcode)
            unknown.append(barcode)

    return {
        "valid_for_target_shipping": valid,
        "contaminated_barcodes": contaminated,
        "unknown_barcodes": unknown,
        "counts": {
            "valid": len(valid),
            "contaminated": len(contaminated),
            "unknown": len(unknown),
        },
    }


def _status_from_diff(expected_units: int, observed_units: int) -> str:
    if expected_units == observed_units:
        return "matched"
    if observed_units == 0 and expected_units > 0:
        return "missing"
    if 0 < observed_units < expected_units:
        return "partial"
    if observed_units > expected_units:
        return "excess"
    return "unknown"


def compare_expected_vs_observed(
    summary_payload: Dict[str, Any],
    observed_payload: Dict[str, Any],
    contaminated_barcodes: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    summary = extract_final_summary(summary_payload)
    expected_idx = build_expected_index_from_summary(summary)
    observed_by_sku = observed_payload.get("observed_by_sku") or {}
    unknown_barcodes = observed_payload.get("unknown_barcodes") or []
    contaminated_barcodes = contaminated_barcodes or []

    results: List[Dict[str, Any]] = []
    unexpected_skus: List[Dict[str, Any]] = []

    total_expected = 0
    total_observed = 0
    matched_count = 0
    partial_count = 0
    missing_count = 0
    excess_count = 0

    for sku, exp in expected_idx.items():
        obs = observed_by_sku.get(sku, {})
        expected_units = _safe_int(exp.get("expected_units"), default=0)
        observed_units = _safe_int(obs.get("observed_units"), default=0)
        diff_units = observed_units - expected_units
        status = _status_from_diff(expected_units, observed_units)

        total_expected += expected_units
        total_observed += observed_units

        if status == "matched":
            matched_count += 1
        elif status == "partial":
            partial_count += 1
        elif status == "missing":
            missing_count += 1
        elif status == "excess":
            excess_count += 1

        results.append(
            {
                "sku": sku,
                "descripcion": exp.get("descripcion"),
                "expected_units": expected_units,
                "observed_units": observed_units,
                "difference_units": diff_units,
                "status": status,
                "matched": status == "matched",
                "packstructure_match_status": exp.get("packstructure_match_status"),
                "shipping_values": exp.get("shipping_values") or [],
                "expected_pack_levels": {
                    "EA": {
                        "ean": exp.get("ean_pack"),
                        "units": exp.get("qty_ea_pack"),
                    },
                    "IN": {
                        "ean": exp.get("ean_in_pack"),
                        "units": exp.get("qty_inn_pack"),
                    },
                    "CS": {
                        "ean": exp.get("ean_cs_pack"),
                        "units": exp.get("qty_cs_pack"),
                    },
                },
                "observed_breakdown": obs.get(
                    "breakdown_by_level",
                    {
                        "EA": {"count": 0, "units": 0},
                        "IN": {"count": 0, "units": 0},
                        "CS": {"count": 0, "units": 0},
                    },
                ),
                "observed_barcodes": obs.get("observed_barcodes", []),
            }
        )

    for sku, obs in observed_by_sku.items():
        if sku in expected_idx:
            continue
        unexpected_skus.append(
            {
                "sku": sku,
                "descripcion": obs.get("descripcion"),
                "observed_units": _safe_int(obs.get("observed_units"), default=0),
                "status": "unexpected_sku",
                "observed_breakdown": obs.get("breakdown_by_level", {}),
                "observed_barcodes": obs.get("observed_barcodes", []),
            }
        )

    route_value = summary.get("route")
    if route_value is None:
        metadata = summary.get("metadata") or {}
        route_value = metadata.get("ruta")

    has_contamination = len(contaminated_barcodes) > 0
    has_unknown = len(unknown_barcodes) > 0
    has_unexpected = len(unexpected_skus) > 0

    if has_contamination:
        closure_status = "contaminated"
    elif (
        not has_unexpected
        and not has_unknown
        and matched_count == len(expected_idx)
        and len(expected_idx) > 0
    ):
        closure_status = "complete_match"
    elif total_observed == 0:
        closure_status = "no_detection"
    else:
        closure_status = "partial_match"

    return {
        "status": "success",
        "closure_status": closure_status,
        "route": route_value,
        "metadata": summary.get("metadata") or {},
        "products": results,
        "unexpected_skus": unexpected_skus,
        "unknown_barcodes": unknown_barcodes,
        "contaminated_barcodes": contaminated_barcodes,
        "counts": {
            "expected_products": len(expected_idx),
            "matched_products": matched_count,
            "partial_products": partial_count,
            "missing_products": missing_count,
            "excess_products": excess_count,
            "unexpected_skus": len(unexpected_skus),
            "unknown_barcodes": len(unknown_barcodes),
            "contaminated_barcodes": len(contaminated_barcodes),
        },
        "totals": {
            "expected_units": total_expected,
            "observed_units": total_observed,
            "difference_units": total_observed - total_expected,
        },
        "flags": {
            "has_unknown_barcodes": has_unknown,
            "has_unexpected_skus": has_unexpected,
            "has_contaminated_barcodes": has_contamination,
            "all_products_matched": matched_count == len(expected_idx) and len(expected_idx) > 0,
        },
    }


def build_readout_trace(readout_payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    payload = _unwrap_readout_payload(readout_payload or {})
    if not isinstance(payload, dict):
        return {}

    detected_items = collect_detected_items_from_readout(payload)
    debug = payload.get("debug") if isinstance(payload.get("debug"), dict) else {}

    trace = {
        "status": payload.get("status"),
        "backend": payload.get("backend"),
        "total_reported": _safe_int(payload.get("total"), default=len(detected_items)),
        "total_items_extracted": len(detected_items),
        "detected_items": detected_items,
        "debug_available": bool(debug),
    }

    if debug:
        trace["debug_keys"] = list(debug.keys())
        trace["yolo_rois_detected"] = len(debug.get("yolo_rois") or [])

        # Compatibilidad con salida antigua y nueva hybrid.
        hybrid_full = debug.get("hybrid_full_image") or debug.get("dynamsoft_full_image") or {}
        trace["hybrid_full_image_total"] = _safe_int(hybrid_full.get("total"), default=0)
        trace["dynamsoft_full_image_total"] = trace["hybrid_full_image_total"]

        trace["hybrid_label_rois_count"] = len(debug.get("hybrid_label_rois") or [])
        trace["hybrid_barcode_rois_count"] = len(debug.get("hybrid_barcode_rois") or [])

        trace["yolo_pipeline_status"] = (debug.get("yolo_pipeline") or {}).get("status")
        trace["dynamsoft_yolo_rois_count"] = len(debug.get("dynamsoft_yolo_rois") or [])

    return trace


def _extract_target_shipping_from_context(event_context: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(event_context, dict):
        return None

    for key in (
        "target_shipping",
        "detected_shipping",
        "shipping",
        "picking_shipping",
    ):
        value = event_context.get(key)
        if value:
            norm = _norm_shipping(value)
            if norm:
                return norm

    return None


def run_packstructure_closure(
    *,
    summary_payload: Dict[str, Any],
    readout_payload: Optional[Dict[str, Any]] = None,
    detected_barcodes: Optional[List[str]] = None,
    event_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Ejecuta el cierre completo.

    Puedes pasar:
    - readout_payload (wrapped_result.json, result interno o salida directa del híbrido)
    - o detected_barcodes directamente.

    Mejora principal:
    - si event_context trae target_shipping, compara contra el summary filtrado por ese shipping.
    - mantiene un índice global para detectar contaminación:
      barcode existe globalmente, pero no pertenece al shipping actual.
    """
    summary = extract_final_summary(summary_payload)
    if not summary:
        return {
            "status": "error",
            "reason": "summary_payload_invalido",
        }

    event_context = event_context or {}
    target_shipping = _extract_target_shipping_from_context(event_context)

    readout_trace = build_readout_trace(readout_payload)

    if detected_barcodes is None:
        detected_items = collect_detected_items_from_readout(readout_payload or {})
    else:
        detected_items = [{"barcode": _norm_barcode(x), "text": x} for x in detected_barcodes]

    detected_items = [
        x for x in detected_items
        if _norm_barcode(x.get("barcode") or x.get("text"))
    ]

    # 1) Índice global: sirve para saber si un barcode existe en el universo.
    global_barcode_index = build_barcode_index_from_summary(summary)

    # 2) Summary filtrado por shipping: sirve para expected/observed del pedido actual.
    filtered_summary_payload = filter_summary_by_shipping(summary, target_shipping)
    filtered_summary = extract_final_summary(filtered_summary_payload)

    filtered_barcode_index = build_barcode_index_from_summary(filtered_summary)

    # 3) Clasificación por shipping.
    classification = classify_detected_barcodes_for_shipping(
        detected_items,
        global_barcode_index=global_barcode_index,
        filtered_barcode_index=filtered_barcode_index,
        target_shipping=target_shipping,
    )

    # 4) Observado se calcula SOLO contra el índice filtrado.
    observed_payload = aggregate_observed_units(detected_items, filtered_barcode_index)

    # 5) Unknown debe venir desde clasificación global, no desde índice filtrado.
    observed_payload["unknown_barcodes"] = classification.get("unknown_barcodes", [])

    comparison = compare_expected_vs_observed(
        filtered_summary,
        observed_payload,
        contaminated_barcodes=classification.get("contaminated_barcodes", []),
    )

    detected_barcodes_clean = [
        _norm_barcode(x.get("barcode") or x.get("text"))
        for x in detected_items
        if _norm_barcode(x.get("barcode") or x.get("text"))
    ]

    comparison["detected_barcodes"] = detected_barcodes_clean
    comparison["detected_items"] = detected_items
    comparison["barcode_hits"] = observed_payload.get("barcode_hits", {})
    comparison["barcode_index_size"] = len(filtered_barcode_index)
    comparison["barcode_index_global_size"] = len(global_barcode_index)
    comparison["event_context"] = {
        **event_context,
        "target_shipping_normalized": target_shipping,
        "shipping_filter_applied": bool(target_shipping),
        "products_before_shipping_filter": len((summary.get("products") or [])),
        "products_after_shipping_filter": len((filtered_summary.get("products") or [])),
    }
    comparison["readout_trace"] = readout_trace
    comparison["barcode_shipping_classification"] = classification

    return comparison