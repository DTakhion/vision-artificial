# app/packstructure_closure.py
from __future__ import annotations

from collections import defaultdict
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
    # Conservamos solo dígitos para evitar ruido visual/ocr
    return "".join(ch for ch in s if ch.isdigit())


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


def extract_final_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Acepta:
    1) el final_summary directamente
    2) el payload completo generado por run_picking_match(...)
    """
    if not isinstance(payload, dict):
        return {}

    nested = payload.get("summary_pickingVision_fillRate_packStructure")
    if isinstance(nested, dict) and nested.get("products") is not None:
        return nested

    if payload.get("products") is not None and payload.get("metadata") is not None:
        return payload

    return {}


def collect_detected_barcodes_from_readout(readout_payload: Dict[str, Any]) -> List[str]:
    """
    Acepta:
    1) wrapped_result.json completo generado por process_event()
    2) result interno de vision_readout
    """
    if not isinstance(readout_payload, dict):
        return []

    root = readout_payload.get("result")
    if isinstance(root, dict):
        payload = root
    else:
        payload = readout_payload

    out: List[str] = []
    seen = set()

    barcode1d = payload.get("barcode1d") or {}
    confirmed_items = barcode1d.get("confirmed_items") or []

    for item in confirmed_items:
        if not isinstance(item, dict):
            continue
        raw = item.get("text")
        code = _norm_barcode(raw)
        if code:
            out.append(code)
            seen.add(code)

    best = payload.get("best") or {}
    if isinstance(best, dict):
        best_kind = best.get("kind")
        best_text = _norm_barcode(best.get("text"))
        if best_kind == "barcode1d" and best_text and best_text not in seen:
            out.append(best_text)

    return out


def build_barcode_index_from_summary(summary_payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Construye índice:
    barcode -> {
        sku, descripcion, level, units_per_barcode, source_field
    }
    """
    summary = extract_final_summary(summary_payload)
    products = summary.get("products") or []

    idx: Dict[str, Dict[str, Any]] = {}

    for prod in products:
        sku = _norm_code(prod.get("codigo_item"))
        descripcion = prod.get("descripcion_ocr")
        pack = prod.get("packstructure") or {}

        levels = [
            ("EA", pack.get("ean_pack"), pack.get("qty_ea_pack"), "ean_pack"),
            ("IN", pack.get("ean_in_pack"), pack.get("qty_inn_pack"), "ean_in_pack"),
            ("CS", pack.get("ean_cs_pack"), pack.get("qty_cs_pack"), "ean_cs_pack"),
        ]

        for level, raw_barcode, raw_units, source_field in levels:
            barcode = _norm_barcode(raw_barcode)
            units = _safe_int(raw_units, default=0)

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
            }

    return idx


def build_expected_index_from_summary(summary_payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    summary = extract_final_summary(summary_payload)
    products = summary.get("products") or []

    idx: Dict[str, Dict[str, Any]] = {}

    for prod in products:
        sku = _norm_code(prod.get("codigo_item"))
        if not sku:
            continue

        expected_units = _safe_int(prod.get("unidades_ocr"), default=0)
        descripcion = prod.get("descripcion_ocr")
        pack = prod.get("packstructure") or {}

        idx[sku] = {
            "sku": sku,
            "descripcion": descripcion,
            "expected_units": expected_units,
            "packstructure_match_status": pack.get("match_status"),
            "ean_pack": _norm_barcode(pack.get("ean_pack")),
            "ean_in_pack": _norm_barcode(pack.get("ean_in_pack")),
            "ean_cs_pack": _norm_barcode(pack.get("ean_cs_pack")),
            "qty_ea_pack": _safe_int(pack.get("qty_ea_pack"), default=0),
            "qty_inn_pack": _safe_int(pack.get("qty_inn_pack"), default=0),
            "qty_cs_pack": _safe_int(pack.get("qty_cs_pack"), default=0),
        }

    return idx


def aggregate_observed_units(
    detected_barcodes: List[str],
    barcode_index: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    observed_by_sku: Dict[str, Dict[str, Any]] = {}
    unknown_barcodes: List[str] = []
    barcode_hits: Dict[str, int] = defaultdict(int)

    for raw in detected_barcodes:
        barcode = _norm_barcode(raw)
        if not barcode:
            continue

        barcode_hits[barcode] += 1
        meta = barcode_index.get(barcode)

        if not meta:
            unknown_barcodes.append(barcode)
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
            }
        )

        if level not in entry["breakdown_by_level"]:
            entry["breakdown_by_level"][level] = {"count": 0, "units": 0}

        entry["breakdown_by_level"][level]["count"] += 1
        entry["breakdown_by_level"][level]["units"] += units_per_barcode

    return {
        "observed_by_sku": observed_by_sku,
        "unknown_barcodes": unknown_barcodes,
        "barcode_hits": dict(barcode_hits),
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
) -> Dict[str, Any]:
    summary = extract_final_summary(summary_payload)
    expected_idx = build_expected_index_from_summary(summary)
    observed_by_sku = observed_payload.get("observed_by_sku") or {}
    unknown_barcodes = observed_payload.get("unknown_barcodes") or []

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

    if (
        len(unexpected_skus) == 0
        and len(unknown_barcodes) == 0
        and matched_count == len(expected_idx)
    ):
        closure_status = "complete_match"
    elif total_observed == 0:
        closure_status = "no_detection"
    else:
        closure_status = "partial_match"

    return {
        "status": "success",
        "closure_status": closure_status,
        "route": (summary.get("metadata") or {}).get("ruta"),
        "metadata": summary.get("metadata") or {},
        "products": results,
        "unexpected_skus": unexpected_skus,
        "unknown_barcodes": unknown_barcodes,
        "counts": {
            "expected_products": len(expected_idx),
            "matched_products": matched_count,
            "partial_products": partial_count,
            "missing_products": missing_count,
            "excess_products": excess_count,
            "unexpected_skus": len(unexpected_skus),
            "unknown_barcodes": len(unknown_barcodes),
        },
        "totals": {
            "expected_units": total_expected,
            "observed_units": total_observed,
            "difference_units": total_observed - total_expected,
        },
        "flags": {
            "has_unknown_barcodes": len(unknown_barcodes) > 0,
            "has_unexpected_skus": len(unexpected_skus) > 0,
            "all_products_matched": matched_count == len(expected_idx) and len(expected_idx) > 0,
        },
    }


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
    - readout_payload (wrapped_result.json o result interno)
    - o detected_barcodes directamente
    """
    summary = extract_final_summary(summary_payload)
    if not summary:
        return {
            "status": "error",
            "reason": "summary_payload_invalido",
        }

    if detected_barcodes is None:
        detected_barcodes = collect_detected_barcodes_from_readout(readout_payload or {})

    normalized_barcodes = [_norm_barcode(x) for x in (detected_barcodes or []) if _norm_barcode(x)]

    barcode_index = build_barcode_index_from_summary(summary)
    observed_payload = aggregate_observed_units(normalized_barcodes, barcode_index)
    comparison = compare_expected_vs_observed(summary, observed_payload)

    comparison["detected_barcodes"] = normalized_barcodes
    comparison["barcode_index_size"] = len(barcode_index)
    comparison["event_context"] = event_context or {}

    return comparison