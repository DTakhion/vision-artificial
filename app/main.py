# app/main.py
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, List

def safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def safe_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def iter_event_jsons(captures_root: Path) -> Iterable[Path]:
    if not captures_root.exists():
        return []
    return sorted(captures_root.glob("frames_*/events/event_*/event.json"))


def resolve_target_frame(event_payload: Dict[str, Any], event_json_path: Path) -> Optional[Path]:
    # 1) Prioridad máxima: frame canónico del evento
    candidate = event_json_path.parent / "frame.jpg"
    if candidate.exists():
        return candidate

    # 2) Luego, si el event.json ya trae paths.frame
    paths = event_payload.get("paths") or {}
    frame_path = paths.get("frame")
    if frame_path:
        p = Path(frame_path)
        if not p.is_absolute():
            p = Path.cwd() / p
        if p.exists():
            return p

    # 3) Luego, burst.main_frame_path
    burst = event_payload.get("burst") or {}
    main_frame_path = burst.get("main_frame_path")
    if main_frame_path:
        p = Path(main_frame_path)
        if not p.is_absolute():
            p = Path.cwd() / p
        if p.exists():
            return p

    # 4) Fallback razonable
    candidate = event_json_path.parent / "frames" / "frame_02.jpg"
    if candidate.exists():
        return candidate

    return None


def build_readout_cmd(
    image_path: Path,
    *,
    readout_module: str,
    mode: str,
    budget: int,
    barcode_mode: str,
    barcode_budget: int,
    no_ocr: bool,
    no_qr: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        readout_module,
        str(image_path),
    ]

    # Compatibilidad con módulos antiguos y nuevo híbrido
    if readout_module == "utils.vision_readout":
        cmd.extend(
            [
                "--mode",
                mode,
                "--budget",
                str(budget),
                "--barcode_mode",
                barcode_mode,
                "--barcode_budget",
                str(barcode_budget),
            ]
        )
        if no_ocr:
            cmd.append("--no-ocr")
        if no_qr:
            cmd.append("--no-qr")

    return cmd


def summarize_readout(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resume tanto salida legacy (vision_readout) como salida nueva híbrida.
    """
    items = payload.get("items") or []
    if isinstance(items, list) and items:
        texts: List[str] = []
        formats: List[str] = []
        sources: List[str] = []

        for item in items:
            if not isinstance(item, dict):
                continue

            text = item.get("text")
            fmt = item.get("format")
            src = item.get("source")

            if text and text not in texts:
                texts.append(text)
            if fmt and fmt not in formats:
                formats.append(fmt)
            if src and src not in sources:
                sources.append(src)

        best_text = texts[0] if texts else None

        return {
            "status": payload.get("status"),
            "backend": payload.get("backend"),
            "best_kind": "hybrid",
            "best_text": best_text,
            "confirmed_count": len(texts),
            "confirmed_texts": texts,
            "formats": formats,
            "sources": sources,
            "total_reported": payload.get("total"),
        }

    # Fallback legacy
    barcode = payload.get("barcode1d") or {}
    confirmed_items = barcode.get("confirmed_items") or []
    texts = []
    for item in confirmed_items:
        text = item.get("text")
        if text and text not in texts:
            texts.append(text)

    best = payload.get("best") or {}
    best_text = None
    best_kind = None
    if isinstance(best, dict):
        best_text = best.get("text")
        best_kind = best.get("kind")

    return {
        "status": payload.get("status"),
        "backend": payload.get("backend"),
        "best_kind": best_kind,
        "best_text": best_text,
        "confirmed_count": len(confirmed_items),
        "confirmed_texts": texts,
    }


def parse_readout_stdout(stdout: str) -> Optional[Dict[str, Any]]:
    stdout = stdout.strip()
    if not stdout:
        return None

    # 1) Intento directo: JSON puro
    try:
        parsed = json.loads(stdout)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # 2) Intento directo: literal Python puro
    try:
        parsed = ast.literal_eval(stdout)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # 3) Intentar extraer el último bloque JSON válido dentro del stdout
    #    útil cuando el módulo imprime logs antes/después del JSON final
    decoder = json.JSONDecoder()
    candidates: List[Dict[str, Any]] = []

    for i, ch in enumerate(stdout):
        if ch != "{":
            continue
        try:
            obj, end = decoder.raw_decode(stdout[i:])
            if isinstance(obj, dict):
                candidates.append(obj)
        except Exception:
            continue

    if candidates:
        return candidates[-1]

    # 4) Intentar por líneas, por si el JSON viene en una sola línea al final
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue

        try:
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        try:
            parsed = ast.literal_eval(line)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    return None


def process_event(
    event_json_path: Path,
    *,
    readout_module: str,
    mode: str,
    budget: int,
    barcode_mode: str,
    barcode_budget: int,
    no_ocr: bool,
    no_qr: bool,
    overwrite: bool,
) -> bool:
    event_payload = safe_read_json(event_json_path)
    if not event_payload:
        print(f"[WARN] No pude leer JSON: {event_json_path}")
        return False

    ev_dir = event_json_path.parent
    result_path = ev_dir / "readout_result.json"
    marker_path = ev_dir / ".processed"

    if result_path.exists() and not overwrite:
        return False
    if marker_path.exists() and not overwrite:
        return False

    image_path = resolve_target_frame(event_payload, event_json_path)
    if image_path is None or not image_path.exists():
        print(f"[WARN] No encontré frame objetivo para: {event_json_path}")
        return False

    cmd = build_readout_cmd(
        image_path,
        readout_module=readout_module,
        mode=mode,
        budget=budget,
        barcode_mode=barcode_mode,
        barcode_budget=barcode_budget,
        no_ocr=no_ocr,
        no_qr=no_qr,
    )

    print(f"[INFO] Procesando evento: {ev_dir.name}")
    print(f"[INFO] Frame objetivo: {image_path}")
    print(f"[INFO] Ejecutando: {' '.join(cmd)}")

    started_at = time.time()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(Path.cwd()),
    )
    elapsed_ms = int((time.time() - started_at) * 1000)

    if proc.returncode != 0:
        error_payload = {
            "status": "error",
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "elapsed_ms": elapsed_ms,
            "image_path": str(image_path),
            "cmd": cmd,
        }
        safe_write_json(result_path, error_payload)
        marker_path.write_text("error\n", encoding="utf-8")
        print(f"[WARN] Falló readout en {ev_dir.name}")
        return False

    readout_payload = parse_readout_stdout(proc.stdout)

    if readout_payload is None:
        error_payload = {
            "status": "error",
            "reason": "stdout_no_es_json_ni_python_literal",
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "elapsed_ms": elapsed_ms,
            "image_path": str(image_path),
            "cmd": cmd,
        }
        safe_write_json(result_path, error_payload)
        marker_path.write_text("error\n", encoding="utf-8")
        print(f"[WARN] La salida no fue parseable en {ev_dir.name}")
        return False

    wrapped_result = {
        "status": "ok",
        "processed_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_ms": elapsed_ms,
        "image_path": str(image_path),
        "readout_module": readout_module,
        "cmd": cmd,
        "result": readout_payload,
        "summary": summarize_readout(readout_payload),
    }
    safe_write_json(result_path, wrapped_result)
    marker_path.write_text("ok\n", encoding="utf-8")

    event_payload["readout"] = {
        "status": "attempted",
        "image_path": str(image_path),
        "result_path": str(result_path),
        "summary": wrapped_result["summary"],
    }
    safe_write_json(event_json_path, event_payload)

    print(f"[OK] Resultado guardado en: {result_path}")
    return True


def watch_loop(
    captures_root: Path,
    *,
    poll_s: float,
    readout_module: str,
    mode: str,
    budget: int,
    barcode_mode: str,
    barcode_budget: int,
    no_ocr: bool,
    no_qr: bool,
    overwrite: bool,
) -> None:
    print(f"[INFO] Watch mode ON en: {captures_root}")
    print(f"[INFO] Poll cada {poll_s:.1f}s")
    print("[INFO] Ctrl+C para salir")

    while True:
        try:
            event_paths = list(iter_event_jsons(captures_root))
            for event_json_path in event_paths:
                process_event(
                    event_json_path,
                    readout_module=readout_module,
                    mode=mode,
                    budget=budget,
                    barcode_mode=barcode_mode,
                    barcode_budget=barcode_budget,
                    no_ocr=no_ocr,
                    no_qr=no_qr,
                    overwrite=overwrite,
                )
            time.sleep(poll_s)
        except KeyboardInterrupt:
            print("\n[INFO] Watch detenido por usuario.")
            break
        except Exception as e:
            print(f"[WARN] Error en watch loop: {e}")
            time.sleep(poll_s)


# ============================================================
# Helpers consolidación fillRate + packStructure
# ============================================================
def _norm_code(txt: Any) -> str:
    s = "" if txt is None else str(txt)
    s = s.upper().strip()
    s = s.replace(" ", "").replace(".", "").replace(",", "")
    return s


# def _norm_barcode(txt: Any) -> str:
#     s = "" if txt is None else str(txt)
#     return "".join(ch for ch in s if ch.isdigit())

def _norm_barcode(x: Any) -> Optional[str]:
    if x is None:
        return None

    s = str(x).strip()

    # SOLO números
    s = "".join(c for c in s if c.isdigit())

    # 👉 VALIDACIÓN CRÍTICA
    if len(s) not in (8, 12, 13, 14):
        return None

    return s


def _norm_estado_orden_display(txt: Any) -> str:
    s = "" if txt is None else str(txt).strip()
    if not s:
        return ""
    try:
        n = int(float(s))
        return f"{n:03d}"
    except Exception:
        only_digits = "".join(ch for ch in s if ch.isdigit())
        return only_digits.zfill(3) if only_digits else s


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in items:
        value = _norm_barcode(raw)
        if not value:
            continue
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _dedupe_strings_preserve_order(items: List[Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in items:
        value = "" if raw is None else str(raw).strip()
        if not value:
            continue
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out

def _extract_summary_products(summary_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(summary_payload, dict):
        return []

    summary = summary_payload.get("summary_fillRate_packStructure")
    if isinstance(summary, dict) and isinstance(summary.get("products"), list):
        return summary.get("products") or []

    if isinstance(summary_payload.get("products"), list):
        return summary_payload.get("products") or []

    return []


def _build_barcode_to_shipping_index(summary_payload: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Índice:
    barcode -> [
      {
        "sku": ...,
        "shipping": ...,
        "ruta": ...,
        "units_per_barcode": ...,
        "cant_original_total_sku": ...,
        "shipping_expected_units": ...,
      },
      ...
    ]
    """
    products = _extract_summary_products(summary_payload)

    # 1) total esperado por shipping
    shipping_expected_totals: Dict[str, int] = {}
    for prod in products:
        shipping_values = prod.get("shipping_values") or []
        if not isinstance(shipping_values, list):
            shipping_values = []

        # Para MVP: usamos shipping único por SKU
        if len(shipping_values) != 1:
            continue

        shipping = str(shipping_values[0]).strip()
        if not shipping:
            continue

        try:
            qty = int(prod.get("cant_original_total") or 0)
        except Exception:
            qty = 0

        shipping_expected_totals[shipping] = shipping_expected_totals.get(shipping, 0) + qty

    # 2) barcode -> shipping candidates
    idx: Dict[str, List[Dict[str, Any]]] = {}

    for prod in products:
        sku = _norm_code(prod.get("sku"))
        if not sku:
            continue

        shipping_values = prod.get("shipping_values") or []
        ruta_values = prod.get("ruta_values") or []

        shipping = str(shipping_values[0]).strip() if isinstance(shipping_values, list) and len(shipping_values) == 1 else None
        ruta = str(ruta_values[0]).strip() if isinstance(ruta_values, list) and len(ruta_values) == 1 else None

        try:
            cant_original_total = int(prod.get("cant_original_total") or 0)
        except Exception:
            cant_original_total = 0

        barcode_fields = [
            ("EA", prod.get("ean_ea"), prod.get("qty_ea")),
            ("IN", prod.get("ean_in"), prod.get("qty_inn")),
            ("CS", prod.get("ean_cs"), prod.get("qty_cs")),
        ]

        for level, raw_barcode, raw_units in barcode_fields:
            barcode = _norm_barcode(raw_barcode)
            if not barcode:
                continue

            try:
                units_per_barcode = int(raw_units or 0)
            except Exception:
                units_per_barcode = 0

            if units_per_barcode <= 0:
                continue

            idx.setdefault(barcode, []).append(
                {
                    "sku": sku,
                    "shipping": shipping,
                    "ruta": ruta,
                    "level": level,
                    "units_per_barcode": units_per_barcode,
                    "cant_original_total_sku": cant_original_total,
                    "shipping_expected_units": shipping_expected_totals.get(shipping, 0) if shipping else 0,
                }
            )

    return idx


def _resolve_target_shipping_from_barcodes(
    *,
    summary_payload: Dict[str, Any],
    detected_barcodes_all: List[str],
) -> Dict[str, Any]:
    """
    Regla MVP:
    - si un barcode apunta a un único shipping -> resolved_unique
    - si aparecen varios shipping distintos -> ambiguous
    - si no matchea nada -> not_found
    """
    barcode_to_shipping = _build_barcode_to_shipping_index(summary_payload)

    resolved_candidates: List[Dict[str, Any]] = []
    distinct_shippings = set()

    for raw in detected_barcodes_all:
        barcode = _norm_barcode(raw)
        if not barcode:
            continue

        candidates = barcode_to_shipping.get(barcode, [])
        if not candidates:
            continue

        valid = [c for c in candidates if c.get("shipping")]
        unique_shipping_values = sorted({c["shipping"] for c in valid if c.get("shipping")})

        if len(unique_shipping_values) == 1:
            shipping = unique_shipping_values[0]
            distinct_shippings.add(shipping)

            chosen = None
            for c in valid:
                if c.get("shipping") == shipping:
                    chosen = c
                    break

            if chosen:
                resolved_candidates.append(
                    {
                        "barcode": barcode,
                        "shipping": shipping,
                        "ruta": chosen.get("ruta"),
                        "sku": chosen.get("sku"),
                        "level": chosen.get("level"),
                        "units_per_barcode": chosen.get("units_per_barcode"),
                        "shipping_expected_units": chosen.get("shipping_expected_units", 0),
                    }
                )

    if not resolved_candidates:
        return {
            "status": "not_found",
            "target_shipping": None,
            "target_ruta": None,
            "target_sku": None,
            "target_shipping_expected_units": 0,
            "resolved_from_barcode": None,
            "resolved_candidates": [],
        }

    if len(distinct_shippings) > 1:
        return {
            "status": "ambiguous",
            "target_shipping": None,
            "target_ruta": None,
            "target_sku": None,
            "target_shipping_expected_units": 0,
            "resolved_from_barcode": None,
            "resolved_candidates": resolved_candidates,
        }

    chosen = resolved_candidates[0]
    return {
        "status": "resolved_unique",
        "target_shipping": chosen.get("shipping"),
        "target_ruta": chosen.get("ruta"),
        "target_sku": chosen.get("sku"),
        "target_shipping_expected_units": int(chosen.get("shipping_expected_units") or 0),
        "resolved_from_barcode": chosen.get("barcode"),
        "resolved_candidates": resolved_candidates,
    }


def _compute_observed_units_for_target_shipping(
    *,
    summary_payload: Dict[str, Any],
    detected_barcodes_all: List[str],
    target_shipping: Optional[str],
) -> Dict[str, Any]:
    if not target_shipping:
        return {
            "observed_units": 0,
            "barcode_hits_in_target_shipping": {},
            "matched_items": [],
        }

    barcode_to_shipping = _build_barcode_to_shipping_index(summary_payload)

    observed_units = 0
    barcode_hits_in_target_shipping: Dict[str, int] = {}
    matched_items: List[Dict[str, Any]] = []

    for raw in detected_barcodes_all:
        barcode = _norm_barcode(raw)
        if not barcode:
            continue

        candidates = barcode_to_shipping.get(barcode, [])
        matched = None
        for c in candidates:
            if c.get("shipping") == target_shipping:
                matched = c
                break

        if not matched:
            continue

        units_per_barcode = int(matched.get("units_per_barcode") or 0)
        observed_units += units_per_barcode
        barcode_hits_in_target_shipping[barcode] = barcode_hits_in_target_shipping.get(barcode, 0) + 1
        matched_items.append(
            {
                "barcode": barcode,
                "sku": matched.get("sku"),
                "level": matched.get("level"),
                "units_per_barcode": units_per_barcode,
                "shipping": target_shipping,
            }
        )

    return {
        "observed_units": observed_units,
        "barcode_hits_in_target_shipping": barcode_hits_in_target_shipping,
        "matched_items": matched_items,
    }

def _build_fillrate_full_summary(row: Dict[str, Any]) -> Dict[str, Any]:
    excel_full = row.get("excel_full", {}) or {}

    return {
        "Fecha Creación": excel_full.get("Fecha Creación", ""),
        "Hora Creación": excel_full.get("Hora Creación", ""),
        "Fecha Última Actualización": excel_full.get("Fecha Última Actualización", ""),
        "Hora Última Actualización": excel_full.get("Hora Última Actualización", ""),
        "Fecha Agenda": excel_full.get("Fecha Agenda", ""),
        "Hora Agenda": excel_full.get("Hora Agenda", ""),
        "Shipping": excel_full.get("Shipping", ""),
        "Orden Compra": excel_full.get("Orden Compra", ""),
        "Cliente": excel_full.get("Cliente", ""),
        "Dirección": excel_full.get("Dirección", ""),
        "Tipo Orden": excel_full.get("Tipo Orden", ""),
        "Estado Orden": excel_full.get("Estado Orden", ""),
        "Descripción Estado": excel_full.get("Descripción Estado", ""),
        "Ruta": excel_full.get("Ruta", ""),
        "Articulo": excel_full.get("Articulo", ""),
        "Descripcion": excel_full.get("Descripcion", ""),
        "Estado": excel_full.get("Estado", ""),
        "Cant. Original": excel_full.get("Cant. Original", ""),
        "Cant. Trabajada": excel_full.get("Cant. Trabajada", ""),
        "Diferencia": excel_full.get("Diferencia", ""),
    }


# def _load_detected_barcodes_json(path: Path) -> Optional[List[str]]:
#     raw_payload = safe_read_json(path)
#     if raw_payload is None:
#         return None

#     if isinstance(raw_payload, dict):
#         if isinstance(raw_payload.get("detected_barcodes"), list):
#             return _dedupe_preserve_order(raw_payload.get("detected_barcodes"))

#         if isinstance(raw_payload.get("barcodes"), list):
#             return _dedupe_preserve_order(raw_payload.get("barcodes"))

#         # Salida directa híbrida: {"items": [...]}
#         items = raw_payload.get("items")
#         if isinstance(items, list):
#             texts = []
#             for item in items:
#                 if not isinstance(item, dict):
#                     continue
#                 text = item.get("text")
#                 if text:
#                     texts.append(text)
#             return _dedupe_preserve_order(texts)

#         # wrapped_result.json: {"result": {"items": [...]}}
#         result = raw_payload.get("result")
#         if isinstance(result, dict):
#             items = result.get("items")
#             if isinstance(items, list):
#                 texts = []
#                 for item in items:
#                     if not isinstance(item, dict):
#                         continue
#                     text = item.get("text")
#                     if text:
#                         texts.append(text)
#                 return _dedupe_preserve_order(texts)

#         return None

#     if isinstance(raw_payload, list):
#         return _dedupe_preserve_order(raw_payload)

#     return None

def _load_detected_barcodes_json(path: Path) -> Optional[List[str]]:
    raw_payload = safe_read_json(path)
    if raw_payload is None:
        return None

    def _normalize_barcode_list(values: List[Any]) -> List[str]:
        out: List[str] = []
        for x in values:
            code = _norm_barcode(x)
            if code:
                out.append(code)
        return out

    if isinstance(raw_payload, dict):
        if isinstance(raw_payload.get("detected_barcodes"), list):
            return _normalize_barcode_list(raw_payload.get("detected_barcodes"))

        if isinstance(raw_payload.get("barcodes"), list):
            return _normalize_barcode_list(raw_payload.get("barcodes"))

        # Salida directa híbrida: {"items": [...]}
        items = raw_payload.get("items")
        if isinstance(items, list):
            texts = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                text = item.get("text")
                code = _norm_barcode(text)
                if code:
                    texts.append(code)
            return texts

        # wrapped_result.json: {"result": {"items": [...]}}
        result = raw_payload.get("result")
        if isinstance(result, dict):
            items = result.get("items")
            if isinstance(items, list):
                texts = []
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    text = item.get("text")
                    code = _norm_barcode(text)
                    if code:
                        texts.append(code)
                return texts

        return None

    if isinstance(raw_payload, list):
        return _normalize_barcode_list(raw_payload)

    return None

# def _parse_manual_barcodes(manual_barcodes: Optional[str]) -> List[str]:
#     if not manual_barcodes:
#         return []
#     parts = [x.strip() for x in manual_barcodes.split(",")]
#     return _dedupe_preserve_order(parts)

def _parse_manual_barcodes(manual_barcodes: Optional[str]) -> List[str]:
    if not manual_barcodes:
        return []

    parts = [x.strip() for x in manual_barcodes.split(",")]
    return [_norm_barcode(x) for x in parts if _norm_barcode(x)]


def _build_fillrate_index(match_result: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    products = match_result.get("products", []) or []
    for prod in products:
        key = _norm_code(prod.get("sku_fillrate"))
        if key and key not in idx:
            idx[key] = prod
    return idx


def _build_pack_index(match_result: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    products = match_result.get("products", []) or []
    for prod in products:
        key = _norm_code(prod.get("sku_fillrate"))
        if key and key not in idx:
            idx[key] = prod
    return idx


def build_final_picking_summary(
    *,
    fillrate_result: Dict[str, Any],
    fillrate_packstructure_match: Dict[str, Any],
    fillrate_excel: Path,
    packstructure_excel: Path,
) -> Dict[str, Any]:
    fillrate_records = fillrate_result.get("records", []) or []
    products = fillrate_packstructure_match.get("products", []) or []

    total_cant_original = 0
    total_cant_trabajada = 0
    total_diferencia = 0
    matched_count = 0

    consolidated_products: List[Dict[str, Any]] = []

    for prod in products:
        fillrate_matches = prod.get("fillrate_matches", []) or []
        pack = prod.get("packstructure") or {}

        if prod.get("match_status") == "matched":
            matched_count += 1

        sku = prod.get("sku_fillrate")
        descripcion_fillrate = prod.get("descripcion_fillrate")

        sku_cant_original = 0
        sku_cant_trabajada = 0
        sku_diferencia = 0

        shipping_values: List[str] = []
        ruta_values: List[str] = []
        estado_orden_values: List[str] = []
        descripcion_estado_values: List[str] = []

        fillrate_full_list: List[Dict[str, Any]] = []

        for row in fillrate_matches:
            try:
                sku_cant_original += int(row.get("cant_original") or 0)
            except Exception:
                pass
            try:
                sku_cant_trabajada += int(row.get("cant_trabajada") or 0)
            except Exception:
                pass
            try:
                sku_diferencia += int(row.get("diferencia") or 0)
            except Exception:
                pass

            sh = row.get("shipping")
            rt = row.get("ruta")
            eo = _norm_estado_orden_display(row.get("estado_orden"))
            de = row.get("descripcion_estado")

            if sh:
                shipping_values.append(str(sh))
            if rt:
                ruta_values.append(str(rt))
            if eo:
                estado_orden_values.append(eo)
            if de:
                descripcion_estado_values.append(str(de))

            excel_full_summary = _build_fillrate_full_summary(row)
            if excel_full_summary:
                fillrate_full_list.append(excel_full_summary)

        shipping_values = _dedupe_strings_preserve_order(shipping_values)
        ruta_values = _dedupe_strings_preserve_order(ruta_values)
        estado_orden_values = _dedupe_strings_preserve_order(estado_orden_values)
        descripcion_estado_values = _dedupe_strings_preserve_order(descripcion_estado_values)

        total_cant_original += sku_cant_original
        total_cant_trabajada += sku_cant_trabajada
        total_diferencia += sku_diferencia

        packstructure_full = pack.get("packstructure_full", {}) if isinstance(pack, dict) else {}

        consolidated_products.append(
            {
                "sku": sku,
                "descripcion_fillrate": descripcion_fillrate,
                "descripcion_packstructure": prod.get("descripcion_packstructure"),
                "match_status": prod.get("match_status"),
                "fillrate_match_count": prod.get("fillrate_match_count"),
                "packstructure_match_count": prod.get("packstructure_match_count"),
                "shipping_values": shipping_values,
                "ruta_values": ruta_values,
                "estado_orden_values": estado_orden_values,
                "descripcion_estado_values": descripcion_estado_values,
                "cant_original_total": sku_cant_original,
                "cant_trabajada_total": sku_cant_trabajada,
                "diferencia_total": sku_diferencia,
                "ean_ea": prod.get("ean_ea"),
                "ean_in": prod.get("ean_in"),
                "ean_cs": prod.get("ean_cs"),
                "qty_ea": prod.get("qty_ea"),
                "qty_inn": prod.get("qty_inn"),
                "qty_cs": prod.get("qty_cs"),
                "qty_pal": prod.get("qty_pal"),
                "fillrate_full": fillrate_full_list,
                "packstructure_full": packstructure_full,
            }
        )

    return {
        "status": "success",
        "processed_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "document_type": "summary_fillRate_packStructure",
        "inputs": {
            "fillrate_excel": str(fillrate_excel),
            "packstructure_excel": str(packstructure_excel),
        },
        "filter_applied": fillrate_packstructure_match.get("filter_applied", {}),
        "products": consolidated_products,
        "counts": {
            "fillrate_rows_total": len(fillrate_records),
            "filtered_unique_products": len(products),
            "packstructure_matched_products": matched_count,
            "packstructure_not_found_products": sum(
                1 for p in products if p.get("match_status") == "not_found_in_packstructure"
            ),
        },
        "totals": {
            "cant_original_total": total_cant_original,
            "cant_trabajada_total": total_cant_trabajada,
            "diferencia_total": total_diferencia,
        },
    }


# ============================================================
# FillRate + PackStructure
# ============================================================
def run_picking_match(
    *,
    picking_image: Optional[Path] = None,
    picking_excel: Path,
    packstructure_excel: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> int:
    try:
        from utils.vision_excel_picking import (
            load_excel_picking,
            summarize_fillrate_for_packstructure,
            match_fillrate_with_packstructure,
            save_json,
        )
        from utils.vision_excel_packStructure import load_excel_packstructure
    except Exception as e:
        print(f"[ERROR] No pude importar módulos de fillrate/packstructure: {e}")
        return 2

    if not picking_excel.exists():
        print(f"[ERROR] No existe excel FillRate: {picking_excel}")
        return 2

    if packstructure_excel is None:
        print("[ERROR] Debes indicar --packstructure_excel para este flujo")
        return 2

    if not packstructure_excel.exists():
        print(f"[ERROR] No existe excel packStructure: {packstructure_excel}")
        return 2

    print(f"[INFO] Cargando FillRate: {picking_excel}")
    fillrate_result = load_excel_picking(str(picking_excel))

    if fillrate_result.get("status") != "success":
        print("[ERROR] Falló carga de excel FillRate")
        print(json.dumps(fillrate_result, ensure_ascii=False, indent=2))
        return 2

    print(f"[INFO] Cargando PackStructure: {packstructure_excel}")
    packstructure_result = load_excel_packstructure(str(packstructure_excel))

    if packstructure_result.get("status") != "success":
        print("[ERROR] Falló carga de excel PackStructure")
        print(json.dumps(packstructure_result, ensure_ascii=False, indent=2))
        return 2

    fillrate_summary = summarize_fillrate_for_packstructure(
        fillrate_result.get("records", []),
        allowed_estado_orden=["040", "045"],
    )

    fillrate_packstructure_match = match_fillrate_with_packstructure(
        fillrate_result,
        packstructure_result,
        allowed_estado_orden=["040", "045"],
    )

    final_summary = build_final_picking_summary(
        fillrate_result=fillrate_result,
        fillrate_packstructure_match=fillrate_packstructure_match,
        fillrate_excel=picking_excel,
        packstructure_excel=packstructure_excel,
    )

    payload = {
        "status": "success",
        "processed_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode_app": "picking_match",
        "flow_note": "legacy_name_kept_for_compatibility_fillrate_packstructure_only",
        "picking_image": str(picking_image) if picking_image else None,
        "fillrate_excel": str(picking_excel),
        "packstructure_excel": str(packstructure_excel),
        "fillrate_result_summary": {
            "sheet_name": fillrate_result.get("sheet_name"),
            "rows_loaded": fillrate_result.get("rows_loaded"),
            "important_fields_presence": fillrate_result.get("important_fields_presence"),
            "summary_for_packstructure": fillrate_summary,
        },
        "packstructure_result_summary": {
            "sheet_name": packstructure_result.get("sheet_name"),
            "rows_loaded": packstructure_result.get("rows_loaded"),
        },
        "fillrate_packstructure_match_summary": {
            "status": fillrate_packstructure_match.get("status"),
            "filter_applied": fillrate_packstructure_match.get("filter_applied"),
            "counts": fillrate_packstructure_match.get("counts"),
        },
        "summary_fillRate_packStructure": final_summary,
    }

    if output_path is None:
        output_dir = Path("data/picking/summary_fillRate_packStructure")
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = picking_excel.stem
        output_path = output_dir / f"{stem}_summary_fillRate_packStructure.json"

    save_json(payload, str(output_path))

    print(f"[OK] Consolidado guardado en: {output_path}")

    try:
        print(json.dumps(final_summary, ensure_ascii=False, indent=2))
    except UnicodeEncodeError:
        print(json.dumps(final_summary, ensure_ascii=True, indent=2))

    return 0

    #save_json(payload, str(output_path))

    #print(f"[OK] Consolidado guardado en: {output_path}")
    #print(json.dumps(final_summary, ensure_ascii=False, indent=2))
    #summary_text = json.dumps(final_summary, ensure_ascii=False, indent=2)
    #try:
    #    print(summary_text)
    #except UnicodeEncodeError:
    #    print(summary_text.encode("utf-8", errors="replace").decode("utf-8"))
    #return 0

# def _extract_detected_items_for_frontend(
#     readout_payload: Optional[Dict[str, Any]],
#     detected_barcodes: Optional[List[str]],
# ) -> List[Dict[str, Any]]:
#     items_out: List[Dict[str, Any]] = []
#     seen = set()

#     if isinstance(readout_payload, dict):
#         source_payload = readout_payload

#         # Caso wrapped_result.json -> {"result": {...}}
#         if isinstance(readout_payload.get("result"), dict):
#             source_payload = readout_payload["result"]

#         items = source_payload.get("items") or []
#         if isinstance(items, list):
#             for item in items:
#                 if not isinstance(item, dict):
#                     continue

#                 barcode = _norm_barcode(item.get("text"))
#                 if not barcode:
#                     continue

#                 key = (
#                     barcode,
#                     str(item.get("format") or "").strip(),
#                     str(item.get("source") or "").strip(),
#                 )
#                 if key in seen:
#                     continue
#                 seen.add(key)

#                 items_out.append(
#                     {
#                         "barcode": barcode,
#                         "serial": barcode,  # por ahora serial = barcode detectado
#                         "format": item.get("format"),
#                         "source": item.get("source"),
#                     }
#                 )

#     if detected_barcodes:
#         for raw in detected_barcodes:
#             barcode = _norm_barcode(raw)
#             if not barcode:
#                 continue

#             key = (barcode, "", "")
#             already_present = any(x.get("barcode") == barcode for x in items_out)
#             if already_present:
#                 continue

#             if key in seen:
#                 continue
#             seen.add(key)

#             items_out.append(
#                 {
#                     "barcode": barcode,
#                     "serial": barcode,
#                     "format": None,
#                     "source": "detected_barcodes_json",
#                 }
#             )

#     return items_out

def _extract_detected_items_for_frontend(
    readout_payload: Optional[Dict[str, Any]],
    detected_barcodes: Optional[List[str]],
) -> List[Dict[str, Any]]:
    items_out: List[Dict[str, Any]] = []
    seen = set()

    if isinstance(readout_payload, dict):
        source_payload = readout_payload

        # Caso wrapped_result.json -> {"result": {...}}
        if isinstance(readout_payload.get("result"), dict):
            source_payload = readout_payload["result"]

        items = source_payload.get("items") or []
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue

                barcode = _norm_barcode(item.get("text"))
                if not barcode:
                    continue

                bbox = item.get("bbox")
                bbox_key = json.dumps(bbox, ensure_ascii=False, sort_keys=False) if bbox is not None else None

                key = (
                    barcode,
                    str(item.get("format") or "").strip(),
                    str(item.get("source") or "").strip(),
                    bbox_key,
                )
                if key in seen:
                    continue
                seen.add(key)

                items_out.append(
                    {
                        "barcode": barcode,
                        "serial": barcode,
                        "format": item.get("format"),
                        "source": item.get("source"),
                        "bbox": item.get("bbox"),
                    }
                )

    if detected_barcodes:
        for idx, raw in enumerate(detected_barcodes):
            barcode = _norm_barcode(raw)
            if not barcode:
                continue

            # Aquí NO deduplicamos por barcode solo.
            # Si viene una lista con repetidos válidos, deben sobrevivir.
            key = (barcode, "", "", f"manual_idx_{idx}")
            if key in seen:
                continue
            seen.add(key)

            items_out.append(
                {
                    "barcode": barcode,
                    "serial": barcode,
                    "format": None,
                    "source": "detected_barcodes_json",
                    "bbox": None,
                }
            )

    return items_out

def build_frontend_closure_summary(
    *,
    closure_result: Dict[str, Any],
    readout_payload: Optional[Dict[str, Any]] = None,
    detected_barcodes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    products = closure_result.get("products") or []
    detected_items = _extract_detected_items_for_frontend(readout_payload, detected_barcodes)

    ground_truth_products: List[Dict[str, Any]] = []
    matches: List[Dict[str, Any]] = []
    matched_detected_barcodes = set()

    matched_count = 0
    partial_count = 0
    missing_count = 0

    for prod in products:
        if not isinstance(prod, dict):
            continue

        status = prod.get("status")
        if status == "matched":
            matched_count += 1
        elif status == "partial":
            partial_count += 1
        elif status == "missing":
            missing_count += 1

        expected_pack_levels = prod.get("expected_pack_levels") or {}
        observed_breakdown = prod.get("observed_breakdown") or {}
        observed_barcodes = prod.get("observed_barcodes") or []

        ean_ea = ((expected_pack_levels.get("EA") or {}).get("ean")) or ""
        ean_in = ((expected_pack_levels.get("IN") or {}).get("ean")) or ""
        ean_cs = ((expected_pack_levels.get("CS") or {}).get("ean")) or ""

        ground_truth_products.append(
            {
                "sku": prod.get("sku"),
                "descripcion": prod.get("descripcion"),
                "expected_units": prod.get("expected_units"),
                "ean_ea": ean_ea,
                "ean_in": ean_in,
                "ean_cs": ean_cs,
            }
        )

        barcode_to_level: Dict[str, str] = {}
        for level_name, level_info in expected_pack_levels.items():
            if not isinstance(level_info, dict):
                continue
            ean = _norm_barcode(level_info.get("ean"))
            if ean:
                barcode_to_level[ean] = level_name

        for raw_barcode in observed_barcodes:
            barcode = _norm_barcode(raw_barcode)
            if not barcode:
                continue

            matched_detected_barcodes.add(barcode)

            matched_level = barcode_to_level.get(barcode)
            units_per_barcode = None
            observed_count = None
            observed_units = None

            if matched_level:
                level_expected = expected_pack_levels.get(matched_level) or {}
                level_observed = observed_breakdown.get(matched_level) or {}

                units_per_barcode = level_expected.get("units")
                observed_count = level_observed.get("count")
                observed_units = level_observed.get("units")

            matches.append(
                {
                    "detected_barcode": barcode,
                    "matched_sku": prod.get("sku"),
                    "descripcion": prod.get("descripcion"),
                    "matched_level": matched_level,
                    "units_per_barcode": units_per_barcode,
                    "observed_count": observed_count,
                    "observed_units": observed_units,
                    "expected_units": prod.get("expected_units"),
                    "difference_units": prod.get("difference_units"),
                    "product_status": status,
                }
            )

    unmatched_detected_barcodes = [
        item["barcode"]
        for item in detected_items
        if item.get("barcode") and item["barcode"] not in matched_detected_barcodes
    ]

    unmatched_detected_barcodes = _dedupe_preserve_order(unmatched_detected_barcodes)

    return {
        "closure_status": closure_result.get("closure_status"),
        "route": closure_result.get("route"),
        "totals": {
            "products_expected": len(ground_truth_products),
            "products_matched": matched_count,
            "products_partial": partial_count,
            "products_missing": missing_count,
            "detected_barcodes_count": len(detected_items),
            "matched_detected_barcodes_count": len(matched_detected_barcodes),
            "unmatched_detected_barcodes_count": len(unmatched_detected_barcodes),
        },
        "ground_truth_products": ground_truth_products,
        "detected_items": detected_items,
        "matches": matches,
        "unmatched_detected_barcodes": unmatched_detected_barcodes,
    }

DEFAULT_CLOSURE_SUMMARY_JSON = Path(
    "data/picking/summary_fillRate_packStructure/fillrate_latest_summary_fillRate_packStructure.json"
)

DEFAULT_CLOSURE_OUTPUT_JSON = Path(
    "data/closure/fillrate_latest_summary_fillRate_packStructure_closure_result.json"
)

DEFAULT_CLOSURE_ITERATIVE_OUTPUT_JSON = Path(
    "data/closure/fillrate_latest_summary_fillRate_packStructure_closure_iterative_result.json"
)

# ============================================================
# Closure Match (summary + readout -> cierre final)
# ============================================================
def run_closure_match(
    *,
    summary_json: Path,
    readout_json: Optional[Path] = None,
    detected_barcodes_json: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> int:
    try:
        from app.packstructure_closure import run_packstructure_closure
    except Exception as e:
        print(f"[ERROR] No pude importar app.packstructure_closure: {e}")
        return 2

    if not summary_json.exists():
        print(f"[ERROR] No existe summary_json: {summary_json}")
        return 2

    summary_payload = safe_read_json(summary_json)
    if not summary_payload:
        print(f"[ERROR] No pude leer summary_json: {summary_json}")
        return 2

    readout_payload: Optional[Dict[str, Any]] = None
    detected_barcodes: Optional[List[str]] = None

    if readout_json is not None:
        if not readout_json.exists():
            print(f"[ERROR] No existe readout_json: {readout_json}")
            return 2

        readout_payload = safe_read_json(readout_json)
        if not readout_payload:
            print(f"[ERROR] No pude leer readout_json: {readout_json}")
            return 2

    if detected_barcodes_json is not None:
        if not detected_barcodes_json.exists():
            print(f"[ERROR] No existe detected_barcodes_json: {detected_barcodes_json}")
            return 2

        detected_barcodes = _load_detected_barcodes_json(detected_barcodes_json)
        if detected_barcodes is None:
            print("[ERROR] detected_barcodes_json debe ser lista o contener 'detected_barcodes' / 'barcodes' como lista")
            return 2

    print(f"[INFO] Ejecutando closure match")
    print(f"[INFO] summary_json: {summary_json}")
    if readout_json:
        print(f"[INFO] readout_json: {readout_json}")
    if detected_barcodes_json:
        print(f"[INFO] detected_barcodes_json: {detected_barcodes_json}")

    closure_result = run_packstructure_closure(
        summary_payload=summary_payload,
        readout_payload=readout_payload,
        detected_barcodes=detected_barcodes,
        event_context={
            "summary_json": str(summary_json),
            "readout_json": str(readout_json) if readout_json else None,
            "detected_barcodes_json": str(detected_barcodes_json) if detected_barcodes_json else None,
        },
    )
    
    frontend_summary = build_frontend_closure_summary(
        closure_result=closure_result,
        readout_payload=readout_payload,
        detected_barcodes=detected_barcodes,
    )

    payload = {
        "status": "success" if closure_result.get("status") == "success" else "error",
        "processed_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary_json": str(summary_json),
        "readout_json": str(readout_json) if readout_json else None,
        "detected_barcodes_json": str(detected_barcodes_json) if detected_barcodes_json else None,
        "frontend_summary": frontend_summary,
        "closure_result": closure_result,
    }

    if output_path is None:
        output_dir = Path("data/closure")
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = summary_json.stem
        output_path = output_dir / f"{stem}_closure_result.json"

    safe_write_json(output_path, payload)

    print(f"[OK] Cierre guardado en: {output_path}")
    #print(json.dumps(closure_result, ensure_ascii=False, indent=2))

    try:
        print(json.dumps(closure_result, ensure_ascii=False, indent=2))
    except UnicodeEncodeError:
        print(json.dumps(closure_result, ensure_ascii=True, indent=2))

    return 0 if closure_result.get("status") == "success" else 2


# ============================================================
# Closure Iterative (acumulado de barcodes por sesión)
# ============================================================
# class ClosureSession:
#     def __init__(
#         self,
#         summary_payload: Dict[str, Any],
#         state_path: Optional[Path] = None,
#         summary_json_path: Optional[Path] = None,
#     ):
#         self.summary_payload = summary_payload
#         self.state_path = state_path
#         self.summary_json_path = summary_json_path

#         now_local = time.strftime("%Y-%m-%d %H:%M:%S")

#         self.session_id: str = f"closure_session_{time.strftime('%Y%m%d_%H%M%S')}"
#         self.session_status: str = "open"
#         self.created_at_local: str = now_local
#         self.updated_at_local: str = now_local
#         self.closed_at_local: Optional[str] = None

#         self.detected_barcodes: List[str] = []
#         self.captures_processed: List[Dict[str, Any]] = []
#         self.last_readout_json: Optional[str] = None
#         self.last_detected_barcodes_json: Optional[str] = None
#         self.last_manual_barcodes: List[str] = []
#         self.last_closure_status: Optional[str] = None

#         if self.state_path and self.state_path.exists():
#             state_payload = safe_read_json(self.state_path)
#             if isinstance(state_payload, dict):
#                 self._load_from_payload(state_payload)

#     def _load_from_payload(self, payload: Dict[str, Any]) -> None:
#         self.session_id = str(payload.get("session_id") or self.session_id)
#         self.session_status = str(payload.get("session_status") or "open")
#         self.created_at_local = str(payload.get("created_at_local") or self.created_at_local)
#         self.updated_at_local = str(payload.get("updated_at_local") or self.updated_at_local)

#         closed_at = payload.get("closed_at_local")
#         self.closed_at_local = str(closed_at) if closed_at else None

#         prev = payload.get("detected_barcodes", [])
#         if isinstance(prev, list):
#             self.detected_barcodes = _dedupe_preserve_order(prev)

#         captures = payload.get("captures_processed", [])
#         if isinstance(captures, list):
#             clean_captures: List[Dict[str, Any]] = []
#             for item in captures:
#                 if isinstance(item, dict):
#                     clean_captures.append(item)
#             self.captures_processed = clean_captures

#         self.last_readout_json = payload.get("last_readout_json")
#         self.last_detected_barcodes_json = payload.get("last_detected_barcodes_json")

#         last_manual = payload.get("last_manual_barcodes", [])
#         if isinstance(last_manual, list):
#             self.last_manual_barcodes = _dedupe_preserve_order(last_manual)

#         self.last_closure_status = payload.get("last_closure_status")

#     def get_barcodes(self) -> List[str]:
#         return list(self.detected_barcodes)

#     def get_captures_processed(self) -> List[Dict[str, Any]]:
#         return list(self.captures_processed)

#     def update(self, new_barcodes: List[str]) -> None:
#         merged = list(self.detected_barcodes) + list(new_barcodes)
#         self.detected_barcodes = _dedupe_preserve_order(merged)
#         self.updated_at_local = time.strftime("%Y-%m-%d %H:%M:%S")

#     def register_capture(
#         self,
#         *,
#         readout_json: Optional[Path] = None,
#         detected_barcodes_json: Optional[Path] = None,
#         manual_barcodes: Optional[List[str]] = None,
#         new_barcodes: Optional[List[str]] = None,
#     ) -> None:
#         ts_local = time.strftime("%Y-%m-%d %H:%M:%S")

#         readout_json_str = str(readout_json) if readout_json else None
#         detected_barcodes_json_str = str(detected_barcodes_json) if detected_barcodes_json else None
#         manual_list = _dedupe_preserve_order(manual_barcodes or [])
#         new_barcodes_list = _dedupe_preserve_order(new_barcodes or [])

#         event_dir = None
#         if readout_json is not None:
#             try:
#                 event_dir = str(readout_json.parent)
#             except Exception:
#                 event_dir = None

#         entry = {
#             "processed_at_local": ts_local,
#             "readout_json": readout_json_str,
#             "event_dir": event_dir,
#             "detected_barcodes_json": detected_barcodes_json_str,
#             "manual_barcodes": manual_list,
#             "new_barcodes": new_barcodes_list,
#             "new_barcodes_count": len(new_barcodes_list),
#         }

#         duplicate = False
#         if readout_json_str:
#             for prev in self.captures_processed:
#                 if not isinstance(prev, dict):
#                     continue
#                 if prev.get("readout_json") == readout_json_str:
#                     duplicate = True
#                     break

#         if not duplicate:
#             self.captures_processed.append(entry)

#         self.last_readout_json = readout_json_str
#         self.last_detected_barcodes_json = detected_barcodes_json_str
#         self.last_manual_barcodes = manual_list
#         self.updated_at_local = ts_local

#     def set_closure_status(self, closure_status: Optional[str]) -> None:
#         self.last_closure_status = closure_status
#         self.updated_at_local = time.strftime("%Y-%m-%d %H:%M:%S")

#     def mark_closed(self) -> None:
#         now_local = time.strftime("%Y-%m-%d %H:%M:%S")
#         self.session_status = "closed"
#         self.closed_at_local = now_local
#         self.updated_at_local = now_local

#     def reopen(self) -> None:
#         self.session_status = "open"
#         self.closed_at_local = None
#         self.updated_at_local = time.strftime("%Y-%m-%d %H:%M:%S")

#     def save(self) -> None:
#         if not self.state_path:
#             return

#         payload = {
#             "status": "success",
#             "processed_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
#             "session_id": self.session_id,
#             "session_status": self.session_status,
#             "created_at_local": self.created_at_local,
#             "updated_at_local": self.updated_at_local,
#             "closed_at_local": self.closed_at_local,
#             "summary_json": str(self.summary_json_path) if self.summary_json_path else None,
#             "last_readout_json": self.last_readout_json,
#             "last_detected_barcodes_json": self.last_detected_barcodes_json,
#             "last_manual_barcodes": self.last_manual_barcodes,
#             "last_closure_status": self.last_closure_status,
#             "detected_barcodes": self.get_barcodes(),
#             "captures_processed": self.get_captures_processed(),
#             "counts": {
#                 "detected_barcodes": len(self.detected_barcodes),
#                 "captures_processed": len(self.captures_processed),
#             },
#         }
#         safe_write_json(self.state_path, payload)

#     def clear(self) -> None:
#         now_local = time.strftime("%Y-%m-%d %H:%M:%S")

#         self.session_id = f"closure_session_{time.strftime('%Y%m%d_%H%M%S')}"
#         self.session_status = "open"
#         self.created_at_local = now_local
#         self.updated_at_local = now_local
#         self.closed_at_local = None

#         self.detected_barcodes = []
#         self.captures_processed = []
#         self.last_readout_json = None
#         self.last_detected_barcodes_json = None
#         self.last_manual_barcodes = []
#         self.last_closure_status = None

#         self.save()

class ClosureSession:
    def __init__(
        self,
        summary_payload: Dict[str, Any],
        state_path: Optional[Path] = None,
        summary_json_path: Optional[Path] = None,
    ):
        self.summary_payload = summary_payload
        self.state_path = state_path
        self.summary_json_path = summary_json_path

        now_local = time.strftime("%Y-%m-%d %H:%M:%S")

        self.session_id: str = f"closure_session_{time.strftime('%Y%m%d_%H%M%S')}"
        self.session_status: str = "open"
        self.created_at_local: str = now_local
        self.updated_at_local: str = now_local
        self.closed_at_local: Optional[str] = None

        # NUEVO: acumulado real con repetidos
        self.detected_barcodes_all: List[str] = []
        # apoyo/debug
        self.detected_barcodes_unique: List[str] = []

        self.captures_processed: List[Dict[str, Any]] = []
        self.last_readout_json: Optional[str] = None
        self.last_detected_barcodes_json: Optional[str] = None
        self.last_manual_barcodes: List[str] = []
        self.last_closure_status: Optional[str] = None

        # NUEVO: resolución de shipping objetivo
        self.shipping_resolution_status: str = "pending"
        self.target_shipping: Optional[str] = None
        self.target_ruta: Optional[str] = None
        self.target_sku: Optional[str] = None
        self.target_shipping_expected_units: int = 0
        self.target_shipping_observed_units: int = 0
        self.resolved_from_barcode: Optional[str] = None

        if self.state_path and self.state_path.exists():
            state_payload = safe_read_json(self.state_path)
            if isinstance(state_payload, dict):
                self._load_from_payload(state_payload)

    def _load_from_payload(self, payload: Dict[str, Any]) -> None:
        self.session_id = str(payload.get("session_id") or self.session_id)
        self.session_status = str(payload.get("session_status") or "open")
        self.created_at_local = str(payload.get("created_at_local") or self.created_at_local)
        self.updated_at_local = str(payload.get("updated_at_local") or self.updated_at_local)

        closed_at = payload.get("closed_at_local")
        self.closed_at_local = str(closed_at) if closed_at else None

        prev_all = payload.get("detected_barcodes_all", [])
        if isinstance(prev_all, list):
            self.detected_barcodes_all = [_norm_barcode(x) for x in prev_all if _norm_barcode(x)]

        # backward compatibility
        if not self.detected_barcodes_all:
            prev = payload.get("detected_barcodes", [])
            if isinstance(prev, list):
                self.detected_barcodes_all = [_norm_barcode(x) for x in prev if _norm_barcode(x)]

        self.detected_barcodes_unique = _dedupe_preserve_order(self.detected_barcodes_all)

        captures = payload.get("captures_processed", [])
        if isinstance(captures, list):
            clean_captures: List[Dict[str, Any]] = []
            for item in captures:
                if isinstance(item, dict):
                    clean_captures.append(item)
            self.captures_processed = clean_captures

        self.last_readout_json = payload.get("last_readout_json")
        self.last_detected_barcodes_json = payload.get("last_detected_barcodes_json")

        last_manual = payload.get("last_manual_barcodes", [])
        if isinstance(last_manual, list):
            self.last_manual_barcodes = _dedupe_preserve_order(last_manual)

        self.last_closure_status = payload.get("last_closure_status")

        self.shipping_resolution_status = str(payload.get("shipping_resolution_status") or "pending")
        self.target_shipping = payload.get("target_shipping")
        self.target_ruta = payload.get("target_ruta")
        self.target_sku = payload.get("target_sku")
        self.target_shipping_expected_units = int(payload.get("target_shipping_expected_units") or 0)
        self.target_shipping_observed_units = int(payload.get("target_shipping_observed_units") or 0)
        self.resolved_from_barcode = payload.get("resolved_from_barcode")

    def get_barcodes(self) -> List[str]:
        return list(self.detected_barcodes_all)

    def get_unique_barcodes(self) -> List[str]:
        return list(self.detected_barcodes_unique)

    def get_captures_processed(self) -> List[Dict[str, Any]]:
        return list(self.captures_processed)

    def update(self, new_barcodes: List[str]) -> None:
        clean_new = [_norm_barcode(x) for x in new_barcodes if _norm_barcode(x)]
        self.detected_barcodes_all.extend(clean_new)
        self.detected_barcodes_unique = _dedupe_preserve_order(self.detected_barcodes_all)
        self.updated_at_local = time.strftime("%Y-%m-%d %H:%M:%S")

    def register_capture(
        self,
        *,
        readout_json: Optional[Path] = None,
        detected_barcodes_json: Optional[Path] = None,
        manual_barcodes: Optional[List[str]] = None,
        new_barcodes: Optional[List[str]] = None,
    ) -> None:
        ts_local = time.strftime("%Y-%m-%d %H:%M:%S")

        readout_json_str = str(readout_json) if readout_json else None
        detected_barcodes_json_str = str(detected_barcodes_json) if detected_barcodes_json else None
        manual_list = [_norm_barcode(x) for x in (manual_barcodes or []) if _norm_barcode(x)]
        new_barcodes_list = [_norm_barcode(x) for x in (new_barcodes or []) if _norm_barcode(x)]

        event_dir = None
        if readout_json is not None:
            try:
                event_dir = str(readout_json.parent)
            except Exception:
                event_dir = None

        entry = {
            "processed_at_local": ts_local,
            "readout_json": readout_json_str,
            "event_dir": event_dir,
            "detected_barcodes_json": detected_barcodes_json_str,
            "manual_barcodes": manual_list,
            "new_barcodes": new_barcodes_list,
            "new_barcodes_count": len(new_barcodes_list),
            "new_barcodes_unique_count": len(_dedupe_preserve_order(new_barcodes_list)),
        }

        duplicate = False
        if readout_json_str:
            for prev in self.captures_processed:
                if not isinstance(prev, dict):
                    continue
                if prev.get("readout_json") == readout_json_str:
                    duplicate = True
                    break

        if not duplicate:
            self.captures_processed.append(entry)

        self.last_readout_json = readout_json_str
        self.last_detected_barcodes_json = detected_barcodes_json_str
        self.last_manual_barcodes = _dedupe_preserve_order(manual_list)
        self.updated_at_local = ts_local

    def set_closure_status(self, closure_status: Optional[str]) -> None:
        self.last_closure_status = closure_status
        self.updated_at_local = time.strftime("%Y-%m-%d %H:%M:%S")

    def set_target_shipping_resolution(
        self,
        *,
        status: str,
        target_shipping: Optional[str] = None,
        target_ruta: Optional[str] = None,
        target_sku: Optional[str] = None,
        target_shipping_expected_units: int = 0,
        target_shipping_observed_units: int = 0,
        resolved_from_barcode: Optional[str] = None,
    ) -> None:
        self.shipping_resolution_status = status
        self.target_shipping = target_shipping
        self.target_ruta = target_ruta
        self.target_sku = target_sku
        self.target_shipping_expected_units = int(target_shipping_expected_units or 0)
        self.target_shipping_observed_units = int(target_shipping_observed_units or 0)
        self.resolved_from_barcode = resolved_from_barcode
        self.updated_at_local = time.strftime("%Y-%m-%d %H:%M:%S")

    def mark_closed(self) -> None:
        now_local = time.strftime("%Y-%m-%d %H:%M:%S")
        self.session_status = "closed"
        self.closed_at_local = now_local
        self.updated_at_local = now_local

    def reopen(self) -> None:
        self.session_status = "open"
        self.closed_at_local = None
        self.updated_at_local = time.strftime("%Y-%m-%d %H:%M:%S")

    def save(self) -> None:
        if not self.state_path:
            return

        payload = {
            "status": "success",
            "processed_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
            "session_id": self.session_id,
            "session_status": self.session_status,
            "created_at_local": self.created_at_local,
            "updated_at_local": self.updated_at_local,
            "closed_at_local": self.closed_at_local,
            "summary_json": str(self.summary_json_path) if self.summary_json_path else None,
            "last_readout_json": self.last_readout_json,
            "last_detected_barcodes_json": self.last_detected_barcodes_json,
            "last_manual_barcodes": self.last_manual_barcodes,
            "last_closure_status": self.last_closure_status,
            "shipping_resolution_status": self.shipping_resolution_status,
            "target_shipping": self.target_shipping,
            "target_ruta": self.target_ruta,
            "target_sku": self.target_sku,
            "target_shipping_expected_units": self.target_shipping_expected_units,
            "target_shipping_observed_units": self.target_shipping_observed_units,
            "resolved_from_barcode": self.resolved_from_barcode,
            "detected_barcodes_all": self.get_barcodes(),
            "detected_barcodes_unique": self.get_unique_barcodes(),
            "captures_processed": self.get_captures_processed(),
            "counts": {
                "detected_barcodes_all": len(self.detected_barcodes_all),
                "detected_barcodes_unique": len(self.detected_barcodes_unique),
                "captures_processed": len(self.captures_processed),
            },
        }
        safe_write_json(self.state_path, payload)

    def clear(self) -> None:
        now_local = time.strftime("%Y-%m-%d %H:%M:%S")

        self.session_id = f"closure_session_{time.strftime('%Y%m%d_%H%M%S')}"
        self.session_status = "open"
        self.created_at_local = now_local
        self.updated_at_local = now_local
        self.closed_at_local = None

        self.detected_barcodes_all = []
        self.detected_barcodes_unique = []
        self.captures_processed = []
        self.last_readout_json = None
        self.last_detected_barcodes_json = None
        self.last_manual_barcodes = []
        self.last_closure_status = None

        self.shipping_resolution_status = "pending"
        self.target_shipping = None
        self.target_ruta = None
        self.target_sku = None
        self.target_shipping_expected_units = 0
        self.target_shipping_observed_units = 0
        self.resolved_from_barcode = None

        self.save()


# def run_closure_iterative(
#     *,
#     summary_json: Path,
#     readout_json: Optional[Path] = None,
#     detected_barcodes_json: Optional[Path] = None,
#     manual_barcodes: Optional[str] = None,
#     session_state_json: Optional[Path] = None,
#     output_path: Optional[Path] = None,
#     reset_session: bool = False,
# ) -> int:
#     try:
#         from app.packstructure_closure import (
#             run_packstructure_closure,
#             collect_detected_barcodes_from_readout,
#         )
#     except Exception as e:
#         print(f"[ERROR] No pude importar app.packstructure_closure: {e}")
#         return 2

#     if not summary_json.exists():
#         print(f"[ERROR] No existe summary_json: {summary_json}")
#         return 2

#     summary_payload = safe_read_json(summary_json)
#     if not summary_payload:
#         print(f"[ERROR] No pude leer summary_json: {summary_json}")
#         return 2
    
#     session = ClosureSession(
#         summary_payload=summary_payload,
#         state_path=session_state_json,
#         summary_json_path=summary_json,
#     )

#     #session = ClosureSession(summary_payload=summary_payload, state_path=session_state_json)

#     if reset_session:
#         print("[INFO] Reiniciando sesión iterativa")
#         session.clear()

#     readout_payload: Optional[Dict[str, Any]] = None
#     new_barcodes: List[str] = []

#     if readout_json is not None:
#         if not readout_json.exists():
#             print(f"[ERROR] No existe readout_json: {readout_json}")
#             return 2

#         readout_payload = safe_read_json(readout_json)
#         if not readout_payload:
#             print(f"[ERROR] No pude leer readout_json: {readout_json}")
#             return 2

#         readout_barcodes = collect_detected_barcodes_from_readout(readout_payload)
#         new_barcodes.extend(readout_barcodes)

#     if detected_barcodes_json is not None:
#         if not detected_barcodes_json.exists():
#             print(f"[ERROR] No existe detected_barcodes_json: {detected_barcodes_json}")
#             return 2

#         json_barcodes = _load_detected_barcodes_json(detected_barcodes_json)
#         if json_barcodes is None:
#             print("[ERROR] detected_barcodes_json debe ser lista o contener 'detected_barcodes' / 'barcodes' como lista")
#             return 2

#         new_barcodes.extend(json_barcodes)

#     manual_list = _parse_manual_barcodes(manual_barcodes)
#     new_barcodes.extend(manual_list)

#     new_barcodes = _dedupe_preserve_order(new_barcodes)
#     previous_barcodes = session.get_barcodes()

#     session.update(new_barcodes)
#     session.save()

#     accumulated_barcodes = session.get_barcodes()

#     print("[INFO] Ejecutando closure iterativo")
#     print(f"[INFO] summary_json: {summary_json}")
#     if readout_json:
#         print(f"[INFO] readout_json: {readout_json}")
#     if detected_barcodes_json:
#         print(f"[INFO] detected_barcodes_json: {detected_barcodes_json}")
#     if manual_list:
#         print(f"[INFO] manual_barcodes: {manual_list}")
#     if session_state_json:
#         print(f"[INFO] session_state_json: {session_state_json}")

#     closure_result = run_packstructure_closure(
#         summary_payload=summary_payload,
#         detected_barcodes=accumulated_barcodes,
#         event_context={
#             "mode": "closure_iterative",
#             "summary_json": str(summary_json),
#             "readout_json": str(readout_json) if readout_json else None,
#             "detected_barcodes_json": str(detected_barcodes_json) if detected_barcodes_json else None,
#             "manual_barcodes": manual_list,
#             "session_state_json": str(session_state_json) if session_state_json else None,
#             "reset_session": reset_session,
#         },
#     )
    
#     frontend_summary = build_frontend_closure_summary(
#         closure_result=closure_result,
#         readout_payload=readout_payload,
#         detected_barcodes=accumulated_barcodes,
#     )

#     missing_products = [
#         p for p in (closure_result.get("products") or [])
#         if p.get("status") == "missing"
#     ]
#     partial_products = [
#         p for p in (closure_result.get("products") or [])
#         if p.get("status") == "partial"
#     ]

#     payload = {
#         "status": "success" if closure_result.get("status") == "success" else "error",
#         "processed_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
#         "mode_app": "closure_iterative",
#         "summary_json": str(summary_json),
#         "readout_json": str(readout_json) if readout_json else None,
#         "detected_barcodes_json": str(detected_barcodes_json) if detected_barcodes_json else None,
#         "manual_barcodes": manual_list,
#         "session_state_json": str(session_state_json) if session_state_json else None,
#         "frontend_summary": frontend_summary,
#         "session": {
#             "previous_barcodes": previous_barcodes,
#             "new_barcodes": new_barcodes,
#             "accumulated_barcodes": accumulated_barcodes,
#             "counts": {
#                 "previous": len(previous_barcodes),
#                 "new": len(new_barcodes),
#                 "accumulated": len(accumulated_barcodes),
#             },
#         },
#         "operator_feedback": {
#             "missing_products_count": len(missing_products),
#             "partial_products_count": len(partial_products),
#             "missing_products": [
#                 {
#                     "sku": p.get("sku"),
#                     "descripcion": p.get("descripcion"),
#                     "expected_units": p.get("expected_units"),
#                     "observed_units": p.get("observed_units"),
#                     "status": p.get("status"),
#                 }
#                 for p in missing_products
#             ],
#             "partial_products": [
#                 {
#                     "sku": p.get("sku"),
#                     "descripcion": p.get("descripcion"),
#                     "expected_units": p.get("expected_units"),
#                     "observed_units": p.get("observed_units"),
#                     "status": p.get("status"),
#                 }
#                 for p in partial_products
#             ],
#         },
#         "closure_result": closure_result,
#     }

#     if output_path is None:
#         output_dir = Path("data/closure")
#         output_dir.mkdir(parents=True, exist_ok=True)
#         stem = summary_json.stem
#         output_path = output_dir / f"{stem}_closure_iterative_result.json"

#     safe_write_json(output_path, payload)

#     print(f"[OK] Cierre iterativo guardado en: {output_path}")
#     print(json.dumps(payload, ensure_ascii=False, indent=2))
#     return 0 if closure_result.get("status") == "success" else 2

# def run_closure_iterative(
#     *,
#     summary_json: Path,
#     readout_json: Optional[Path] = None,
#     detected_barcodes_json: Optional[Path] = None,
#     manual_barcodes: Optional[str] = None,
#     session_state_json: Optional[Path] = None,
#     output_path: Optional[Path] = None,
#     reset_session: bool = False,
# ) -> int:
#     try:
#         from app.packstructure_closure import (
#             run_packstructure_closure,
#             collect_detected_barcodes_from_readout,
#         )
#     except Exception as e:
#         print(f"[ERROR] No pude importar app.packstructure_closure: {e}")
#         return 2

#     if not summary_json.exists():
#         print(f"[ERROR] No existe summary_json: {summary_json}")
#         return 2

#     summary_payload = safe_read_json(summary_json)
#     if not summary_payload:
#         print(f"[ERROR] No pude leer summary_json: {summary_json}")
#         return 2

#     session = ClosureSession(
#         summary_payload=summary_payload,
#         state_path=session_state_json,
#         summary_json_path=summary_json,
#     )

#     if reset_session:
#         print("[INFO] Reiniciando sesión iterativa")
#         session.clear()

#     readout_payload: Optional[Dict[str, Any]] = None
#     new_barcodes: List[str] = []

#     if readout_json is not None:
#         if not readout_json.exists():
#             print(f"[ERROR] No existe readout_json: {readout_json}")
#             return 2

#         readout_payload = safe_read_json(readout_json)
#         if not readout_payload:
#             print(f"[ERROR] No pude leer readout_json: {readout_json}")
#             return 2

#         readout_barcodes = collect_detected_barcodes_from_readout(readout_payload)
#         new_barcodes.extend(readout_barcodes)

#     if detected_barcodes_json is not None:
#         if not detected_barcodes_json.exists():
#             print(f"[ERROR] No existe detected_barcodes_json: {detected_barcodes_json}")
#             return 2

#         json_barcodes = _load_detected_barcodes_json(detected_barcodes_json)
#         if json_barcodes is None:
#             print("[ERROR] detected_barcodes_json debe ser lista o contener 'detected_barcodes' / 'barcodes' como lista")
#             return 2

#         new_barcodes.extend(json_barcodes)

#     manual_list = _parse_manual_barcodes(manual_barcodes)
#     new_barcodes.extend(manual_list)

#     new_barcodes = _dedupe_preserve_order(new_barcodes)
#     previous_barcodes = session.get_barcodes()

#     session.update(new_barcodes)

#     # Registrar la evidencia consumida en esta iteración
#     session.register_capture(
#         readout_json=readout_json,
#         detected_barcodes_json=detected_barcodes_json,
#         manual_barcodes=manual_list,
#         new_barcodes=new_barcodes,
#     )

#     accumulated_barcodes = session.get_barcodes()

#     print("[INFO] Ejecutando closure iterativo")
#     print(f"[INFO] summary_json: {summary_json}")
#     if readout_json:
#         print(f"[INFO] readout_json: {readout_json}")
#     if detected_barcodes_json:
#         print(f"[INFO] detected_barcodes_json: {detected_barcodes_json}")
#     if manual_list:
#         print(f"[INFO] manual_barcodes: {manual_list}")
#     if session_state_json:
#         print(f"[INFO] session_state_json: {session_state_json}")

#     closure_result = run_packstructure_closure(
#         summary_payload=summary_payload,
#         detected_barcodes=accumulated_barcodes,
#         event_context={
#             "mode": "closure_iterative",
#             "summary_json": str(summary_json),
#             "readout_json": str(readout_json) if readout_json else None,
#             "detected_barcodes_json": str(detected_barcodes_json) if detected_barcodes_json else None,
#             "manual_barcodes": manual_list,
#             "session_state_json": str(session_state_json) if session_state_json else None,
#             "reset_session": reset_session,
#         },
#     )

#     session.set_closure_status(closure_result.get("closure_status"))
#     session.save()

#     frontend_summary = build_frontend_closure_summary(
#         closure_result=closure_result,
#         readout_payload=readout_payload,
#         detected_barcodes=accumulated_barcodes,
#     )

#     missing_products = [
#         p for p in (closure_result.get("products") or [])
#         if p.get("status") == "missing"
#     ]
#     partial_products = [
#         p for p in (closure_result.get("products") or [])
#         if p.get("status") == "partial"
#     ]

#     payload = {
#         "status": "success" if closure_result.get("status") == "success" else "error",
#         "processed_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
#         "mode_app": "closure_iterative",
#         "summary_json": str(summary_json),
#         "readout_json": str(readout_json) if readout_json else None,
#         "detected_barcodes_json": str(detected_barcodes_json) if detected_barcodes_json else None,
#         "manual_barcodes": manual_list,
#         "session_state_json": str(session_state_json) if session_state_json else None,
#         "frontend_summary": frontend_summary,
#         "session": {
#             "session_id": session.session_id,
#             "session_status": session.session_status,
#             "created_at_local": session.created_at_local,
#             "updated_at_local": session.updated_at_local,
#             "closed_at_local": session.closed_at_local,
#             "last_closure_status": session.last_closure_status,
#             "last_readout_json": session.last_readout_json,
#             "last_detected_barcodes_json": session.last_detected_barcodes_json,
#             "last_manual_barcodes": session.last_manual_barcodes,
#             "captures_processed": session.get_captures_processed(),
#             "previous_barcodes": previous_barcodes,
#             "new_barcodes": new_barcodes,
#             "accumulated_barcodes": accumulated_barcodes,
#             "counts": {
#                 "previous": len(previous_barcodes),
#                 "new": len(new_barcodes),
#                 "accumulated": len(accumulated_barcodes),
#                 "captures_processed": len(session.get_captures_processed()),
#             },
#         },
#         "operator_feedback": {
#             "missing_products_count": len(missing_products),
#             "partial_products_count": len(partial_products),
#             "missing_products": [
#                 {
#                     "sku": p.get("sku"),
#                     "descripcion": p.get("descripcion"),
#                     "expected_units": p.get("expected_units"),
#                     "observed_units": p.get("observed_units"),
#                     "status": p.get("status"),
#                 }
#                 for p in missing_products
#             ],
#             "partial_products": [
#                 {
#                     "sku": p.get("sku"),
#                     "descripcion": p.get("descripcion"),
#                     "expected_units": p.get("expected_units"),
#                     "observed_units": p.get("observed_units"),
#                     "status": p.get("status"),
#                 }
#                 for p in partial_products
#             ],
#         },
#         "closure_result": closure_result,
#     }

#     if output_path is None:
#         output_dir = Path("data/closure")
#         output_dir.mkdir(parents=True, exist_ok=True)
#         stem = summary_json.stem
#         output_path = output_dir / f"{stem}_closure_iterative_result.json"

#     safe_write_json(output_path, payload)

#     print(f"[OK] Cierre iterativo guardado en: {output_path}")
#     print(json.dumps(payload, ensure_ascii=False, indent=2))
#     return 0 if closure_result.get("status") == "success" else 2

def run_closure_iterative(
    *,
    summary_json: Path,
    readout_json: Optional[Path] = None,
    detected_barcodes_json: Optional[Path] = None,
    manual_barcodes: Optional[str] = None,
    session_state_json: Optional[Path] = None,
    output_path: Optional[Path] = None,
    reset_session: bool = False,
) -> int:
    try:
        from app.packstructure_closure import (
            run_packstructure_closure,
            collect_detected_barcodes_from_readout,
        )
    except Exception as e:
        print(f"[ERROR] No pude importar app.packstructure_closure: {e}")
        return 2

    if not summary_json.exists():
        print(f"[ERROR] No existe summary_json: {summary_json}")
        return 2

    summary_payload = safe_read_json(summary_json)
    if not summary_payload:
        print(f"[ERROR] No pude leer summary_json: {summary_json}")
        return 2

    session = ClosureSession(
        summary_payload=summary_payload,
        state_path=session_state_json,
        summary_json_path=summary_json,
    )

    if reset_session:
        print("[INFO] Reiniciando sesión iterativa")
        session.clear()

    readout_payload: Optional[Dict[str, Any]] = None
    new_barcodes: List[str] = []

    if readout_json is not None:
        if not readout_json.exists():
            print(f"[ERROR] No existe readout_json: {readout_json}")
            return 2

        readout_payload = safe_read_json(readout_json)
        if not readout_payload:
            print(f"[ERROR] No pude leer readout_json: {readout_json}")
            return 2

        readout_barcodes = collect_detected_barcodes_from_readout(readout_payload)
        new_barcodes.extend(readout_barcodes)

    if detected_barcodes_json is not None:
        if not detected_barcodes_json.exists():
            print(f"[ERROR] No existe detected_barcodes_json: {detected_barcodes_json}")
            return 2

        json_barcodes = _load_detected_barcodes_json(detected_barcodes_json)
        if json_barcodes is None:
            print("[ERROR] detected_barcodes_json debe ser lista o contener 'detected_barcodes' / 'barcodes' como lista")
            return 2

        new_barcodes.extend(json_barcodes)

    manual_list = _parse_manual_barcodes(manual_barcodes)
    new_barcodes.extend(manual_list)

    # OJO: acá ya NO deduplicamos
    new_barcodes = [_norm_barcode(x) for x in new_barcodes if _norm_barcode(x)]
    previous_barcodes_all = session.get_barcodes()
    previous_barcodes_unique = session.get_unique_barcodes()

    session.update(new_barcodes)

    session.register_capture(
        readout_json=readout_json,
        detected_barcodes_json=detected_barcodes_json,
        manual_barcodes=manual_list,
        new_barcodes=new_barcodes,
    )

    accumulated_barcodes_all = session.get_barcodes()
    accumulated_barcodes_unique = session.get_unique_barcodes()

    print("[INFO] Ejecutando closure iterativo")
    print(f"[INFO] summary_json: {summary_json}")
    if readout_json:
        print(f"[INFO] readout_json: {readout_json}")
    if detected_barcodes_json:
        print(f"[INFO] detected_barcodes_json: {detected_barcodes_json}")
    if manual_list:
        print(f"[INFO] manual_barcodes: {manual_list}")
    if session_state_json:
        print(f"[INFO] session_state_json: {session_state_json}")

    # 1) Resolver shipping objetivo
    shipping_resolution = _resolve_target_shipping_from_barcodes(
        summary_payload=summary_payload,
        detected_barcodes_all=accumulated_barcodes_all,
    )

    # Si la sesión ya traía shipping resuelto, lo respetamos
    # if session.target_shipping and session.shipping_resolution_status == "resolved_unique":
    #     shipping_resolution = {
    #         "status": "resolved_unique",
    #         "target_shipping": session.target_shipping,
    #         "target_ruta": session.target_ruta,
    #         "target_sku": session.target_sku,
    #         "target_shipping_expected_units": session.target_shipping_expected_units,
    #         "resolved_from_barcode": session.resolved_from_barcode,
    #         "resolved_candidates": [],
    #     }
    
    # Si la sesión ya traía shipping resuelto, lo respetamos
    if session.target_shipping and session.shipping_resolution_status in (
        "resolved_unique",
        "resolved_from_picking_sheet",
    ):
        shipping_resolution = {
            "status": session.shipping_resolution_status,
            "target_shipping": session.target_shipping,
            "target_ruta": session.target_ruta,
            "target_sku": session.target_sku,
            "target_shipping_expected_units": session.target_shipping_expected_units,
            "resolved_from_barcode": session.resolved_from_barcode,
            "resolved_candidates": [],
        }

    # 2) Calcular observado real para esa orden
    target_observation = _compute_observed_units_for_target_shipping(
        summary_payload=summary_payload,
        detected_barcodes_all=accumulated_barcodes_all,
        target_shipping=shipping_resolution.get("target_shipping"),
    )

    session.set_target_shipping_resolution(
        status=shipping_resolution.get("status") or "pending",
        target_shipping=shipping_resolution.get("target_shipping"),
        target_ruta=shipping_resolution.get("target_ruta"),
        target_sku=shipping_resolution.get("target_sku"),
        target_shipping_expected_units=int(shipping_resolution.get("target_shipping_expected_units") or 0),
        target_shipping_observed_units=int(target_observation.get("observed_units") or 0),
        resolved_from_barcode=shipping_resolution.get("resolved_from_barcode"),
    )

    # 3) Mantener closure_result actual como apoyo de comparación SKU-level
    closure_result = run_packstructure_closure(
        summary_payload=summary_payload,
        detected_barcodes=accumulated_barcodes_all,
        event_context={
            "mode": "closure_iterative",
            "summary_json": str(summary_json),
            "readout_json": str(readout_json) if readout_json else None,
            "detected_barcodes_json": str(detected_barcodes_json) if detected_barcodes_json else None,
            "manual_barcodes": manual_list,
            "session_state_json": str(session_state_json) if session_state_json else None,
            "reset_session": reset_session,
            "target_shipping": session.target_shipping,
        },
    )

    # 4) Criterio real de cierre de sesión
    # target_shipping_complete = (
    #     session.shipping_resolution_status == "resolved_unique"
    #     and session.target_shipping is not None
    #     and session.target_shipping_expected_units > 0
    #     and session.target_shipping_observed_units >= session.target_shipping_expected_units
    # )
    target_shipping_complete = (
        session.shipping_resolution_status in ("resolved_unique", "resolved_from_picking_sheet")
        and session.target_shipping is not None
        and session.target_shipping_expected_units > 0
        and session.target_shipping_observed_units >= session.target_shipping_expected_units
    )

    if target_shipping_complete:
        session.mark_closed()

    effective_closure_status = closure_result.get("closure_status")
    if target_shipping_complete:
        effective_closure_status = "target_shipping_complete"
    elif session.shipping_resolution_status == "ambiguous":
        effective_closure_status = "target_shipping_ambiguous"
    elif session.shipping_resolution_status == "not_found":
        effective_closure_status = "target_shipping_not_found"

    session.set_closure_status(effective_closure_status)
    session.save()

    frontend_summary = build_frontend_closure_summary(
        closure_result=closure_result,
        readout_payload=readout_payload,
        detected_barcodes=accumulated_barcodes_all,
    )

    missing_products = [
        p for p in (closure_result.get("products") or [])
        if p.get("status") == "missing"
    ]
    partial_products = [
        p for p in (closure_result.get("products") or [])
        if p.get("status") == "partial"
    ]

    payload = {
        "status": "success" if closure_result.get("status") == "success" else "error",
        "processed_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode_app": "closure_iterative",
        "summary_json": str(summary_json),
        "readout_json": str(readout_json) if readout_json else None,
        "detected_barcodes_json": str(detected_barcodes_json) if detected_barcodes_json else None,
        "manual_barcodes": manual_list,
        "session_state_json": str(session_state_json) if session_state_json else None,
        "frontend_summary": frontend_summary,
        "session": {
            "session_id": session.session_id,
            "session_status": session.session_status,
            "created_at_local": session.created_at_local,
            "updated_at_local": session.updated_at_local,
            "closed_at_local": session.closed_at_local,
            "last_closure_status": session.last_closure_status,
            "last_readout_json": session.last_readout_json,
            "last_detected_barcodes_json": session.last_detected_barcodes_json,
            "last_manual_barcodes": session.last_manual_barcodes,
            "captures_processed": session.get_captures_processed(),
            "shipping_resolution_status": session.shipping_resolution_status,
            "target_shipping": session.target_shipping,
            "target_ruta": session.target_ruta,
            "target_sku": session.target_sku,
            "target_shipping_expected_units": session.target_shipping_expected_units,
            "target_shipping_observed_units": session.target_shipping_observed_units,
            "resolved_from_barcode": session.resolved_from_barcode,
            "previous_barcodes_all": previous_barcodes_all,
            "previous_barcodes_unique": previous_barcodes_unique,
            "new_barcodes": new_barcodes,
            "accumulated_barcodes_all": accumulated_barcodes_all,
            "accumulated_barcodes_unique": accumulated_barcodes_unique,
            "counts": {
                "previous_all": len(previous_barcodes_all),
                "previous_unique": len(previous_barcodes_unique),
                "new": len(new_barcodes),
                "accumulated_all": len(accumulated_barcodes_all),
                "accumulated_unique": len(accumulated_barcodes_unique),
                "captures_processed": len(session.get_captures_processed()),
            },
        },
        #"target_shipping_resolution": shipping_resolution,
        #"target_shipping_observation": target_observation,
        #"target_shipping_complete": target_shipping_complete,
        #"effective_closure_status": effective_closure_status,
        
        "target_shipping_resolution": shipping_resolution,
        "target_shipping_observation": target_observation,
        "target_shipping_complete": target_shipping_complete,
        "effective_closure_status": effective_closure_status,
        "target_shipping_summary": {
            "status": effective_closure_status,
            "shipping": session.target_shipping,
            "ruta": session.target_ruta,
            "sku": session.target_sku,
            "expected_units": session.target_shipping_expected_units,
            "observed_units": session.target_shipping_observed_units,
            "difference_units": (
                session.target_shipping_expected_units
                - session.target_shipping_observed_units
            ),
            "matched_items": target_observation.get("matched_items", []),
            "barcode_hits_in_target_shipping": target_observation.get(
                "barcode_hits_in_target_shipping",
                {},
            ),
            "resolution_status": session.shipping_resolution_status,
            "resolved_from_barcode": session.resolved_from_barcode,
        },
                
        "operator_feedback": {
            "missing_products_count": len(missing_products),
            "partial_products_count": len(partial_products),
            "missing_products": [
                {
                    "sku": p.get("sku"),
                    "descripcion": p.get("descripcion"),
                    "expected_units": p.get("expected_units"),
                    "observed_units": p.get("observed_units"),
                    "status": p.get("status"),
                }
                for p in missing_products
            ],
            "partial_products": [
                {
                    "sku": p.get("sku"),
                    "descripcion": p.get("descripcion"),
                    "expected_units": p.get("expected_units"),
                    "observed_units": p.get("observed_units"),
                    "status": p.get("status"),
                }
                for p in partial_products
            ],
        },
        "closure_result": closure_result,
    }

    if output_path is None:
        output_dir = Path("data/closure")
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = summary_json.stem
        output_path = output_dir / f"{stem}_closure_iterative_result.json"

    safe_write_json(output_path, payload)

    print(f"[OK] Cierre iterativo guardado en: {output_path}")
    #print(json.dumps(payload, ensure_ascii=False, indent=2))

    try:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    except UnicodeEncodeError:
        print(json.dumps(payload, ensure_ascii=True, indent=2))

    return 0 if closure_result.get("status") == "success" else 2

# def run_picking_shipping(
#     *,
#     picking_image: Path,
#     summary_json: Optional[Path] = None,
#     session_state_json: Optional[Path] = None,
#     output_path: Optional[Path] = None,
#     env_file: str = ".env",
#     reset_session: bool = False,
# ) -> int:
#     try:
#         from utils.vision_picking import (
#             read_picking_shipping,
#             PickingSheetConfig,
#             build_picking_debug_images,
#             save_debug_images,
#         )
#         #from utils.vision_picking import read_picking_shipping, PickingSheetConfig
#     except Exception as e:
#         print(f"[ERROR] No pude importar utils.vision_picking: {e}")
#         return 2
    
#     import cv2

#     if not picking_image.exists():
#         print(f"[ERROR] No existe picking_image: {picking_image}")
#         return 2

#     summary_payload: Dict[str, Any] = {}
#     if summary_json is not None:
#         if not summary_json.exists():
#             print(f"[ERROR] No existe summary_json: {summary_json}")
#             return 2

#         summary_payload = safe_read_json(summary_json) or {}
#         if not summary_payload:
#             print(f"[ERROR] No pude leer summary_json: {summary_json}")
#             return 2

#     img = cv2.imread(str(picking_image))
#     if img is None:
#         print(f"[ERROR] No se pudo cargar la imagen: {picking_image}")
#         return 2

#     cfg = PickingSheetConfig(
#         dynamsoft_env_file=env_file,
#     )

#     print("[INFO] Ejecutando lectura de hoja de picking")
#     print(f"[INFO] picking_image: {picking_image}")
#     if summary_json:
#         print(f"[INFO] summary_json: {summary_json}")
#     if session_state_json:
#         print(f"[INFO] session_state_json: {session_state_json}")

#     picking_result = read_picking_shipping(img, cfg=cfg)

#     session = ClosureSession(
#         summary_payload=summary_payload,
#         state_path=session_state_json,
#         summary_json_path=summary_json,
#     )

#     if reset_session:
#         print("[INFO] Reiniciando sesión antes de registrar hoja de picking")
#         session.clear()

#     detected_shipping = picking_result.get("shipping")
#     detected_source = picking_result.get("source")

#     resolution_status = "not_found"
#     if detected_shipping:
#         resolution_status = "resolved_from_picking_sheet"

#     session.set_target_shipping_resolution(
#         status=resolution_status,
#         target_shipping=detected_shipping,
#         target_ruta=None,
#         target_sku=None,
#         target_shipping_expected_units=0,
#         target_shipping_observed_units=0,
#         resolved_from_barcode=None,
#     )
#     session.save()

#     payload = {
#         "status": "success" if detected_shipping else "not_found",
#         "processed_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
#         "mode_app": "picking_shipping",
#         "picking_image": str(picking_image),
#         "summary_json": str(summary_json) if summary_json else None,
#         "session_state_json": str(session_state_json) if session_state_json else None,
#         "shipping_result": picking_result,
#         "session": {
#             "session_id": session.session_id,
#             "session_status": session.session_status,
#             "shipping_resolution_status": session.shipping_resolution_status,
#             "target_shipping": session.target_shipping,
#             "target_ruta": session.target_ruta,
#             "target_sku": session.target_sku,
#             "target_shipping_expected_units": session.target_shipping_expected_units,
#             "target_shipping_observed_units": session.target_shipping_observed_units,
#         },
#         "event_context": {
#             "detected_shipping": detected_shipping,
#             "detected_source": detected_source,
#             "sheet_found": picking_result.get("sheet_found"),
#         },
#     }

#     if output_path is None:
#         output_dir = Path("data/picking")
#         output_dir.mkdir(parents=True, exist_ok=True)
#         output_path = output_dir / f"{picking_image.stem}_picking_shipping.json"

#     safe_write_json(output_path, payload)

#     print(f"[OK] Resultado picking_shipping guardado en: {output_path}")
#     print(json.dumps(payload, ensure_ascii=False, indent=2))
#     return 0 if detected_shipping else 2

def run_picking_shipping(
    *,
    picking_image: Path,
    summary_json: Optional[Path] = None,
    session_state_json: Optional[Path] = None,
    output_path: Optional[Path] = None,
    env_file: str = ".env",
    reset_session: bool = False,
) -> int:
    try:
        from utils.vision_picking import (
            read_picking_shipping,
            PickingSheetConfig,
            build_picking_debug_images,
            save_debug_images,
        )
    except Exception as e:
        print(f"[ERROR] No pude importar utils.vision_picking: {e}")
        return 2

    import cv2

    if not picking_image.exists():
        print(f"[ERROR] No existe picking_image: {picking_image}")
        return 2

    summary_payload: Dict[str, Any] = {}
    if summary_json is not None:
        if not summary_json.exists():
            print(f"[ERROR] No existe summary_json: {summary_json}")
            return 2

        summary_payload = safe_read_json(summary_json) or {}
        if not summary_payload:
            print(f"[ERROR] No pude leer summary_json: {summary_json}")
            return 2

    img = cv2.imread(str(picking_image))
    if img is None:
        print(f"[ERROR] No se pudo cargar la imagen: {picking_image}")
        return 2

    cfg = PickingSheetConfig(
        dynamsoft_env_file=env_file,
    )

    print("[INFO] Ejecutando lectura de hoja de picking")
    print(f"[INFO] picking_image: {picking_image}")
    if summary_json:
        print(f"[INFO] summary_json: {summary_json}")
    if session_state_json:
        print(f"[INFO] session_state_json: {session_state_json}")

    picking_result = read_picking_shipping(img, cfg=cfg)

    # ------------------------------------------------------------
    # NUEVO: guardar imágenes debug del picking
    # ------------------------------------------------------------
    debug_dir = Path("results/picking/debug")
    debug_images = build_picking_debug_images(img, picking_result, cfg=cfg)
    saved_debug_paths = save_debug_images(
        debug_images,
        str(debug_dir),
        stem=picking_image.stem,
    )

    debug_images_map: Dict[str, str] = {}
    for p in saved_debug_paths:
        pp = Path(p)
        debug_images_map[pp.name] = str(pp)

    input_image_path = debug_images_map.get(f"{picking_image.stem}_input.png")
    input_labeled_path = debug_images_map.get(f"{picking_image.stem}_input_labeled.png")
    input_detected_path = debug_images_map.get(f"{picking_image.stem}_input_detected.png")
    input_hybrid_path = debug_images_map.get(f"{picking_image.stem}_input_hybrid.png")

    session = ClosureSession(
        summary_payload=summary_payload,
        state_path=session_state_json,
        summary_json_path=summary_json,
    )

    if reset_session:
        print("[INFO] Reiniciando sesión antes de registrar hoja de picking")
        session.clear()

    detected_shipping = picking_result.get("shipping")
    detected_source = picking_result.get("source")

    resolution_status = "not_found"
    if detected_shipping:
        resolution_status = "resolved_from_picking_sheet"

    session.set_target_shipping_resolution(
        status=resolution_status,
        target_shipping=detected_shipping,
        target_ruta=None,
        target_sku=None,
        target_shipping_expected_units=0,
        target_shipping_observed_units=0,
        resolved_from_barcode=None,
    )
    session.save()

    payload = {
        "status": "success" if detected_shipping else "not_found",
        "processed_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode_app": "picking_shipping",
        "picking_image": str(picking_image),
        "summary_json": str(summary_json) if summary_json else None,
        "session_state_json": str(session_state_json) if session_state_json else None,
        "shipping_result": picking_result,
        "picking_debug": {
            "debug_dir": str(debug_dir),
            "images": {
                "input": input_image_path,
                "input_labeled": input_labeled_path,
                "input_detected": input_detected_path,
                "input_hybrid": input_hybrid_path,
            },
            "saved_paths": saved_debug_paths,
        },
        "session": {
            "session_id": session.session_id,
            "session_status": session.session_status,
            "shipping_resolution_status": session.shipping_resolution_status,
            "target_shipping": session.target_shipping,
            "target_ruta": session.target_ruta,
            "target_sku": session.target_sku,
            "target_shipping_expected_units": session.target_shipping_expected_units,
            "target_shipping_observed_units": session.target_shipping_observed_units,
        },
        "event_context": {
            "detected_shipping": detected_shipping,
            "detected_source": detected_source,
            "sheet_found": picking_result.get("sheet_found"),
        },
    }

    if output_path is None:
        output_dir = Path("data/picking")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{picking_image.stem}_picking_shipping.json"

    safe_write_json(output_path, payload)

    #print(f"[OK] Resultado picking_shipping guardado en: {output_path}")
    print(f"[OK] Resultado picking_shipping guardado en: {output_path}")

    if saved_debug_paths:
        print("[OK] Debug picking guardado en:")
        for s in saved_debug_paths:
            print(f" - {s}")

    try:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    except UnicodeEncodeError:
        print(json.dumps(payload, ensure_ascii=True, indent=2))

    return 0 if detected_shipping else 2

def run_picking_flow(
    *,
    picking_image: Path,
    picking_excel: Optional[Path] = None,
    packstructure_excel: Optional[Path] = None,
    summary_json: Optional[Path] = None,
    readout_json: Optional[Path] = None,
    detected_barcodes_json: Optional[Path] = None,
    manual_barcodes: Optional[str] = None,
    session_state_json: Optional[Path] = None,
    closure_output: Optional[Path] = None,
    env_file: str = ".env",
    reset_session: bool = False,
) -> int:
    """
    Flujo completo:
    1) Construye o reutiliza summary FillRate + PackStructure
    2) Lee hoja de picking y resuelve shipping objetivo
    3) Ejecuta cierre iterativo usando ese shipping ya fijado en sesión
    """

    if not picking_image.exists():
        print(f"[ERROR] No existe picking_image: {picking_image}")
        return 2

    if (
        readout_json is None
        and detected_barcodes_json is None
        and not manual_barcodes
    ):
        print("[ERROR] Para picking_flow debes indicar --readout_json, --detected_barcodes_json o --manual_barcodes")
        return 2

    # ------------------------------------------------------------
    # Paso 1: obtener summary_json
    # ------------------------------------------------------------
    effective_summary_json: Path

    if summary_json is not None:
        effective_summary_json = summary_json
        if not effective_summary_json.exists():
            print(f"[ERROR] No existe summary_json: {effective_summary_json}")
            return 2
        print(f"[INFO] Reutilizando summary_json existente: {effective_summary_json}")
    else:
        if picking_excel is None:
            print("[ERROR] Para picking_flow sin --summary_json debes indicar --picking_excel")
            return 2
        if packstructure_excel is None:
            print("[ERROR] Para picking_flow sin --summary_json debes indicar --packstructure_excel")
            return 2
        if not picking_excel.exists():
            print(f"[ERROR] No existe picking_excel: {picking_excel}")
            return 2
        if not packstructure_excel.exists():
            print(f"[ERROR] No existe packstructure_excel: {packstructure_excel}")
            return 2

        summary_dir = Path("data/picking/summary_fillRate_packStructure")
        summary_dir.mkdir(parents=True, exist_ok=True)
        effective_summary_json = summary_dir / f"{picking_excel.stem}_summary_fillRate_packStructure.json"

        rc_summary = run_picking_match(
            picking_image=picking_image,
            picking_excel=picking_excel,
            packstructure_excel=packstructure_excel,
            output_path=effective_summary_json,
        )
        if rc_summary != 0:
            print("[ERROR] Falló run_picking_match dentro de picking_flow")
            return rc_summary

    # ------------------------------------------------------------
    # Paso 2: resolver shipping desde hoja de picking
    # ------------------------------------------------------------
    shipping_output_dir = Path("data/picking")
    shipping_output_dir.mkdir(parents=True, exist_ok=True)
    shipping_output = shipping_output_dir / f"{picking_image.stem}_picking_shipping.json"

    #rc_shipping = run_picking_shipping(
    #    picking_image=picking_image,
    #    summary_json=effective_summary_json,
    #    session_state_json=session_state_json,
    #    output_path=shipping_output,
    #    env_file=env_file,
    #    reset_session=reset_session,
    #)
    #if rc_shipping != 0:
    #    print("[ERROR] Falló run_picking_shipping dentro de picking_flow")
    #    return rc_shipping

    rc_shipping = run_picking_shipping(
        picking_image=picking_image,
        summary_json=effective_summary_json,
        session_state_json=session_state_json,
        output_path=shipping_output,
        env_file=env_file,
        reset_session=reset_session,
    )

    if rc_shipping != 0:
        print("[WARN] run_picking_shipping no resolvió shipping; continúo igual con closure_iterative")

    # ------------------------------------------------------------
    # Paso 3: ejecutar cierre iterativo usando la misma sesión
    # ------------------------------------------------------------
    rc_closure = run_closure_iterative(
        summary_json=effective_summary_json,
        readout_json=readout_json,
        detected_barcodes_json=detected_barcodes_json,
        manual_barcodes=manual_barcodes,
        session_state_json=session_state_json,
        output_path=closure_output,
        reset_session=False,  # ya se aplicó antes, si correspondía
    )
    if rc_closure != 0:
        print("[ERROR] Falló run_closure_iterative dentro de picking_flow")
        return rc_closure

    print("[OK] Flujo completo picking_flow ejecutado correctamente")
    print(f"[OK] summary_json: {effective_summary_json}")
    print(f"[OK] picking_shipping_json: {shipping_output}")
    if closure_output is not None:
        print(f"[OK] closure_output: {closure_output}")

    return 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Procesa automáticamente eventos capturados y ejecuta vision_readout, flujo picking+excel, closure_match o closure_iterative."
    )

    ap.add_argument(
        "--mode_app",
        type=str,
        choices=["readout", "picking_match", "closure_match", "closure_iterative", "picking_shipping", "picking_flow"],
        default="readout",
        help="Flujo de aplicación principal.",
    )

    ap.add_argument(
        "--captures_root",
        type=str,
        default="data/captures/opencv",
        help="Raíz donde capture_opencv guarda las sesiones.",
    )
    ap.add_argument(
        "--poll_s",
        type=float,
        default=2.0,
        help="Intervalo de polling en segundos para detectar eventos nuevos.",
    )
    ap.add_argument(
        "--once",
        action="store_true",
        help="Procesa una pasada y termina. Sin este flag, queda en modo watch.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocesa aunque ya exista resultado.",
    )

    ap.add_argument("--mode", type=str, default="retry")
    ap.add_argument("--budget", type=int, default=6500)
    ap.add_argument("--barcode_mode", type=str, default="collect_plus")
    ap.add_argument("--barcode_budget", type=int, default=6000)
    ap.add_argument("--no_ocr", action="store_true", default=True)
    ap.add_argument("--no_qr", action="store_true", default=True)

    ap.add_argument("--picking_image", type=str, default=None, help="Ruta a imagen de hoja de picking.")
    ap.add_argument("--picking_excel", type=str, default=None, help="Ruta a excel asociado al picking.")
    ap.add_argument("--packstructure_excel", type=str, default=None, help="Ruta a excel PackStructure.")
    ap.add_argument("--picking_output", type=str, default=None, help="Ruta de salida JSON consolidado.")

    ap.add_argument("--summary_json", type=str, default=None, help="Ruta al JSON consolidado de picking/fillRate/PackStructure.")
    ap.add_argument("--readout_json", type=str, default=None, help="Ruta al readout_result.json del evento.")
    
    ap.add_argument(
        "--readout_module",
        type=str,
        default="utils.vision_readout_hybrid",
        help="Módulo de readout a ejecutar. Ej: utils.vision_readout_hybrid o utils.vision_readout",
    )
    
    ap.add_argument("--detected_barcodes_json", type=str, default=None, help="Ruta a JSON con lista de barcodes detectados.")
    ap.add_argument("--closure_output", type=str, default=None, help="Ruta de salida JSON del cierre final.")

    ap.add_argument(
        "--manual_barcodes",
        type=str,
        default=None,
        help="Lista manual de barcodes separada por coma. Ej: 123,456,789",
    )
    ap.add_argument(
        "--session_state_json",
        type=str,
        default=None,
        help="Ruta a JSON de estado acumulado para cierre iterativo.",
    )
    ap.add_argument(
        "--reset_session",
        action="store_true",
        help="Reinicia la sesión acumulada antes de agregar nuevos barcodes.",
    )

    args = ap.parse_args()

    if args.mode_app == "picking_match":
        if not args.picking_excel:
            print("[ERROR] Para --mode_app picking_match debes indicar --picking_excel")
            raise SystemExit(2)

        rc = run_picking_match(
            picking_image=Path(args.picking_image) if args.picking_image else None,
            picking_excel=Path(args.picking_excel),
            packstructure_excel=Path(args.packstructure_excel) if args.packstructure_excel else None,
            output_path=Path(args.picking_output) if args.picking_output else None,
        )
        raise SystemExit(rc)

    if args.mode_app == "closure_match":
        summary_json = Path(args.summary_json) if args.summary_json else DEFAULT_CLOSURE_SUMMARY_JSON
        
        if not args.readout_json and not args.detected_barcodes_json:
            print("[ERROR] Para --mode_app closure_match debes indicar --readout_json o --detected_barcodes_json")
            raise SystemExit(2)
        
        rc = run_closure_match(
            summary_json=summary_json,
            readout_json=Path(args.readout_json) if args.readout_json else None,
            detected_barcodes_json=Path(args.detected_barcodes_json) if args.detected_barcodes_json else None,
            output_path=Path(args.closure_output) if args.closure_output else DEFAULT_CLOSURE_OUTPUT_JSON,
        )
        
        raise SystemExit(rc)

    if args.mode_app == "closure_iterative":
        summary_json = Path(args.summary_json) if args.summary_json else DEFAULT_CLOSURE_SUMMARY_JSON
        
        if (
            not args.readout_json
            and not args.detected_barcodes_json
            and not args.manual_barcodes
            and not args.reset_session
        ):
            print("[ERROR] Para --mode_app closure_iterative debes indicar --readout_json, --detected_barcodes_json, --manual_barcodes o --reset_session")
            raise SystemExit(2)
        
        rc = run_closure_iterative(
            summary_json=summary_json,
            readout_json=Path(args.readout_json) if args.readout_json else None,
            detected_barcodes_json=Path(args.detected_barcodes_json) if args.detected_barcodes_json else None,
            manual_barcodes=args.manual_barcodes,
            session_state_json=Path(args.session_state_json) if args.session_state_json else None,
            output_path=Path(args.closure_output) if args.closure_output else DEFAULT_CLOSURE_ITERATIVE_OUTPUT_JSON,
            reset_session=args.reset_session,
        )
        raise SystemExit(rc)
    
    if args.mode_app == "picking_shipping":
        if not args.picking_image:
            print("[ERROR] Para --mode_app picking_shipping debes indicar --picking_image")
            raise SystemExit(2)
        
        rc = run_picking_shipping(
            picking_image=Path(args.picking_image),
            summary_json=Path(args.summary_json) if args.summary_json else None,
            session_state_json=Path(args.session_state_json) if args.session_state_json else None,
            output_path=Path(args.closure_output) if args.closure_output else None,
            env_file=".env",
            reset_session=args.reset_session,
        )
        raise SystemExit(rc)
    
    # if args.mode_app == "picking_shipping":
    #     print("[INFO] Modo picking_shipping aún no implementado.")
    #     raise SystemExit(0)
    
    if args.mode_app == "picking_flow":
        if not args.picking_image:
            print("[ERROR] Para --mode_app picking_flow debes indicar --picking_image")
            raise SystemExit(2)

        if (
            not args.summary_json
            and not args.picking_excel
        ):
            print("[ERROR] Para --mode_app picking_flow debes indicar --summary_json o --picking_excel")
            raise SystemExit(2)

        if (
            not args.readout_json
            and not args.detected_barcodes_json
            and not args.manual_barcodes
        ):
            print("[ERROR] Para --mode_app picking_flow debes indicar --readout_json, --detected_barcodes_json o --manual_barcodes")
            raise SystemExit(2)

        rc = run_picking_flow(
            picking_image=Path(args.picking_image),
            picking_excel=Path(args.picking_excel) if args.picking_excel else None,
            packstructure_excel=Path(args.packstructure_excel) if args.packstructure_excel else None,
            summary_json=Path(args.summary_json) if args.summary_json else None,
            readout_json=Path(args.readout_json) if args.readout_json else None,
            detected_barcodes_json=Path(args.detected_barcodes_json) if args.detected_barcodes_json else None,
            manual_barcodes=args.manual_barcodes,
            session_state_json=Path(args.session_state_json) if args.session_state_json else None,
            closure_output=Path(args.closure_output) if args.closure_output else DEFAULT_CLOSURE_ITERATIVE_OUTPUT_JSON,
            env_file=".env",
            reset_session=args.reset_session,
        )
        raise SystemExit(rc)

    captures_root = Path(args.captures_root)

    if args.once:
        processed = 0
        for event_json_path in iter_event_jsons(captures_root):
            ok = process_event(
                event_json_path,
                readout_module=args.readout_module,
                mode=args.mode,
                budget=args.budget,
                barcode_mode=args.barcode_mode,
                barcode_budget=args.barcode_budget,
                no_ocr=args.no_ocr,
                no_qr=args.no_qr,
                overwrite=args.overwrite,
            )
            if ok:
                processed += 1
        print(f"[DONE] Eventos procesados en esta pasada: {processed}")
        return

    watch_loop(
        captures_root,
        poll_s=args.poll_s,
        readout_module=args.readout_module,
        mode=args.mode,
        budget=args.budget,
        barcode_mode=args.barcode_mode,
        barcode_budget=args.barcode_budget,
        no_ocr=args.no_ocr,
        no_qr=args.no_qr,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()