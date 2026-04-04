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


def _norm_barcode(txt: Any) -> str:
    s = "" if txt is None else str(txt)
    return "".join(ch for ch in s if ch.isdigit())


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


def _load_detected_barcodes_json(path: Path) -> Optional[List[str]]:
    raw_payload = safe_read_json(path)
    if raw_payload is None:
        return None

    if isinstance(raw_payload, dict):
        if isinstance(raw_payload.get("detected_barcodes"), list):
            return _dedupe_preserve_order(raw_payload.get("detected_barcodes"))

        if isinstance(raw_payload.get("barcodes"), list):
            return _dedupe_preserve_order(raw_payload.get("barcodes"))

        # Salida directa híbrida: {"items": [...]}
        items = raw_payload.get("items")
        if isinstance(items, list):
            texts = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                text = item.get("text")
                if text:
                    texts.append(text)
            return _dedupe_preserve_order(texts)

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
                    if text:
                        texts.append(text)
                return _dedupe_preserve_order(texts)

        return None

    if isinstance(raw_payload, list):
        return _dedupe_preserve_order(raw_payload)

    return None


def _parse_manual_barcodes(manual_barcodes: Optional[str]) -> List[str]:
    if not manual_barcodes:
        return []
    parts = [x.strip() for x in manual_barcodes.split(",")]
    return _dedupe_preserve_order(parts)


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
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))
    return 0

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

                key = (
                    barcode,
                    str(item.get("format") or "").strip(),
                    str(item.get("source") or "").strip(),
                )
                if key in seen:
                    continue
                seen.add(key)

                items_out.append(
                    {
                        "barcode": barcode,
                        "serial": barcode,  # por ahora serial = barcode detectado
                        "format": item.get("format"),
                        "source": item.get("source"),
                    }
                )

    if detected_barcodes:
        for raw in detected_barcodes:
            barcode = _norm_barcode(raw)
            if not barcode:
                continue

            key = (barcode, "", "")
            already_present = any(x.get("barcode") == barcode for x in items_out)
            if already_present:
                continue

            if key in seen:
                continue
            seen.add(key)

            items_out.append(
                {
                    "barcode": barcode,
                    "serial": barcode,
                    "format": None,
                    "source": "detected_barcodes_json",
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
    print(json.dumps(closure_result, ensure_ascii=False, indent=2))
    return 0 if closure_result.get("status") == "success" else 2


# ============================================================
# Closure Iterative (acumulado de barcodes por sesión)
# ============================================================
class ClosureSession:
    def __init__(self, summary_payload: Dict[str, Any], state_path: Optional[Path] = None):
        self.summary_payload = summary_payload
        self.state_path = state_path
        self.detected_barcodes: List[str] = []

        if self.state_path and self.state_path.exists():
            state_payload = safe_read_json(self.state_path)
            if isinstance(state_payload, dict):
                prev = state_payload.get("detected_barcodes", [])
                if isinstance(prev, list):
                    self.detected_barcodes = _dedupe_preserve_order(prev)

    def get_barcodes(self) -> List[str]:
        return list(self.detected_barcodes)

    def update(self, new_barcodes: List[str]) -> None:
        merged = list(self.detected_barcodes) + list(new_barcodes)
        self.detected_barcodes = _dedupe_preserve_order(merged)

    def save(self) -> None:
        if not self.state_path:
            return

        payload = {
            "status": "success",
            "processed_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
            "detected_barcodes": self.get_barcodes(),
            "counts": {
                "detected_barcodes": len(self.detected_barcodes),
            },
        }
        safe_write_json(self.state_path, payload)

    def clear(self) -> None:
        self.detected_barcodes = []
        self.save()


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

    session = ClosureSession(summary_payload=summary_payload, state_path=session_state_json)

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

    new_barcodes = _dedupe_preserve_order(new_barcodes)
    previous_barcodes = session.get_barcodes()

    session.update(new_barcodes)
    session.save()

    accumulated_barcodes = session.get_barcodes()

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

    closure_result = run_packstructure_closure(
        summary_payload=summary_payload,
        detected_barcodes=accumulated_barcodes,
        event_context={
            "mode": "closure_iterative",
            "summary_json": str(summary_json),
            "readout_json": str(readout_json) if readout_json else None,
            "detected_barcodes_json": str(detected_barcodes_json) if detected_barcodes_json else None,
            "manual_barcodes": manual_list,
            "session_state_json": str(session_state_json) if session_state_json else None,
            "reset_session": reset_session,
        },
    )
    
    frontend_summary = build_frontend_closure_summary(
        closure_result=closure_result,
        readout_payload=readout_payload,
        detected_barcodes=accumulated_barcodes,
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
            "previous_barcodes": previous_barcodes,
            "new_barcodes": new_barcodes,
            "accumulated_barcodes": accumulated_barcodes,
            "counts": {
                "previous": len(previous_barcodes),
                "new": len(new_barcodes),
                "accumulated": len(accumulated_barcodes),
            },
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
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if closure_result.get("status") == "success" else 2


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Procesa automáticamente eventos capturados y ejecuta vision_readout, flujo picking+excel, closure_match o closure_iterative."
    )

    ap.add_argument(
        "--mode_app",
        type=str,
        choices=["readout", "picking_match", "closure_match", "closure_iterative"],
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
        if not args.summary_json:
            print("[ERROR] Para --mode_app closure_iterative debes indicar --summary_json")
            raise SystemExit(2)

        if (
            not args.readout_json
            and not args.detected_barcodes_json
            and not args.manual_barcodes
            and not args.reset_session
        ):
            print("[ERROR] Para --mode_app closure_iterative debes indicar --readout_json, --detected_barcodes_json, --manual_barcodes o --reset_session")
            raise SystemExit(2)

        rc = run_closure_iterative(
            summary_json=Path(args.summary_json),
            readout_json=Path(args.readout_json) if args.readout_json else None,
            detected_barcodes_json=Path(args.detected_barcodes_json) if args.detected_barcodes_json else None,
            manual_barcodes=args.manual_barcodes,
            session_state_json=Path(args.session_state_json) if args.session_state_json else None,
            output_path=Path(args.closure_output) if args.closure_output else None,
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