# # app/main.py
# from __future__ import annotations

# import argparse
# import ast
# import json
# import subprocess
# import sys
# import time
# from pathlib import Path
# from typing import Any, Dict, Iterable, Optional, List


# def safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
#     try:
#         with path.open("r", encoding="utf-8") as f:
#             return json.load(f)
#     except Exception:
#         return None


# def safe_write_json(path: Path, payload: Dict[str, Any]) -> None:
#     path.parent.mkdir(parents=True, exist_ok=True)
#     tmp = path.with_suffix(path.suffix + ".tmp")
#     with tmp.open("w", encoding="utf-8") as f:
#         json.dump(payload, f, ensure_ascii=False, indent=2)
#     tmp.replace(path)


# def iter_event_jsons(captures_root: Path) -> Iterable[Path]:
#     if not captures_root.exists():
#         return []
#     return sorted(captures_root.glob("frames_*/events/event_*/event.json"))


# def resolve_target_frame(event_payload: Dict[str, Any], event_json_path: Path) -> Optional[Path]:
#     burst = event_payload.get("burst") or {}
#     main_frame_path = burst.get("main_frame_path")
#     if main_frame_path:
#         p = Path(main_frame_path)
#         if not p.is_absolute():
#             p = Path.cwd() / p
#         if p.exists():
#             return p

#     paths = event_payload.get("paths") or {}
#     frame_path = paths.get("frame")
#     if frame_path:
#         p = Path(frame_path)
#         if not p.is_absolute():
#             p = Path.cwd() / p
#         if p.exists():
#             return p

#     candidate = event_json_path.parent / "frames" / "frame_02.jpg"
#     if candidate.exists():
#         return candidate

#     candidate = event_json_path.parent / "frame.jpg"
#     if candidate.exists():
#         return candidate

#     return None


# def build_readout_cmd(
#     image_path: Path,
#     *,
#     mode: str,
#     budget: int,
#     barcode_mode: str,
#     barcode_budget: int,
#     no_ocr: bool,
#     no_qr: bool,
# ) -> list[str]:
#     cmd = [
#         sys.executable,
#         "-m",
#         "utils.vision_readout",
#         str(image_path),
#         "--mode",
#         mode,
#         "--budget",
#         str(budget),
#         "--barcode_mode",
#         barcode_mode,
#         "--barcode_budget",
#         str(barcode_budget),
#     ]
#     if no_ocr:
#         cmd.append("--no-ocr")
#     if no_qr:
#         cmd.append("--no-qr")
#     return cmd


# def summarize_readout(payload: Dict[str, Any]) -> Dict[str, Any]:
#     barcode = payload.get("barcode1d") or {}
#     confirmed_items = barcode.get("confirmed_items") or []
#     texts = []
#     for item in confirmed_items:
#         text = item.get("text")
#         if text and text not in texts:
#             texts.append(text)

#     best = payload.get("best") or {}
#     best_text = None
#     best_kind = None
#     if isinstance(best, dict):
#         best_text = best.get("text")
#         best_kind = best.get("kind")

#     return {
#         "status": payload.get("status"),
#         "best_kind": best_kind,
#         "best_text": best_text,
#         "confirmed_count": len(confirmed_items),
#         "confirmed_texts": texts,
#     }


# def parse_readout_stdout(stdout: str) -> Optional[Dict[str, Any]]:
#     stdout = stdout.strip()
#     if not stdout:
#         return None

#     try:
#         parsed = json.loads(stdout)
#         if isinstance(parsed, dict):
#             return parsed
#     except json.JSONDecodeError:
#         pass

#     try:
#         parsed = ast.literal_eval(stdout)
#         if isinstance(parsed, dict):
#             return parsed
#     except Exception:
#         pass

#     return None


# def process_event(
#     event_json_path: Path,
#     *,
#     mode: str,
#     budget: int,
#     barcode_mode: str,
#     barcode_budget: int,
#     no_ocr: bool,
#     no_qr: bool,
#     overwrite: bool,
# ) -> bool:
#     event_payload = safe_read_json(event_json_path)
#     if not event_payload:
#         print(f"[WARN] No pude leer JSON: {event_json_path}")
#         return False

#     ev_dir = event_json_path.parent
#     result_path = ev_dir / "readout_result.json"
#     marker_path = ev_dir / ".processed"

#     if result_path.exists() and not overwrite:
#         return False
#     if marker_path.exists() and not overwrite:
#         return False

#     image_path = resolve_target_frame(event_payload, event_json_path)
#     if image_path is None or not image_path.exists():
#         print(f"[WARN] No encontré frame objetivo para: {event_json_path}")
#         return False

#     cmd = build_readout_cmd(
#         image_path,
#         mode=mode,
#         budget=budget,
#         barcode_mode=barcode_mode,
#         barcode_budget=barcode_budget,
#         no_ocr=no_ocr,
#         no_qr=no_qr,
#     )

#     print(f"[INFO] Procesando evento: {ev_dir.name}")
#     print(f"[INFO] Frame objetivo: {image_path}")
#     print(f"[INFO] Ejecutando: {' '.join(cmd)}")

#     started_at = time.time()
#     proc = subprocess.run(
#         cmd,
#         capture_output=True,
#         text=True,
#         cwd=str(Path.cwd()),
#     )
#     elapsed_ms = int((time.time() - started_at) * 1000)

#     if proc.returncode != 0:
#         error_payload = {
#             "status": "error",
#             "returncode": proc.returncode,
#             "stdout": proc.stdout,
#             "stderr": proc.stderr,
#             "elapsed_ms": elapsed_ms,
#             "image_path": str(image_path),
#             "cmd": cmd,
#         }
#         safe_write_json(result_path, error_payload)
#         marker_path.write_text("error\n", encoding="utf-8")
#         print(f"[WARN] Falló readout en {ev_dir.name}")
#         return False

#     readout_payload = parse_readout_stdout(proc.stdout)

#     if readout_payload is None:
#         error_payload = {
#             "status": "error",
#             "reason": "stdout_no_es_json_ni_python_literal",
#             "stdout": proc.stdout,
#             "stderr": proc.stderr,
#             "elapsed_ms": elapsed_ms,
#             "image_path": str(image_path),
#             "cmd": cmd,
#         }
#         safe_write_json(result_path, error_payload)
#         marker_path.write_text("error\n", encoding="utf-8")
#         print(f"[WARN] La salida no fue parseable en {ev_dir.name}")
#         return False

#     wrapped_result = {
#         "status": "ok",
#         "processed_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
#         "elapsed_ms": elapsed_ms,
#         "image_path": str(image_path),
#         "cmd": cmd,
#         "result": readout_payload,
#         "summary": summarize_readout(readout_payload),
#     }
#     safe_write_json(result_path, wrapped_result)
#     marker_path.write_text("ok\n", encoding="utf-8")

#     event_payload["readout"] = {
#         "status": "attempted",
#         "image_path": str(image_path),
#         "result_path": str(result_path),
#         "summary": wrapped_result["summary"],
#     }
#     safe_write_json(event_json_path, event_payload)

#     print(f"[OK] Resultado guardado en: {result_path}")
#     return True


# def watch_loop(
#     captures_root: Path,
#     *,
#     poll_s: float,
#     mode: str,
#     budget: int,
#     barcode_mode: str,
#     barcode_budget: int,
#     no_ocr: bool,
#     no_qr: bool,
#     overwrite: bool,
# ) -> None:
#     print(f"[INFO] Watch mode ON en: {captures_root}")
#     print(f"[INFO] Poll cada {poll_s:.1f}s")
#     print("[INFO] Ctrl+C para salir")

#     while True:
#         try:
#             event_paths = list(iter_event_jsons(captures_root))
#             for event_json_path in event_paths:
#                 process_event(
#                     event_json_path,
#                     mode=mode,
#                     budget=budget,
#                     barcode_mode=barcode_mode,
#                     barcode_budget=barcode_budget,
#                     no_ocr=no_ocr,
#                     no_qr=no_qr,
#                     overwrite=overwrite,
#                 )
#             time.sleep(poll_s)
#         except KeyboardInterrupt:
#             print("\n[INFO] Watch detenido por usuario.")
#             break
#         except Exception as e:
#             print(f"[WARN] Error en watch loop: {e}")
#             time.sleep(poll_s)


# # ============================================================
# # Helpers consolidación picking + fillRate + packStructure
# # ============================================================
# def _norm_code(txt: Any) -> str:
#     s = "" if txt is None else str(txt)
#     s = s.upper().strip()
#     s = s.replace(" ", "").replace(".", "").replace(",", "")
#     return s


# def _build_fillrate_index(match_result: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
#     idx: Dict[str, Dict[str, Any]] = {}
#     products = match_result.get("products", []) or []
#     for prod in products:
#         key = _norm_code(prod.get("codigo_item_ocr"))
#         if key:
#             idx[key] = prod
#     return idx


# def _build_pack_index(match_result: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
#     idx: Dict[str, Dict[str, Any]] = {}
#     products = match_result.get("products", []) or []
#     for prod in products:
#         key = _norm_code(prod.get("codigo_item_ocr"))
#         if key:
#             idx[key] = prod
#     return idx


# def build_final_picking_summary(
#     *,
#     picking_result: Dict[str, Any],
#     fillrate_match: Dict[str, Any],
#     packstructure_match: Dict[str, Any],
#     picking_image: Path,
#     picking_excel: Path,
#     packstructure_excel: Path,
# ) -> Dict[str, Any]:
#     metadata = picking_result.get("metadata", {}) or {}
#     products = picking_result.get("products", []) or []

#     fill_idx = _build_fillrate_index(fillrate_match)
#     pack_idx = _build_pack_index(packstructure_match)

#     consolidated_products: List[Dict[str, Any]] = []

#     total_unidades_ocr = 0
#     total_cant_original = 0
#     total_cant_trabajada = 0
#     fillrate_matched = 0
#     packstructure_matched = 0

#     for prod in products:
#         codigo = prod.get("codigo_item")
#         codigo_n = _norm_code(codigo)

#         fill = fill_idx.get(codigo_n, {})
#         pack = pack_idx.get(codigo_n, {})

#         unidades_ocr = prod.get("unidades") or 0
#         cant_original = fill.get("cant_original_excel") or 0
#         cant_trabajada = fill.get("cant_trabajada_excel") or 0

#         total_unidades_ocr += int(unidades_ocr or 0)
#         total_cant_original += int(cant_original or 0)
#         total_cant_trabajada += int(cant_trabajada or 0)

#         if fill.get("match_status") == "matched":
#             fillrate_matched += 1
#         if pack.get("match_status") == "matched":
#             packstructure_matched += 1

#         consolidated_products.append(
#             {
#                 "codigo_item": codigo,
#                 "descripcion_ocr": prod.get("descripcion"),
#                 "unidades_ocr": prod.get("unidades"),

#                 "fillrate": {
#                     "match_status": fill.get("match_status"),
#                     "codigo_item_excel": fill.get("codigo_item_excel"),
#                     "descripcion_excel": fill.get("descripcion_excel"),
#                     "shipping_excel": fill.get("shipping_excel"),
#                     "ruta_excel": fill.get("ruta_excel"),
#                     "orden_compra_excel": fill.get("orden_compra_excel"),
#                     "cliente_excel": fill.get("cliente_excel"),
#                     "direccion_excel": fill.get("direccion_excel"),
#                     "cant_original_excel": fill.get("cant_original_excel"),
#                     "cant_trabajada_excel": fill.get("cant_trabajada_excel"),
#                     "diferencia_excel": fill.get("diferencia_excel"),
#                 },

#                 "packstructure": {
#                     "match_status": pack.get("match_status"),
#                     "codigo_item_pack": pack.get("codigo_item_pack"),
#                     "descripcion_pack": pack.get("descripcion_pack"),
#                     "item_type_pack": pack.get("item_type_pack"),

#                     "qty_ea_pack": pack.get("qty_ea_pack"),
#                     "largo_ea_pack": pack.get("largo_ea_pack"),
#                     "ancho_ea_pack": pack.get("ancho_ea_pack"),
#                     "alto_ea_pack": pack.get("alto_ea_pack"),
#                     "peso_ea_pack": pack.get("peso_ea_pack"),
#                     "ean_pack": pack.get("ean_pack"),

#                     "qty_inn_pack": pack.get("qty_inn_pack"),
#                     "largo_inn_pack": pack.get("largo_inn_pack"),
#                     "ancho_inn_pack": pack.get("ancho_inn_pack"),
#                     "alto_inn_pack": pack.get("alto_inn_pack"),
#                     "peso_inn_pack": pack.get("peso_inn_pack"),
#                     "ean_in_pack": pack.get("ean_in_pack"),

#                     "qty_cs_pack": pack.get("qty_cs_pack"),
#                     "largo_cs_pack": pack.get("largo_cs_pack"),
#                     "ancho_cs_pack": pack.get("ancho_cs_pack"),
#                     "alto_cs_pack": pack.get("alto_cs_pack"),
#                     "peso_cs_pack": pack.get("peso_cs_pack"),
#                     "ean_cs_pack": pack.get("ean_cs_pack"),

#                     "qty_pal_pack": pack.get("qty_pal_pack"),
#                     "largo_pal_pack": pack.get("largo_pal_pack"),
#                     "ancho_pal_pack": pack.get("ancho_pal_pack"),
#                     "alto_pal_pack": pack.get("alto_pal_pack"),
#                     "peso_pal_pack": pack.get("peso_pal_pack"),
#                 },
#             }
#         )

#     return {
#         "status": "success",
#         "processed_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
#         "document_type": "summary_pickingVision_fillRate_packStructure",
#         "inputs": {
#             "picking_image": str(picking_image),
#             "fillrate_excel": str(picking_excel),
#             "packstructure_excel": str(packstructure_excel),
#         },
#         "metadata": {
#             "centro": metadata.get("centro"),
#             "mo": metadata.get("nm"),
#             "entrega": metadata.get("entrega"),
#             "wave": metadata.get("wave_id"),
#             "ruta": metadata.get("ruta"),
#             "cliente": metadata.get("cliente"),
#             "direccion": metadata.get("direccion"),
#             "fecha_impresion": metadata.get("fecha_impresion"),
#             "hora_impresion": metadata.get("hora_impresion"),
#         },
#         "products": consolidated_products,
#         "counts": {
#             "picking_products": len(products),
#             "fillrate_matched_products": fillrate_matched,
#             "packstructure_matched_products": packstructure_matched,
#         },
#         "totals": {
#             "unidades_ocr_total": total_unidades_ocr,
#             "cant_original_excel_total": total_cant_original,
#             "cant_trabajada_excel_total": total_cant_trabajada,
#         },
#     }


# # ============================================================
# # Picking + Excel + PackStructure
# # ============================================================
# def run_picking_match(
#     *,
#     picking_image: Path,
#     picking_excel: Path,
#     packstructure_excel: Optional[Path] = None,
#     output_path: Optional[Path] = None,
# ) -> int:
#     try:
#         from utils.vision_picking import extract_picking_sheet_from_path
#         from utils.vision_excel_picking import (
#             load_excel_picking,
#             match_picking_with_excel,
#             save_json,
#         )
#         from utils.vision_excel_packStructure import (
#             load_excel_packstructure,
#             match_picking_with_packstructure,
#         )
#     except Exception as e:
#         print(f"[ERROR] No pude importar módulos de picking/excel: {e}")
#         return 2

#     if not picking_image.exists():
#         print(f"[ERROR] No existe imagen picking: {picking_image}")
#         return 2

#     if not picking_excel.exists():
#         print(f"[ERROR] No existe excel picking: {picking_excel}")
#         return 2

#     if packstructure_excel is not None and not packstructure_excel.exists():
#         print(f"[ERROR] No existe excel packStructure: {packstructure_excel}")
#         return 2

#     print(f"[INFO] Procesando hoja de picking: {picking_image}")
#     picking_result = extract_picking_sheet_from_path(str(picking_image))

#     if picking_result.get("status") != "success":
#         print("[ERROR] Falló extracción de hoja de picking")
#         print(json.dumps(picking_result, ensure_ascii=False, indent=2))
#         return 2

#     print(f"[INFO] Cargando excel picking: {picking_excel}")
#     excel_result = load_excel_picking(str(picking_excel))

#     if excel_result.get("status") != "success":
#         print("[ERROR] Falló carga de excel FillRate")
#         print(json.dumps(excel_result, ensure_ascii=False, indent=2))
#         return 2

#     fillrate_match = match_picking_with_excel(picking_result, excel_result)

#     packstructure_result = None
#     packstructure_match = {"status": "success", "products": [], "counts": {}}

#     if packstructure_excel is not None:
#         print(f"[INFO] Cargando excel PackStructure: {packstructure_excel}")
#         packstructure_result = load_excel_packstructure(str(packstructure_excel))

#         if packstructure_result.get("status") != "success":
#             print("[ERROR] Falló carga de excel PackStructure")
#             print(json.dumps(packstructure_result, ensure_ascii=False, indent=2))
#             return 2

#         packstructure_match = match_picking_with_packstructure(
#             picking_result,
#             packstructure_result,
#         )

#     final_summary = build_final_picking_summary(
#         picking_result=picking_result,
#         fillrate_match=fillrate_match,
#         packstructure_match=packstructure_match,
#         picking_image=picking_image,
#         picking_excel=picking_excel,
#         packstructure_excel=packstructure_excel or Path(""),
#     )

#     payload = {
#         "status": "success",
#         "processed_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
#         "picking_image": str(picking_image),
#         "picking_excel": str(picking_excel),
#         "packstructure_excel": str(packstructure_excel) if packstructure_excel else None,

#         "picking_result_summary": picking_result.get("summary"),
#         "excel_result_summary": {
#             "sheet_name": excel_result.get("sheet_name"),
#             "rows_loaded": excel_result.get("rows_loaded"),
#         },
#         "packstructure_result_summary": {
#             "sheet_name": packstructure_result.get("sheet_name") if packstructure_result else None,
#             "rows_loaded": packstructure_result.get("rows_loaded") if packstructure_result else None,
#         },

#         "fillrate_match_result": fillrate_match,
#         "packstructure_match_result": packstructure_match,
#         "summary_pickingVision_fillRate_packStructure": final_summary,
#     }

#     if output_path is None:
#         output_dir = Path("data/picking/summary_pickingVision_fillRate_packStructure")
#         output_dir.mkdir(parents=True, exist_ok=True)
#         stem = picking_image.stem
#         output_path = output_dir / f"{stem}_summary_pickingVision_fillRate_packStructure.json"

#     save_json(payload, str(output_path))

#     print(f"[OK] Consolidado guardado en: {output_path}")
#     print(json.dumps(final_summary, ensure_ascii=False, indent=2))
#     return 0


# def main() -> None:
#     ap = argparse.ArgumentParser(
#         description="Procesa automáticamente eventos capturados y ejecuta vision_readout o flujo picking+excel."
#     )

#     ap.add_argument(
#         "--mode_app",
#         type=str,
#         choices=["readout", "picking_match"],
#         default="readout",
#         help="Flujo de aplicación principal.",
#     )

#     ap.add_argument(
#         "--captures_root",
#         type=str,
#         default="data/captures/opencv",
#         help="Raíz donde capture_opencv guarda las sesiones.",
#     )
#     ap.add_argument(
#         "--poll_s",
#         type=float,
#         default=2.0,
#         help="Intervalo de polling en segundos para detectar eventos nuevos.",
#     )
#     ap.add_argument(
#         "--once",
#         action="store_true",
#         help="Procesa una pasada y termina. Sin este flag, queda en modo watch.",
#     )
#     ap.add_argument(
#         "--overwrite",
#         action="store_true",
#         help="Reprocesa aunque ya exista resultado.",
#     )

#     ap.add_argument("--mode", type=str, default="retry")
#     ap.add_argument("--budget", type=int, default=6500)
#     ap.add_argument("--barcode_mode", type=str, default="collect_plus")
#     ap.add_argument("--barcode_budget", type=int, default=6000)
#     ap.add_argument("--no_ocr", action="store_true", default=True)
#     ap.add_argument("--no_qr", action="store_true", default=True)

#     ap.add_argument("--picking_image", type=str, default=None, help="Ruta a imagen de hoja de picking.")
#     ap.add_argument("--picking_excel", type=str, default=None, help="Ruta a excel asociado al picking.")
#     ap.add_argument("--packstructure_excel", type=str, default=None, help="Ruta a excel PackStructure.")
#     ap.add_argument("--picking_output", type=str, default=None, help="Ruta de salida JSON consolidado.")

#     args = ap.parse_args()

#     if args.mode_app == "picking_match":
#         if not args.picking_image or not args.picking_excel:
#             print("[ERROR] Para --mode_app picking_match debes indicar --picking_image y --picking_excel")
#             raise SystemExit(2)

#         rc = run_picking_match(
#             picking_image=Path(args.picking_image),
#             picking_excel=Path(args.picking_excel),
#             packstructure_excel=Path(args.packstructure_excel) if args.packstructure_excel else None,
#             output_path=Path(args.picking_output) if args.picking_output else None,
#         )
#         raise SystemExit(rc)

#     captures_root = Path(args.captures_root)

#     if args.once:
#         processed = 0
#         for event_json_path in iter_event_jsons(captures_root):
#             ok = process_event(
#                 event_json_path,
#                 mode=args.mode,
#                 budget=args.budget,
#                 barcode_mode=args.barcode_mode,
#                 barcode_budget=args.barcode_budget,
#                 no_ocr=args.no_ocr,
#                 no_qr=args.no_qr,
#                 overwrite=args.overwrite,
#             )
#             if ok:
#                 processed += 1
#         print(f"[DONE] Eventos procesados en esta pasada: {processed}")
#         return

#     watch_loop(
#         captures_root,
#         poll_s=args.poll_s,
#         mode=args.mode,
#         budget=args.budget,
#         barcode_mode=args.barcode_mode,
#         barcode_budget=args.barcode_budget,
#         no_ocr=args.no_ocr,
#         no_qr=args.no_qr,
#         overwrite=args.overwrite,
#     )


# if __name__ == "__main__":
#     main()

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
    burst = event_payload.get("burst") or {}
    main_frame_path = burst.get("main_frame_path")
    if main_frame_path:
        p = Path(main_frame_path)
        if not p.is_absolute():
            p = Path.cwd() / p
        if p.exists():
            return p

    paths = event_payload.get("paths") or {}
    frame_path = paths.get("frame")
    if frame_path:
        p = Path(frame_path)
        if not p.is_absolute():
            p = Path.cwd() / p
        if p.exists():
            return p

    candidate = event_json_path.parent / "frames" / "frame_02.jpg"
    if candidate.exists():
        return candidate

    candidate = event_json_path.parent / "frame.jpg"
    if candidate.exists():
        return candidate

    return None


def build_readout_cmd(
    image_path: Path,
    *,
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
        "utils.vision_readout",
        str(image_path),
        "--mode",
        mode,
        "--budget",
        str(budget),
        "--barcode_mode",
        barcode_mode,
        "--barcode_budget",
        str(barcode_budget),
    ]
    if no_ocr:
        cmd.append("--no-ocr")
    if no_qr:
        cmd.append("--no-qr")
    return cmd


def summarize_readout(payload: Dict[str, Any]) -> Dict[str, Any]:
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
        "best_kind": best_kind,
        "best_text": best_text,
        "confirmed_count": len(confirmed_items),
        "confirmed_texts": texts,
    }


def parse_readout_stdout(stdout: str) -> Optional[Dict[str, Any]]:
    stdout = stdout.strip()
    if not stdout:
        return None

    try:
        parsed = json.loads(stdout)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    try:
        parsed = ast.literal_eval(stdout)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    return None


def process_event(
    event_json_path: Path,
    *,
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
# Helpers consolidación picking + fillRate + packStructure
# ============================================================
def _norm_code(txt: Any) -> str:
    s = "" if txt is None else str(txt)
    s = s.upper().strip()
    s = s.replace(" ", "").replace(".", "").replace(",", "")
    return s


def _build_fillrate_index(match_result: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    products = match_result.get("products", []) or []
    for prod in products:
        key = _norm_code(prod.get("codigo_item_ocr"))
        if key and key not in idx:
            idx[key] = prod
    return idx


def _build_pack_index(match_result: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    products = match_result.get("products", []) or []
    for prod in products:
        key = _norm_code(prod.get("codigo_item_ocr"))
        if key and key not in idx:
            idx[key] = prod
    return idx


def build_final_picking_summary(
    *,
    picking_result: Dict[str, Any],
    fillrate_match: Dict[str, Any],
    packstructure_match: Dict[str, Any],
    picking_image: Path,
    picking_excel: Path,
    packstructure_excel: Path,
) -> Dict[str, Any]:
    metadata = picking_result.get("metadata", {}) or {}
    products = picking_result.get("products", []) or []

    fill_idx = _build_fillrate_index(fillrate_match)
    pack_idx = _build_pack_index(packstructure_match)

    consolidated_products: List[Dict[str, Any]] = []

    total_unidades_ocr = 0
    total_cant_original = 0
    total_cant_trabajada = 0
    fillrate_matched = 0
    packstructure_matched = 0

    for prod in products:
        codigo = prod.get("codigo_item")
        codigo_n = _norm_code(codigo)

        fill = fill_idx.get(codigo_n, {})
        pack = pack_idx.get(codigo_n, {})

        unidades_ocr = prod.get("unidades") or 0
        cant_original = fill.get("cant_original_excel") or 0
        cant_trabajada = fill.get("cant_trabajada_excel") or 0

        try:
            total_unidades_ocr += int(unidades_ocr or 0)
        except Exception:
            pass

        try:
            total_cant_original += int(cant_original or 0)
        except Exception:
            pass

        try:
            total_cant_trabajada += int(cant_trabajada or 0)
        except Exception:
            pass

        if fill.get("match_status") == "matched":
            fillrate_matched += 1
        if pack.get("match_status") == "matched":
            packstructure_matched += 1

        consolidated_products.append(
            {
                "codigo_item": codigo,
                "descripcion_ocr": prod.get("descripcion"),
                "unidades_ocr": prod.get("unidades"),

                "fillrate": {
                    "match_status": fill.get("match_status"),
                    "excel_match_count": fill.get("excel_match_count"),

                    "codigo_item_excel": fill.get("codigo_item_excel"),
                    "descripcion_excel": fill.get("descripcion_excel"),
                    "shipping_excel": fill.get("shipping_excel"),
                    "ruta_excel": fill.get("ruta_excel"),
                    "orden_compra_excel": fill.get("orden_compra_excel"),
                    "cliente_excel": fill.get("cliente_excel"),
                    "direccion_excel": fill.get("direccion_excel"),

                    "estado_excel": fill.get("estado_excel"),
                    "estado_orden_excel": fill.get("estado_orden_excel"),
                    "descripcion_estado_excel": fill.get("descripcion_estado_excel"),

                    "cant_original_excel": fill.get("cant_original_excel"),
                    "cant_trabajada_excel": fill.get("cant_trabajada_excel"),
                    "diferencia_excel": fill.get("diferencia_excel"),

                    # trazabilidad
                    "excel": fill.get("excel"),
                    "excel_matches": fill.get("excel_matches", []),
                },

                "packstructure": {
                    "match_status": pack.get("match_status"),
                    "packstructure_match_count": pack.get("packstructure_match_count"),

                    "codigo_item_pack": pack.get("codigo_item_pack"),
                    "descripcion_pack": pack.get("descripcion_pack"),
                    "item_type_pack": pack.get("item_type_pack"),

                    "qty_ea_pack": pack.get("qty_ea_pack"),
                    "largo_ea_pack": pack.get("largo_ea_pack"),
                    "ancho_ea_pack": pack.get("ancho_ea_pack"),
                    "alto_ea_pack": pack.get("alto_ea_pack"),
                    "peso_ea_pack": pack.get("peso_ea_pack"),
                    "ean_pack": pack.get("ean_pack"),

                    "qty_inn_pack": pack.get("qty_inn_pack"),
                    "largo_inn_pack": pack.get("largo_inn_pack"),
                    "ancho_inn_pack": pack.get("ancho_inn_pack"),
                    "alto_inn_pack": pack.get("alto_inn_pack"),
                    "peso_inn_pack": pack.get("peso_inn_pack"),
                    "ean_in_pack": pack.get("ean_in_pack"),

                    "qty_cs_pack": pack.get("qty_cs_pack"),
                    "largo_cs_pack": pack.get("largo_cs_pack"),
                    "ancho_cs_pack": pack.get("ancho_cs_pack"),
                    "alto_cs_pack": pack.get("alto_cs_pack"),
                    "peso_cs_pack": pack.get("peso_cs_pack"),
                    "ean_cs_pack": pack.get("ean_cs_pack"),

                    "qty_pal_pack": pack.get("qty_pal_pack"),
                    "largo_pal_pack": pack.get("largo_pal_pack"),
                    "ancho_pal_pack": pack.get("ancho_pal_pack"),
                    "alto_pal_pack": pack.get("alto_pal_pack"),
                    "peso_pal_pack": pack.get("peso_pal_pack"),

                    # trazabilidad
                    "packstructure": pack.get("packstructure"),
                    "packstructure_matches": pack.get("packstructure_matches", []),
                },
            }
        )

    return {
        "status": "success",
        "processed_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "document_type": "summary_pickingVision_fillRate_packStructure",
        "inputs": {
            "picking_image": str(picking_image),
            "fillrate_excel": str(picking_excel),
            "packstructure_excel": str(packstructure_excel) if str(packstructure_excel) else None,
        },
        "metadata": {
            "centro": metadata.get("centro"),
            "mo": metadata.get("nm"),
            "entrega": metadata.get("entrega"),
            "wave": metadata.get("wave_id"),
            "ruta": metadata.get("ruta"),
            "cliente": metadata.get("cliente"),
            "direccion": metadata.get("direccion"),
            "fecha_impresion": metadata.get("fecha_impresion"),
            "hora_impresion": metadata.get("hora_impresion"),
        },
        "products": consolidated_products,
        "counts": {
            "picking_products": len(products),
            "fillrate_matched_products": fillrate_matched,
            "packstructure_matched_products": packstructure_matched,
        },
        "totals": {
            "unidades_ocr_total": total_unidades_ocr,
            "cant_original_excel_total": total_cant_original,
            "cant_trabajada_excel_total": total_cant_trabajada,
        },
    }


# ============================================================
# Picking + Excel + PackStructure
# ============================================================
def run_picking_match(
    *,
    picking_image: Path,
    picking_excel: Path,
    packstructure_excel: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> int:
    try:
        from utils.vision_picking import extract_picking_sheet_from_path
        from utils.vision_excel_picking import (
            load_excel_picking,
            match_picking_with_excel,
            save_json,
        )
        from utils.vision_excel_packStructure import (
            load_excel_packstructure,
            match_picking_with_packstructure,
        )
    except Exception as e:
        print(f"[ERROR] No pude importar módulos de picking/excel: {e}")
        return 2

    if not picking_image.exists():
        print(f"[ERROR] No existe imagen picking: {picking_image}")
        return 2

    if not picking_excel.exists():
        print(f"[ERROR] No existe excel picking: {picking_excel}")
        return 2

    if packstructure_excel is not None and not packstructure_excel.exists():
        print(f"[ERROR] No existe excel packStructure: {packstructure_excel}")
        return 2

    print(f"[INFO] Procesando hoja de picking: {picking_image}")
    picking_result = extract_picking_sheet_from_path(str(picking_image))

    if picking_result.get("status") != "success":
        print("[ERROR] Falló extracción de hoja de picking")
        print(json.dumps(picking_result, ensure_ascii=False, indent=2))
        return 2

    print(f"[INFO] Cargando excel picking: {picking_excel}")
    excel_result = load_excel_picking(str(picking_excel))

    if excel_result.get("status") != "success":
        print("[ERROR] Falló carga de excel FillRate")
        print(json.dumps(excel_result, ensure_ascii=False, indent=2))
        return 2

    fillrate_match = match_picking_with_excel(picking_result, excel_result)

    packstructure_result = None
    packstructure_match = {"status": "success", "products": [], "counts": {}}

    if packstructure_excel is not None:
        print(f"[INFO] Cargando excel PackStructure: {packstructure_excel}")
        packstructure_result = load_excel_packstructure(str(packstructure_excel))

        if packstructure_result.get("status") != "success":
            print("[ERROR] Falló carga de excel PackStructure")
            print(json.dumps(packstructure_result, ensure_ascii=False, indent=2))
            return 2

        packstructure_match = match_picking_with_packstructure(
            picking_result,
            packstructure_result,
        )

    final_summary = build_final_picking_summary(
        picking_result=picking_result,
        fillrate_match=fillrate_match,
        packstructure_match=packstructure_match,
        picking_image=picking_image,
        picking_excel=picking_excel,
        packstructure_excel=packstructure_excel or Path(""),
    )

    payload = {
        "status": "success",
        "processed_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "picking_image": str(picking_image),
        "picking_excel": str(picking_excel),
        "packstructure_excel": str(packstructure_excel) if packstructure_excel else None,

        "picking_result_summary": picking_result.get("summary"),
        "excel_result_summary": {
            "sheet_name": excel_result.get("sheet_name"),
            "rows_loaded": excel_result.get("rows_loaded"),
        },
        "packstructure_result_summary": {
            "sheet_name": packstructure_result.get("sheet_name") if packstructure_result else None,
            "rows_loaded": packstructure_result.get("rows_loaded") if packstructure_result else None,
        },

        "fillrate_match_result": fillrate_match,
        "packstructure_match_result": packstructure_match,
        "summary_pickingVision_fillRate_packStructure": final_summary,
    }

    if output_path is None:
        output_dir = Path("data/picking/summary_pickingVision_fillRate_packStructure")
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = picking_image.stem
        output_path = output_dir / f"{stem}_summary_pickingVision_fillRate_packStructure.json"

    save_json(payload, str(output_path))

    print(f"[OK] Consolidado guardado en: {output_path}")
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))
    return 0


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

        raw_payload = safe_read_json(detected_barcodes_json)
        if raw_payload is None:
            print(f"[ERROR] No pude leer detected_barcodes_json: {detected_barcodes_json}")
            return 2

        if isinstance(raw_payload, dict):
            if isinstance(raw_payload.get("detected_barcodes"), list):
                detected_barcodes = raw_payload.get("detected_barcodes")
            elif isinstance(raw_payload.get("barcodes"), list):
                detected_barcodes = raw_payload.get("barcodes")
            else:
                print("[ERROR] detected_barcodes_json debe contener 'detected_barcodes' o 'barcodes' como lista")
                return 2
        else:
            print("[ERROR] detected_barcodes_json debe ser un JSON tipo objeto")
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

    payload = {
        "status": "success" if closure_result.get("status") == "success" else "error",
        "processed_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary_json": str(summary_json),
        "readout_json": str(readout_json) if readout_json else None,
        "detected_barcodes_json": str(detected_barcodes_json) if detected_barcodes_json else None,
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


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Procesa automáticamente eventos capturados y ejecuta vision_readout, flujo picking+excel o closure_match."
    )

    ap.add_argument(
        "--mode_app",
        type=str,
        choices=["readout", "picking_match", "closure_match"],
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
    ap.add_argument("--detected_barcodes_json", type=str, default=None, help="Ruta a JSON con lista de barcodes detectados.")
    ap.add_argument("--closure_output", type=str, default=None, help="Ruta de salida JSON del cierre final.")

    args = ap.parse_args()

    if args.mode_app == "picking_match":
        if not args.picking_image or not args.picking_excel:
            print("[ERROR] Para --mode_app picking_match debes indicar --picking_image y --picking_excel")
            raise SystemExit(2)

        rc = run_picking_match(
            picking_image=Path(args.picking_image),
            picking_excel=Path(args.picking_excel),
            packstructure_excel=Path(args.packstructure_excel) if args.packstructure_excel else None,
            output_path=Path(args.picking_output) if args.picking_output else None,
        )
        raise SystemExit(rc)

    if args.mode_app == "closure_match":
        if not args.summary_json:
            print("[ERROR] Para --mode_app closure_match debes indicar --summary_json")
            raise SystemExit(2)

        if not args.readout_json and not args.detected_barcodes_json:
            print("[ERROR] Para --mode_app closure_match debes indicar --readout_json o --detected_barcodes_json")
            raise SystemExit(2)

        rc = run_closure_match(
            summary_json=Path(args.summary_json),
            readout_json=Path(args.readout_json) if args.readout_json else None,
            detected_barcodes_json=Path(args.detected_barcodes_json) if args.detected_barcodes_json else None,
            output_path=Path(args.closure_output) if args.closure_output else None,
        )
        raise SystemExit(rc)

    captures_root = Path(args.captures_root)

    if args.once:
        processed = 0
        for event_json_path in iter_event_jsons(captures_root):
            ok = process_event(
                event_json_path,
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