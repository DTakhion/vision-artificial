# app/main.py
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


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


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Procesa automáticamente eventos capturados y ejecuta vision_readout."
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

    args = ap.parse_args()

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