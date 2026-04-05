# backend/main.py

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


app = FastAPI()

# CORS para frontend local (React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # luego restringir
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Paths base
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent  # vision-artificial/
CAPTURE_SCRIPT_PATH = BASE_DIR / "scripts" / "capture_opencv.py"
APP_MAIN_PATH = BASE_DIR / "app" / "main.py"

CAPTURES_DIR = BASE_DIR / "data" / "captures" / "opencv"
CLOSURE_DIR = BASE_DIR / "data" / "closure"

DEFAULT_SESSION_STATE_JSON = CLOSURE_DIR / "session_state_latest.json"

app.mount("/data", StaticFiles(directory=BASE_DIR / "data"), name="data")


# ============================================================
# Request models
# ============================================================
class ProcessRequest(BaseModel):
    event_dir: Optional[str] = None


# ============================================================
# Helpers
# ============================================================
def safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return payload
        return None
    except Exception:
        return None


def path_to_data_url(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None

    try:
        rel = path.resolve().relative_to((BASE_DIR / "data").resolve())
        return f"/data/{rel.as_posix()}"
    except Exception:
        return None


def run_subprocess(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )


def get_latest_session_dir() -> Path:
    sessions = sorted(CAPTURES_DIR.glob("frames_*"), reverse=True)
    if not sessions:
        raise HTTPException(status_code=404, detail="No se encontraron sesiones de captura")
    return sessions[0]


def get_last_manual_event_from_session(session_dir: Path) -> Dict[str, Any]:
    session_file = session_dir / "session.json"
    if not session_file.exists():
        raise HTTPException(status_code=404, detail="No se encontró session.json en la sesión capturada")

    session_data = safe_read_json(session_file)
    if not session_data:
        raise HTTPException(status_code=500, detail="No se pudo leer session.json")

    last_event = session_data.get("last_manual_event")
    if not isinstance(last_event, dict):
        raise HTTPException(status_code=404, detail="No se encontró last_manual_event en session.json")

    return last_event


def resolve_event_dir(event_dir: Optional[str] = None) -> Path:
    """
    Si event_dir viene explícito, lo usamos.
    Si no, resolvemos el último evento manual de la última sesión.
    """
    if event_dir:
        p = Path(event_dir)
        if not p.is_absolute():
            p = (BASE_DIR / p).resolve()
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"No existe event_dir: {p}")
        return p

    latest_session = get_latest_session_dir()
    last_event = get_last_manual_event_from_session(latest_session)

    event_dir_value = last_event.get("event_dir") or last_event.get("dir")
    if not event_dir_value:
        raise HTTPException(status_code=404, detail="last_manual_event no contiene event_dir")

    p = Path(event_dir_value)
    if not p.is_absolute():
        p = (BASE_DIR / p).resolve()

    if not p.exists():
        raise HTTPException(status_code=404, detail=f"No existe el event_dir resuelto: {p}")

    return p


def resolve_target_frame(event_dir: Path) -> Path:
    """
    Intentamos usar el frame canónico del evento.
    """
    candidates = [
        event_dir / "frame.jpg",
        event_dir / "roi.jpg",
        event_dir / "frames" / "frame_02.jpg",
        event_dir / "frames" / "frame_03.jpg",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise HTTPException(
        status_code=404,
        detail=f"No se encontró frame objetivo en el evento: {event_dir}",
    )


def build_capture_response(event_dir: Path) -> Dict[str, Any]:
    frame_path = resolve_target_frame(event_dir)
    event_json_path = event_dir / "event.json"
    readout_json_path = event_dir / "readout_result.json"
    readout_vis_path = event_dir / "readout_vis.jpg"

    return {
        "event_dir": str(event_dir),
        "event_json": str(event_json_path) if event_json_path.exists() else None,
        "frame_path": str(frame_path),
        "frame_url": path_to_data_url(frame_path),
        "readout_json": str(readout_json_path) if readout_json_path.exists() else None,
        "readout_json_url": path_to_data_url(readout_json_path) if readout_json_path.exists() else None,
        "readout_vis": str(readout_vis_path) if readout_vis_path.exists() else None,
        "readout_vis_url": path_to_data_url(readout_vis_path) if readout_vis_path.exists() else None,
    }


# ============================================================
# Endpoints
# ============================================================
@app.get("/vision/health")
def health():
    return {"status": "ok"}


@app.post("/vision/capture")
def capture():
    """
    Solo captura.
    Abre OpenCV, el operario presiona 'e', y devolvemos el evento generado.
    """
    try:
        print("[INFO] Ejecutando captura OpenCV...")

        cmd = [
            sys.executable,
            str(CAPTURE_SCRIPT_PATH),
            "--device", "0",
            "--width", "1920",
            "--height", "1080",
            "--fps", "30",
            "--out_dir", "data/captures/opencv",
            "--events",
            "--roi", "80", "80", "1700", "900",
            "--every", "0",
        ]

        proc = run_subprocess(cmd, cwd=BASE_DIR)

        if proc.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Falló la captura OpenCV",
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                    "cmd": cmd,
                },
            )

        latest_session = get_latest_session_dir()
        last_event = get_last_manual_event_from_session(latest_session)

        event_dir_value = last_event.get("event_dir") or last_event.get("dir")
        if not event_dir_value:
            raise HTTPException(status_code=404, detail="No se encontró event_dir en last_manual_event")

        event_dir = Path(event_dir_value)
        if not event_dir.is_absolute():
            event_dir = (BASE_DIR / event_dir).resolve()

        if not event_dir.exists():
            raise HTTPException(status_code=404, detail=f"No existe el directorio del evento: {event_dir}")

        event_info = build_capture_response(event_dir)

        return {
            "status": "success",
            "message": "Captura realizada correctamente",
            "session_dir": str(latest_session),
            "session_dir_url": path_to_data_url(latest_session / "session.json"),
            "event": event_info,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/vision/process")
def process_capture(payload: ProcessRequest):
    """
    Procesa una captura concreta:
    1) readout híbrido sobre el frame del evento
    2) closure_iterative con el readout generado
    """
    try:
        event_dir = resolve_event_dir(payload.event_dir)
        frame_path = resolve_target_frame(event_dir)

        readout_json_path = event_dir / "readout_result.json"
        readout_vis_path = event_dir / "readout_vis.jpg"
        closure_output_path = event_dir / "closure_iterative_result.json"

        print(f"[INFO] Procesando evento: {event_dir}")
        print(f"[INFO] Frame objetivo: {frame_path}")

        # --------------------------------------------------------
        # 1) Readout híbrido
        # --------------------------------------------------------
        readout_cmd = [
            sys.executable,
            "-m",
            "utils.vision_readout_hybrid",
            str(frame_path),
            "--save-json",
            "--json-out",
            str(readout_json_path),
            "--save-vis",
            "--vis-out",
            str(readout_vis_path),
        ]

        readout_proc = run_subprocess(readout_cmd, cwd=BASE_DIR)

        if readout_proc.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Falló el readout híbrido",
                    "stdout": readout_proc.stdout,
                    "stderr": readout_proc.stderr,
                    "cmd": readout_cmd,
                },
            )

        if not readout_json_path.exists():
            raise HTTPException(
                status_code=500,
                detail="El readout terminó, pero no se generó readout_result.json",
            )

        # --------------------------------------------------------
        # 2) Closure iterative
        # --------------------------------------------------------
        closure_cmd = [
            sys.executable,
            "-m",
            "app.main",
            "--mode_app",
            "closure_iterative",
            "--readout_json",
            str(readout_json_path),
            "--session_state_json",
            str(DEFAULT_SESSION_STATE_JSON),
            "--closure_output",
            str(closure_output_path),
        ]

        closure_proc = run_subprocess(closure_cmd, cwd=BASE_DIR)

        if closure_proc.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Falló closure_iterative",
                    "stdout": closure_proc.stdout,
                    "stderr": closure_proc.stderr,
                    "cmd": closure_cmd,
                },
            )

        closure_payload = safe_read_json(closure_output_path)
        if not closure_payload:
            raise HTTPException(
                status_code=500,
                detail="closure_iterative terminó, pero no se pudo leer el JSON de salida",
            )

        frontend_summary = closure_payload.get("frontend_summary")
        operator_feedback = closure_payload.get("operator_feedback")
        session_payload = closure_payload.get("session")
        closure_result = closure_payload.get("closure_result")

        return {
            "status": "success",
            "message": "Captura procesada correctamente",
            "event": {
                "event_dir": str(event_dir),
                "frame_path": str(frame_path),
                "frame_url": path_to_data_url(frame_path),
                "readout_json": str(readout_json_path),
                "readout_json_url": path_to_data_url(readout_json_path),
                "readout_vis": str(readout_vis_path) if readout_vis_path.exists() else None,
                "readout_vis_url": path_to_data_url(readout_vis_path) if readout_vis_path.exists() else None,
                "closure_output": str(closure_output_path),
                "closure_output_url": path_to_data_url(closure_output_path),
            },
            "session_state_json": str(DEFAULT_SESSION_STATE_JSON),
            "session_state_json_url": path_to_data_url(DEFAULT_SESSION_STATE_JSON),
            "frontend_summary": frontend_summary,
            "operator_feedback": operator_feedback,
            "session": session_payload,
            "closure_result": closure_result,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/vision/session/reset")
def reset_closure_session():
    """
    Reinicia la sesión iterativa para comenzar un nuevo picking.
    """
    try:
        closure_output_path = CLOSURE_DIR / "closure_iterative_reset_result.json"

        reset_cmd = [
            sys.executable,
            "-m",
            "app.main",
            "--mode_app",
            "closure_iterative",
            "--session_state_json",
            str(DEFAULT_SESSION_STATE_JSON),
            "--closure_output",
            str(closure_output_path),
            "--reset_session",
        ]

        proc = run_subprocess(reset_cmd, cwd=BASE_DIR)

        if proc.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Falló el reset de la sesión iterativa",
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                    "cmd": reset_cmd,
                },
            )

        session_payload = safe_read_json(DEFAULT_SESSION_STATE_JSON)

        return {
            "status": "success",
            "message": "Sesión iterativa reiniciada correctamente",
            "session_state_json": str(DEFAULT_SESSION_STATE_JSON),
            "session_state_json_url": path_to_data_url(DEFAULT_SESSION_STATE_JSON),
            "session": session_payload,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))