# backend/main.py 

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fastapi.staticfiles import StaticFiles



import subprocess
import json
from pathlib import Path

app = FastAPI()

# CORS para frontend local (React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # luego lo puedes restringir
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths base
BASE_DIR = Path(__file__).resolve().parent.parent  # vision-artificial/
SCRIPT_PATH = BASE_DIR / "scripts" / "capture_opencv.py"
CAPTURES_DIR = BASE_DIR / "data" / "captures" / "opencv"

app.mount("/data", StaticFiles(directory=BASE_DIR / "data"), name="data")


@app.get("/vision/health")
def health():
    return {"status": "ok"}


@app.post("/vision/capture")
def capture():
    try:
        print("[INFO] Ejecutando captura OpenCV...")

        # 🚀 Ejecuta tu script (bloqueante)
        subprocess.run(
            [
                "python3",
                str(SCRIPT_PATH),
                "--device", "0",
                "--width", "1920",
                "--height", "1080",
                "--fps", "30",
                "--out_dir", "data/captures/opencv",
                "--events",
                "--roi", "80", "80", "1700", "900",
                "--every", "0"
            ],
            cwd=BASE_DIR
        )

        print("[INFO] Captura finalizada, buscando session.json...")

        # 🔍 Buscar la última carpeta de sesión
        sessions = sorted(CAPTURES_DIR.glob("frames_*"), reverse=True)

        if not sessions:
            return {"error": "No se encontraron sesiones"}

        latest_session = sessions[0]
        session_file = latest_session / "session.json"

        if not session_file.exists():
            return {"error": "No se encontró session.json"}

        with open(session_file, "r", encoding="utf-8") as f:
            session_data = json.load(f)

        last_event = session_data.get("last_manual_event")

        if not last_event:
            return {"error": "No se encontró last_manual_event"}

        print("[INFO] Evento encontrado:", last_event)

        return {
            "status": "success",
            "event": last_event
        }

    except Exception as e:
        return {"error": str(e)}