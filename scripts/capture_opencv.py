from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2


def open_camera(device: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    # macOS: AVFoundation
    cap = cv2.VideoCapture(device, cv2.CAP_AVFOUNDATION)

    # A veces conviene setear DESPUÉS de abrir; si la cam no soporta, lo ignorará
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    cap.set(cv2.CAP_PROP_FPS, int(fps))

    return cap


def warmup_read(cap: cv2.VideoCapture, tries: int = 60, sleep_s: float = 0.03):
    """
    En macOS es común que los primeros reads fallen.
    Intentamos varias veces hasta obtener un frame válido.
    """
    last = None
    for _ in range(tries):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            return True, frame
        last = frame
        time.sleep(sleep_s)
    return False, last


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0, help="Index de cámara (0,1,2...)")
    ap.add_argument("--width", type=int, default=640, help="Recomendado en Mac para partir: 640")
    ap.add_argument("--height", type=int, default=480, help="Recomendado en Mac para partir: 480")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out_dir", type=str, default="data/captures/opencv", help="Carpeta de salida")
    ap.add_argument("--save_video", action="store_true", help="Guarda un mp4 además de frames")
    ap.add_argument("--every", type=int, default=15, help="Guarda 1 frame cada N frames (0 desactiva)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = open_camera(args.device, args.width, args.height, args.fps)
    if not cap.isOpened():
        raise SystemExit(
            f"No pude abrir la cámara device={args.device}. "
            f"Prueba con --device 1 o 2, o revisa permisos de cámara."
        )

    # Warm-up / primer frame estable
    ok, frame = warmup_read(cap)
    if not ok:
        cap.release()
        raise SystemExit(
            "Abrí la cámara pero no pude leer frames (warm-up falló). "
            "Cierra Zoom/Teams/Chrome/FaceTime y prueba otra resolución (640x480) o otro device."
        )

    h, w = frame.shape[:2]
    print(f"[INFO] Cámara OK. Resolución real: {w}x{h}")

    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        ts = time.strftime("%Y%m%d_%H%M%S")
        video_path = out_dir / f"capture_{ts}.mp4"
        writer = cv2.VideoWriter(str(video_path), fourcc, args.fps, (w, h))
        print(f"[INFO] Grabando video en: {video_path}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    frame_dir = out_dir / f"frames_{ts}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Guardando frames en: {frame_dir}")
    print("[INFO] Controles: 'q' salir | 's' guardar frame manual")

    idx = 0
    saved = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            # Reintento breve en vez de morir al primer fallo
            ok2, frame2 = warmup_read(cap, tries=10, sleep_s=0.02)
            if not ok2:
                print("[WARN] No pude leer frame; saliendo.")
                break
            frame = frame2

        if writer is not None:
            writer.write(frame)

        if args.every > 0 and (idx % args.every == 0):
            fp = frame_dir / f"frame_{idx:06d}.jpg"
            cv2.imwrite(str(fp), frame)
            saved += 1

        disp = frame.copy()
        cv2.putText(
            disp,
            f"device={args.device}  {w}x{h}  idx={idx}  saved={saved}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Capture (OpenCV)", disp)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("s"):
            fp = frame_dir / f"manual_{idx:06d}.jpg"
            cv2.imwrite(str(fp), frame)
            saved += 1
            print(f"[INFO] Guardado manual: {fp}")

        idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print("[DONE] Captura finalizada.")


if __name__ == "__main__":
    main()
