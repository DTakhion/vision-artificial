# # scripts/capture_opencv.py
# from __future__ import annotations

# import argparse
# import json
# import time
# from pathlib import Path
# from typing import Any, Dict, Optional, Tuple

# import cv2


# def open_camera(device: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
#     # macOS: AVFoundation
#     cap = cv2.VideoCapture(device, cv2.CAP_AVFOUNDATION)

#     # A veces conviene setear DESPUÉS de abrir; si la cam no soporta, lo ignorará
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
#     cap.set(cv2.CAP_PROP_FPS, int(fps))

#     return cap


# def warmup_read(cap: cv2.VideoCapture, tries: int = 60, sleep_s: float = 0.03):
#     """
#     En macOS es común que los primeros reads fallen.
#     Intentamos varias veces hasta obtener un frame válido.
#     """
#     last = None
#     for _ in range(tries):
#         ok, frame = cap.read()
#         if ok and frame is not None and frame.size > 0:
#             return True, frame
#         last = frame
#         time.sleep(sleep_s)
#     return False, last


# def safe_write_json(path: Path, payload: dict) -> None:
#     path.parent.mkdir(parents=True, exist_ok=True)
#     tmp = path.with_suffix(path.suffix + ".tmp")
#     with tmp.open("w", encoding="utf-8") as f:
#         json.dump(payload, f, ensure_ascii=False, indent=2)
#     tmp.replace(path)


# def crop_roi(img, roi: Optional[Tuple[int, int, int, int]]):
#     if roi is None:
#         return None
#     x, y, rw, rh = roi
#     x = max(0, int(x))
#     y = max(0, int(y))
#     rw = max(1, int(rw))
#     rh = max(1, int(rh))
#     return img[y : y + rh, x : x + rw].copy()


# def to_gray_blur(img):
#     g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     g = cv2.GaussianBlur(g, (7, 7), 0)
#     return g


# def motion_ratio(prev_gray, curr_gray, thresh: int = 25) -> float:
#     diff = cv2.absdiff(prev_gray, curr_gray)
#     _, bw = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
#     changed = cv2.countNonZero(bw)
#     total = bw.shape[0] * bw.shape[1]
#     return changed / max(1, total)


# def save_event(
#     *,
#     frame_dir: Path,
#     events_dir: Path,
#     event_id: int,
#     frame,
#     idx: int,
#     roi: Optional[Tuple[int, int, int, int]],
#     trigger: str,
#     auto_metrics: Optional[Dict[str, Any]] = None,
# ) -> Path:
#     epoch_ms = int(time.time() * 1000)
#     ev_dir = events_dir / f"event_{event_id:06d}"
#     ev_dir.mkdir(parents=True, exist_ok=True)

#     frame_path = ev_dir / "frame.jpg"
#     cv2.imwrite(str(frame_path), frame)

#     roi_path = None
#     roi_img = crop_roi(frame, roi) if roi else None
#     if roi_img is not None and roi_img.size > 0:
#         roi_path = ev_dir / "roi.jpg"
#         cv2.imwrite(str(roi_path), roi_img)

#     ev = {
#         "event_id": event_id,
#         "trigger": trigger,  # "manual" | "auto"
#         "event_time_epoch_ms": epoch_ms,
#         "event_time_local": time.strftime("%Y-%m-%d %H:%M:%S"),
#         "frame_idx": idx,
#         "paths": {"frame": str(frame_path), "roi": (str(roi_path) if roi_path else None)},
#         "roi": ({"x": roi[0], "y": roi[1], "w": roi[2], "h": roi[3]} if roi else None),
#         # Lecturas: en PoC puede ser "not_attempted" y está perfecto (trazabilidad primero)
#         "readout": {"barcode": None, "qrcode": None, "serial": None, "status": "not_attempted"},
#         "auto_metrics": auto_metrics,
#     }
#     safe_write_json(ev_dir / "event.json", ev)
#     return ev_dir


# def main() -> None:
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--device", type=int, default=0, help="Index de cámara (0,1,2...)")
#     ap.add_argument("--width", type=int, default=640, help="Recomendado en Mac para partir: 640")
#     ap.add_argument("--height", type=int, default=480, help="Recomendado en Mac para partir: 480")
#     ap.add_argument("--fps", type=int, default=30)
#     ap.add_argument("--out_dir", type=str, default="data/captures/opencv", help="Carpeta de salida")
#     ap.add_argument("--save_video", action="store_true", help="Guarda un mp4 además de frames")
#     ap.add_argument("--every", type=int, default=15, help="Guarda 1 frame cada N frames (0 desactiva)")
#     ap.add_argument("--no_display", action="store_true", help="Modo headless (sin ventana). Cortar con Ctrl+C.")
#     ap.add_argument("--fps_window", type=int, default=30, help="Ventana (en frames) para estimar FPS real")

#     # --- Eventos (trazabilidad del “gesto” en mesón) ---
#     ap.add_argument("--events", action="store_true", help="Habilita guardado de eventos (frame+event.json)")
#     ap.add_argument(
#         "--roi",
#         type=int,
#         nargs=4,
#         metavar=("X", "Y", "W", "H"),
#         default=None,
#         help="ROI opcional para el código/gesto (x y w h). Ej: --roi 200 200 800 300",
#     )

#     # --- Auto-eventos por estabilidad en ROI (PoC demostrable sin modelos) ---
#     ap.add_argument("--auto_events", action="store_true", help="Disparo automático de eventos por estabilidad en ROI")
#     ap.add_argument("--enter_thr", type=float, default=0.08, help="Motion ratio para detectar entrada/movimiento fuerte (0-1)")
#     ap.add_argument("--stable_thr", type=float, default=0.01, help="Motion ratio para considerar ROI estable (0-1)")
#     ap.add_argument("--stable_frames", type=int, default=25, help="Frames estables consecutivos para disparar evento")
#     ap.add_argument("--cooldown_s", type=float, default=2.0, help="Cooldown tras disparar un evento automático")

#     args = ap.parse_args()

#     out_dir = Path(args.out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     roi: Optional[Tuple[int, int, int, int]] = tuple(args.roi) if args.roi else None

#     cap = open_camera(args.device, args.width, args.height, args.fps)
#     if not cap.isOpened():
#         raise SystemExit(
#             f"No pude abrir la cámara device={args.device}. "
#             f"Prueba con --device 1 o 2, o revisa permisos de cámara."
#         )

#     # Warm-up / primer frame estable
#     ok, frame = warmup_read(cap)
#     if not ok:
#         cap.release()
#         raise SystemExit(
#             "Abrí la cámara pero no pude leer frames (warm-up falló). "
#             "Cierra Zoom/Teams/Chrome/FaceTime y prueba otra resolución (640x480) o otro device."
#         )

#     h, w = frame.shape[:2]
#     print(f"[INFO] Cámara OK. Resolución real: {w}x{h}")

#     # --- session folder ---
#     ts = time.strftime("%Y%m%d_%H%M%S")
#     frame_dir = out_dir / f"frames_{ts}"
#     frame_dir.mkdir(parents=True, exist_ok=True)
#     print(f"[INFO] Guardando frames en: {frame_dir}")

#     # --- events folder ---
#     events_dir = frame_dir / "events"
#     if args.events:
#         events_dir.mkdir(parents=True, exist_ok=True)
#         print(f"[INFO] Eventos habilitados. Carpeta: {events_dir}")
#         if args.auto_events:
#             if roi is None:
#                 print("[WARN] auto_events sin ROI: funcionará, pero puede ser más ruidoso. Recomiendo definir --roi.")
#             print(
#                 f"[INFO] Auto-eventos ON | enter_thr={args.enter_thr} stable_thr={args.stable_thr} "
#                 f"stable_frames={args.stable_frames} cooldown_s={args.cooldown_s}"
#             )

#     # --- metadata / trace ---
#     session_path = frame_dir / "session.json"
#     session: Dict[str, Any] = {
#         "session_id": ts,
#         "start_time_local": time.strftime("%Y-%m-%d %H:%M:%S"),
#         "start_time_epoch": time.time(),
#         "camera": {
#             "device": args.device,
#             "requested": {"width": args.width, "height": args.height, "fps": args.fps},
#             "actual": {"width": w, "height": h},
#         },
#         "capture": {
#             "out_dir": str(out_dir),
#             "frame_dir": str(frame_dir),
#             "every": args.every,
#             "save_video": bool(args.save_video),
#             "video_path": None,
#         },
#         "events": {
#             "enabled": bool(args.events),
#             "auto_enabled": bool(args.events and args.auto_events),
#             "roi": ({"x": roi[0], "y": roi[1], "w": roi[2], "h": roi[3]} if roi else None),
#             "counts": {"total": 0, "manual": 0, "auto": 0},
#         },
#         "runtime": {
#             "no_display": bool(args.no_display),
#             "fps_window": args.fps_window,
#             "fps_real_last": None,
#             "frames_total": 0,
#             "frames_saved": 0,
#             "manual_saved": 0,
#         },
#         "end_time_local": None,
#         "end_time_epoch": None,
#         "status": "running",
#     }
#     safe_write_json(session_path, session)

#     # --- optional video ---
#     writer = None
#     if args.save_video:
#         fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#         video_path = frame_dir / f"capture_{ts}.mp4"
#         writer = cv2.VideoWriter(str(video_path), fourcc, args.fps, (w, h))
#         if not writer.isOpened():
#             print("[WARN] No pude abrir VideoWriter con mp4v. Continuaré sin video.")
#             writer = None
#         else:
#             session["capture"]["video_path"] = str(video_path)
#             safe_write_json(session_path, session)
#             print(f"[INFO] Grabando video en: {video_path}")

#     if not args.no_display:
#         if args.events:
#             print("[INFO] Controles: 'q' salir | 's' guardar frame manual | 'e' guardar EVENTO (manual)")
#         else:
#             print("[INFO] Controles: 'q' salir | 's' guardar frame manual")
#     else:
#         print("[INFO] Headless activo: Ctrl+C para salir")

#     idx = 0
#     saved = 0
#     manual_saved = 0

#     # --- FPS rolling ---
#     fps_win = max(5, int(args.fps_window))
#     win_start_t = time.time()
#     win_start_idx = 0
#     fps_real: Optional[float] = None

#     # --- AUTO EVENT state ---
#     prev_roi_gray = None
#     stable_count = 0
#     armed = True
#     cooldown_until = 0.0
#     last_motion: Optional[float] = None

#     # --- EVENT counters ---
#     event_id = 0

#     try:
#         while True:
#             ok, frame = cap.read()
#             if not ok or frame is None:
#                 # Reintento breve en vez de morir al primer fallo
#                 ok2, frame2 = warmup_read(cap, tries=10, sleep_s=0.02)
#                 if not ok2:
#                     print("[WARN] No pude leer frame; saliendo.")
#                     break
#                 frame = frame2

#             if writer is not None:
#                 writer.write(frame)

#             # --- periodic frame saving ---
#             if args.every > 0 and (idx % args.every == 0):
#                 epoch_ms = int(time.time() * 1000)
#                 fp = frame_dir / f"frame_{idx:06d}_{epoch_ms}.jpg"
#                 cv2.imwrite(str(fp), frame)
#                 saved += 1

#             # --- FPS rolling ---
#             if (idx - win_start_idx) >= fps_win:
#                 dt = time.time() - win_start_t
#                 if dt > 1e-6:
#                     fps_real = (idx - win_start_idx) / dt
#                 win_start_t = time.time()
#                 win_start_idx = idx

#             # --- AUTO EVENT LOGIC (ROI motion/stability) ---
#             auto_trigger = False
#             auto_metrics = None

#             if args.events and args.auto_events:
#                 roi_img = crop_roi(frame, roi) if roi else frame
#                 if roi_img is not None and roi_img.size > 0:
#                     curr = to_gray_blur(roi_img)

#                     if prev_roi_gray is not None:
#                         mr = motion_ratio(prev_roi_gray, curr)
#                         last_motion = mr

#                         now = time.time()
#                         if now < cooldown_until:
#                             stable_count = 0
#                             armed = False
#                         else:
#                             # Movimiento fuerte => re-armar (objeto entrando o reacomodo)
#                             if mr > args.enter_thr:
#                                 armed = True
#                                 stable_count = 0

#                             # Si está armado y estable, contamos
#                             if armed and mr < args.stable_thr:
#                                 stable_count += 1
#                             else:
#                                 stable_count = 0

#                             if armed and stable_count >= args.stable_frames:
#                                 auto_trigger = True
#                                 auto_metrics = {
#                                     "motion_ratio": float(mr),
#                                     "stable_frames_required": int(args.stable_frames),
#                                     "enter_thr": float(args.enter_thr),
#                                     "stable_thr": float(args.stable_thr),
#                                 }
#                                 armed = False
#                                 stable_count = 0
#                                 cooldown_until = time.time() + float(args.cooldown_s)

#                     prev_roi_gray = curr

#             # --- FIRE AUTO EVENT ---
#             if auto_trigger and args.events:
#                 event_id += 1
#                 ev_dir = save_event(
#                     frame_dir=frame_dir,
#                     events_dir=events_dir,
#                     event_id=event_id,
#                     frame=frame,
#                     idx=idx,
#                     roi=roi,
#                     trigger="auto",
#                     auto_metrics=auto_metrics,
#                 )
#                 session["events"]["counts"]["total"] += 1
#                 session["events"]["counts"]["auto"] += 1
#                 safe_write_json(session_path, session)
#                 print(f"[AUTO] Evento #{event_id} guardado: {ev_dir}")

#             # --- update session stats sparse ---
#             if idx % fps_win == 0:
#                 session["runtime"]["fps_real_last"] = (round(fps_real, 2) if fps_real is not None else None)
#                 session["runtime"]["frames_total"] = idx
#                 session["runtime"]["frames_saved"] = saved
#                 session["runtime"]["manual_saved"] = manual_saved
#                 safe_write_json(session_path, session)

#             # --- UI / key handling ---
#             if not args.no_display:
#                 disp = frame.copy()

#                 # draw ROI box if set
#                 if roi is not None:
#                     x, y, rw, rh = roi
#                     cv2.rectangle(disp, (x, y), (x + rw, y + rh), (0, 255, 255), 2)

#                 fps_txt = f"{fps_real:.1f}" if fps_real is not None else "..."
#                 mr_txt = f"{last_motion:.3f}" if last_motion is not None else "..."
#                 cv2.putText(
#                     disp,
#                     f"device={args.device} {w}x{h} idx={idx} saved={saved} fps={fps_txt}",
#                     (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.75,
#                     (0, 255, 0),
#                     2,
#                     cv2.LINE_AA,
#                 )

#                 if args.events and args.auto_events:
#                     cv2.putText(
#                         disp,
#                         f"AUTO motion={mr_txt} stable={stable_count}/{args.stable_frames} armed={1 if armed else 0}",
#                         (10, 60),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.65,
#                         (0, 255, 255),
#                         2,
#                         cv2.LINE_AA,
#                     )

#                 cv2.imshow("Capture (OpenCV)", disp)
#                 key = cv2.waitKey(1) & 0xFF

#                 if key == ord("q"):
#                     break

#                 if key == ord("s"):
#                     epoch_ms = int(time.time() * 1000)
#                     fp = frame_dir / f"manual_{idx:06d}_{epoch_ms}.jpg"
#                     cv2.imwrite(str(fp), frame)
#                     saved += 1
#                     manual_saved += 1
#                     print(f"[INFO] Guardado manual: {fp}")

#                 if key == ord("e") and args.events:
#                     event_id += 1
#                     ev_dir = save_event(
#                         frame_dir=frame_dir,
#                         events_dir=events_dir,
#                         event_id=event_id,
#                         frame=frame,
#                         idx=idx,
#                         roi=roi,
#                         trigger="manual",
#                         auto_metrics=None,
#                     )
#                     session["events"]["counts"]["total"] += 1
#                     session["events"]["counts"]["manual"] += 1
#                     safe_write_json(session_path, session)
#                     print(f"[EVENT] Manual #{event_id} guardado: {ev_dir}")

#             idx += 1

#     except KeyboardInterrupt:
#         print("\n[INFO] Interrumpido por usuario (Ctrl+C). Cerrando...")

#     finally:
#         cap.release()
#         if writer is not None:
#             writer.release()
#         if not args.no_display:
#             cv2.destroyAllWindows()

#         session["runtime"]["fps_real_last"] = (round(fps_real, 2) if fps_real is not None else None)
#         session["runtime"]["frames_total"] = idx
#         session["runtime"]["frames_saved"] = saved
#         session["runtime"]["manual_saved"] = manual_saved
#         session["end_time_local"] = time.strftime("%Y-%m-%d %H:%M:%S")
#         session["end_time_epoch"] = time.time()
#         session["status"] = "done"
#         safe_write_json(session_path, session)

#         print("[DONE] Captura finalizada.")
#         print(f"[TRACE] session.json: {session_path}")


# if __name__ == "__main__":
#     main()

# scripts/capture_opencv.py
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2


def open_camera(device: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(device, cv2.CAP_AVFOUNDATION)  # macOS: AVFoundation
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    cap.set(cv2.CAP_PROP_FPS, int(fps))
    return cap


def warmup_read(cap: cv2.VideoCapture, tries: int = 60, sleep_s: float = 0.03):
    last = None
    for _ in range(tries):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            return True, frame
        last = frame
        time.sleep(sleep_s)
    return False, last


def safe_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def crop_roi(img, roi: Optional[Tuple[int, int, int, int]]):
    if roi is None:
        return None
    x, y, rw, rh = roi
    x = max(0, int(x))
    y = max(0, int(y))
    rw = max(1, int(rw))
    rh = max(1, int(rh))
    return img[y : y + rh, x : x + rw].copy()


def to_gray_blur(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (7, 7), 0)
    return g


def motion_ratio(prev_gray, curr_gray, thresh: int = 25) -> float:
    diff = cv2.absdiff(prev_gray, curr_gray)
    _, bw = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
    changed = cv2.countNonZero(bw)
    total = bw.shape[0] * bw.shape[1]
    return changed / max(1, total)


def save_event(
    *,
    frame_dir: Path,
    events_dir: Path,
    event_id: int,
    frame,
    idx: int,
    roi: Optional[Tuple[int, int, int, int]],
    trigger: str,
    auto_metrics: Optional[Dict[str, Any]] = None,
) -> Path:
    epoch_ms = int(time.time() * 1000)
    ev_dir = events_dir / f"event_{event_id:06d}"
    ev_dir.mkdir(parents=True, exist_ok=True)

    frame_path = ev_dir / "frame.jpg"
    cv2.imwrite(str(frame_path), frame)

    roi_path = None
    roi_img = crop_roi(frame, roi) if roi else None
    if roi_img is not None and roi_img.size > 0:
        roi_path = ev_dir / "roi.jpg"
        cv2.imwrite(str(roi_path), roi_img)

    ev = {
        "event_id": event_id,
        "trigger": trigger,  # "manual" | "auto"
        "event_time_epoch_ms": epoch_ms,
        "event_time_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "frame_idx": idx,
        "paths": {"frame": str(frame_path), "roi": (str(roi_path) if roi_path else None)},
        "roi": ({"x": roi[0], "y": roi[1], "w": roi[2], "h": roi[3]} if roi else None),
        "readout": {"barcode": None, "qrcode": None, "serial": None, "status": "not_attempted"},
        "auto_metrics": auto_metrics,
    }
    safe_write_json(ev_dir / "event.json", ev)
    return ev_dir


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out_dir", type=str, default="data/captures/opencv")
    ap.add_argument("--save_video", action="store_true")
    ap.add_argument("--every", type=int, default=15, help="Guarda 1 frame cada N frames (0 desactiva)")
    ap.add_argument("--no_display", action="store_true", help="Headless. Cortar con Ctrl+C.")
    ap.add_argument("--fps_window", type=int, default=30)

    # Eventos
    ap.add_argument("--events", action="store_true", help="Guarda eventos (frame+event.json)")
    ap.add_argument(
        "--roi",
        type=int,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        default=None,
        help="ROI (x y w h) para el mesón / gesto. Ej: --roi 240 180 800 360",
    )

    # Auto-eventos
    ap.add_argument("--auto_events", action="store_true", help="Disparo automático de eventos")
    ap.add_argument(
        "--auto_method",
        type=str,
        default="bg",
        choices=["bg", "motion"],
        help="Método auto: bg (MOG2+contornos) o motion (diff simple)",
    )

    # Auto por MOTION (tu método original)
    ap.add_argument("--enter_thr", type=float, default=0.08, help="motion_ratio para re-armar (0-1)")
    ap.add_argument("--stable_thr", type=float, default=0.01, help="motion_ratio para considerar estable (0-1)")
    ap.add_argument("--stable_frames", type=int, default=25, help="frames estables para disparar")
    ap.add_argument("--cooldown_s", type=float, default=2.0, help="cooldown tras disparar")

    # Auto por BG (robusto)
    ap.add_argument("--bg_warmup", type=int, default=45, help="frames para aprender fondo antes de disparar")
    ap.add_argument("--min_fg_ratio", type=float, default=0.02, help="ratio de foreground para 'objeto presente'")
    ap.add_argument("--min_contour_area", type=int, default=2500, help="área mínima de contorno para 'objeto presente'")
    ap.add_argument("--present_frames", type=int, default=10, help="frames de presencia para disparar evento")
    ap.add_argument("--bg_history", type=int, default=200)
    ap.add_argument("--bg_var_threshold", type=int, default=16)
    ap.add_argument("--bg_detect_shadows", action="store_true", help="MOG2 detectShadows=True")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    roi: Optional[Tuple[int, int, int, int]] = tuple(args.roi) if args.roi else None

    cap = open_camera(args.device, args.width, args.height, args.fps)
    if not cap.isOpened():
        raise SystemExit(f"No pude abrir cámara device={args.device}. Prueba --device 1/2 y permisos.")

    ok, frame = warmup_read(cap)
    if not ok:
        cap.release()
        raise SystemExit("Warm-up falló. Cierra Zoom/Teams/Chrome/FaceTime y prueba 640x480 u otro device.")

    h, w = frame.shape[:2]
    print(f"[INFO] Cámara OK. Resolución real: {w}x{h}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    frame_dir = out_dir / f"frames_{ts}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Guardando en: {frame_dir}")

    # Events dir
    events_dir = frame_dir / "events"
    if args.events:
        events_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Eventos habilitados: {events_dir}")
        if args.auto_events:
            print(f"[INFO] Auto-eventos ON | method={args.auto_method} | cooldown_s={args.cooldown_s}")
            if roi is None:
                print("[WARN] auto_events sin ROI: funcionará, pero para PoC industrial se recomienda ROI en el mesón.")

    # Session trace
    session_path = frame_dir / "session.json"
    session: Dict[str, Any] = {
        "session_id": ts,
        "start_time_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "start_time_epoch": time.time(),
        "camera": {
            "device": args.device,
            "requested": {"width": args.width, "height": args.height, "fps": args.fps},
            "actual": {"width": w, "height": h},
        },
        "capture": {
            "out_dir": str(out_dir),
            "frame_dir": str(frame_dir),
            "every": args.every,
            "save_video": bool(args.save_video),
            "video_path": None,
        },
        "events": {
            "enabled": bool(args.events),
            "auto_enabled": bool(args.events and args.auto_events),
            "auto_method": (args.auto_method if (args.events and args.auto_events) else None),
            "roi": ({"x": roi[0], "y": roi[1], "w": roi[2], "h": roi[3]} if roi else None),
            "counts": {"total": 0, "manual": 0, "auto": 0},
        },
        "runtime": {
            "no_display": bool(args.no_display),
            "fps_window": args.fps_window,
            "fps_real_last": None,
            "frames_total": 0,
            "frames_saved": 0,
            "manual_saved": 0,
        },
        "end_time_local": None,
        "end_time_epoch": None,
        "status": "running",
    }
    safe_write_json(session_path, session)

    # Optional video
    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_path = frame_dir / f"capture_{ts}.mp4"
        writer = cv2.VideoWriter(str(video_path), fourcc, args.fps, (w, h))
        if not writer.isOpened():
            print("[WARN] VideoWriter mp4v falló. Continuaré sin video.")
            writer = None
        else:
            session["capture"]["video_path"] = str(video_path)
            safe_write_json(session_path, session)
            print(f"[INFO] Grabando video: {video_path}")

    if not args.no_display:
        if args.events:
            print("[INFO] Controles: 'q' salir | 's' frame manual | 'e' EVENTO manual")
        else:
            print("[INFO] Controles: 'q' salir | 's' frame manual")
    else:
        print("[INFO] Headless: Ctrl+C para salir")

    idx = 0
    saved = 0
    manual_saved = 0

    # FPS rolling
    fps_win = max(5, int(args.fps_window))
    win_start_t = time.time()
    win_start_idx = 0
    fps_real: Optional[float] = None

    # Event counters
    event_id = 0

    # Auto state common
    cooldown_until = 0.0
    armed = True

    # Auto state (motion)
    prev_gray = None
    stable_count = 0
    last_motion: Optional[float] = None

    # Auto state (bg)
    bg_sub = None
    bg_warmup_left = int(args.bg_warmup)
    present_count = 0
    last_fg_ratio: Optional[float] = None
    last_max_area: Optional[int] = None

    if args.events and args.auto_events and args.auto_method == "bg":
        bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=int(args.bg_history),
            varThreshold=int(args.bg_var_threshold),
            detectShadows=bool(args.bg_detect_shadows),
        )

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                ok2, frame2 = warmup_read(cap, tries=10, sleep_s=0.02)
                if not ok2:
                    print("[WARN] No pude leer frame; saliendo.")
                    break
                frame = frame2

            if writer is not None:
                writer.write(frame)

            # periodic frames
            if args.every > 0 and (idx % args.every == 0):
                epoch_ms = int(time.time() * 1000)
                fp = frame_dir / f"frame_{idx:06d}_{epoch_ms}.jpg"
                cv2.imwrite(str(fp), frame)
                saved += 1

            # FPS
            if (idx - win_start_idx) >= fps_win:
                dt = time.time() - win_start_t
                if dt > 1e-6:
                    fps_real = (idx - win_start_idx) / dt
                win_start_t = time.time()
                win_start_idx = idx

            # -------- AUTO EVENT ----------
            auto_trigger = False
            auto_metrics = None

            if args.events and args.auto_events:
                now = time.time()
                if now < cooldown_until:
                    # In cooldown
                    armed = False
                    stable_count = 0
                    present_count = 0
                else:
                    roi_img = crop_roi(frame, roi) if roi else frame
                    if roi_img is not None and roi_img.size > 0:
                        if args.auto_method == "motion":
                            curr = to_gray_blur(roi_img)
                            if prev_gray is not None:
                                mr = motion_ratio(prev_gray, curr)
                                last_motion = mr

                                if mr > args.enter_thr:
                                    armed = True
                                    stable_count = 0

                                if armed and mr < args.stable_thr:
                                    stable_count += 1
                                else:
                                    stable_count = 0

                                if armed and stable_count >= args.stable_frames:
                                    auto_trigger = True
                                    auto_metrics = {
                                        "method": "motion",
                                        "motion_ratio": float(mr),
                                        "enter_thr": float(args.enter_thr),
                                        "stable_thr": float(args.stable_thr),
                                        "stable_frames": int(args.stable_frames),
                                    }
                                    armed = False
                                    stable_count = 0
                                    cooldown_until = time.time() + float(args.cooldown_s)

                            prev_gray = curr

                        else:
                            # bg method: MOG2 + contornos grandes
                            assert bg_sub is not None

                            # Warmup de fondo (aprende entorno sin disparar)
                            fg = bg_sub.apply(roi_img)
                            if bg_warmup_left > 0:
                                bg_warmup_left -= 1
                                present_count = 0
                                armed = True  # queda listo al terminar warmup
                            else:
                                # limpiar máscara: quitar sombras si existieran (valor 127)
                                _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

                                # morfología para robustez
                                fg = cv2.medianBlur(fg, 5)
                                fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, None, iterations=1)
                                fg = cv2.morphologyEx(fg, cv2.MORPH_DILATE, None, iterations=1)

                                fg_pixels = cv2.countNonZero(fg)
                                total = fg.shape[0] * fg.shape[1]
                                fg_ratio = fg_pixels / max(1, total)
                                last_fg_ratio = fg_ratio

                                contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                max_area = 0
                                for c in contours:
                                    a = int(cv2.contourArea(c))
                                    if a > max_area:
                                        max_area = a
                                last_max_area = max_area

                                present = (fg_ratio >= float(args.min_fg_ratio)) or (max_area >= int(args.min_contour_area))

                                if present:
                                    present_count += 1
                                else:
                                    present_count = 0
                                    armed = True  # si desaparece, re-armamos

                                # Disparo cuando hay presencia sostenida
                                if armed and present_count >= int(args.present_frames):
                                    auto_trigger = True
                                    auto_metrics = {
                                        "method": "bg",
                                        "fg_ratio": float(fg_ratio),
                                        "max_contour_area": int(max_area),
                                        "min_fg_ratio": float(args.min_fg_ratio),
                                        "min_contour_area": int(args.min_contour_area),
                                        "present_frames": int(args.present_frames),
                                        "bg_warmup": int(args.bg_warmup),
                                        "bg_history": int(args.bg_history),
                                        "bg_var_threshold": int(args.bg_var_threshold),
                                        "detect_shadows": bool(args.bg_detect_shadows),
                                    }
                                    armed = False
                                    present_count = 0
                                    cooldown_until = time.time() + float(args.cooldown_s)

            if auto_trigger and args.events:
                event_id += 1
                ev_dir = save_event(
                    frame_dir=frame_dir,
                    events_dir=events_dir,
                    event_id=event_id,
                    frame=frame,
                    idx=idx,
                    roi=roi,
                    trigger="auto",
                    auto_metrics=auto_metrics,
                )
                session["events"]["counts"]["total"] += 1
                session["events"]["counts"]["auto"] += 1
                safe_write_json(session_path, session)
                print(f"[AUTO] Evento #{event_id} guardado: {ev_dir}")

            # update session sparse
            if idx % fps_win == 0:
                session["runtime"]["fps_real_last"] = (round(fps_real, 2) if fps_real is not None else None)
                session["runtime"]["frames_total"] = idx
                session["runtime"]["frames_saved"] = saved
                session["runtime"]["manual_saved"] = manual_saved
                safe_write_json(session_path, session)

            # UI
            if not args.no_display:
                disp = frame.copy()
                if roi is not None:
                    x, y, rw, rh = roi
                    cv2.rectangle(disp, (x, y), (x + rw, y + rh), (0, 255, 255), 2)

                fps_txt = f"{fps_real:.1f}" if fps_real is not None else "..."
                cv2.putText(
                    disp,
                    f"{args.auto_method.upper()} device={args.device} {w}x{h} idx={idx} saved={saved} fps={fps_txt}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                if args.events and args.auto_events:
                    if args.auto_method == "motion":
                        mr_txt = f"{last_motion:.3f}" if last_motion is not None else "..."
                        cv2.putText(
                            disp,
                            f"AUTO(motion) motion={mr_txt} stable={stable_count}/{args.stable_frames} armed={1 if armed else 0}",
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.62,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                    else:
                        fg_txt = f"{last_fg_ratio:.3f}" if last_fg_ratio is not None else "..."
                        ma_txt = f"{last_max_area}" if last_max_area is not None else "..."
                        cv2.putText(
                            disp,
                            f"AUTO(bg) fg={fg_txt} maxA={ma_txt} present={present_count}/{args.present_frames} warmup_left={bg_warmup_left}",
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.62,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )

                cv2.imshow("Capture (OpenCV)", disp)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break
                if key == ord("s"):
                    epoch_ms = int(time.time() * 1000)
                    fp = frame_dir / f"manual_{idx:06d}_{epoch_ms}.jpg"
                    cv2.imwrite(str(fp), frame)
                    saved += 1
                    manual_saved += 1
                    print(f"[INFO] Guardado manual: {fp}")
                if key == ord("e") and args.events:
                    event_id += 1
                    ev_dir = save_event(
                        frame_dir=frame_dir,
                        events_dir=events_dir,
                        event_id=event_id,
                        frame=frame,
                        idx=idx,
                        roi=roi,
                        trigger="manual",
                        auto_metrics=None,
                    )
                    session["events"]["counts"]["total"] += 1
                    session["events"]["counts"]["manual"] += 1
                    safe_write_json(session_path, session)
                    print(f"[EVENT] Manual #{event_id} guardado: {ev_dir}")

            idx += 1

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C. Cerrando...")

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

        session["runtime"]["fps_real_last"] = (round(fps_real, 2) if fps_real is not None else None)
        session["runtime"]["frames_total"] = idx
        session["runtime"]["frames_saved"] = saved
        session["runtime"]["manual_saved"] = manual_saved
        session["end_time_local"] = time.strftime("%Y-%m-%d %H:%M:%S")
        session["end_time_epoch"] = time.time()
        session["status"] = "done"
        safe_write_json(session_path, session)

        print("[DONE] Captura finalizada.")
        print(f"[TRACE] session.json: {session_path}")


if __name__ == "__main__":
    main()