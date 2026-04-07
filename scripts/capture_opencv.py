# # scripts/capture_opencv.py

# from __future__ import annotations

# import argparse
# import json
# import time
# from collections import deque
# from pathlib import Path
# from typing import Any, Deque, Dict, List, Optional, Tuple

# import cv2
# import platform


# # def open_camera(device: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
# #     cap = cv2.VideoCapture(device, cv2.CAP_AVFOUNDATION)  # macOS: AVFoundation
# #     cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
# #     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
# #     cap.set(cv2.CAP_PROP_FPS, int(fps))
# #     return cap

# def open_camera(device: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
#     system = platform.system()

#     # Orden de backends por sistema
#     if system == "Windows":
#         candidates = [
#             ("DSHOW", cv2.CAP_DSHOW),
#             ("MSMF", cv2.CAP_MSMF),
#             ("DEFAULT", None),
#         ]
#     elif system == "Darwin":  # macOS
#         candidates = [
#             ("AVFOUNDATION", cv2.CAP_AVFOUNDATION),
#             ("DEFAULT", None),
#         ]
#     else:  # Linux / otros
#         candidates = [
#             ("DEFAULT", None),
#         ]

#     for backend_name, backend in candidates:
#         if backend is None:
#             cap = cv2.VideoCapture(device)
#         else:
#             cap = cv2.VideoCapture(device, backend)

#         if cap is not None and cap.isOpened():
#             cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
#             cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
#             cap.set(cv2.CAP_PROP_FPS, int(fps))

#             # Revalidamos después de setear propiedades
#             ok, frame = cap.read()
#             if ok and frame is not None and frame.size > 0:
#                 print(f"[INFO] Cámara abierta con backend: {backend_name}")
#                 return cap

#             cap.release()

#     # Si nada funcionó, devolvemos un capture inválido
#     return cv2.VideoCapture()

# def warmup_read(cap: cv2.VideoCapture, tries: int = 60, sleep_s: float = 0.03):
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
#     x2 = min(img.shape[1], x + rw)
#     y2 = min(img.shape[0], y + rh)
#     if x >= x2 or y >= y2:
#         return None
#     return img[y:y2, x:x2].copy()


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


# def get_backend_name(cap: cv2.VideoCapture) -> Optional[str]:
#     try:
#         name = cap.getBackendName()
#         return str(name) if name else None
#     except Exception:
#         return None


# def get_fourcc_str(cap: cv2.VideoCapture) -> Optional[str]:
#     try:
#         fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
#         if fourcc <= 0:
#             return None
#         chars = [
#             chr((fourcc >> 0) & 0xFF),
#             chr((fourcc >> 8) & 0xFF),
#             chr((fourcc >> 16) & 0xFF),
#             chr((fourcc >> 24) & 0xFF),
#         ]
#         s = "".join(chars).strip("\x00").strip()
#         return s or None
#     except Exception:
#         return None


# def frame_record(idx: int, frame, epoch_ms: int) -> Dict[str, Any]:
#     return {
#         "idx": int(idx),
#         "epoch_ms": int(epoch_ms),
#         "frame": frame.copy(),
#     }


# def save_burst_frames(
#     *,
#     ev_dir: Path,
#     burst_records: List[Dict[str, Any]],
#     roi: Optional[Tuple[int, int, int, int]],
# ) -> Dict[str, Any]:
#     frames_dir = ev_dir / "frames"
#     frames_dir.mkdir(parents=True, exist_ok=True)

#     saved_frames = []
#     for i, rec in enumerate(burst_records, start=1):
#         fp = frames_dir / f"frame_{i:02d}.jpg"
#         cv2.imwrite(str(fp), rec["frame"])
#         saved_frames.append(
#             {
#                 "index_in_burst": i,
#                 "frame_idx": int(rec["idx"]),
#                 "epoch_ms": int(rec["epoch_ms"]),
#                 "path": str(fp),
#             }
#         )

#     roi_paths = []
#     if roi:
#         roi_dir = ev_dir / "roi_frames"
#         roi_dir.mkdir(parents=True, exist_ok=True)
#         for i, rec in enumerate(burst_records, start=1):
#             roi_img = crop_roi(rec["frame"], roi)
#             if roi_img is not None and roi_img.size > 0:
#                 rp = roi_dir / f"roi_{i:02d}.jpg"
#                 cv2.imwrite(str(rp), roi_img)
#                 roi_paths.append(
#                     {
#                         "index_in_burst": i,
#                         "frame_idx": int(rec["idx"]),
#                         "path": str(rp),
#                     }
#                 )

#     mid_idx = len(saved_frames) // 2 if saved_frames else 0
#     main_frame_path = saved_frames[mid_idx]["path"] if saved_frames else None
#     main_roi_path = None
#     if roi_paths:
#         mid_roi_idx = len(roi_paths) // 2
#         main_roi_path = roi_paths[mid_roi_idx]["path"]

#     return {
#         "main_frame": main_frame_path,
#         "main_roi": main_roi_path,
#         "frames": saved_frames,
#         "roi_frames": roi_paths,
#     }


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
#     burst_records: Optional[List[Dict[str, Any]]] = None,
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

#     burst_info = None
#     if burst_records:
#         burst_info = save_burst_frames(ev_dir=ev_dir, burst_records=burst_records, roi=roi)

#     ev = {
#         "event_id": event_id,
#         "trigger": trigger,
#         "event_time_epoch_ms": epoch_ms,
#         "event_time_local": time.strftime("%Y-%m-%d %H:%M:%S"),
#         "frame_idx": idx,
#         "paths": {
#             "frame": str(frame_path),
#             "roi": (str(roi_path) if roi_path else None),
#         },
#         "roi": ({"x": roi[0], "y": roi[1], "w": roi[2], "h": roi[3]} if roi else None),
#         "readout": {
#             "barcode": None,
#             "qrcode": None,
#             "serial": None,
#             "status": "not_attempted",
#         },
#         "auto_metrics": auto_metrics,
#         "burst": (
#             {
#                 "enabled": True,
#                 "count": len(burst_records),
#                 "main_frame_path": burst_info["main_frame"] if burst_info else None,
#                 "main_roi_path": burst_info["main_roi"] if burst_info else None,
#                 "frames": burst_info["frames"] if burst_info else [],
#                 "roi_frames": burst_info["roi_frames"] if burst_info else [],
#             }
#             if burst_records
#             else {
#                 "enabled": False,
#                 "count": 0,
#                 "main_frame_path": None,
#                 "main_roi_path": None,
#                 "frames": [],
#                 "roi_frames": [],
#             }
#         ),
#     }
#     safe_write_json(ev_dir / "event.json", ev)
#     return ev_dir


# def capture_auto_window(
#     *,
#     cap: cv2.VideoCapture,
#     idx_start: int,
#     duration_s: float,
#     interval_s: float,
# ) -> Tuple[List[Dict[str, Any]], int]:
#     """
#     ## MANTENIDO SOLO COMO REFERENCIA
#     ## Este helper pertenece al flujo automático anterior.
#     ## No se usa en el MVP manual controlado.
#     """
#     records: List[Dict[str, Any]] = []

#     duration_s = max(0.5, float(duration_s))
#     interval_s = max(0.1, float(interval_s))

#     start_t = time.time()
#     next_capture_t = start_t
#     idx = int(idx_start)

#     while True:
#         now = time.time()
#         elapsed = now - start_t
#         if elapsed > duration_s:
#             break

#         ok, frame = cap.read()
#         if not ok or frame is None:
#             ok2, frame2 = warmup_read(cap, tries=5, sleep_s=0.01)
#             if not ok2:
#                 break
#             frame = frame2

#         now_epoch_ms = int(time.time() * 1000)

#         if now >= next_capture_t:
#             records.append(frame_record(idx=idx, frame=frame, epoch_ms=now_epoch_ms))
#             next_capture_t += interval_s

#         idx += 1
#         time.sleep(0.005)

#     return records, idx


# def point_in_rect(px: int, py: int, rect: Tuple[int, int, int, int]) -> bool:
#     x1, y1, x2, y2 = rect
#     return x1 <= px <= x2 and y1 <= py <= y2


# def draw_button(
#     img,
#     rect: Tuple[int, int, int, int],
#     text: str,
#     fill_color: Tuple[int, int, int],
#     text_color: Tuple[int, int, int] = (0, 0, 0),
# ) -> None:
#     x1, y1, x2, y2 = rect
#     cv2.rectangle(img, (x1, y1), (x2, y2), fill_color, -1)
#     cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)

#     font = cv2.FONT_HERSHEY_SIMPLEX
#     scale = 0.75
#     thick = 2
#     (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
#     tx = x1 + max(10, (x2 - x1 - tw) // 2)
#     ty = y1 + max(th + 8, (y2 - y1 + th) // 2)
#     cv2.putText(img, text, (tx, ty), font, scale, text_color, thick, cv2.LINE_AA)


# def main() -> None:
#     ap = argparse.ArgumentParser()

#     ap.add_argument("--device", type=int, default=0)
#     ap.add_argument("--width", type=int, default=640)
#     ap.add_argument("--height", type=int, default=480)
#     ap.add_argument("--fps", type=int, default=30)
#     ap.add_argument("--out_dir", type=str, default="data/captures/opencv")
#     ap.add_argument("--save_video", action="store_true")
#     ap.add_argument("--every", type=int, default=0, help="Guarda 1 frame cada N frames (0 desactiva)")
#     ap.add_argument("--no_display", action="store_true", help="Headless. Cortar con Ctrl+C.")
#     ap.add_argument("--fps_window", type=int, default=30)

#     # Eventos
#     ap.add_argument("--events", action="store_true", help="Guarda eventos (frame+event.json)")
#     ap.add_argument(
#         "--roi",
#         type=int,
#         nargs=4,
#         metavar=("X", "Y", "W", "H"),
#         default=None,
#         help="ROI (x y w h). Ej: --roi 240 180 800 360",
#     )

#     # Manual
#     ap.add_argument(
#         "--manual_burst",
#         type=int,
#         default=3,
#         help="Cantidad de frames a guardar en evento manual (default=3).",
#     )
#     ap.add_argument(
#         "--manual_buffer",
#         type=int,
#         default=5,
#         help="Tamaño del buffer reciente para construir burst manual (default=5).",
#     )

#     ## ==========================================================
#     ## AUTO-EVENTOS (MANTENIDOS SOLO COMO REFERENCIA FUTURA)
#     ## Para el MVP del lunes NO se usarán.
#     ## Dejamos los argumentos para no romper compatibilidad.
#     ## ==========================================================
#     ap.add_argument("--auto_events", action="store_true", help="Disparo automático de eventos")
#     ap.add_argument(
#         "--auto_method",
#         type=str,
#         default="bg",
#         choices=["bg", "motion"],
#         help="Método auto: bg (MOG2+contornos) o motion (diff simple)",
#     )
#     ap.add_argument(
#         "--auto_use_window_capture",
#         action="store_true",
#         help="Cuando detecta el trigger automático, captura una secuencia temporal en vez de un solo frame.",
#     )
#     ap.add_argument(
#         "--auto_window_s",
#         type=float,
#         default=20.0,
#         help="Duración de la captura temporal automática en segundos.",
#     )
#     ap.add_argument(
#         "--auto_interval_s",
#         type=float,
#         default=1.0,
#         help="Intervalo entre frames en la captura temporal automática.",
#     )

#     ap.add_argument("--enter_thr", type=float, default=0.08, help="motion_ratio para re-armar (0-1)")
#     ap.add_argument("--stable_thr", type=float, default=0.01, help="motion_ratio para considerar estable (0-1)")
#     ap.add_argument("--stable_frames", type=int, default=25, help="frames estables para disparar")
#     ap.add_argument("--cooldown_s", type=float, default=2.0, help="cooldown tras disparar")

#     ap.add_argument("--bg_warmup", type=int, default=45, help="frames para aprender fondo antes de disparar")
#     ap.add_argument("--min_fg_ratio", type=float, default=0.02, help="ratio de foreground para 'objeto presente'")
#     ap.add_argument("--min_contour_area", type=int, default=2500, help="área mínima de contorno para 'objeto presente'")
#     ap.add_argument("--present_frames", type=int, default=10, help="frames de presencia para disparar evento")
#     ap.add_argument("--bg_history", type=int, default=200)
#     ap.add_argument("--bg_var_threshold", type=int, default=16)
#     ap.add_argument("--bg_detect_shadows", action="store_true", help="MOG2 detectShadows=True")

#     args = ap.parse_args()

#     out_dir = Path(args.out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     roi: Optional[Tuple[int, int, int, int]] = tuple(args.roi) if args.roi else None

#     cap = open_camera(args.device, args.width, args.height, args.fps)
#     if not cap.isOpened():
#         raise SystemExit(f"No pude abrir cámara device={args.device}. Prueba --device 1/2 y permisos.")

#     ok, frame = warmup_read(cap)
#     if not ok:
#         cap.release()
#         raise SystemExit("Warm-up falló. Cierra Zoom/Teams/Chrome/FaceTime y prueba 640x480 u otro device.")

#     h, w = frame.shape[:2]
#     backend_name = get_backend_name(cap)
#     actual_fps_reported = cap.get(cv2.CAP_PROP_FPS)
#     fourcc_str = get_fourcc_str(cap)

#     print(f"[INFO] Cámara OK. Resolución real: {w}x{h}")
#     if backend_name:
#         print(f"[INFO] Backend: {backend_name}")
#     if actual_fps_reported:
#         print(f"[INFO] FPS reportado por OpenCV: {actual_fps_reported:.2f}")
#     if fourcc_str:
#         print(f"[INFO] FOURCC: {fourcc_str}")

#     ts = time.strftime("%Y%m%d_%H%M%S")
#     frame_dir = out_dir / f"frames_{ts}"
#     frame_dir.mkdir(parents=True, exist_ok=True)
#     print(f"[INFO] Guardando en: {frame_dir}")

#     events_dir = frame_dir / "events"
#     if args.events:
#         events_dir.mkdir(parents=True, exist_ok=True)
#         print(f"[INFO] Eventos habilitados: {events_dir}")

#     session_path = frame_dir / "session.json"
#     session: Dict[str, Any] = {
#         "session_id": ts,
#         "start_time_local": time.strftime("%Y-%m-%d %H:%M:%S"),
#         "start_time_epoch": time.time(),
#         "camera": {
#             "device": args.device,
#             "backend": backend_name,
#             "requested": {"width": args.width, "height": args.height, "fps": args.fps},
#             "actual": {
#                 "width": w,
#                 "height": h,
#                 "fps_reported": (round(float(actual_fps_reported), 2) if actual_fps_reported else None),
#                 "fourcc": fourcc_str,
#             },
#         },
#         "capture": {
#             "out_dir": str(out_dir),
#             "frame_dir": str(frame_dir),
#             "every": args.every,
#             "save_video": bool(args.save_video),
#             "video_path": None,
#             "manual_burst": int(max(1, args.manual_burst)),
#             "manual_buffer": int(max(3, args.manual_buffer)),
#             "auto_use_window_capture": False,
#             "auto_window_s": float(args.auto_window_s),
#             "auto_interval_s": float(args.auto_interval_s),
#         },
#         "events": {
#             "enabled": bool(args.events),
#             "auto_enabled": False,
#             "auto_method": None,
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
#         "last_manual_event": None,
#         "end_time_local": None,
#         "end_time_epoch": None,
#         "status": "running",
#     }
#     safe_write_json(session_path, session)

#     writer = None
#     if args.save_video:
#         fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#         video_path = frame_dir / f"capture_{ts}.mp4"
#         writer = cv2.VideoWriter(str(video_path), fourcc, args.fps, (w, h))
#         if not writer.isOpened():
#             print("[WARN] VideoWriter mp4v falló. Continuaré sin video.")
#             writer = None
#         else:
#             session["capture"]["video_path"] = str(video_path)
#             safe_write_json(session_path, session)
#             print(f"[INFO] Grabando video: {video_path}")

#     if not args.no_display:
#         if args.events:
#             print("[INFO] Controles MVP: 'q' salir | 's' frame manual | 'e' CAPTURAR evento manual y cerrar | touch botones")
#         else:
#             print("[INFO] Controles MVP: 'q' salir | 's' frame manual | touch botón SALIR")
#     else:
#         print("[INFO] Headless: Ctrl+C para salir")

#     idx = 0
#     saved = 0
#     manual_saved = 0

#     fps_win = max(5, int(args.fps_window))
#     win_start_t = time.time()
#     win_start_idx = 0
#     fps_real: Optional[float] = None

#     event_id = 0

#     cooldown_until = 0.0
#     armed = True

#     prev_gray = None
#     stable_count = 0
#     last_motion: Optional[float] = None

#     bg_sub = None
#     bg_warmup_left = int(args.bg_warmup)
#     present_count = 0
#     last_fg_ratio: Optional[float] = None
#     last_max_area: Optional[int] = None

#     recent_buffer_size = max(3, int(args.manual_buffer))
#     recent_frames: Deque[Dict[str, Any]] = deque(maxlen=recent_buffer_size)

#     pending_auto_capture = False
#     pending_event_id: Optional[int] = None
#     pending_auto_metrics: Optional[Dict[str, Any]] = None
#     pending_trigger_idx: Optional[int] = None
#     pending_trigger_frame = None
#     pending_records: List[Dict[str, Any]] = []
#     pending_start_t = 0.0
#     pending_next_capture_t = 0.0

#     captured_event_dir: Optional[str] = None
#     captured_frame_path: Optional[str] = None
#     captured_roi_path: Optional[str] = None
#     captured_event_json_path: Optional[str] = None

#     window_name = "Capture (OpenCV)"
#     touch_action: Optional[str] = None

#     def on_mouse(event, x, y, flags, param):
#         nonlocal touch_action
#         if event != cv2.EVENT_LBUTTONDOWN:
#             return

#         rect_capture = param["rect_capture"]
#         rect_exit = param["rect_exit"]
#         events_enabled = param["events_enabled"]

#         if events_enabled and point_in_rect(x, y, rect_capture):
#             touch_action = "capture"
#         elif point_in_rect(x, y, rect_exit):
#             touch_action = "quit"

#     if not args.no_display:
#         cv2.namedWindow(window_name)

#     try:
#         while True:
#             ok, frame = cap.read()
#             if not ok or frame is None:
#                 ok2, frame2 = warmup_read(cap, tries=10, sleep_s=0.02)
#                 if not ok2:
#                     print("[WARN] No pude leer frame; saliendo.")
#                     break
#                 frame = frame2

#             now_epoch_ms = int(time.time() * 1000)
#             recent_frames.append(frame_record(idx=idx, frame=frame, epoch_ms=now_epoch_ms))

#             if writer is not None:
#                 writer.write(frame)

#             if args.every > 0 and (idx % args.every == 0):
#                 fp = frame_dir / f"frame_{idx:06d}_{now_epoch_ms}.jpg"
#                 cv2.imwrite(str(fp), frame)
#                 saved += 1

#             if (idx - win_start_idx) >= fps_win:
#                 dt = time.time() - win_start_t
#                 if dt > 1e-6:
#                     fps_real = (idx - win_start_idx) / dt
#                 win_start_t = time.time()
#                 win_start_idx = idx

#             if idx % fps_win == 0:
#                 session["runtime"]["fps_real_last"] = (round(fps_real, 2) if fps_real is not None else None)
#                 session["runtime"]["frames_total"] = idx
#                 session["runtime"]["frames_saved"] = saved
#                 session["runtime"]["manual_saved"] = manual_saved
#                 safe_write_json(session_path, session)

#             if not args.no_display:
#                 disp = frame.copy()

#                 if roi is not None:
#                     x, y, rw, rh = roi
#                     cv2.rectangle(disp, (x, y), (x + rw, y + rh), (0, 255, 255), 2)

#                 fps_txt = f"{fps_real:.1f}" if fps_real is not None else "..."
#                 cv2.putText(
#                     disp,
#                     f"MVP MANUAL | device={args.device} {w}x{h} idx={idx} saved={saved} fps={fps_txt}",
#                     (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.72,
#                     (0, 255, 0),
#                     2,
#                     cv2.LINE_AA,
#                 )

#                 cv2.putText(
#                     disp,
#                     "Controles: q=salir | s=guardar frame | e=capturar evento y cerrar | touch botones",
#                     (10, 60),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.62,
#                     (0, 255, 255),
#                     2,
#                     cv2.LINE_AA,
#                 )

#                 if roi is not None:
#                     cv2.putText(
#                         disp,
#                         "ROI activa para captura controlada",
#                         (10, 90),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.62,
#                         (255, 200, 0),
#                         2,
#                         cv2.LINE_AA,
#                     )

#                 btn_h = 60
#                 btn_w_capture = 220
#                 btn_w_exit = 140
#                 btn_y1 = h - btn_h - 20
#                 btn_y2 = h - 20
#                 btn_capture = (20, btn_y1, 20 + btn_w_capture, btn_y2)
#                 btn_exit = (20 + btn_w_capture + 20, btn_y1, 20 + btn_w_capture + 20 + btn_w_exit, btn_y2)

#                 if args.events:
#                     draw_button(disp, btn_capture, "CAPTURAR", (0, 180, 0), (255, 255, 255))
#                 draw_button(disp, btn_exit, "SALIR", (50, 50, 200), (255, 255, 255))

#                 cv2.setMouseCallback(
#                     window_name,
#                     on_mouse,
#                     {
#                         "rect_capture": btn_capture,
#                         "rect_exit": btn_exit,
#                         "events_enabled": bool(args.events),
#                     },
#                 )

#                 cv2.imshow(window_name, disp)
#                 key = cv2.waitKey(1) & 0xFF

#                 do_quit = (key == ord("q")) or (touch_action == "quit")
#                 do_save = (key == ord("s"))
#                 do_capture = ((key == ord("e")) or (touch_action == "capture")) and args.events

#                 touch_action = None

#                 if do_quit:
#                     break

#                 if do_save:
#                     epoch_ms = int(time.time() * 1000)
#                     fp = frame_dir / f"manual_{idx:06d}_{epoch_ms}.jpg"
#                     cv2.imwrite(str(fp), frame)
#                     saved += 1
#                     manual_saved += 1
#                     print(f"[INFO] Guardado manual: {fp}")

#                 if do_capture:
#                     event_id += 1

#                     burst_n = max(1, int(args.manual_burst))
#                     burst_records = list(recent_frames)[-burst_n:]

#                     ev_dir = save_event(
#                         frame_dir=frame_dir,
#                         events_dir=events_dir,
#                         event_id=event_id,
#                         frame=frame,
#                         idx=idx,
#                         roi=roi,
#                         trigger="manual",
#                         auto_metrics=None,
#                         burst_records=burst_records,
#                     )

#                     captured_event_dir = str(ev_dir)
#                     captured_frame_path = str(ev_dir / "frame.jpg")
#                     captured_roi_path = str(ev_dir / "roi.jpg") if (ev_dir / "roi.jpg").exists() else None
#                     captured_event_json_path = str(ev_dir / "event.json")

#                     session["events"]["counts"]["total"] += 1
#                     session["events"]["counts"]["manual"] += 1
#                     session["last_manual_event"] = {
#                         "event_id": event_id,
#                         "event_dir": captured_event_dir,
#                         "frame_path": captured_frame_path,
#                         "roi_path": captured_roi_path,
#                         "event_json_path": captured_event_json_path,
#                     }
#                     safe_write_json(session_path, session)

#                     print(f"[EVENT] Manual #{event_id} guardado: {ev_dir} | burst={len(burst_records)}")
#                     print("[INFO] Captura manual realizada. Cerrando ventana para retornar control al frontend...")
#                     break

#             idx += 1

#     except KeyboardInterrupt:
#         print("\n[INFO] Ctrl+C. Cerrando...")

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

#         if session.get("last_manual_event"):
#             print("[TRACE] last_manual_event:")
#             print(json.dumps(session["last_manual_event"], ensure_ascii=False, indent=2))


# if __name__ == "__main__":
#     main()

# scripts/capture_opencv.py

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import platform
import ctypes
from ctypes import wintypes


def open_camera(device: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    system = platform.system()

    # Orden de backends por sistema
    if system == "Windows":
        candidates = [
            ("DSHOW", cv2.CAP_DSHOW),
            ("MSMF", cv2.CAP_MSMF),
            ("DEFAULT", None),
        ]
    elif system == "Darwin":  # macOS
        candidates = [
            ("AVFOUNDATION", cv2.CAP_AVFOUNDATION),
            ("DEFAULT", None),
        ]
    else:  # Linux / otros
        candidates = [
            ("DEFAULT", None),
        ]

    for backend_name, backend in candidates:
        if backend is None:
            cap = cv2.VideoCapture(device)
        else:
            cap = cv2.VideoCapture(device, backend)

        if cap is not None and cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
            cap.set(cv2.CAP_PROP_FPS, int(fps))

            # Revalidamos después de setear propiedades
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                print(f"[INFO] Cámara abierta con backend: {backend_name}")
                return cap

            cap.release()

    # Si nada funcionó, devolvemos un capture inválido
    return cv2.VideoCapture()


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
    x2 = min(img.shape[1], x + rw)
    y2 = min(img.shape[0], y + rh)
    if x >= x2 or y >= y2:
        return None
    return img[y:y2, x:x2].copy()


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


def get_backend_name(cap: cv2.VideoCapture) -> Optional[str]:
    try:
        name = cap.getBackendName()
        return str(name) if name else None
    except Exception:
        return None


def get_fourcc_str(cap: cv2.VideoCapture) -> Optional[str]:
    try:
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        if fourcc <= 0:
            return None
        chars = [
            chr((fourcc >> 0) & 0xFF),
            chr((fourcc >> 8) & 0xFF),
            chr((fourcc >> 16) & 0xFF),
            chr((fourcc >> 24) & 0xFF),
        ]
        s = "".join(chars).strip("\x00").strip()
        return s or None
    except Exception:
        return None


def frame_record(idx: int, frame, epoch_ms: int) -> Dict[str, Any]:
    return {
        "idx": int(idx),
        "epoch_ms": int(epoch_ms),
        "frame": frame.copy(),
    }


def fit_size(src_w: int, src_h: int, max_w: int, max_h: int) -> Tuple[int, int, float]:
    if src_w <= 0 or src_h <= 0:
        return max_w, max_h, 1.0

    scale = min(max_w / src_w, max_h / src_h, 1.0)
    out_w = max(1, int(round(src_w * scale)))
    out_h = max(1, int(round(src_h * scale)))
    return out_w, out_h, scale


def save_burst_frames(
    *,
    ev_dir: Path,
    burst_records: List[Dict[str, Any]],
    roi: Optional[Tuple[int, int, int, int]],
) -> Dict[str, Any]:
    frames_dir = ev_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    saved_frames = []
    for i, rec in enumerate(burst_records, start=1):
        fp = frames_dir / f"frame_{i:02d}.jpg"
        cv2.imwrite(str(fp), rec["frame"])
        saved_frames.append(
            {
                "index_in_burst": i,
                "frame_idx": int(rec["idx"]),
                "epoch_ms": int(rec["epoch_ms"]),
                "path": str(fp),
            }
        )

    roi_paths = []
    if roi:
        roi_dir = ev_dir / "roi_frames"
        roi_dir.mkdir(parents=True, exist_ok=True)
        for i, rec in enumerate(burst_records, start=1):
            roi_img = crop_roi(rec["frame"], roi)
            if roi_img is not None and roi_img.size > 0:
                rp = roi_dir / f"roi_{i:02d}.jpg"
                cv2.imwrite(str(rp), roi_img)
                roi_paths.append(
                    {
                        "index_in_burst": i,
                        "frame_idx": int(rec["idx"]),
                        "path": str(rp),
                    }
                )

    mid_idx = len(saved_frames) // 2 if saved_frames else 0
    main_frame_path = saved_frames[mid_idx]["path"] if saved_frames else None
    main_roi_path = None
    if roi_paths:
        mid_roi_idx = len(roi_paths) // 2
        main_roi_path = roi_paths[mid_roi_idx]["path"]

    return {
        "main_frame": main_frame_path,
        "main_roi": main_roi_path,
        "frames": saved_frames,
        "roi_frames": roi_paths,
    }


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
    burst_records: Optional[List[Dict[str, Any]]] = None,
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

    burst_info = None
    if burst_records:
        burst_info = save_burst_frames(ev_dir=ev_dir, burst_records=burst_records, roi=roi)

    ev = {
        "event_id": event_id,
        "trigger": trigger,
        "event_time_epoch_ms": epoch_ms,
        "event_time_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "frame_idx": idx,
        "paths": {
            "frame": str(frame_path),
            "roi": (str(roi_path) if roi_path else None),
        },
        "roi": ({"x": roi[0], "y": roi[1], "w": roi[2], "h": roi[3]} if roi else None),
        "readout": {
            "barcode": None,
            "qrcode": None,
            "serial": None,
            "status": "not_attempted",
        },
        "auto_metrics": auto_metrics,
        "burst": (
            {
                "enabled": True,
                "count": len(burst_records),
                "main_frame_path": burst_info["main_frame"] if burst_info else None,
                "main_roi_path": burst_info["main_roi"] if burst_info else None,
                "frames": burst_info["frames"] if burst_info else [],
                "roi_frames": burst_info["roi_frames"] if burst_info else [],
            }
            if burst_records
            else {
                "enabled": False,
                "count": 0,
                "main_frame_path": None,
                "main_roi_path": None,
                "frames": [],
                "roi_frames": [],
            }
        ),
    }
    safe_write_json(ev_dir / "event.json", ev)
    return ev_dir


def capture_auto_window(
    *,
    cap: cv2.VideoCapture,
    idx_start: int,
    duration_s: float,
    interval_s: float,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    ## MANTENIDO SOLO COMO REFERENCIA
    ## Este helper pertenece al flujo automático anterior.
    ## No se usa en el MVP manual controlado.
    """
    records: List[Dict[str, Any]] = []

    duration_s = max(0.5, float(duration_s))
    interval_s = max(0.1, float(interval_s))

    start_t = time.time()
    next_capture_t = start_t
    idx = int(idx_start)

    while True:
        now = time.time()
        elapsed = now - start_t
        if elapsed > duration_s:
            break

        ok, frame = cap.read()
        if not ok or frame is None:
            ok2, frame2 = warmup_read(cap, tries=5, sleep_s=0.01)
            if not ok2:
                break
            frame = frame2

        now_epoch_ms = int(time.time() * 1000)

        if now >= next_capture_t:
            records.append(frame_record(idx=idx, frame=frame, epoch_ms=now_epoch_ms))
            next_capture_t += interval_s

        idx += 1
        time.sleep(0.005)

    return records, idx


def point_in_rect(px: int, py: int, rect: Tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = rect
    return x1 <= px <= x2 and y1 <= py <= y2


def draw_button(
    img,
    rect: Tuple[int, int, int, int],
    text: str,
    fill_color: Tuple[int, int, int],
    text_color: Tuple[int, int, int] = (0, 0, 0),
) -> None:
    x1, y1, x2, y2 = rect
    cv2.rectangle(img, (x1, y1), (x2, y2), fill_color, -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.75
    thick = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    tx = x1 + max(10, (x2 - x1 - tw) // 2)
    ty = y1 + max(th + 8, (y2 - y1 + th) // 2)
    cv2.putText(img, text, (tx, ty), font, scale, text_color, thick, cv2.LINE_AA)

def _find_window_by_title_contains(title_substring: str):
    if platform.system() != "Windows":
        return None

    user32 = ctypes.windll.user32
    found = []

    EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

    def enum_proc(hwnd, lParam):
        if not user32.IsWindowVisible(hwnd):
            return True

        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True

        buff = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buff, length + 1)
        title = buff.value or ""

        if title_substring.lower() in title.lower():
            found.append(hwnd)
            return False

        return True

    user32.EnumWindows(EnumWindowsProc(enum_proc), 0)
    return found[0] if found else None


def bring_window_to_front_strong(window_name: str) -> bool:
    """
    Intenta traer la ventana OpenCV al frente de forma más agresiva en Windows.
    Devuelve True si encontró la ventana; False si no.
    """
    if platform.system() != "Windows":
        return False

    try:
        user32 = ctypes.windll.user32
        hwnd = _find_window_by_title_contains(window_name)
        if not hwnd:
            return False

        SW_RESTORE = 9
        SW_SHOW = 5
        SWP_NOMOVE = 0x0002
        SWP_NOSIZE = 0x0001
        SWP_SHOWWINDOW = 0x0040
        HWND_TOPMOST = -1
        HWND_NOTOPMOST = -2

        user32.ShowWindow(hwnd, SW_RESTORE)
        user32.ShowWindow(hwnd, SW_SHOW)

        user32.SetWindowPos(
            hwnd,
            HWND_TOPMOST,
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW,
        )
        user32.SetWindowPos(
            hwnd,
            HWND_NOTOPMOST,
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW,
        )

        try:
            user32.BringWindowToTop(hwnd)
        except Exception:
            pass

        try:
            user32.SetForegroundWindow(hwnd)
        except Exception:
            pass

        try:
            user32.SetActiveWindow(hwnd)
        except Exception:
            pass

        try:
            user32.SetFocus(hwnd)
        except Exception:
            pass

        return True

    except Exception:
        return False

def bring_window_to_front(window_name: str) -> None:
    """
    Intenta traer la ventana de OpenCV al frente en Windows.
    En otros sistemas no hace nada.
    """
    if platform.system() != "Windows":
        return

    try:
        import ctypes

        user32 = ctypes.windll.user32

        hwnd = user32.FindWindowW(None, window_name)
        if not hwnd:
            return

        SW_RESTORE = 9
        SWP_NOMOVE = 0x0002
        SWP_NOSIZE = 0x0001
        SWP_SHOWWINDOW = 0x0040
        HWND_TOPMOST = -1
        HWND_NOTOPMOST = -2

        user32.ShowWindow(hwnd, SW_RESTORE)
        user32.SetWindowPos(
            hwnd,
            HWND_TOPMOST,
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW,
        )
        user32.SetWindowPos(
            hwnd,
            HWND_NOTOPMOST,
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW,
        )
        user32.SetForegroundWindow(hwnd)

    except Exception:
        # Si falla, seguimos normal sin romper el flujo.
        pass

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out_dir", type=str, default="data/captures/opencv")
    ap.add_argument("--save_video", action="store_true")
    ap.add_argument("--every", type=int, default=0, help="Guarda 1 frame cada N frames (0 desactiva)")
    ap.add_argument("--no_display", action="store_true", help="Headless. Cortar con Ctrl+C.")
    ap.add_argument("--fps_window", type=int, default=30)
    ap.add_argument("--preview_max_w", type=int, default=1280, help="Ancho máximo de preview en pantalla")
    ap.add_argument("--preview_max_h", type=int, default=720, help="Alto máximo de preview en pantalla")

    # Eventos
    ap.add_argument("--events", action="store_true", help="Guarda eventos (frame+event.json)")
    ap.add_argument(
        "--roi",
        type=int,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        default=None,
        help="ROI (x y w h). Ej: --roi 240 180 800 360",
    )

    # Manual
    ap.add_argument(
        "--manual_burst",
        type=int,
        default=3,
        help="Cantidad de frames a guardar en evento manual (default=3).",
    )
    ap.add_argument(
        "--manual_buffer",
        type=int,
        default=5,
        help="Tamaño del buffer reciente para construir burst manual (default=5).",
    )

    ## ==========================================================
    ## AUTO-EVENTOS (MANTENIDOS SOLO COMO REFERENCIA FUTURA)
    ## Para el MVP del lunes NO se usarán.
    ## Dejamos los argumentos para no romper compatibilidad.
    ## ==========================================================
    ap.add_argument("--auto_events", action="store_true", help="Disparo automático de eventos")
    ap.add_argument(
        "--auto_method",
        type=str,
        default="bg",
        choices=["bg", "motion"],
        help="Método auto: bg (MOG2+contornos) o motion (diff simple)",
    )
    ap.add_argument(
        "--auto_use_window_capture",
        action="store_true",
        help="Cuando detecta el trigger automático, captura una secuencia temporal en vez de un solo frame.",
    )
    ap.add_argument(
        "--auto_window_s",
        type=float,
        default=20.0,
        help="Duración de la captura temporal automática en segundos.",
    )
    ap.add_argument(
        "--auto_interval_s",
        type=float,
        default=1.0,
        help="Intervalo entre frames en la captura temporal automática.",
    )

    ap.add_argument("--enter_thr", type=float, default=0.08, help="motion_ratio para re-armar (0-1)")
    ap.add_argument("--stable_thr", type=float, default=0.01, help="motion_ratio para considerar estable (0-1)")
    ap.add_argument("--stable_frames", type=int, default=25, help="frames estables para disparar")
    ap.add_argument("--cooldown_s", type=float, default=2.0, help="cooldown tras disparar")

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
        raise SystemExit(
            f"No pude abrir cámara device={args.device}. "
            "En Windows se intentó DSHOW/MSMF/DEFAULT; revisa permisos y que la cámara no esté ocupada."
        )

    ok, frame = warmup_read(cap)
    if not ok:
        cap.release()
        raise SystemExit("Warm-up falló. Cierra Zoom/Teams/Chrome/FaceTime y prueba 640x480 u otro device.")

    h, w = frame.shape[:2]
    backend_name = get_backend_name(cap)
    actual_fps_reported = cap.get(cv2.CAP_PROP_FPS)
    fourcc_str = get_fourcc_str(cap)

    print(f"[INFO] Cámara OK. Resolución real: {w}x{h}")
    if backend_name:
        print(f"[INFO] Backend: {backend_name}")
    if actual_fps_reported:
        print(f"[INFO] FPS reportado por OpenCV: {actual_fps_reported:.2f}")
    if fourcc_str:
        print(f"[INFO] FOURCC: {fourcc_str}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    frame_dir = out_dir / f"frames_{ts}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Guardando en: {frame_dir}")

    events_dir = frame_dir / "events"
    if args.events:
        events_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Eventos habilitados: {events_dir}")

    session_path = frame_dir / "session.json"
    session: Dict[str, Any] = {
        "session_id": ts,
        "start_time_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "start_time_epoch": time.time(),
        "camera": {
            "device": args.device,
            "backend": backend_name,
            "requested": {"width": args.width, "height": args.height, "fps": args.fps},
            "actual": {
                "width": w,
                "height": h,
                "fps_reported": (round(float(actual_fps_reported), 2) if actual_fps_reported else None),
                "fourcc": fourcc_str,
            },
        },
        "capture": {
            "out_dir": str(out_dir),
            "frame_dir": str(frame_dir),
            "every": args.every,
            "save_video": bool(args.save_video),
            "video_path": None,
            "manual_burst": int(max(1, args.manual_burst)),
            "manual_buffer": int(max(3, args.manual_buffer)),
            "auto_use_window_capture": False,
            "auto_window_s": float(args.auto_window_s),
            "auto_interval_s": float(args.auto_interval_s),
            "preview_max_w": int(args.preview_max_w),
            "preview_max_h": int(args.preview_max_h),
        },
        "events": {
            "enabled": bool(args.events),
            "auto_enabled": False,
            "auto_method": None,
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
        "last_manual_event": None,
        "end_time_local": None,
        "end_time_epoch": None,
        "status": "running",
    }
    safe_write_json(session_path, session)

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
            print("[INFO] Controles MVP: 'q' salir | 's' frame manual | 'e' CAPTURAR evento manual y cerrar | touch botones")
        else:
            print("[INFO] Controles MVP: 'q' salir | 's' frame manual | touch botón SALIR")
    else:
        print("[INFO] Headless: Ctrl+C para salir")

    idx = 0
    saved = 0
    manual_saved = 0

    fps_win = max(5, int(args.fps_window))
    win_start_t = time.time()
    win_start_idx = 0
    fps_real: Optional[float] = None

    event_id = 0

    cooldown_until = 0.0
    armed = True

    prev_gray = None
    stable_count = 0
    last_motion: Optional[float] = None

    bg_sub = None
    bg_warmup_left = int(args.bg_warmup)
    present_count = 0
    last_fg_ratio: Optional[float] = None
    last_max_area: Optional[int] = None

    recent_buffer_size = max(3, int(args.manual_buffer))
    recent_frames: Deque[Dict[str, Any]] = deque(maxlen=recent_buffer_size)

    pending_auto_capture = False
    pending_event_id: Optional[int] = None
    pending_auto_metrics: Optional[Dict[str, Any]] = None
    pending_trigger_idx: Optional[int] = None
    pending_trigger_frame = None
    pending_records: List[Dict[str, Any]] = []
    pending_start_t = 0.0
    pending_next_capture_t = 0.0

    captured_event_dir: Optional[str] = None
    captured_frame_path: Optional[str] = None
    captured_roi_path: Optional[str] = None
    captured_event_json_path: Optional[str] = None

    window_name = "Capture (OpenCV)"
    touch_action: Optional[str] = None

    def on_mouse(event, x, y, flags, param):
        nonlocal touch_action
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        scale = float(param["scale"])
        real_x = int(round(x / scale))
        real_y = int(round(y / scale))

        rect_capture = param["rect_capture"]
        rect_exit = param["rect_exit"]
        events_enabled = param["events_enabled"]

        if events_enabled and point_in_rect(real_x, real_y, rect_capture):
            touch_action = "capture"
        elif point_in_rect(real_x, real_y, rect_exit):
            touch_action = "quit"

    if not args.no_display:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, min(args.preview_max_w, w), min(args.preview_max_h, h))
        cv2.waitKey(1)

        focus_window_pending = True
        focus_window_attempts = 0
        max_focus_window_attempts = 40
    else:
        focus_window_pending = False
        focus_window_attempts = 0
        max_focus_window_attempts = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                ok2, frame2 = warmup_read(cap, tries=10, sleep_s=0.02)
                if not ok2:
                    print("[WARN] No pude leer frame; saliendo.")
                    break
                frame = frame2

            now_epoch_ms = int(time.time() * 1000)
            recent_frames.append(frame_record(idx=idx, frame=frame, epoch_ms=now_epoch_ms))

            if writer is not None:
                writer.write(frame)

            if args.every > 0 and (idx % args.every == 0):
                fp = frame_dir / f"frame_{idx:06d}_{now_epoch_ms}.jpg"
                cv2.imwrite(str(fp), frame)
                saved += 1

            if (idx - win_start_idx) >= fps_win:
                dt = time.time() - win_start_t
                if dt > 1e-6:
                    fps_real = (idx - win_start_idx) / dt
                win_start_t = time.time()
                win_start_idx = idx

            if idx % fps_win == 0:
                session["runtime"]["fps_real_last"] = (round(fps_real, 2) if fps_real is not None else None)
                session["runtime"]["frames_total"] = idx
                session["runtime"]["frames_saved"] = saved
                session["runtime"]["manual_saved"] = manual_saved
                safe_write_json(session_path, session)

            if not args.no_display:
                disp = frame.copy()

                preview_w, preview_h, preview_scale = fit_size(
                    w, h, args.preview_max_w, args.preview_max_h
                )

                fps_txt = f"{fps_real:.1f}" if fps_real is not None else "..."

                if roi is not None:
                    x, y, rw, rh = roi
                    cv2.rectangle(disp, (x, y), (x + rw, y + rh), (0, 255, 255), 2)

                cv2.putText(
                    disp,
                    f"MVP MANUAL | device={args.device} {w}x{h} idx={idx} saved={saved} fps={fps_txt}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    disp,
                    "Controles: q=salir | s=guardar frame | e=capturar evento y cerrar | touch botones",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                if roi is not None:
                    cv2.putText(
                        disp,
                        "ROI activa para captura controlada",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.62,
                        (255, 200, 0),
                        2,
                        cv2.LINE_AA,
                    )

                btn_h = 60
                btn_w_capture = 220
                btn_w_exit = 140
                margin = 20

                btn_y1 = h - btn_h - margin
                btn_y2 = h - margin
                btn_capture = (20, btn_y1, 20 + btn_w_capture, btn_y2)
                btn_exit = (20 + btn_w_capture + 20, btn_y1, 20 + btn_w_capture + 20 + btn_w_exit, btn_y2)

                if args.events:
                    draw_button(disp, btn_capture, "CAPTURAR", (0, 180, 0), (255, 255, 255))
                draw_button(disp, btn_exit, "SALIR", (50, 50, 200), (255, 255, 255))

                if preview_scale < 1.0:
                    disp_show = cv2.resize(disp, (preview_w, preview_h), interpolation=cv2.INTER_AREA)
                else:
                    disp_show = disp

                cv2.setMouseCallback(
                    window_name,
                    on_mouse,
                    {
                        "rect_capture": btn_capture,
                        "rect_exit": btn_exit,
                        "events_enabled": bool(args.events),
                        "scale": preview_scale,
                    },
                )

                cv2.imshow(window_name, disp_show)
                
                if focus_window_pending:
                    found = bring_window_to_front_strong(window_name)
                    focus_window_attempts += 1

                    # seguimos insistiendo varios frames, porque a veces OpenCV crea/actualiza
                    # la ventana unos instantes después del primer imshow
                    if found and focus_window_attempts >= 10:
                        focus_window_pending = False
                    elif focus_window_attempts >= max_focus_window_attempts:
                        focus_window_pending = False
                
                key = cv2.waitKey(1) & 0xFF

                do_quit = (key == ord("q")) or (touch_action == "quit")
                do_save = (key == ord("s"))
                do_capture = ((key == ord("e")) or (touch_action == "capture")) and args.events

                touch_action = None

                if do_quit:
                    break

                if do_save:
                    epoch_ms = int(time.time() * 1000)
                    fp = frame_dir / f"manual_{idx:06d}_{epoch_ms}.jpg"
                    cv2.imwrite(str(fp), frame)
                    saved += 1
                    manual_saved += 1
                    print(f"[INFO] Guardado manual: {fp}")

                if do_capture:
                    event_id += 1

                    burst_n = max(1, int(args.manual_burst))
                    burst_records = list(recent_frames)[-burst_n:]

                    ev_dir = save_event(
                        frame_dir=frame_dir,
                        events_dir=events_dir,
                        event_id=event_id,
                        frame=frame,
                        idx=idx,
                        roi=roi,
                        trigger="manual",
                        auto_metrics=None,
                        burst_records=burst_records,
                    )

                    captured_event_dir = str(ev_dir)
                    captured_frame_path = str(ev_dir / "frame.jpg")
                    captured_roi_path = str(ev_dir / "roi.jpg") if (ev_dir / "roi.jpg").exists() else None
                    captured_event_json_path = str(ev_dir / "event.json")

                    session["events"]["counts"]["total"] += 1
                    session["events"]["counts"]["manual"] += 1
                    session["last_manual_event"] = {
                        "event_id": event_id,
                        "event_dir": captured_event_dir,
                        "frame_path": captured_frame_path,
                        "roi_path": captured_roi_path,
                        "event_json_path": captured_event_json_path,
                    }
                    safe_write_json(session_path, session)

                    print(f"[EVENT] Manual #{event_id} guardado: {ev_dir} | burst={len(burst_records)}")
                    print("[INFO] Captura manual realizada. Cerrando ventana para retornar control al frontend...")
                    break

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

        if session.get("last_manual_event"):
            print("[TRACE] last_manual_event:")
            print(json.dumps(session["last_manual_event"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()