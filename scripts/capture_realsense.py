# scripts/capture_realsense.py
from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except Exception as e:
    raise SystemExit(
        "No pude importar pyrealsense2. Instálalo en tu entorno antes de ejecutar este script.\n"
        f"Detalle: {e}"
    )


# -----------------------------
# RealSense helpers
# -----------------------------
def open_realsense(
    width: int,
    height: int,
    fps: int,
) -> Tuple[Any, Any, str, Optional[str], Optional[Any]]:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, int(width), int(height), rs.format.bgr8, int(fps))

    profile = pipeline.start(config)

    device = profile.get_device()
    device_name = device.get_info(rs.camera_info.name) if device else "Unknown"
    serial = device.get_info(rs.camera_info.serial_number) if device else None

    color_sensor = None
    try:
        sensors = device.query_sensors()
        for sensor in sensors:
            try:
                sensor_name = str(sensor.get_info(rs.camera_info.name)).lower()
                if "color" in sensor_name:
                    color_sensor = sensor
                    break
            except Exception:
                continue
    except Exception:
        color_sensor = None

    return pipeline, profile, device_name, serial, color_sensor


def warmup_read_rs(
    pipeline: Any,
    tries: int = 90,
    sleep_s: float = 0.03,
) -> Tuple[bool, Optional[np.ndarray]]:
    last: Optional[np.ndarray] = None
    for _ in range(tries):
        try:
            frames = pipeline.wait_for_frames(timeout_ms=2000)
            color_frame = frames.get_color_frame()
            if color_frame:
                img = np.asanyarray(color_frame.get_data())
                if img is not None and img.size > 0:
                    return True, img
                last = img
        except Exception:
            pass
        time.sleep(sleep_s)
    return False, last


def read_frame_rs(pipeline: Any) -> Tuple[bool, Optional[np.ndarray]]:
    frames = pipeline.wait_for_frames(timeout_ms=2000)
    color_frame = frames.get_color_frame()
    if not color_frame:
        return False, None

    img = np.asanyarray(color_frame.get_data())
    if img is None or img.size == 0:
        return False, None

    return True, img


def get_rs_stream_info(profile: Any) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "stream": "color",
        "format": None,
        "fps": None,
        "width": None,
        "height": None,
    }
    try:
        stream = profile.get_stream(rs.stream.color)
        vsp = stream.as_video_stream_profile()
        info["format"] = str(vsp.format())
        info["fps"] = int(vsp.fps())
        info["width"] = int(vsp.width())
        info["height"] = int(vsp.height())
    except Exception:
        pass
    return info


# -----------------------------
# Generic helpers
# -----------------------------
def safe_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def crop_roi(
    img: np.ndarray,
    roi: Optional[Tuple[int, int, int, int]],
) -> Optional[np.ndarray]:
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


def to_gray_blur(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    return gray


def motion_ratio(prev_gray: np.ndarray, curr_gray: np.ndarray, thresh: int = 25) -> float:
    diff = cv2.absdiff(prev_gray, curr_gray)
    _, bw = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
    changed = cv2.countNonZero(bw)
    total = bw.shape[0] * bw.shape[1]
    return changed / max(1, total)


def frame_record(idx: int, frame: np.ndarray, epoch_ms: int) -> Dict[str, Any]:
    return {
        "idx": int(idx),
        "epoch_ms": int(epoch_ms),
        "frame": frame.copy(),
    }


def save_burst_frames(
    *,
    ev_dir: Path,
    burst_records: List[Dict[str, Any]],
    roi: Optional[Tuple[int, int, int, int]],
) -> Dict[str, Any]:
    frames_dir = ev_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    saved_frames: List[Dict[str, Any]] = []
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

    roi_paths: List[Dict[str, Any]] = []
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
    events_dir: Path,
    event_id: int,
    frame: np.ndarray,
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


def capture_auto_window_rs(
    *,
    pipeline: Any,
    idx_start: int,
    duration_s: float,
    interval_s: float,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Mantenida solo por compatibilidad / referencia.
    En el flujo principal ya no se usa de forma bloqueante.
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

        ok, frame = read_frame_rs(pipeline)
        if not ok or frame is None:
            ok2, frame2 = warmup_read_rs(pipeline, tries=5, sleep_s=0.01)
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


def update_auto_trigger_motion(
    *,
    roi_img: np.ndarray,
    prev_gray: Optional[np.ndarray],
    armed: bool,
    stable_count: int,
    enter_thr: float,
    stable_thr: float,
    stable_frames: int,
) -> Tuple[bool, Optional[Dict[str, Any]], np.ndarray, bool, int, Optional[float]]:
    auto_trigger = False
    auto_metrics = None
    last_motion: Optional[float] = None

    curr = to_gray_blur(roi_img)

    if prev_gray is not None:
        mr = motion_ratio(prev_gray, curr)
        last_motion = mr

        if mr > enter_thr:
            armed = True
            stable_count = 0

        if armed and mr < stable_thr:
            stable_count += 1
        else:
            stable_count = 0

        if armed and stable_count >= stable_frames:
            auto_trigger = True
            auto_metrics = {
                "method": "motion",
                "motion_ratio": float(mr),
                "enter_thr": float(enter_thr),
                "stable_thr": float(stable_thr),
                "stable_frames": int(stable_frames),
            }
            armed = False
            stable_count = 0

    return auto_trigger, auto_metrics, curr, armed, stable_count, last_motion


def update_auto_trigger_bg(
    *,
    roi_img: np.ndarray,
    bg_sub: Any,
    bg_warmup_left: int,
    present_count: int,
    armed: bool,
    min_fg_ratio: float,
    min_contour_area: int,
    present_frames: int,
    bg_warmup: int,
    bg_history: int,
    bg_var_threshold: int,
    bg_detect_shadows: bool,
    morph_kernel: np.ndarray,
) -> Tuple[
    bool,
    Optional[Dict[str, Any]],
    int,
    int,
    bool,
    Optional[float],
    Optional[int],
]:
    auto_trigger = False
    auto_metrics = None
    last_fg_ratio: Optional[float] = None
    last_max_area: Optional[int] = None

    fg = bg_sub.apply(roi_img)

    if bg_warmup_left > 0:
        bg_warmup_left -= 1
        present_count = 0
        armed = True
        return auto_trigger, auto_metrics, bg_warmup_left, present_count, armed, last_fg_ratio, last_max_area

    _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
    fg = cv2.medianBlur(fg, 5)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, morph_kernel, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_DILATE, morph_kernel, iterations=1)

    fg_pixels = cv2.countNonZero(fg)
    total = fg.shape[0] * fg.shape[1]
    fg_ratio = fg_pixels / max(1, total)
    last_fg_ratio = fg_ratio

    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    for contour in contours:
        area = int(cv2.contourArea(contour))
        if area > max_area:
            max_area = area
    last_max_area = max_area

    present = (fg_ratio >= float(min_fg_ratio)) or (max_area >= int(min_contour_area))

    if present:
        present_count += 1
    else:
        present_count = 0
        armed = True

    if armed and present_count >= int(present_frames):
        auto_trigger = True
        auto_metrics = {
            "method": "bg",
            "fg_ratio": float(fg_ratio),
            "max_contour_area": int(max_area),
            "min_fg_ratio": float(min_fg_ratio),
            "min_contour_area": int(min_contour_area),
            "present_frames": int(present_frames),
            "bg_warmup": int(bg_warmup),
            "bg_history": int(bg_history),
            "bg_var_threshold": int(bg_var_threshold),
            "detect_shadows": bool(bg_detect_shadows),
        }
        armed = False
        present_count = 0

    return auto_trigger, auto_metrics, bg_warmup_left, present_count, armed, last_fg_ratio, last_max_area


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out_dir", type=str, default="data/captures/realsense")
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

    # Auto-eventos
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

    # Auto por MOTION
    ap.add_argument("--enter_thr", type=float, default=0.08, help="motion_ratio para re-armar (0-1)")
    ap.add_argument("--stable_thr", type=float, default=0.01, help="motion_ratio para considerar estable (0-1)")
    ap.add_argument("--stable_frames", type=int, default=25, help="frames estables para disparar")
    ap.add_argument("--cooldown_s", type=float, default=2.0, help="cooldown tras disparar")

    # Auto por BG
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

    pipeline = None
    profile = None
    writer = None

    frame_dir: Optional[Path] = None
    events_dir: Optional[Path] = None
    session_path: Optional[Path] = None

    session: Dict[str, Any] = {}
    idx = 0
    saved = 0
    manual_saved = 0
    fps_real: Optional[float] = None

    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    try:
        pipeline, profile, device_name, serial, color_sensor = open_realsense(args.width, args.height, args.fps)

        ok, frame = warmup_read_rs(pipeline)
        if not ok or frame is None:
            raise SystemExit("Warm-up falló con RealSense. Verifica cámara, cable, permisos y resolución/FPS.")

        h, w = frame.shape[:2]
        stream_info = get_rs_stream_info(profile)

        print(f"[INFO] RealSense OK. Resolución real: {w}x{h}")
        print(f"[INFO] Device: {device_name}")
        if serial:
            print(f"[INFO] Serial: {serial}")
        print(f"[INFO] Stream color: {stream_info}")

        frame_ts = time.strftime("%Y%m%d_%H%M%S")
        frame_dir = out_dir / f"frames_{frame_ts}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Guardando en: {frame_dir}")

        events_dir = frame_dir / "events"
        if args.events:
            events_dir.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] Eventos habilitados: {events_dir}")
            if args.auto_events:
                print(f"[INFO] Auto-eventos ON | method={args.auto_method} | cooldown_s={args.cooldown_s}")
                if args.auto_use_window_capture:
                    print(
                        f"[INFO] Auto-window ON | duration={args.auto_window_s:.1f}s | interval={args.auto_interval_s:.2f}s"
                    )
                if roi is None:
                    print("[WARN] auto_events sin ROI: funcionará, pero para PoC industrial se recomienda ROI en el mesón.")

        session_path = frame_dir / "session.json"
        session = {
            "session_id": frame_ts,
            "start_time_local": time.strftime("%Y-%m-%d %H:%M:%S"),
            "start_time_epoch": time.time(),
            "camera": {
                "type": "realsense",
                "device_name": device_name,
                "serial": serial,
                "requested": {"width": args.width, "height": args.height, "fps": args.fps},
                "actual": {
                    "width": w,
                    "height": h,
                    "fps_reported": stream_info.get("fps"),
                    "format": stream_info.get("format"),
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
                "auto_use_window_capture": bool(args.auto_use_window_capture),
                "auto_window_s": float(args.auto_window_s),
                "auto_interval_s": float(args.auto_interval_s),
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

        if args.save_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_path = frame_dir / f"capture_{frame_ts}.mp4"
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

        fps_win = max(5, int(args.fps_window))
        win_start_t = time.time()
        win_start_idx = 0

        event_id = 0
        cooldown_until = 0.0
        armed = True

        prev_gray: Optional[np.ndarray] = None
        stable_count = 0
        last_motion: Optional[float] = None

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

        recent_buffer_size = max(3, int(args.manual_buffer))
        recent_frames: Deque[Dict[str, Any]] = deque(maxlen=recent_buffer_size)

        # Estado para auto-window no bloqueante
        pending_auto_capture = False
        pending_event_id: Optional[int] = None
        pending_auto_metrics: Optional[Dict[str, Any]] = None
        pending_trigger_idx: Optional[int] = None
        pending_trigger_frame: Optional[np.ndarray] = None
        pending_records: List[Dict[str, Any]] = []
        pending_start_t = 0.0
        pending_next_capture_t = 0.0

        while True:
            ok, frame = read_frame_rs(pipeline)
            if not ok or frame is None:
                ok2, frame2 = warmup_read_rs(pipeline, tries=10, sleep_s=0.02)
                if not ok2 or frame2 is None:
                    print("[WARN] No pude leer frame desde RealSense; saliendo.")
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

            auto_trigger = False
            auto_metrics = None

            if args.events and args.auto_events:
                now = time.time()
                if now < cooldown_until:
                    armed = False
                    stable_count = 0
                    present_count = 0
                else:
                    roi_img = crop_roi(frame, roi) if roi else frame
                    if roi_img is not None and roi_img.size > 0:
                        if args.auto_method == "motion":
                            (
                                auto_trigger,
                                auto_metrics,
                                prev_gray,
                                armed,
                                stable_count,
                                last_motion,
                            ) = update_auto_trigger_motion(
                                roi_img=roi_img,
                                prev_gray=prev_gray,
                                armed=armed,
                                stable_count=stable_count,
                                enter_thr=float(args.enter_thr),
                                stable_thr=float(args.stable_thr),
                                stable_frames=int(args.stable_frames),
                            )
                        else:
                            assert bg_sub is not None
                            (
                                auto_trigger,
                                auto_metrics,
                                bg_warmup_left,
                                present_count,
                                armed,
                                last_fg_ratio,
                                last_max_area,
                            ) = update_auto_trigger_bg(
                                roi_img=roi_img,
                                bg_sub=bg_sub,
                                bg_warmup_left=bg_warmup_left,
                                present_count=present_count,
                                armed=armed,
                                min_fg_ratio=float(args.min_fg_ratio),
                                min_contour_area=int(args.min_contour_area),
                                present_frames=int(args.present_frames),
                                bg_warmup=int(args.bg_warmup),
                                bg_history=int(args.bg_history),
                                bg_var_threshold=int(args.bg_var_threshold),
                                bg_detect_shadows=bool(args.bg_detect_shadows),
                                morph_kernel=morph_kernel,
                            )

                        if auto_trigger:
                            cooldown_until = time.time() + float(args.cooldown_s)

            if auto_trigger and args.events:
                event_id += 1

                if args.auto_use_window_capture:
                    pending_auto_capture = True
                    pending_event_id = event_id
                    pending_auto_metrics = auto_metrics
                    pending_trigger_idx = idx
                    pending_trigger_frame = frame.copy()
                    pending_records = []
                    pending_start_t = time.time()
                    pending_next_capture_t = pending_start_t
                    print(f"[AUTO] Trigger #{event_id} iniciado | window={args.auto_window_s:.1f}s")
                else:
                    ev_dir = save_event(
                        events_dir=events_dir,
                        event_id=event_id,
                        frame=frame,
                        idx=idx,
                        roi=roi,
                        trigger="auto",
                        auto_metrics=auto_metrics,
                        burst_records=None,
                    )
                    session["events"]["counts"]["total"] += 1
                    session["events"]["counts"]["auto"] += 1
                    safe_write_json(session_path, session)
                    print(f"[AUTO] Evento #{event_id} guardado: {ev_dir}")

            if pending_auto_capture:
                now_cap = time.time()

                if now_cap >= pending_next_capture_t:
                    pending_records.append(
                        frame_record(
                            idx=idx,
                            frame=frame,
                            epoch_ms=int(time.time() * 1000),
                        )
                    )
                    pending_next_capture_t += float(args.auto_interval_s)

                if (now_cap - pending_start_t) >= float(args.auto_window_s):
                    if pending_records:
                        mid = len(pending_records) // 2
                        frame_for_event = pending_records[mid]["frame"]
                        idx_for_event = pending_records[mid]["idx"]
                    else:
                        frame_for_event = pending_trigger_frame if pending_trigger_frame is not None else frame
                        idx_for_event = pending_trigger_idx if pending_trigger_idx is not None else idx

                    ev_dir = save_event(
                        events_dir=events_dir,
                        event_id=int(pending_event_id),
                        frame=frame_for_event,
                        idx=int(idx_for_event),
                        roi=roi,
                        trigger="auto_window",
                        auto_metrics=pending_auto_metrics,
                        burst_records=pending_records,
                    )

                    session["events"]["counts"]["total"] += 1
                    session["events"]["counts"]["auto"] += 1
                    safe_write_json(session_path, session)
                    print(f"[AUTO] Evento #{pending_event_id} guardado: {ev_dir} | burst={len(pending_records)}")

                    pending_auto_capture = False
                    pending_event_id = None
                    pending_auto_metrics = None
                    pending_trigger_idx = None
                    pending_trigger_frame = None
                    pending_records = []
                    pending_start_t = 0.0
                    pending_next_capture_t = 0.0

            if idx > 0 and idx % fps_win == 0:
                session["runtime"]["fps_real_last"] = (round(fps_real, 2) if fps_real is not None else None)
                session["runtime"]["frames_total"] = idx
                session["runtime"]["frames_saved"] = saved
                session["runtime"]["manual_saved"] = manual_saved
                safe_write_json(session_path, session)

            if not args.no_display:
                disp = frame.copy()

                if roi is not None:
                    x, y, rw, rh = roi
                    cv2.rectangle(disp, (x, y), (x + rw, y + rh), (0, 255, 255), 2)

                fps_txt = f"{fps_real:.1f}" if fps_real is not None else "..."
                cv2.putText(
                    disp,
                    f"REALSENSE {w}x{h} idx={idx} saved={saved} fps={fps_txt}",
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

                    if args.auto_use_window_capture:
                        if pending_auto_capture:
                            remaining = max(0.0, float(args.auto_window_s) - (time.time() - pending_start_t))
                            cv2.putText(
                                disp,
                                f"AUTO_WINDOW REC {remaining:.1f}s burst={len(pending_records)}",
                                (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.62,
                                (0, 128, 255),
                                2,
                                cv2.LINE_AA,
                            )
                        else:
                            cv2.putText(
                                disp,
                                f"AUTO_WINDOW {args.auto_window_s:.1f}s / {args.auto_interval_s:.2f}s",
                                (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.62,
                                (255, 200, 0),
                                2,
                                cv2.LINE_AA,
                            )

                cv2.imshow("Capture (RealSense)", disp)
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

                    burst_n = max(1, int(args.manual_burst))
                    burst_records = list(recent_frames)[-burst_n:]

                    ev_dir = save_event(
                        events_dir=events_dir,
                        event_id=event_id,
                        frame=frame,
                        idx=idx,
                        roi=roi,
                        trigger="manual",
                        auto_metrics=None,
                        burst_records=burst_records,
                    )
                    session["events"]["counts"]["total"] += 1
                    session["events"]["counts"]["manual"] += 1
                    safe_write_json(session_path, session)
                    print(f"[EVENT] Manual #{event_id} guardado: {ev_dir} | burst={len(burst_records)}")

            idx += 1

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C. Cerrando...")

    finally:
        if writer is not None:
            writer.release()

        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass

        if not args.no_display:
            cv2.destroyAllWindows()

        if session and session_path is not None:
            try:
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
            except Exception:
                print("[DONE] Captura finalizada.")
        else:
            print("[DONE] Captura finalizada.")


if __name__ == "__main__":
    main()