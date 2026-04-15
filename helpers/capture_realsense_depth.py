# scripts/capture_realsense_depth.py
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
# JSON helpers
# -----------------------------
def safe_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


# -----------------------------
# Image / ROI helpers
# -----------------------------
def crop_roi(img: np.ndarray, roi: Optional[Tuple[int, int, int, int]]) -> Optional[np.ndarray]:
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


def crop_roi_depth(depth_img: np.ndarray, roi: Optional[Tuple[int, int, int, int]]) -> Optional[np.ndarray]:
    if roi is None:
        return None
    x, y, rw, rh = roi
    x = max(0, int(x))
    y = max(0, int(y))
    rw = max(1, int(rw))
    rh = max(1, int(rh))
    x2 = min(depth_img.shape[1], x + rw)
    y2 = min(depth_img.shape[0], y + rh)
    if x >= x2 or y >= y2:
        return None
    return depth_img[y:y2, x:x2].copy()


def to_gray_blur(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (7, 7), 0)
    return g


def motion_ratio(prev_gray: np.ndarray, curr_gray: np.ndarray, thresh: int = 25) -> float:
    diff = cv2.absdiff(prev_gray, curr_gray)
    _, bw = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
    changed = cv2.countNonZero(bw)
    total = bw.shape[0] * bw.shape[1]
    return changed / max(1, total)


def frame_record(
    idx: int,
    color: np.ndarray,
    depth: np.ndarray,
    epoch_ms: int,
) -> Dict[str, Any]:
    return {
        "idx": int(idx),
        "epoch_ms": int(epoch_ms),
        "color": color.copy(),
        "depth": depth.copy(),
    }


def colorize_depth_for_preview(
    depth_image: np.ndarray,
    *,
    min_depth_m: float = 0.1,
    max_depth_m: float = 2.0,
    depth_scale: float = 1.0,
) -> np.ndarray:
    if depth_image is None or depth_image.size == 0:
        return np.zeros((240, 320, 3), dtype=np.uint8)

    depth_m = depth_image.astype(np.float32) * float(depth_scale)
    depth_m = np.clip(depth_m, min_depth_m, max_depth_m)

    denom = max(1e-6, max_depth_m - min_depth_m)
    norm = ((depth_m - min_depth_m) / denom * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(255 - norm, cv2.COLORMAP_JET)
    return colored


# -----------------------------
# RealSense helpers
# -----------------------------
def open_realsense_depth(width: int, height: int, fps: int):
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, int(width), int(height), rs.format.bgr8, int(fps))
    config.enable_stream(rs.stream.depth, int(width), int(height), rs.format.z16, int(fps))

    profile = pipeline.start(config)

    device = profile.get_device()
    device_name = device.get_info(rs.camera_info.name) if device else "Unknown"
    serial = device.get_info(rs.camera_info.serial_number) if device else None

    depth_sensor = None
    color_sensor = None

    try:
        sensors = device.query_sensors()
        for s in sensors:
            try:
                s_name = s.get_info(rs.camera_info.name).lower()
                if "depth" in s_name:
                    depth_sensor = s
                if "color" in s_name:
                    color_sensor = s
            except Exception:
                continue
    except Exception:
        pass

    return pipeline, profile, device_name, serial, color_sensor, depth_sensor


def get_depth_scale(depth_sensor) -> float:
    try:
        return float(depth_sensor.get_depth_scale())
    except Exception:
        return 0.001


def warmup_read_rs_depth(
    pipeline,
    align,
    tries: int = 90,
    sleep_s: float = 0.03,
):
    last_color = None
    last_depth = None

    for _ in range(tries):
        try:
            frames = pipeline.wait_for_frames(timeout_ms=2000)
            frames = align.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if color_frame and depth_frame:
                color = np.asanyarray(color_frame.get_data())
                depth = np.asanyarray(depth_frame.get_data())

                if color is not None and color.size > 0 and depth is not None and depth.size > 0:
                    return True, color, depth

                last_color = color
                last_depth = depth
        except Exception:
            pass

        time.sleep(sleep_s)

    return False, last_color, last_depth


def read_frame_rs_depth(pipeline, align):
    frames = pipeline.wait_for_frames(timeout_ms=2000)
    frames = align.process(frames)

    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    if not color_frame or not depth_frame:
        return False, None, None

    color = np.asanyarray(color_frame.get_data())
    depth = np.asanyarray(depth_frame.get_data())

    if color is None or color.size == 0 or depth is None or depth.size == 0:
        return False, None, None

    return True, color, depth


def get_rs_stream_info(profile) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "color": {"format": None, "fps": None, "width": None, "height": None},
        "depth": {"format": None, "fps": None, "width": None, "height": None},
    }

    try:
        s = profile.get_stream(rs.stream.color)
        vsp = s.as_video_stream_profile()
        info["color"]["format"] = str(vsp.format())
        info["color"]["fps"] = int(vsp.fps())
        info["color"]["width"] = int(vsp.width())
        info["color"]["height"] = int(vsp.height())
    except Exception:
        pass

    try:
        s = profile.get_stream(rs.stream.depth)
        vsp = s.as_video_stream_profile()
        info["depth"]["format"] = str(vsp.format())
        info["depth"]["fps"] = int(vsp.fps())
        info["depth"]["width"] = int(vsp.width())
        info["depth"]["height"] = int(vsp.height())
    except Exception:
        pass

    return info


def get_color_intrinsics(profile) -> Dict[str, float]:
    try:
        stream = profile.get_stream(rs.stream.color)
        vsp = stream.as_video_stream_profile()
        intr = vsp.get_intrinsics()
        return {
            "fx": float(intr.fx),
            "fy": float(intr.fy),
            "cx": float(intr.ppx),
            "cy": float(intr.ppy),
            "width": int(intr.width),
            "height": int(intr.height),
        }
    except Exception:
        return {
            "fx": 0.0,
            "fy": 0.0,
            "cx": 0.0,
            "cy": 0.0,
            "width": 0,
            "height": 0,
        }


# -----------------------------
# Burst save helpers
# -----------------------------
def save_burst_frames_depth(
    *,
    ev_dir: Path,
    burst_records: List[Dict[str, Any]],
    roi: Optional[Tuple[int, int, int, int]],
    depth_scale: float,
    preview_min_depth_m: float,
    preview_max_depth_m: float,
) -> Dict[str, Any]:
    color_dir = ev_dir / "frames_color"
    depth_dir = ev_dir / "frames_depth"
    preview_dir = ev_dir / "frames_depth_preview"

    color_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    saved_frames = []

    for i, rec in enumerate(burst_records, start=1):
        color_fp = color_dir / f"frame_{i:02d}.jpg"
        depth_fp = depth_dir / f"frame_{i:02d}.npy"
        preview_fp = preview_dir / f"frame_{i:02d}.png"

        cv2.imwrite(str(color_fp), rec["color"])
        np.save(str(depth_fp), rec["depth"])

        preview = colorize_depth_for_preview(
            rec["depth"],
            min_depth_m=preview_min_depth_m,
            max_depth_m=preview_max_depth_m,
            depth_scale=depth_scale,
        )
        cv2.imwrite(str(preview_fp), preview)

        saved_frames.append(
            {
                "index_in_burst": i,
                "frame_idx": int(rec["idx"]),
                "epoch_ms": int(rec["epoch_ms"]),
                "color_path": str(color_fp),
                "depth_npy_path": str(depth_fp),
                "depth_preview_path": str(preview_fp),
            }
        )

    roi_color_paths = []
    roi_depth_paths = []
    roi_preview_paths = []

    if roi:
        roi_color_dir = ev_dir / "roi_color"
        roi_depth_dir = ev_dir / "roi_depth"
        roi_preview_dir = ev_dir / "roi_depth_preview"

        roi_color_dir.mkdir(parents=True, exist_ok=True)
        roi_depth_dir.mkdir(parents=True, exist_ok=True)
        roi_preview_dir.mkdir(parents=True, exist_ok=True)

        for i, rec in enumerate(burst_records, start=1):
            roi_color = crop_roi(rec["color"], roi)
            roi_depth = crop_roi_depth(rec["depth"], roi)

            if roi_color is not None and roi_color.size > 0:
                rp = roi_color_dir / f"roi_{i:02d}.jpg"
                cv2.imwrite(str(rp), roi_color)
                roi_color_paths.append(
                    {
                        "index_in_burst": i,
                        "frame_idx": int(rec["idx"]),
                        "path": str(rp),
                    }
                )

            if roi_depth is not None and roi_depth.size > 0:
                dp = roi_depth_dir / f"roi_{i:02d}.npy"
                np.save(str(dp), roi_depth)
                roi_depth_paths.append(
                    {
                        "index_in_burst": i,
                        "frame_idx": int(rec["idx"]),
                        "path": str(dp),
                    }
                )

                preview = colorize_depth_for_preview(
                    roi_depth,
                    min_depth_m=preview_min_depth_m,
                    max_depth_m=preview_max_depth_m,
                    depth_scale=depth_scale,
                )
                pp = roi_preview_dir / f"roi_{i:02d}.png"
                cv2.imwrite(str(pp), preview)
                roi_preview_paths.append(
                    {
                        "index_in_burst": i,
                        "frame_idx": int(rec["idx"]),
                        "path": str(pp),
                    }
                )

    mid_idx = len(saved_frames) // 2 if saved_frames else 0
    main_color_path = saved_frames[mid_idx]["color_path"] if saved_frames else None
    main_depth_path = saved_frames[mid_idx]["depth_npy_path"] if saved_frames else None
    main_preview_path = saved_frames[mid_idx]["depth_preview_path"] if saved_frames else None

    main_roi_color = roi_color_paths[len(roi_color_paths) // 2]["path"] if roi_color_paths else None
    main_roi_depth = roi_depth_paths[len(roi_depth_paths) // 2]["path"] if roi_depth_paths else None
    main_roi_preview = roi_preview_paths[len(roi_preview_paths) // 2]["path"] if roi_preview_paths else None

    return {
        "main_color_path": main_color_path,
        "main_depth_npy_path": main_depth_path,
        "main_depth_preview_path": main_preview_path,
        "main_roi_color_path": main_roi_color,
        "main_roi_depth_npy_path": main_roi_depth,
        "main_roi_depth_preview_path": main_roi_preview,
        "frames": saved_frames,
        "roi_color_frames": roi_color_paths,
        "roi_depth_frames": roi_depth_paths,
        "roi_depth_preview_frames": roi_preview_paths,
    }


def save_event_depth(
    *,
    events_dir: Path,
    event_id: int,
    color: np.ndarray,
    depth: np.ndarray,
    idx: int,
    roi: Optional[Tuple[int, int, int, int]],
    trigger: str,
    intrinsics: Dict[str, float],
    depth_scale: float,
    auto_metrics: Optional[Dict[str, Any]] = None,
    burst_records: Optional[List[Dict[str, Any]]] = None,
    preview_min_depth_m: float = 0.1,
    preview_max_depth_m: float = 2.0,
) -> Path:
    epoch_ms = int(time.time() * 1000)
    ev_dir = events_dir / f"event_{event_id:06d}"
    ev_dir.mkdir(parents=True, exist_ok=True)

    color_path = ev_dir / "color.jpg"
    depth_npy_path = ev_dir / "depth.npy"
    depth_preview_path = ev_dir / "depth_preview.png"

    cv2.imwrite(str(color_path), color)
    np.save(str(depth_npy_path), depth)

    preview = colorize_depth_for_preview(
        depth,
        min_depth_m=preview_min_depth_m,
        max_depth_m=preview_max_depth_m,
        depth_scale=depth_scale,
    )
    cv2.imwrite(str(depth_preview_path), preview)

    roi_color_path = None
    roi_depth_path = None
    roi_depth_preview_path = None

    roi_color = crop_roi(color, roi) if roi else None
    roi_depth = crop_roi_depth(depth, roi) if roi else None

    if roi_color is not None and roi_color.size > 0:
        roi_color_path = ev_dir / "roi_color.jpg"
        cv2.imwrite(str(roi_color_path), roi_color)

    if roi_depth is not None and roi_depth.size > 0:
        roi_depth_path = ev_dir / "roi_depth.npy"
        np.save(str(roi_depth_path), roi_depth)

        roi_preview = colorize_depth_for_preview(
            roi_depth,
            min_depth_m=preview_min_depth_m,
            max_depth_m=preview_max_depth_m,
            depth_scale=depth_scale,
        )
        roi_depth_preview_path = ev_dir / "roi_depth_preview.png"
        cv2.imwrite(str(roi_depth_preview_path), roi_preview)

    burst_info = None
    if burst_records:
        burst_info = save_burst_frames_depth(
            ev_dir=ev_dir,
            burst_records=burst_records,
            roi=roi,
            depth_scale=depth_scale,
            preview_min_depth_m=preview_min_depth_m,
            preview_max_depth_m=preview_max_depth_m,
        )

    ev = {
        "event_id": event_id,
        "trigger": trigger,
        "event_time_epoch_ms": epoch_ms,
        "event_time_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "frame_idx": int(idx),
        "paths": {
            "color": str(color_path),
            "depth_npy": str(depth_npy_path),
            "depth_preview": str(depth_preview_path),
            "roi_color": str(roi_color_path) if roi_color_path else None,
            "roi_depth_npy": str(roi_depth_path) if roi_depth_path else None,
            "roi_depth_preview": str(roi_depth_preview_path) if roi_depth_preview_path else None,
        },
        "roi": ({"x": roi[0], "y": roi[1], "w": roi[2], "h": roi[3]} if roi else None),
        "measurement": {
            "status": "not_attempted",
            "result_path": None,
        },
        "camera": {
            "intrinsics_color": intrinsics,
            "depth_scale": float(depth_scale),
        },
        "auto_metrics": auto_metrics,
        "burst": (
            {
                "enabled": True,
                "count": len(burst_records),
                "main_color_path": burst_info["main_color_path"] if burst_info else None,
                "main_depth_npy_path": burst_info["main_depth_npy_path"] if burst_info else None,
                "main_depth_preview_path": burst_info["main_depth_preview_path"] if burst_info else None,
                "main_roi_color_path": burst_info["main_roi_color_path"] if burst_info else None,
                "main_roi_depth_npy_path": burst_info["main_roi_depth_npy_path"] if burst_info else None,
                "main_roi_depth_preview_path": burst_info["main_roi_depth_preview_path"] if burst_info else None,
                "frames": burst_info["frames"] if burst_info else [],
                "roi_color_frames": burst_info["roi_color_frames"] if burst_info else [],
                "roi_depth_frames": burst_info["roi_depth_frames"] if burst_info else [],
                "roi_depth_preview_frames": burst_info["roi_depth_preview_frames"] if burst_info else [],
            }
            if burst_records
            else {
                "enabled": False,
                "count": 0,
                "main_color_path": None,
                "main_depth_npy_path": None,
                "main_depth_preview_path": None,
                "main_roi_color_path": None,
                "main_roi_depth_npy_path": None,
                "main_roi_depth_preview_path": None,
                "frames": [],
                "roi_color_frames": [],
                "roi_depth_frames": [],
                "roi_depth_preview_frames": [],
            }
        ),
    }

    safe_write_json(ev_dir / "event.json", ev)
    return ev_dir


def capture_auto_window_rs_depth(
    *,
    pipeline,
    align,
    idx_start: int,
    duration_s: float,
    interval_s: float,
) -> Tuple[List[Dict[str, Any]], int]:
    records: List[Dict[str, Any]] = []

    duration_s = max(0.5, float(duration_s))
    interval_s = max(0.1, float(interval_s))

    start_t = time.time()
    next_capture_t = start_t
    idx = int(idx_start)

    while True:
        now = time.time()
        if (now - start_t) > duration_s:
            break

        ok, color, depth = read_frame_rs_depth(pipeline, align)
        if not ok or color is None or depth is None:
            ok2, color2, depth2 = warmup_read_rs_depth(pipeline, align, tries=5, sleep_s=0.01)
            if not ok2:
                break
            color = color2
            depth = depth2

        now_epoch_ms = int(time.time() * 1000)

        if now >= next_capture_t:
            records.append(
                frame_record(
                    idx=idx,
                    color=color,
                    depth=depth,
                    epoch_ms=now_epoch_ms,
                )
            )
            next_capture_t += interval_s

        idx += 1
        time.sleep(0.005)

    return records, idx


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out_dir", type=str, default="data/captures/realsense_depth")
    ap.add_argument("--every", type=int, default=15, help="Guarda 1 color preview cada N frames (0 desactiva)")
    ap.add_argument("--no_display", action="store_true", help="Headless. Cortar con Ctrl+C.")
    ap.add_argument("--fps_window", type=int, default=30)

    ap.add_argument("--preview_min_depth_m", type=float, default=0.10)
    ap.add_argument("--preview_max_depth_m", type=float, default=2.00)

    # Eventos
    ap.add_argument("--events", action="store_true", help="Guarda eventos color+depth+event.json")
    ap.add_argument(
        "--roi",
        type=int,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        default=None,
        help="ROI (x y w h). Ej: --roi 240 180 800 360",
    )

    # Manual
    ap.add_argument("--manual_burst", type=int, default=3, help="Cantidad de frames a guardar en evento manual.")
    ap.add_argument("--manual_buffer", type=int, default=5, help="Tamaño buffer reciente para burst manual.")

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
        help="Cuando detecta trigger, captura secuencia temporal en vez de frame único.",
    )
    ap.add_argument("--auto_window_s", type=float, default=8.0, help="Duración secuencia automática en segundos.")
    ap.add_argument("--auto_interval_s", type=float, default=0.5, help="Intervalo entre frames del burst.")

    # Auto por motion
    ap.add_argument("--enter_thr", type=float, default=0.08)
    ap.add_argument("--stable_thr", type=float, default=0.01)
    ap.add_argument("--stable_frames", type=int, default=25)
    ap.add_argument("--cooldown_s", type=float, default=2.0)

    # Auto por bg
    ap.add_argument("--bg_warmup", type=int, default=45)
    ap.add_argument("--min_fg_ratio", type=float, default=0.02)
    ap.add_argument("--min_contour_area", type=int, default=2500)
    ap.add_argument("--present_frames", type=int, default=10)
    ap.add_argument("--bg_history", type=int, default=200)
    ap.add_argument("--bg_var_threshold", type=int, default=16)
    ap.add_argument("--bg_detect_shadows", action="store_true")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    roi: Optional[Tuple[int, int, int, int]] = tuple(args.roi) if args.roi else None

    pipeline = None

    try:
        pipeline, profile, device_name, serial, color_sensor, depth_sensor = open_realsense_depth(
            args.width, args.height, args.fps
        )
        align = rs.align(rs.stream.color)

        depth_scale = get_depth_scale(depth_sensor)

        ok, color, depth = warmup_read_rs_depth(pipeline, align)
        if not ok:
            raise SystemExit("Warm-up falló con RealSense depth. Verifica cámara, cable, permisos y resolución/FPS.")

        h, w = color.shape[:2]
        stream_info = get_rs_stream_info(profile)
        intrinsics = get_color_intrinsics(profile)

        print(f"[INFO] RealSense Depth OK. Resolución color real: {w}x{h}")
        print(f"[INFO] Device: {device_name}")
        if serial:
            print(f"[INFO] Serial: {serial}")
        print(f"[INFO] Stream info: {stream_info}")
        print(f"[INFO] Depth scale: {depth_scale}")

        ts = time.strftime("%Y%m%d_%H%M%S")
        frame_dir = out_dir / f"frames_{ts}"
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

        session_path = frame_dir / "session.json"
        session: Dict[str, Any] = {
            "session_id": ts,
            "start_time_local": time.strftime("%Y-%m-%d %H:%M:%S"),
            "start_time_epoch": time.time(),
            "camera": {
                "type": "realsense_depth",
                "device_name": device_name,
                "serial": serial,
                "requested": {"width": args.width, "height": args.height, "fps": args.fps},
                "actual": stream_info,
                "intrinsics_color": intrinsics,
                "depth_scale": float(depth_scale),
            },
            "capture": {
                "out_dir": str(out_dir),
                "frame_dir": str(frame_dir),
                "every": args.every,
                "manual_burst": int(max(1, args.manual_burst)),
                "manual_buffer": int(max(3, args.manual_buffer)),
                "auto_use_window_capture": bool(args.auto_use_window_capture),
                "auto_window_s": float(args.auto_window_s),
                "auto_interval_s": float(args.auto_interval_s),
                "preview_min_depth_m": float(args.preview_min_depth_m),
                "preview_max_depth_m": float(args.preview_max_depth_m),
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

        if args.events and args.auto_events and args.auto_method == "bg":
            bg_sub = cv2.createBackgroundSubtractorMOG2(
                history=int(args.bg_history),
                varThreshold=int(args.bg_var_threshold),
                detectShadows=bool(args.bg_detect_shadows),
            )

        recent_frames: Deque[Dict[str, Any]] = deque(maxlen=max(3, int(args.manual_buffer)))

        while True:
            ok, color, depth = read_frame_rs_depth(pipeline, align)
            if not ok or color is None or depth is None:
                ok2, color2, depth2 = warmup_read_rs_depth(pipeline, align, tries=10, sleep_s=0.02)
                if not ok2:
                    print("[WARN] No pude leer frame color/depth desde RealSense; saliendo.")
                    break
                color = color2
                depth = depth2

            now_epoch_ms = int(time.time() * 1000)
            recent_frames.append(frame_record(idx=idx, color=color, depth=depth, epoch_ms=now_epoch_ms))

            if args.every > 0 and (idx % args.every == 0):
                color_fp = frame_dir / f"color_{idx:06d}_{now_epoch_ms}.jpg"
                depth_fp = frame_dir / f"depth_{idx:06d}_{now_epoch_ms}.npy"
                preview_fp = frame_dir / f"depth_preview_{idx:06d}_{now_epoch_ms}.png"

                cv2.imwrite(str(color_fp), color)
                np.save(str(depth_fp), depth)

                preview = colorize_depth_for_preview(
                    depth,
                    min_depth_m=float(args.preview_min_depth_m),
                    max_depth_m=float(args.preview_max_depth_m),
                    depth_scale=depth_scale,
                )
                cv2.imwrite(str(preview_fp), preview)
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
                    roi_img = crop_roi(color, roi) if roi else color
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
                            assert bg_sub is not None

                            fg = bg_sub.apply(roi_img)
                            if bg_warmup_left > 0:
                                bg_warmup_left -= 1
                                present_count = 0
                                armed = True
                            else:
                                _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
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
                                    armed = True

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

                if args.auto_use_window_capture:
                    burst_records, idx = capture_auto_window_rs_depth(
                        pipeline=pipeline,
                        align=align,
                        idx_start=idx,
                        duration_s=float(args.auto_window_s),
                        interval_s=float(args.auto_interval_s),
                    )
                    trigger_name = "auto_window"
                else:
                    burst_records = None
                    trigger_name = "auto"

                color_for_event = burst_records[0]["color"] if burst_records else color
                depth_for_event = burst_records[0]["depth"] if burst_records else depth
                idx_for_event = burst_records[0]["idx"] if burst_records else idx

                ev_dir = save_event_depth(
                    events_dir=events_dir,
                    event_id=event_id,
                    color=color_for_event,
                    depth=depth_for_event,
                    idx=idx_for_event,
                    roi=roi,
                    trigger=trigger_name,
                    intrinsics=intrinsics,
                    depth_scale=depth_scale,
                    auto_metrics=auto_metrics,
                    burst_records=burst_records,
                    preview_min_depth_m=float(args.preview_min_depth_m),
                    preview_max_depth_m=float(args.preview_max_depth_m),
                )
                session["events"]["counts"]["total"] += 1
                session["events"]["counts"]["auto"] += 1
                safe_write_json(session_path, session)
                print(f"[AUTO] Evento depth #{event_id} guardado: {ev_dir}")

            if idx % fps_win == 0:
                session["runtime"]["fps_real_last"] = (round(fps_real, 2) if fps_real is not None else None)
                session["runtime"]["frames_total"] = idx
                session["runtime"]["frames_saved"] = saved
                session["runtime"]["manual_saved"] = manual_saved
                safe_write_json(session_path, session)

            if not args.no_display:
                disp_color = color.copy()
                disp_depth = colorize_depth_for_preview(
                    depth,
                    min_depth_m=float(args.preview_min_depth_m),
                    max_depth_m=float(args.preview_max_depth_m),
                    depth_scale=depth_scale,
                )

                if roi is not None:
                    x, y, rw, rh = roi
                    cv2.rectangle(disp_color, (x, y), (x + rw, y + rh), (0, 255, 255), 2)
                    cv2.rectangle(disp_depth, (x, y), (x + rw, y + rh), (0, 255, 255), 2)

                fps_txt = f"{fps_real:.1f}" if fps_real is not None else "..."

                cv2.putText(
                    disp_color,
                    f"RS DEPTH color {w}x{h} idx={idx} saved={saved} fps={fps_txt}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    disp_depth,
                    f"Depth preview [{args.preview_min_depth_m:.2f}m - {args.preview_max_depth_m:.2f}m]",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                if args.events and args.auto_events:
                    if args.auto_method == "motion":
                        mr_txt = f"{last_motion:.3f}" if last_motion is not None else "..."
                        txt = f"AUTO(motion) motion={mr_txt} stable={stable_count}/{args.stable_frames} armed={1 if armed else 0}"
                    else:
                        fg_txt = f"{last_fg_ratio:.3f}" if last_fg_ratio is not None else "..."
                        ma_txt = f"{last_max_area}" if last_max_area is not None else "..."
                        txt = f"AUTO(bg) fg={fg_txt} maxA={ma_txt} present={present_count}/{args.present_frames} warmup_left={bg_warmup_left}"

                    cv2.putText(
                        disp_color,
                        txt,
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                cv2.imshow("Capture (RealSense Color)", disp_color)
                cv2.imshow("Capture (RealSense Depth Preview)", disp_depth)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

                if key == ord("s"):
                    epoch_ms = int(time.time() * 1000)

                    color_fp = frame_dir / f"manual_color_{idx:06d}_{epoch_ms}.jpg"
                    depth_fp = frame_dir / f"manual_depth_{idx:06d}_{epoch_ms}.npy"
                    preview_fp = frame_dir / f"manual_depth_preview_{idx:06d}_{epoch_ms}.png"

                    cv2.imwrite(str(color_fp), color)
                    np.save(str(depth_fp), depth)

                    preview = colorize_depth_for_preview(
                        depth,
                        min_depth_m=float(args.preview_min_depth_m),
                        max_depth_m=float(args.preview_max_depth_m),
                        depth_scale=depth_scale,
                    )
                    cv2.imwrite(str(preview_fp), preview)

                    saved += 1
                    manual_saved += 1
                    print(f"[INFO] Guardado manual color/depth: {color_fp} | {depth_fp}")

                if key == ord("e") and args.events:
                    event_id += 1

                    burst_n = max(1, int(args.manual_burst))
                    burst_records = list(recent_frames)[-burst_n:]

                    ev_dir = save_event_depth(
                        events_dir=events_dir,
                        event_id=event_id,
                        color=color,
                        depth=depth,
                        idx=idx,
                        roi=roi,
                        trigger="manual",
                        intrinsics=intrinsics,
                        depth_scale=depth_scale,
                        auto_metrics=None,
                        burst_records=burst_records,
                        preview_min_depth_m=float(args.preview_min_depth_m),
                        preview_max_depth_m=float(args.preview_max_depth_m),
                    )
                    session["events"]["counts"]["total"] += 1
                    session["events"]["counts"]["manual"] += 1
                    safe_write_json(session_path, session)
                    print(f"[EVENT] Manual depth #{event_id} guardado: {ev_dir} | burst={len(burst_records)}")

            idx += 1

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C. Cerrando...")

    finally:
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass

        if not args.no_display:
            cv2.destroyAllWindows()

        try:
            session["runtime"]["fps_real_last"] = (round(fps_real, 2) if fps_real is not None else None)
            session["runtime"]["frames_total"] = idx
            session["runtime"]["frames_saved"] = saved
            session["runtime"]["manual_saved"] = manual_saved
            session["end_time_local"] = time.strftime("%Y-%m-%d %H:%M:%S")
            session["end_time_epoch"] = time.time()
            session["status"] = "done"
            safe_write_json(session_path, session)
            print("[DONE] Captura depth finalizada.")
            print(f"[TRACE] session.json: {session_path}")
        except Exception:
            print("[DONE] Captura depth finalizada.")


if __name__ == "__main__":
    main()