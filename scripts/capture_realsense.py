# scripts/capture_realsense.py
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    raise SystemExit(
        "pyrealsense2 no está instalado. Ejecuta: pip install pyrealsense2"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out_dir", type=str, default="data/captures/realsense")
    ap.add_argument("--save_video", action="store_true")
    ap.add_argument("--save_depth", action="store_true")
    ap.add_argument("--every", type=int, default=15, help="Guardar 1 frame cada N")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Inicializando pipeline RealSense...")
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    profile = pipeline.start(config)

    align = rs.align(rs.stream.color)

    print("[INFO] RealSense iniciada correctamente.")

    ts = time.strftime("%Y%m%d_%H%M%S")
    frame_dir = out_dir / f"frames_{ts}"
    frame_dir.mkdir(parents=True, exist_ok=True)

    depth_dir = None
    if args.save_depth:
        depth_dir = out_dir / f"depth_{ts}"
        depth_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_path = out_dir / f"capture_{ts}.mp4"
        writer = cv2.VideoWriter(str(video_path), fourcc, args.fps, (args.width, args.height))
        print(f"[INFO] Grabando video en: {video_path}")

    idx = 0
    saved = 0

    print("[INFO] Controles: 'q' salir | 's' guardar frame manual")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = None

            if depth_frame:
                depth_image = np.asanyarray(depth_frame.get_data())

            if writer is not None:
                writer.write(color_image)

            if args.every > 0 and (idx % args.every == 0):
                fp = frame_dir / f"frame_{idx:06d}.jpg"
                cv2.imwrite(str(fp), color_image)
                saved += 1

                if args.save_depth and depth_image is not None:
                    dp = depth_dir / f"depth_{idx:06d}.png"
                    cv2.imwrite(str(dp), depth_image)

            disp = color_image.copy()
            cv2.putText(
                disp,
                f"RealSense {args.width}x{args.height} idx={idx} saved={saved}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Capture (RealSense)", disp)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("s"):
                fp = frame_dir / f"manual_{idx:06d}.jpg"
                cv2.imwrite(str(fp), color_image)
                print(f"[INFO] Guardado manual: {fp}")
                saved += 1

            idx += 1

    finally:
        print("[INFO] Cerrando pipeline...")
        pipeline.stop()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    print("[DONE] Captura finalizada.")


if __name__ == "__main__":
    main()
