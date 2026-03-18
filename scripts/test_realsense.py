# scripts/test_realsense.py
from __future__ import annotations

import sys
import time

import cv2
import numpy as np
import pyrealsense2 as rs


def main() -> None:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = None
    try:
        profile = pipeline.start(config)
        device = profile.get_device()
        print("[INFO] Device:", device.get_info(rs.camera_info.name))
        print("[INFO] Serial:", device.get_info(rs.camera_info.serial_number))

        for _ in range(30):
            frames = pipeline.wait_for_frames(timeout_ms=3000)
            color_frame = frames.get_color_frame()
            if color_frame:
                img = np.asanyarray(color_frame.get_data())
                print("[INFO] Primer frame:", img.shape)
                break
            time.sleep(0.03)
        else:
            raise RuntimeError("No llegaron frames de color.")

        while True:
            frames = pipeline.wait_for_frames(timeout_ms=3000)
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            img = np.asanyarray(color_frame.get_data())
            cv2.imshow("RealSense RGB Test", img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrumpido por teclado.")
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("[DONE] Test finalizado.")


if __name__ == "__main__":
    main()