En macOS: brew install zbar
En macOS: brew install tesseract
scripts/capture_opencv.py -> python scripts/capture_opencv.py --device 0 --save_video --every 15

pip install pyrealsense2 # una vez instalado el SDK de Intel RealSense en el sistema
scripts/capture_realsense.py -> python scripts/capture_realsense.py --save_video --save_depth