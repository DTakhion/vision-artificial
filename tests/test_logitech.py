# scripts/test_logitech.py

## canal

import cv2

print("Buscando cámaras disponibles...\n")

for i in range(10):  # ampliamos rango por si acaso
    cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
    
    if cap.isOpened():
        print(f"Cámara disponible en índice {i}")
        
        # Intentamos leer un frame (validación real)
        ret, frame = cap.read()
        if ret:
            print(f"→ Frame capturado correctamente ({frame.shape})")
        else:
            print("No se pudo capturar frame")
        
        cap.release()
    else:
        print(f"Índice {i} no disponible")

# ejecución para conocimiento de canal (normalmente 0)
# python scripts/test_logitech.py

# corroboración de canal (ejemplo, 0)
# python scripts/capture_opencv.py --device 0 --width 1920 --height 1080 --fps 30

# accion 4k
# python scripts/capture_opencv.py \
#   --device 0 \
#   --width 3840 \
#   --height 2160 \
#   --fps 30
