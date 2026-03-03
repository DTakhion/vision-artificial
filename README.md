## `scripts/capture_opencv.py` --- Captura OpenCV + trazabilidad (PoC)

Script de captura en tiempo real desde webcam (OpenCV/macOS
AVFoundation) orientado a PoC en terreno.\
Guarda evidencia (frames / video) y trazabilidad (`session.json` +
eventos).

------------------------------------------------------------------------

### Qué hace

-   Abre cámara (`--device`) y realiza *warm-up* para asegurar lectura
    estable en macOS.
-   Captura frames en loop y puede:
    -   Guardar **frames periódicos** (1 cada N frames) para
        dataset/evidencia.
    -   Guardar **video** opcional (`.mp4`) de toda la sesión.
-   Escribe **metadata de sesión** en `session.json` (configuración,
    paths, FPS real, contadores, tiempos y estado).
-   Permite **modo headless** (`--no_display`) para ejecutar sin ventana
    (cortar con `Ctrl+C`).

------------------------------------------------------------------------

### Estructura de salida

Cada ejecución crea una carpeta con timestamp:

data/captures/opencv/frames_YYYYMMDD_HHMMSS/ session.json
capture_YYYYMMDD_HHMMSS.mp4 (opcional) frame_000000_XXXXXXXXXXXX.jpg
(frames periódicos si --every \> 0) manual_000123_XXXXXXXXXXXX.jpg
(frames manuales con tecla 's') events/ (si --events) event_000001/
event.json frame.jpg roi.jpg (si se definió ROI)

------------------------------------------------------------------------

### ROI (Region of Interest)

Opcionalmente se define un rectángulo ROI donde ocurre el "gesto"
(caja/etiqueta en el mesón):

--roi X Y W H

-   Si se define ROI:
    -   Se dibuja en pantalla.
    -   Se guarda `roi.jpg` dentro de cada evento.
-   Si no se define ROI:
    -   El modo automático funciona usando el frame completo (más
        ruidoso).

------------------------------------------------------------------------

### Eventos (trazabilidad del "gesto")

Al habilitar `--events`, el script registra "eventos" (imagen +
`event.json`) para auditoría y validaciones.

#### Evento manual (controlado)

-   Tecla `e`: guarda un evento en `events/event_XXXXXX/`.
-   Incluye:
    -   `frame.jpg` (imagen completa)
    -   `roi.jpg` (recorte ROI si aplica)
    -   `event.json` con:
        -   `trigger="manual"`
        -   `readout="not_attempted"`

------------------------------------------------------------------------

#### Evento automático (demostrable sin modelos)

Se habilita con:

--auto_events

(Requiere `--events`)

Detecta el gesto automáticamente usando **movimiento + estabilidad** en
ROI.

Lógica:

1.  Calcula `motion_ratio` (0..1) como porcentaje de píxeles que cambian
    en la ROI entre frames sucesivos.
2.  Cuando hay movimiento fuerte (`motion_ratio > enter_thr`) → el
    sistema se "arma".
3.  Si luego la ROI queda estable (`motion_ratio < stable_thr`) por
    `stable_frames` consecutivos → dispara evento automático.
4.  Aplica un cooldown (`cooldown_s`) para evitar múltiples disparos
    seguidos.

Cada evento automático se guarda igual que el manual, pero con:

-   `trigger="auto"`
-   `auto_metrics`:
    -   `motion_ratio`
    -   `enter_thr`
    -   `stable_thr`
    -   `stable_frames_required`

------------------------------------------------------------------------

### Overlay (pantalla)

Muestra en vivo:

-   Resolución real
-   Índice de frame
-   Contador de frames guardados
-   FPS real (ventana rolling)
-   Si auto-eventos están activos:
    -   `motion_ratio`
    -   `stable_count`
    -   Estado `armed`
    -   ROI dibujado

------------------------------------------------------------------------

### Controles (con ventana)

q → Salir\
s → Guardar frame manual\
e → Guardar evento manual (si `--events`)


----------------------------------------------------------------------

# 🛠 Instalación (macOS)

## Dependencias del sistema

``` bash
brew install zbar
brew install tesseract
```

## Python

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

# 📷 Captura OpenCV -- scripts/capture_opencv.py

Script de captura en tiempo real desde webcam (OpenCV/macOS
AVFoundation) orientado a PoC en terreno.

Guarda evidencia visual y metadata estructurada para trazabilidad.

------------------------------------------------------------------------

## 🚀 Ejecución básica

### Captura simple

``` bash
python scripts/capture_opencv.py   --device 0 --width 640 --height 480 --fps 30 --every 15
python scripts/capture_opencv.py --device 0 --width 640 --height 480 --fps 30 --save_video --every 15
```

### Headless

``` bash
python scripts/capture_opencv.py   --device 0 --width 640 --height 480 --fps 30 --every 15 --no_display
python scripts/capture_opencv.py --device 0 --width 640 --height 480 --fps 30 --every 15
python scripts/capture_opencv.py --device 0 --width 640 --height 480 --fps 30 --every 15 --no_display
```

### Con video

``` bash
python scripts/capture_opencv.py   --device 0 --width 1280 --height 720 --fps 30   --every 10 --save_video
python scripts/capture_opencv.py --device 0 --width 1280 --height 720 --fps 30 --every 10 --save_video
```

------------------------------------------------------------------------

## 🎯 Eventos

### Manual (tecla e)

``` bash
python scripts/capture_opencv.py   --width 1280 --height 720 --fps 30   --events --roi 200 200 800 300
python scripts/capture_opencv.py --width 1280 --height 720 --fps 30 --events --roi 200 200 800 300
```

### Automático + respaldo manual

``` bash
python scripts/capture_opencv.py   --width 1280 --height 720 --fps 30   --events --auto_events --roi 200 200 800 300
python scripts/capture_opencv.py --width 1280 --height 720 --fps 30 --events --auto_events --roi 200 200 800 300

```

Lógica automática: 1. Detecta movimiento fuerte. 2. Espera estabilidad.
3. Dispara evento. 4. Aplica cooldown.

------------------------------------------------------------------------

# Estructura de salida

    data/captures/opencv/frames_YYYYMMDD_HHMMSS/
        session.json
        capture_YYYYMMDD_HHMMSS.mp4
        frame_XXXXXX.jpg
        manual_XXXXXX.jpg
        events/
            event_000001/
                frame.jpg
                roi.jpg
                event.json

------------------------------------------------------------------------

# RealSense

``` bash
pip install pyrealsense2
python scripts/capture_realsense.py --save_video --save_depth
```

------------------------------------------------------------------------

# Lectura QR / Barcode / Serial

``` bash
python -m utils.vision_qr data/tests_qr/527ca803-4e3c-4547-8cf5-00cb5f406bf7.JPG --variants all --budget 650
python utils/vision_barcode.py data/tests_qr/barcode_ok.jpg
python utils/vision_ocr.py data/tests_qr/serial_ok.jpg
python utils/vision_readout.py data/tests_qr/qr_ok.jpg
```
## Preprocesamiento – Herramientas y parámetros (`utils/vision_preprocess.py`)

| Etapa / herramienta | Flag (cfg) | Parámetros (valores actuales) | Implementación (OpenCV) | Variantes que genera |
|---|---|---|---|---|
| Resize “max side” | `resize_max_side` | `1280` (0 = desactiva) | `cv2.resize(..., INTER_AREA)` si `max(h,w) > 1280` | Afecta a todas (pre-step) |
| Grayscale | (siempre) | — | `cv2.cvtColor(BGR2GRAY)` | Base para todas |
| Contraste (CLAHE) | `clahe=True` | `clahe_clip=2.0`, `clahe_grid=(8,8)` | `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)` | Afecta `gray` y derivadas |
| Denoise (NLMeans) | `denoise=False` | `denoise_h=10`, `templateWindowSize=7`, `searchWindowSize=21` | `cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)` | Afecta `gray` y derivadas (si se activa) |
| Sharpen (Unsharp mask) | `sharpen=True` | `sharpen_amount=0.6`, `sigmaX=1.2` | `GaussianBlur(sigmaX=1.2)` + `addWeighted(gray, 1+amount, blur, -amount, 0)` | `sharp` (desde `gray`) |
| Denoise preservando bordes (Bilateral) | `bilateral=True` | `bilateral_d=9`, `bilateral_sigma_color=75`, `bilateral_sigma_space=75` | `cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)` | `bilateral` (desde `gray`) |
| Sharpen leve post-bilateral | (implícito si `bilateral` y `sharpen`) | `amount=0.35`, `sigmaX=1.2` | Unsharp suave sobre `bilateral` | `bilateral_sharp` |
| Morfología: Close | `morph_close=True` | `morph_kernel=(3,3)`, `iterations=1` | `getStructuringElement(MORPH_RECT,(3,3))` + `morphologyEx(sharp, MORPH_CLOSE, iter=1)` | `morph_close` |
| Binarización adaptativa (Gauss) | `binarize=False` | `blockSize=31`, `C=2` | `cv2.adaptiveThreshold(sharp,255,GAUSSIAN_C,THRESH_BINARY,31,2)` | `bw` (si se activa) |
| Upscale (INTER_CUBIC) | `upscale=True` | `upscale_factors=(2.0,)` | `cv2.resize(..., fx=2.0, fy=2.0, INTER_CUBIC)` | `sharp_x2`, `bilateral_x2`, `morph_close_x2`, `bw_x2` *(según existan)* |
| Métrica de nitidez (Laplacian var) | (función aparte) | — | `cv2.Laplacian(gray, CV_64F).var()` | No crea variante; devuelve score |