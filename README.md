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

python -m utils.vision_barcode data/tests_barcode/527ca803-4e3c-4547-8cf5-00cb5f406bf7.JPG --variants all --budget 650

# Preferir zxingcpp (recomendado si lo tienes instalado)
python -m utils.vision_barcode data/tests_barcode/527ca803-4e3c-4547-8cf5-00cb5f406bf7.JPG --variants all --budget 650 --prefer zxingcpp

# Preferir pyzbar
python -m utils.vision_barcode data/tests_barcode/527ca803-4e3c-4547-8cf5-00cb5f406bf7.JPG --variants all --budget 650 --prefer pyzbar

# Preferir OpenCV barcode (si tu build lo soporta)
python -m utils.vision_barcode data/tests_barcode/527ca803-4e3c-4547-8cf5-00cb5f406bf7.JPG --variants all --budget 650 --prefer opencv_barcode

python utils/vision_ocr.py data/tests_qr/serial_ok.jpg
python -m utils.vision_ocr data/tests_barcode/ALGUNA_IMAGEN.JPG --variants all --budget 650 --aggressive
python -m utils.vision_ocr data/tests_barcode/ALGUNA_IMAGEN.JPG --variants all --budget 650 --prefer tesseract --no_fallback --aggressive
python -m utils.vision_ocr data/tests_barcode/ALGUNA_IMAGEN.JPG --variants all --budget 650 --prefer easyocr --no_fallback --aggressive

python -m utils.vision_ocr data/tests_barcode/2b80dbe7-8ac4-423e-82b5-0289f6ed1684.JPG \
  --variants all --no_budget --max_tries 120 --aggressive --roi all

python -m utils.vision_ocr data/tests_barcode/2b80dbe7-8ac4-423e-82b5-0289f6ed1684.JPG \
  --variants all --no_budget --max_tries 40 --aggressive --roi all --save_debug_rois data/_debug_rois --debug

python utils/vision_readout.py data/tests_qr/qr_ok.jpg
# QR+BarCode
python -m utils.vision_readout data/tests_barcode/2b80dbe7-8ac4-423e-82b5-0289f6ed1684.JPG
# QR+BarCode+OCR
python -m utils.vision_readout data/tests_barcode/2b80dbe7-8ac4-423e-82b5-0289f6ed1684.JPG --mode retry --ocr

# BarCode
## collect mode
python -m utils.vision_barcode data/tests_multibarcode/1f2a0f03-f73c-4da7-a943-a54088c2c799.JPG --mode collect --budget 1800 --variants all --roi_upscale 4.0

## collect mode agresive y recue
python -m utils.vision_barcode data/tests_multibarcode/29788e64-ad4a-4d62-849c-5b22f1cb2e83.JPG --mode collect --budget 1800 --variants all --roi_upscale 4.0 --roi_stage_ratio 0.55 --max_tiles 5

## CodeBar mas presupuesto y todas las variables (indicado)
python -m utils.vision_barcode /ruta/a/imagen.jpg --mode collect --budget 700 --variants all

# sin ROI
python -m utils.vision_barcode /ruta/a/imagen.jpg --mode collect --no_roi

# con ROI pero sin barrido full imagen
python -m utils.vision_barcode /ruta/a/imagen.jpg --mode collect --no_full_image

# utils/vision_barcode_plus.py 
python -m utils.vision_barcode_plus data/tests_multibarcode/29788e64-ad4a-4d62-849c-5b22f1cb2e83.JPG --budget 6500 --variants all --roi_upscale 4.0

# ReadOut

# BarCode
python -m utils.vision_readout /ruta/a/imagen.jpg --mode immediate --barcode --no-ocr --no-qr

# barCode + OCR
python -m utils.vision_readout /ruta/a/imagen.jpg --mode immediate --barcode --ocr --no-qr

# Retry con BarCode Collect
python -m utils.vision_readout data/tests_multibarcode/29788e64-ad4a-4d62-849c-5b22f1cb2e83.JPG --mode retry --budget 6500 --barcode_mode collect_plus --barcode_budget 6000 --no-ocr --no-qr

# especifico Schneider
python -m utils.vision_readout /ruta/a/imagen.jpg --mode retry --barcode --ocr --no-qr --barcode_mode collect

## OCR
# agresivo, modo collect
python -m utils.vision_ocr /ruta/a/imagen.jpg --aggressive --mode collect

# OCR+BarCode like
python -m utils.vision_ocr /ruta/a/imagen.jpg --roi barcode_like --aggressive --mode collect

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

## scripts/capture_opencv.py

``` bash
python scripts/capture_opencv.py \
--device 0 \
--width 1280 \
--height 720 \
--fps 30 \
--events \
--manual_burst 3 \
--manual_buffer 5 \
--every 0
```
e: guarda un evento manual
q: cierra la captura de forma normal
Ctrl + C: también corta y cierra

## app/main.py

# terminal 1
python app/main.py

# terminal 2
capture_opencv.py

# inmediato 
python app/main.py --once
python app/main.py --once --overwrite

# scripts/capture_opencv.py
``` bash
python scripts/capture_opencv.py \
  --device 0 \
  --width 1280 \
  --height 720 \
  --fps 30 \
  --out_dir data/captures/opencv \
  --events \
  --auto_events \
  --auto_method bg \
  --auto_use_window_capture \
  --auto_window_s 20 \
  --auto_interval_s 1.0 \
  --roi 180 120 950 500 \
  --every 0 \
  --present_frames 8 \
  --min_fg_ratio 0.03 \
  --min_contour_area 4000 \
  --cooldown_s 20
```

# scripts/capture_realsense.py
python scripts/capture_opencv.py \
  --device 0 \
  --width 1280 \
  --height 720 \
  --fps 30 \
  --out_dir data/captures/opencv \
  --events \
  --auto_events \
  --auto_method bg \
  --auto_use_window_capture \
  --auto_window_s 7 \
  --auto_interval_s 1.0 \
  --roi 180 120 950 500 \
  --every 0 \
  --present_frames 8 \
  --min_fg_ratio 0.03 \
  --min_contour_area 4000 \
  --cooldown_s 20

# utils/vision_picking.py
``` bash
# Total
python -m utils.vision_picking data/tests_picking/SCHNSCL01_Preparacion_Pickinglist_consolidada_Masivos_V5_2.png

# Sin guardar
python -m utils.vision_picking <ruta_imagen> --no_save_summary

# Cambiar carpeta
python -m utils.vision_picking <ruta_imagen> --summary_output_dir data/picking/summary_results
```

# intel realSense D455
sudo /opt/homebrew/bin/rs-enumerate-devices
sudo "$(pwd)/.venv/bin/python" scripts/test_realsense.py

# main.py
python -m app.main \
  --mode_app picking_match \
  --picking_image data/tests_picking/SCHNSCL01_Preparacion_Pickinglist_consolidada_Masivos_V5_2.png \
  --picking_excel data/tests_excel_picking/SCHNSCL01_Informe_de_Fill_Rate_OUTBOUND_v2.xlsx \                                                 
  --packstructure_excel data/tests_picking/PackStructure.xlsx

# Cierre (desde el orquestador app/main.py)

## etapa 0; captura
``` bash
scripts/capture_realsense.py \
  --width 1280 \
  --height 720 \
  --fps 30 \
  --out_dir data/captures/realsense \
  --events \
  --auto_events \
  --auto_method bg \
  --auto_use_window_capture \
  --auto_window_s 10 \
  --auto_interval_s 1.0 \
  --roi 140 220 1120 360 \
  --every 0 \
  --present_frames 8 \
  --min_fg_ratio 0.03 \
  --min_contour_area 4000 \
  --cooldown_s 20
```

## etapa 1; Primero generamos el consolidado:
``` bash
python -m app.main \
  --mode_app picking_match \
  --picking_image data/tests_picking/SCHNSCL01_Preparacion_Pickinglist_consolidada_Masivos_V5_2.png \
  --picking_excel data/tests_excel_picking/SCHNSCL01_Informe_de_Fill_Rate_OUTBOUND_v2.xlsx \
  --packstructure_excel data/tests_picking/PackStructure.xlsx
```

# Cierre (desde el orquestador app/main.py)
## etapa 1; Primero generamos el consolidado:
python -m app.main \
  --mode_app picking_match \
  --picking_image data/tests_picking/SCHNSCL01_Preparacion_Pickinglist_consolidada_Masivos_V5_2.png \
  --picking_excel data/tests_excel_picking/SCHNSCL01_Informe_de_Fill_Rate_OUTBOUND_v2.xlsx \
  --packstructure_excel data/tests_picking/PackStructure.xlsx

## etapa 2; ejecutamos el cierre con el readout_result.json
python -m app.main \
  --mode_app closure_match \
  --summary_json data/picking/summary_pickingVision_fillRate_packStructure/SCHNSCL01_Preparacion_Pickinglist_consolidada_Masivos_V5_2_summary_pickingVision_fillRate_packStructure.json \
  --readout_json data/captures/opencv/frames_xxx/events/event_xxx/readout_result.json

  ## etapa 3; final, cierre
  python -m app.main \
  --mode_app closure_match \
  --summary_json data/picking/summary_pickingVision_fillRate_packStructure/SCHNSCL01_Preparacion_Pickinglist_consolidada_Masivos_V5_2_summary_pickingVision_fillRate_packStructure.json \
  --detected_barcodes_json data/tests/manual_barcodes.json

  # scripts/capture_realsense_depth.py
