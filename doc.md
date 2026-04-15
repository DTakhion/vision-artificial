## paso 1; Ground Truth (fillRate + PackStructure)

# extracción + normalizaciónn + consolidación + automatización
``` bash
while true; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Ejecutando fetch + matching..."
  python3 -m utils.fillrate_gmail_fetch --out-json results/fillrate_fetch.json >> logs/fillrate_loop.log 2>&1
  python3 -m app.main \
    --mode_app picking_match \
    --picking_excel data/fillrate/latest/fillrate_latest.xlsx \
    --packstructure_excel data/tests_picking/PackStructure.xlsx \
    --picking_out data/picking/summary_fillRate_packStructure/fillrate_latest_summary_fillRate_packStructure.json >> logs/fillrate_loop.log 2>&1
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Esperando 16 minutos..."
  sleep 960
done
```

## paso 3; Cierre (app/main.py function clousure_match/clousure_iterative)
``` bash
python -m app.main \
  --mode_app closure_match \
  --readout_json results/capture_barcode_test_hybrid.json \
  --closure_output data/closure/prueba_closure.json
```

``` bash
python -m app.main \
  --mode_app closure_iterative \
  --summary_json data/picking/summary_fillRate_packStructure/fillrate_latest_summary_fillRate_packStructure.json \
  --readout_json results/capture_barcode_test_hybrid.json \
  --session_state_json data/closure/session_state_schnscl01.json \
  --closure_output data/closure/prueba_closure_iterative.json \
  --reset_session
```

``` bash
python -m app.main \
  --mode_app closure_iterative \
  --summary_json data/picking/summary_fillRate_packStructure/fillrate_latest_summary_fillRate_packStructure.json \
  --readout_json results/capture_barcode_test_hybrid.json \
  --session_state_json data/closure/session_state_schnscl01.json \
  --closure_output data/closure/prueba_closure_iterative.json \
```
# paso 2, para PoC

``` bash
python scripts/capture_opencv.py \     
  --device 0 \
  --width 1920 \
  --height 1080 \
  --fps 30 \                                           
  --out_dir data/captures/opencv \
  --events \                                           
  --auto_events \
  --auto_method bg \
  --auto_use_window_capture \
  --auto_window_s 7 \
  --auto_interval_s 1.0 \
  --roi 80 80 1700 900 \
  --every 0 \
  --present_frames 8 \
  --min_fg_ratio 0.03 \
  --min_contour_area 7000 \
  --cooldown_s 10
```

# Paso 2, 4k, para PoC

``` bash
python scripts/capture_opencv.py \
  --device 0 \
  --width 3840 \
  --height 2160 \
  --fps 30 \
  --out_dir data/captures/opencv \
  --events \
  --auto_events \
  --auto_method bg \
  --auto_use_window_capture \
  --auto_window_s 7 \
  --auto_interval_s 1.0 \
  --roi 160 160 3400 1800 \
  --every 0 \
  --present_frames 8 \
  --min_fg_ratio 0.03 \
  --min_contour_area 7000 \
  --cooldown_s 10
```

# paso 2, para MvP

``` bash
python scripts/capture_opencv.py \
  --device 0 \
  --width 1920 \
  --height 1080 \
  --fps 30 \
  --out_dir data/captures/opencv \
  --events \
  --roi 80 80 1700 900 \
  --every 0
```

# otras herramientas; 

## utils/vision_readout_hybrid.py con filtro de formatos, 

``` bash
python -m utils.vision_readout_hybrid \
  data/tests_picking/capture_barcode_test.png \
  --save-vis \
  --vis-out results/capture_barcode_test_hybrid.png \
  --save-json \
  --json-out results/capture_barcode_test_hybrid.json
```

## utils/vision_readout_hybrid.py con formatos adicionales, 
``` bash
python -m utils.vision_readout_hybrid \
  data/tests_picking/capture_barcode_test.png \
  --allowed-formats EAN_13 CODE_128 ITF CODE_39 \
  --save-json
```


# utils/vision_barcode_yolo.py con ajuste de umbrales

``` bash
python -m utils.vision_barcode_yolo data/captures/opencv/frames_20260407_150039/events/event_000001/frames/frame_01.jpg \
  --conf 0.30 \
  --pad_ratio 0.30 \
  --decoder_mode collect_plus \
  --decoder_budget 2000 \
  --save-vis \
  --save-json \
  --save-crops \
  --debug
```

## utils/vision_barcode_dynamsoft.py

``` bash
python -m utils.vision_barcode_dynamsoft \
  data/captures/opencv/frames_20260402_105555/events/event_000003/frames/frame_03.jpg \
  --json \
  --save-vis \
  --out results/dynamsoft/test_vis.png \
  --save-json \
  --json-out results/dynamsoft/test.json
```

# Levantar Backend; 
## uvicorn backend.main:app --reload

# utils/vision_picking.py
``` bash
python -m utils.vision_picking data/tests_picking/tu_imagen.jpg --json --save-debug
```

# app/main.py -> picking_shipping
``` bash
 python -m app.main \
  --mode_app picking_shipping \
  --picking_image data/captures/opencv/frames_20260402_105555/events/event_000004/frames/frame_01.jpg \                                 
  --session_state_json data/closure/session_state.json \      
  --closure_output data/picking/frame_03_picking_shipping.json
```

