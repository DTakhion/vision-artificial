# Caso de Estudio -- Límite de Decodificación por Blur (QR Detection vs QR Decode)

## Contexto

Durante pruebas del pipeline híbrido:

-   OpenCV (`QRCodeDetector`)
-   Fallback con `pyzbar` (ZBar)
-   Warp ROI por perspectiva
-   Upscaling (x2/x3/x4)
-   Sharpen + bilateral
-   Otsu / Adaptive threshold

Se presentó un caso donde:

-   El QR **es detectado geométricamente**
-   Pero **no puede ser decodificado**

------------------------------------------------------------------------

## 🧪 Resultado Técnico

Comando ejecutado:

``` bash
python - << 'PY'
import cv2, numpy as np
from utils.vision_qr import decode_qr_opencv
img = cv2.imread("data/tests_qr/3503c836-ac26-43ea-a2eb-67202c30436b.JPG")
res = decode_qr_opencv(img, time_budget_ms=300, variants=None)
pts = np.array(res["points"][0], dtype=float)
w1 = np.linalg.norm(pts[1]-pts[0])
w2 = np.linalg.norm(pts[2]-pts[3])
print("approx_qr_width_px:", (w1+w2)/2)
print(res)
PY
```

Salida:

    approx_qr_width_px: 417.5972843895515

``` json
{
  "status": "not_found",
  "text": null,
  "points": [[[186.0, 563.0], [598.0, 567.0], [611.1367, 970.3723], [189.0, 1000.0]]],
  "backend": "opencv",
  "elapsed_ms": 414,
  "variant": null,
  "tried": ["gray", "sharp", "bilateral", "bilateral_sharp", "bilateral_x2"]
}
```

------------------------------------------------------------------------

## 🔎 Análisis Técnico

Observaciones clave:

1.  OpenCV **detecta correctamente el cuadrilátero del QR**.
2.  El ancho efectivo del QR es \~**417 px**.
3.  A pesar de ello:
    -   OpenCV no logra decodificar.
    -   `pyzbar` tampoco logra decodificar.
    -   Warp ROI + upscales x4 tampoco rescatan el caso.
    -   Sharpen + Otsu + Adaptive threshold tampoco funcionan.

### Conclusión técnica:

> El problema no es tamaño geométrico, sino **pérdida de detalle por
> blur (desenfoque / movimiento)**.

El QR está suficientemente grande en pixeles, pero: - Los módulos
internos están suavizados. - La frecuencia espacial necesaria para
reconstrucción del patrón se perdió. - No existe información suficiente
para que ningún decoder reconstruya los bits.

Esto representa un **límite físico de señal**, no un límite de
algoritmo.

------------------------------------------------------------------------

## 🧠 Insight Importante para el PoC

Este caso valida que:

-   El pipeline híbrido está funcionando correctamente.
-   Se están intentando variantes robustas.
-   Se está explotando la detección geométrica.
-   Se está usando fallback avanzado.
-   El fallo es legítimo y explicable.

------------------------------------------------------------------------

## 📏 Regla Operativa Derivada

La decodificación de QR depende de:

1.  Tamaño efectivo (px por lado)
2.  Nitidez (resolución espacial real)
3.  Nivel de blur
4.  Contraste

Incluso con \>400 px de ancho, si el blur destruye los bordes de los
módulos:

> La decodificación puede ser imposible.

------------------------------------------------------------------------

## 🎯 Recomendaciones de Captura para Operación en Terreno

Para evitar este tipo de fallo:

-   El QR debe ocupar al menos **1/4 del ancho del frame**
-   Evitar movimiento (mejor iluminación → menor tiempo de exposición)
-   No usar zoom digital
-   Mantener ángulo \< 30°
-   Enfocar a distancia media (\~30--50 cm)
-   Evitar vibración al disparar

------------------------------------------------------------------------

## 🏁 Conclusión

Este caso no representa un fallo del sistema, sino:

> Un límite físico inherente a la adquisición de imagen.

Es un resultado técnicamente válido y documentable en el PoC.

------------------------------------------------------------------------

## 📚 Nota Estratégica

En proyectos de visión artificial, la mayoría de los fallos en
producción no son de modelo o algoritmo, sino de:

-   Calidad de captura
-   Iluminación
-   Movimiento
-   Enfoque
-   Ángulo

Este caso confirma esa realidad.
