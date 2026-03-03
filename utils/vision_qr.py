# # utils/vision_qr.py
# from __future__ import annotations

# from typing import Any, Dict, Optional, List, Union
# import time

# import cv2
# import numpy as np

# from utils.vision_preprocess import preprocess_variants


# # Variantes recomendadas (ordenadas por probabilidad de éxito + costo)
# DEFAULT_VARIANTS_QR: List[str] = [
#     "gray",
#     "sharp",
#     "bilateral",
#     "bilateral_sharp",
#     "bilateral_x2",
#     "sharp_x2",
#     "morph_close",
#     "morph_close_x2",
#     "bw",       # QR a veces sufre con binarizado, por eso va al final
#     "bw_x2",
# ]


# def _resolve_variants(variants: Optional[Union[List[str], str]], available: List[str]) -> List[str]:
#     """
#     variants:
#       - None -> DEFAULT_VARIANTS_QR filtrado por disponibles
#       - "all" -> todos los disponibles
#       - list[str] -> usa esos nombres (en ese orden), filtrando por disponibles
#     """
#     if variants is None:
#         return [v for v in DEFAULT_VARIANTS_QR if v in available]
#     if isinstance(variants, str):
#         if variants.lower() == "all":
#             return available
#         # permitir "gray,sharp,bilateral"
#         if "," in variants:
#             req = [x.strip() for x in variants.split(",") if x.strip()]
#             return [v for v in req if v in available]
#         # single name
#         return [variants] if variants in available else []
#     # list
#     return [v for v in variants if v in available]


# # -------------------------
# # Warp helpers (ROI by points)
# # -------------------------
# def _order_points(pts: np.ndarray) -> np.ndarray:
#     # pts: (4,2)
#     rect = np.zeros((4, 2), dtype=np.float32)
#     s = pts.sum(axis=1)
#     rect[0] = pts[np.argmin(s)]  # TL
#     rect[2] = pts[np.argmax(s)]  # BR
#     diff = np.diff(pts, axis=1).reshape(-1)
#     rect[1] = pts[np.argmin(diff)]  # TR
#     rect[3] = pts[np.argmax(diff)]  # BL
#     return rect


# def _warp_from_points(img_bgr: np.ndarray, pts: np.ndarray, out_size: int = 950) -> np.ndarray:
#     # pts can be (1,4,2) or (4,2) or list-like
#     pts = np.asarray(pts, dtype=np.float32)
#     if pts.ndim == 3:
#         pts = pts.reshape(-1, 2)

#     rect = _order_points(pts)
#     dst = np.array(
#         [[0, 0], [out_size - 1, 0], [out_size - 1, out_size - 1], [0, out_size - 1]],
#         dtype=np.float32,
#     )
#     M = cv2.getPerspectiveTransform(rect, dst)
#     return cv2.warpPerspective(img_bgr, M, (out_size, out_size), flags=cv2.INTER_CUBIC)


# def decode_qr_opencv(
#     img_bgr: np.ndarray,
#     *,
#     time_budget_ms: int = 160,
#     variants: Optional[Union[List[str], str]] = None,
# ) -> Dict[str, Any]:
#     """
#     Returns:
#       {
#         "status": "success" | "not_found" | "error",
#         "text": str | None,
#         "points": list | None,
#         "backend": "opencv",
#         "elapsed_ms": int,
#         "variant": str | None,
#         "tried": list[str] (only on not_found/error)
#       }

#     Improvement:
#       - Keep best detected points even when decode fails (helps ROI/warp fallback).
#     """
#     t0 = time.perf_counter()
#     detector = cv2.QRCodeDetector()

#     tried: List[str] = []
#     best_pts = None

#     try:
#         ims = preprocess_variants(img_bgr)
#         available = list(ims.keys())
#         to_try = _resolve_variants(variants, available)

#         # fallback defensivo si alguien pide variantes que no existen
#         if not to_try:
#             to_try = [v for v in DEFAULT_VARIANTS_QR if v in available] or available

#         for name in to_try:
#             if (time.perf_counter() - t0) * 1000 > time_budget_ms:
#                 break
#             gray = ims.get(name)
#             if gray is None:
#                 continue

#             tried.append(name)

#             # OpenCV QRCodeDetector expects 8-bit 1-channel or BGR; we use gray
#             txt, pts, _ = detector.detectAndDecode(gray)

#             # ✅ store detection geometry even if txt is empty
#             if pts is not None and best_pts is None:
#                 best_pts = pts

#             if txt:
#                 elapsed = int((time.perf_counter() - t0) * 1000)
#                 return {
#                     "status": "success",
#                     "text": txt,
#                     "points": (pts.tolist() if pts is not None else None),
#                     "backend": "opencv",
#                     "elapsed_ms": elapsed,
#                     "variant": name,
#                 }

#         elapsed = int((time.perf_counter() - t0) * 1000)
#         return {
#             "status": "not_found",
#             "text": None,
#             "points": (best_pts.tolist() if best_pts is not None else None),
#             "backend": "opencv",
#             "elapsed_ms": elapsed,
#             "variant": None,
#             "tried": tried,
#         }
#     except Exception as e:
#         elapsed = int((time.perf_counter() - t0) * 1000)
#         return {
#             "status": "error",
#             "text": None,
#             "points": (best_pts.tolist() if best_pts is not None else None),
#             "backend": "opencv",
#             "elapsed_ms": elapsed,
#             "error": repr(e),
#             "variant": None,
#             "tried": tried,
#         }


# def _resize_gray(gray: np.ndarray, factor: float) -> np.ndarray:
#     if factor <= 1.0:
#         return gray
#     return cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)

# def decode_qr_pyzbar(img_bgr: np.ndarray) -> Dict[str, Any]:
#     """
#     Fallback decoder using pyzbar (ZBar).
#     Stronger version:
#       - More candidate variants for blur: x4, unsharp, Otsu/adaptive binarization.
#     """
#     t0 = time.perf_counter()
#     tried: List[str] = []

#     try:
#         from pyzbar.pyzbar import decode as zbar_decode, ZBarSymbol

#         gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

#         def unsharp(g: np.ndarray, amount: float = 1.0) -> np.ndarray:
#             blur = cv2.GaussianBlur(g, (0, 0), sigmaX=1.2)
#             return cv2.addWeighted(g, 1.0 + amount, blur, -amount, 0)

#         # Base + upscales (x2/x3/x4)
#         gray_x2 = _resize_gray(gray, 2.0)
#         gray_x3 = _resize_gray(gray, 3.0)
#         gray_x4 = _resize_gray(gray, 4.0)

#         # Edge-preserving denoise
#         bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
#         bilateral_x2 = _resize_gray(bilateral, 2.0)
#         bilateral_x4 = _resize_gray(bilateral, 4.0)

#         # Sharpened (helps blur)
#         sharp = unsharp(gray, 1.2)
#         sharp_x2 = _resize_gray(sharp, 2.0)
#         sharp_x4 = _resize_gray(sharp, 4.0)

#         # Binarizations (sometimes ZBar prefers high contrast)
#         def otsu(g: np.ndarray) -> np.ndarray:
#             _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#             return bw

#         def adapt(g: np.ndarray) -> np.ndarray:
#             return cv2.adaptiveThreshold(
#                 g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
#             )

#         candidates: List[tuple[str, np.ndarray]] = [
#             # Order: start with “most likely to rescue blur”
#             ("sharp_x4", sharp_x4),
#             ("bilateral_x4", bilateral_x4),
#             ("gray_x4", gray_x4),
#             ("sharp_x2", sharp_x2),
#             ("bilateral_x2", bilateral_x2),
#             ("gray_x2", gray_x2),

#             # Then simpler
#             ("sharp", sharp),
#             ("bilateral", bilateral),
#             ("gray", gray),

#             # Binarized options (after)
#             ("otsu_gray_x4", otsu(gray_x4)),
#             ("adapt_gray_x4", adapt(gray_x4)),
#             ("otsu_gray", otsu(gray)),
#             ("adapt_gray", adapt(gray)),
#         ]

#         for name, im in candidates:
#             tried.append(name)
#             decoded = zbar_decode(im, symbols=[ZBarSymbol.QRCODE])
#             if decoded:
#                 data = decoded[0].data.decode("utf-8", errors="replace")
#                 elapsed = int((time.perf_counter() - t0) * 1000)
#                 return {
#                     "status": "success",
#                     "text": data,
#                     "points": None,
#                     "backend": "pyzbar",
#                     "elapsed_ms": elapsed,
#                     "variant": name,
#                 }

#         elapsed = int((time.perf_counter() - t0) * 1000)
#         return {
#             "status": "not_found",
#             "text": None,
#             "points": None,
#             "backend": "pyzbar",
#             "elapsed_ms": elapsed,
#             "variant": None,
#             "tried": tried,
#         }
#     except Exception as e:
#         elapsed = int((time.perf_counter() - t0) * 1000)
#         return {
#             "status": "error",
#             "text": None,
#             "points": None,
#             "backend": "pyzbar",
#             "elapsed_ms": elapsed,
#             "error": repr(e),
#             "variant": None,
#             "tried": tried,
#         }


# def decode_qr(
#     img_bgr: np.ndarray,
#     *,
#     time_budget_ms: int = 160,
#     variants: Optional[Union[List[str], str]] = None,
#     enable_fallback: bool = True,
# ) -> Dict[str, Any]:
#     """
#     Wrapper: tries OpenCV first; if not success and enable_fallback=True, tries pyzbar.

#     Improvements:
#       - If OpenCV detected points, warp ROI (try 2 sizes) and try pyzbar on ROI first.
#       - If pyzbar fails, attach pyzbar debug info into the returned dict.
#     """
#     res = decode_qr_opencv(img_bgr, time_budget_ms=time_budget_ms, variants=variants)
#     if res.get("status") == "success" or not enable_fallback:
#         return res

#     pts = res.get("points")

#     # 1) ROI warp attempts (two sizes)
#     if pts is not None:
#         for size in (950, 1300):
#             try:
#                 roi = _warp_from_points(img_bgr, pts, out_size=size)
#                 fb_roi = decode_qr_pyzbar(roi)
#                 if fb_roi.get("status") == "success":
#                     fb_roi["opencv_elapsed_ms"] = res.get("elapsed_ms")
#                     fb_roi["opencv_tried"] = res.get("tried")
#                     fb_roi["opencv_variant"] = res.get("variant")
#                     fb_roi["note"] = f"pyzbar_on_warp_roi_{size}"
#                     return fb_roi

#                 # attach debug from ROI attempt
#                 res["pyzbar_roi_status"] = fb_roi.get("status")
#                 res["pyzbar_roi_elapsed_ms"] = fb_roi.get("elapsed_ms")
#                 res["pyzbar_roi_tried"] = fb_roi.get("tried")
#                 res["pyzbar_roi_variant"] = fb_roi.get("variant")
#                 if fb_roi.get("status") == "error":
#                     res["pyzbar_roi_error"] = fb_roi.get("error")
#             except Exception as e:
#                 res["pyzbar_roi_status"] = "error"
#                 res["pyzbar_roi_error"] = repr(e)

#     # 2) Full-image fallback
#     fb = decode_qr_pyzbar(img_bgr)
#     if fb.get("status") == "success":
#         fb["opencv_elapsed_ms"] = res.get("elapsed_ms")
#         fb["opencv_tried"] = res.get("tried")
#         fb["opencv_variant"] = res.get("variant")
#         return fb

#     # attach debug from full-image pyzbar attempt too
#     res["pyzbar_status"] = fb.get("status")
#     res["pyzbar_elapsed_ms"] = fb.get("elapsed_ms")
#     res["pyzbar_tried"] = fb.get("tried")
#     res["pyzbar_variant"] = fb.get("variant")
#     if fb.get("status") == "error":
#         res["pyzbar_error"] = fb.get("error")

#     return res


# def _cli() -> int:
#     import argparse

#     p = argparse.ArgumentParser(
#         description="Decode QR from an image with OpenCV + preprocess variants (+ optional pyzbar fallback)."
#     )
#     p.add_argument("image_path", type=str, help="Path to image file")
#     p.add_argument("--budget", type=int, default=160, help="Time budget in ms for OpenCV (default: 160)")
#     p.add_argument(
#         "--variants",
#         type=str,
#         default="default",
#         help='Variants to try: "default" | "all" | comma-list e.g. "gray,sharp,bilateral_x2"',
#     )
#     p.add_argument(
#         "--no_fallback",
#         action="store_true",
#         help="Disable pyzbar fallback (OpenCV only).",
#     )
#     args = p.parse_args()

#     img = cv2.imread(args.image_path)
#     if img is None:
#         print({"status": "error", "error": "Could not load image", "path": args.image_path})
#         return 2

#     if args.variants == "default":
#         v: Optional[Union[List[str], str]] = None
#     else:
#         v = args.variants

#     result = decode_qr(
#         img,
#         time_budget_ms=args.budget,
#         variants=v,
#         enable_fallback=(not args.no_fallback),
#     )
#     print(result)
#     return 0


# if __name__ == "__main__":
#     raise SystemExit(_cli())

# utils/vision_qr.py
from __future__ import annotations

from typing import Any, Dict, Optional, List, Union, Tuple
import time

import cv2
import numpy as np

from utils.vision_preprocess import preprocess_variants


# Variantes recomendadas (ordenadas por probabilidad de éxito + costo)
DEFAULT_VARIANTS_QR: List[str] = [
    "gray",
    "sharp",
    "bilateral",
    "bilateral_sharp",
    "bilateral_x2",
    "sharp_x2",
    "morph_close",
    "morph_close_x2",
    "bw",       # QR a veces sufre con binarizado, por eso va al final
    "bw_x2",
]


def _resolve_variants(variants: Optional[Union[List[str], str]], available: List[str]) -> List[str]:
    """
    variants:
      - None -> DEFAULT_VARIANTS_QR filtrado por disponibles
      - "all" -> todos los disponibles
      - list[str] -> usa esos nombres (en ese orden), filtrando por disponibles
    """
    if variants is None:
        return [v for v in DEFAULT_VARIANTS_QR if v in available]
    if isinstance(variants, str):
        if variants.lower() == "all":
            return available
        # permitir "gray,sharp,bilateral"
        if "," in variants:
            req = [x.strip() for x in variants.split(",") if x.strip()]
            return [v for v in req if v in available]
        # single name
        return [variants] if variants in available else []
    # list
    return [v for v in variants if v in available]


# -------------------------
# Warp helpers (ROI by points)
# -------------------------
def _order_points(pts: np.ndarray) -> np.ndarray:
    # pts: (4,2)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # TL
    rect[2] = pts[np.argmax(s)]  # BR
    diff = np.diff(pts, axis=1).reshape(-1)
    rect[1] = pts[np.argmin(diff)]  # TR
    rect[3] = pts[np.argmax(diff)]  # BL
    return rect


def _warp_from_points(img_bgr: np.ndarray, pts: Any, out_size: int = 950) -> np.ndarray:
    """
    pts can be:
      - (1,4,2) from OpenCV
      - (4,2)
      - list-like [[[x,y],...]] or [[x,y],...]
    """
    pts = np.asarray(pts, dtype=np.float32)
    if pts.ndim == 3:
        pts = pts.reshape(-1, 2)
    if pts.shape[0] != 4:
        raise ValueError(f"Expected 4 points, got shape {pts.shape}")

    rect = _order_points(pts)
    dst = np.array(
        [[0, 0], [out_size - 1, 0], [out_size - 1, out_size - 1], [0, out_size - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img_bgr, M, (out_size, out_size), flags=cv2.INTER_CUBIC)


def _roi_from_points_bbox(img_bgr: np.ndarray, pts: Any, pad: int = 20) -> Optional[np.ndarray]:
    """
    Fast ROI crop by bbox (no warp). Useful when warp fails or points are noisy.
    """
    try:
        pts = np.asarray(pts, dtype=np.float32)
        if pts.ndim == 3:
            pts = pts.reshape(-1, 2)
        x0 = int(np.floor(np.min(pts[:, 0]))) - pad
        y0 = int(np.floor(np.min(pts[:, 1]))) - pad
        x1 = int(np.ceil(np.max(pts[:, 0]))) + pad
        y1 = int(np.ceil(np.max(pts[:, 1]))) + pad
        h, w = img_bgr.shape[:2]
        x0 = max(0, x0); y0 = max(0, y0)
        x1 = min(w, x1); y1 = min(h, y1)
        if x1 <= x0 or y1 <= y0:
            return None
        return img_bgr[y0:y1, x0:x1].copy()
    except Exception:
        return None


def decode_qr_opencv(
    img_bgr: np.ndarray,
    *,
    time_budget_ms: int = 160,
    variants: Optional[Union[List[str], str]] = None,
) -> Dict[str, Any]:
    """
    Returns:
      {
        "status": "success" | "not_found" | "error",
        "text": str | None,
        "points": list | None,
        "backend": "opencv",
        "elapsed_ms": int,
        "variant": str | None,
        "tried": list[str] (only on not_found/error)
      }

    Improvement:
      - Keep best detected points even when decode fails (helps ROI/warp fallback).
      - Keep the variant that produced the points (detect variant).
    """
    t0 = time.perf_counter()
    detector = cv2.QRCodeDetector()

    tried: List[str] = []
    best_pts = None
    best_pts_variant = None

    try:
        ims = preprocess_variants(img_bgr)
        available = list(ims.keys())
        to_try = _resolve_variants(variants, available)

        # fallback defensivo si alguien pide variantes que no existen
        if not to_try:
            to_try = [v for v in DEFAULT_VARIANTS_QR if v in available] or available

        for name in to_try:
            if (time.perf_counter() - t0) * 1000 > time_budget_ms:
                break
            gray = ims.get(name)
            if gray is None:
                continue

            tried.append(name)

            txt, pts, _ = detector.detectAndDecode(gray)

            # ✅ store detection geometry even if txt is empty
            if pts is not None and best_pts is None:
                best_pts = pts
                best_pts_variant = name

            if txt:
                elapsed = int((time.perf_counter() - t0) * 1000)
                return {
                    "status": "success",
                    "text": txt,
                    "points": (pts.tolist() if pts is not None else None),
                    "backend": "opencv",
                    "elapsed_ms": elapsed,
                    "variant": name,
                }

        elapsed = int((time.perf_counter() - t0) * 1000)
        return {
            "status": "not_found",
            "text": None,
            "points": (best_pts.tolist() if best_pts is not None else None),
            "backend": "opencv",
            "elapsed_ms": elapsed,
            "variant": None,
            "tried": tried,
            "detect_variant": best_pts_variant,  # 👈 new (useful debug)
        }
    except Exception as e:
        elapsed = int((time.perf_counter() - t0) * 1000)
        return {
            "status": "error",
            "text": None,
            "points": (best_pts.tolist() if best_pts is not None else None),
            "backend": "opencv",
            "elapsed_ms": elapsed,
            "error": repr(e),
            "variant": None,
            "tried": tried,
            "detect_variant": best_pts_variant,
        }


def _resize_gray(gray: np.ndarray, factor: float) -> np.ndarray:
    if factor <= 1.0:
        return gray
    return cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)


def decode_qr_pyzbar(img_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Fallback decoder using pyzbar (ZBar).
    Stronger version:
      - More candidate variants for blur: x4, unsharp, Otsu/adaptive binarization.
    """
    t0 = time.perf_counter()
    tried: List[str] = []

    try:
        from pyzbar.pyzbar import decode as zbar_decode, ZBarSymbol

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        def unsharp(g: np.ndarray, amount: float = 1.0) -> np.ndarray:
            blur = cv2.GaussianBlur(g, (0, 0), sigmaX=1.2)
            return cv2.addWeighted(g, 1.0 + amount, blur, -amount, 0)

        def otsu(g: np.ndarray) -> np.ndarray:
            _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return bw

        def adapt(g: np.ndarray) -> np.ndarray:
            return cv2.adaptiveThreshold(
                g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
            )

        # Base + upscales (x2/x3/x4)
        gray_x2 = _resize_gray(gray, 2.0)
        gray_x3 = _resize_gray(gray, 3.0)
        gray_x4 = _resize_gray(gray, 4.0)

        # Edge-preserving denoise
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        bilateral_x2 = _resize_gray(bilateral, 2.0)
        bilateral_x4 = _resize_gray(bilateral, 4.0)

        # Sharpened (helps blur)
        sharp = unsharp(gray, 1.2)
        sharp_x2 = _resize_gray(sharp, 2.0)
        sharp_x4 = _resize_gray(sharp, 4.0)

        candidates: List[Tuple[str, np.ndarray]] = [
            # Order: start with “most likely to rescue blur”
            ("sharp_x4", sharp_x4),
            ("bilateral_x4", bilateral_x4),
            ("gray_x4", gray_x4),
            ("sharp_x2", sharp_x2),
            ("bilateral_x2", bilateral_x2),
            ("gray_x2", gray_x2),
            ("gray_x3", gray_x3),

            # Then simpler
            ("sharp", sharp),
            ("bilateral", bilateral),
            ("gray", gray),

            # Binarized options (after)
            ("otsu_gray_x4", otsu(gray_x4)),
            ("adapt_gray_x4", adapt(gray_x4)),
            ("otsu_gray", otsu(gray)),
            ("adapt_gray", adapt(gray)),
        ]

        for name, im in candidates:
            tried.append(name)
            decoded = zbar_decode(im, symbols=[ZBarSymbol.QRCODE])
            if decoded:
                data = decoded[0].data.decode("utf-8", errors="replace")
                elapsed = int((time.perf_counter() - t0) * 1000)
                return {
                    "status": "success",
                    "text": data,
                    "points": None,
                    "backend": "pyzbar",
                    "elapsed_ms": elapsed,
                    "variant": name,
                }

        elapsed = int((time.perf_counter() - t0) * 1000)
        return {
            "status": "not_found",
            "text": None,
            "points": None,
            "backend": "pyzbar",
            "elapsed_ms": elapsed,
            "variant": None,
            "tried": tried,
        }
    except Exception as e:
        elapsed = int((time.perf_counter() - t0) * 1000)
        return {
            "status": "error",
            "text": None,
            "points": None,
            "backend": "pyzbar",
            "elapsed_ms": elapsed,
            "error": repr(e),
            "variant": None,
            "tried": tried,
        }


def decode_qr(
    img_bgr: np.ndarray,
    *,
    time_budget_ms: int = 160,
    variants: Optional[Union[List[str], str]] = None,
    enable_fallback: bool = True,
) -> Dict[str, Any]:
    """
    Wrapper: tries OpenCV first; if not success and enable_fallback=True, tries pyzbar.

    Improvements:
      - If OpenCV detected points, warp ROI (try 2 sizes) and try pyzbar on ROI first.
      - Extra: also try a bbox-crop ROI (no warp) before full-image fallback.
      - If pyzbar fails, attach pyzbar debug info into the returned dict.
    """
    res = decode_qr_opencv(img_bgr, time_budget_ms=time_budget_ms, variants=variants)
    if res.get("status") == "success" or not enable_fallback:
        return res

    pts = res.get("points")

    # 1) ROI warp attempts (two sizes)
    if pts is not None:
        for size in (950, 1300):
            try:
                roi = _warp_from_points(img_bgr, pts, out_size=size)
                fb_roi = decode_qr_pyzbar(roi)
                if fb_roi.get("status") == "success":
                    fb_roi["opencv_elapsed_ms"] = res.get("elapsed_ms")
                    fb_roi["opencv_tried"] = res.get("tried")
                    fb_roi["opencv_variant"] = res.get("variant")
                    fb_roi["opencv_detect_variant"] = res.get("detect_variant")
                    fb_roi["note"] = f"pyzbar_on_warp_roi_{size}"
                    return fb_roi

                # attach debug from ROI attempt (keep last attempt)
                res["pyzbar_roi_status"] = fb_roi.get("status")
                res["pyzbar_roi_elapsed_ms"] = fb_roi.get("elapsed_ms")
                res["pyzbar_roi_tried"] = fb_roi.get("tried")
                res["pyzbar_roi_variant"] = fb_roi.get("variant")
                if fb_roi.get("status") == "error":
                    res["pyzbar_roi_error"] = fb_roi.get("error")
            except Exception as e:
                res["pyzbar_roi_status"] = "error"
                res["pyzbar_roi_error"] = repr(e)

        # 1b) bbox ROI crop (no warp) — sometimes points are good enough for crop but warp hurts
        try:
            roi2 = _roi_from_points_bbox(img_bgr, pts, pad=24)
            if roi2 is not None:
                fb_roi2 = decode_qr_pyzbar(roi2)
                if fb_roi2.get("status") == "success":
                    fb_roi2["opencv_elapsed_ms"] = res.get("elapsed_ms")
                    fb_roi2["opencv_tried"] = res.get("tried")
                    fb_roi2["opencv_variant"] = res.get("variant")
                    fb_roi2["opencv_detect_variant"] = res.get("detect_variant")
                    fb_roi2["note"] = "pyzbar_on_bbox_roi"
                    return fb_roi2

                res["pyzbar_bbox_roi_status"] = fb_roi2.get("status")
                res["pyzbar_bbox_roi_elapsed_ms"] = fb_roi2.get("elapsed_ms")
                res["pyzbar_bbox_roi_tried"] = fb_roi2.get("tried")
                res["pyzbar_bbox_roi_variant"] = fb_roi2.get("variant")
                if fb_roi2.get("status") == "error":
                    res["pyzbar_bbox_roi_error"] = fb_roi2.get("error")
        except Exception as e:
            res["pyzbar_bbox_roi_status"] = "error"
            res["pyzbar_bbox_roi_error"] = repr(e)

    # 2) Full-image fallback
    fb = decode_qr_pyzbar(img_bgr)
    if fb.get("status") == "success":
        fb["opencv_elapsed_ms"] = res.get("elapsed_ms")
        fb["opencv_tried"] = res.get("tried")
        fb["opencv_variant"] = res.get("variant")
        fb["opencv_detect_variant"] = res.get("detect_variant")
        return fb

    # attach debug from full-image pyzbar attempt too
    res["pyzbar_status"] = fb.get("status")
    res["pyzbar_elapsed_ms"] = fb.get("elapsed_ms")
    res["pyzbar_tried"] = fb.get("tried")
    res["pyzbar_variant"] = fb.get("variant")
    if fb.get("status") == "error":
        res["pyzbar_error"] = fb.get("error")

    return res


# -------------------------
# Multi-QR decode (list of results)
# -------------------------
def decode_qr_multi(
    img_bgr: np.ndarray,
    *,
    time_budget_ms: int = 650,
    variants: Optional[Union[List[str], str]] = None,
    enable_fallback: bool = True,
    roi_warp_size: int = 950,
) -> Dict[str, Any]:
    """
    Multi-QR decoder.
    Uses OpenCV detectAndDecodeMulti across preprocess variants (gives points for each QR),
    then optionally tries pyzbar on warped ROIs for those that were detected but not decoded.

    Returns:
      {
        "status": "success" | "not_found" | "error",
        "backend": "opencv_multi" | "mixed" | "pyzbar_multi",
        "elapsed_ms": int,
        "items": [
          {"index": int, "status": "...", "text": str|None, "points": list|None, "backend": "...", "variant": str|None}
        ],
        "tried": list[str]
      }
    """
    t0 = time.perf_counter()
    detector = cv2.QRCodeDetector()

    tried: List[str] = []
    items: List[Dict[str, Any]] = []

    try:
        ims = preprocess_variants(img_bgr)
        available = list(ims.keys())
        to_try = _resolve_variants(variants, available)
        if not to_try:
            to_try = [v for v in DEFAULT_VARIANTS_QR if v in available] or available

        found_any = False

        for vname in to_try:
            if (time.perf_counter() - t0) * 1000 > time_budget_ms:
                break

            gray = ims.get(vname)
            if gray is None:
                continue

            tried.append(vname)

            # (ok, decoded_info, points, straight)
            try:
                ok, decoded_info, points, _ = detector.detectAndDecodeMulti(gray)
            except Exception:
                ok, decoded_info, points = False, [], None

            if not ok or points is None or len(points) == 0:
                continue

            found_any = True

            # points: (N,4,2)
            for i in range(len(points)):
                txt = decoded_info[i] if i < len(decoded_info) else ""
                pts = points[i].tolist()

                if txt:
                    items.append(
                        {
                            "index": i,
                            "status": "success",
                            "text": txt,
                            "points": pts,
                            "backend": "opencv",
                            "variant": vname,
                        }
                    )
                else:
                    items.append(
                        {
                            "index": i,
                            "status": "not_found",
                            "text": None,
                            "points": pts,
                            "backend": "opencv",
                            "variant": vname,
                        }
                    )

            # If we got at least one success, stop early (you can remove this if you prefer full sweep)
            if any(it["status"] == "success" for it in items):
                break

        # Fallback on ROIs for detected-but-not-decoded
        if enable_fallback and any(it["backend"] == "opencv" and it["status"] != "success" and it.get("points") for it in items):
            for it in items:
                if (time.perf_counter() - t0) * 1000 > time_budget_ms:
                    break
                if it["backend"] != "opencv" or it["status"] == "success" or not it.get("points"):
                    continue

                try:
                    roi = _warp_from_points(img_bgr, it["points"], out_size=roi_warp_size)
                    fb = decode_qr_pyzbar(roi)
                    if fb.get("status") == "success":
                        it.update(
                            {
                                "status": "success",
                                "text": fb.get("text"),
                                "backend": "pyzbar",
                                "variant": f"roi_from_{it.get('variant')}",
                            }
                        )
                except Exception:
                    pass

        # If OpenCV found nothing at all, try pyzbar multi on full image (rarely helps, but cheap)
        if enable_fallback and (not found_any) and len(items) == 0:
            try:
                from pyzbar.pyzbar import decode as zbar_decode, ZBarSymbol

                gray_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                decoded = zbar_decode(gray_full, symbols=[ZBarSymbol.QRCODE])
                for i, d in enumerate(decoded):
                    txt = d.data.decode("utf-8", errors="replace")
                    items.append(
                        {
                            "index": i,
                            "status": "success",
                            "text": txt,
                            "points": None,
                            "backend": "pyzbar",
                            "variant": "full_gray",
                        }
                    )
            except Exception:
                pass

        # Deduplicate by text (common in multi passes)
        seen_text = set()
        deduped: List[Dict[str, Any]] = []
        for it in items:
            txt = it.get("text")
            key = txt if txt else (tuple(np.array(it.get("points") or []).ravel().tolist()) if it.get("points") else None)
            if key in seen_text:
                continue
            seen_text.add(key)
            deduped.append(it)
        items = deduped

        elapsed = int((time.perf_counter() - t0) * 1000)

        if any(it["status"] == "success" for it in items):
            mixed = any(it["backend"] == "pyzbar" for it in items) and any(it["backend"] == "opencv" for it in items)
            backend = "mixed" if mixed else ("pyzbar_multi" if all(it["backend"] == "pyzbar" for it in items) else "opencv_multi")
            return {
                "status": "success",
                "backend": backend,
                "elapsed_ms": elapsed,
                "items": items,
                "tried": tried,
                "success_count": sum(1 for it in items if it["status"] == "success"),
                "total": len(items),
            }

        return {
            "status": "not_found",
            "backend": "opencv_multi",
            "elapsed_ms": elapsed,
            "items": items,
            "tried": tried,
            "success_count": 0,
            "total": len(items),
        }

    except Exception as e:
        elapsed = int((time.perf_counter() - t0) * 1000)
        return {
            "status": "error",
            "backend": "opencv_multi",
            "elapsed_ms": elapsed,
            "error": repr(e),
            "items": items,
            "tried": tried,
        }


def _cli() -> int:
    import argparse

    p = argparse.ArgumentParser(
        description="Decode QR from an image with OpenCV + preprocess variants (+ optional pyzbar fallback)."
    )
    p.add_argument("image_path", type=str, help="Path to image file")
    p.add_argument("--budget", type=int, default=160, help="Time budget in ms for OpenCV (default: 160)")
    p.add_argument(
        "--variants",
        type=str,
        default="default",
        help='Variants to try: "default" | "all" | comma-list e.g. "gray,sharp,bilateral_x2"',
    )
    p.add_argument(
        "--no_fallback",
        action="store_true",
        help="Disable pyzbar fallback (OpenCV only).",
    )
    p.add_argument(
        "--multi",
        action="store_true",
        help="Decode multiple QRs and return a list of results.",
    )
    args = p.parse_args()

    img = cv2.imread(args.image_path)
    if img is None:
        print({"status": "error", "error": "Could not load image", "path": args.image_path})
        return 2

    if args.variants == "default":
        v: Optional[Union[List[str], str]] = None
    else:
        v = args.variants

    if args.multi:
        result = decode_qr_multi(
            img,
            time_budget_ms=args.budget,
            variants=v,
            enable_fallback=(not args.no_fallback),
        )
    else:
        result = decode_qr(
            img,
            time_budget_ms=args.budget,
            variants=v,
            enable_fallback=(not args.no_fallback),
        )

    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())