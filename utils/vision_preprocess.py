# utils/vision_preprocess.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import cv2
import numpy as np


@dataclass
class PreprocessConfig:
    resize_max_side: int = 1280  # 0 = disable

    # Contrast
    clahe: bool = True
    clahe_clip: float = 2.0
    clahe_grid: Tuple[int, int] = (8, 8)

    # Denoise
    denoise: bool = False
    denoise_h: int = 10

    # ✅ Edge-preserving denoise (great for "dotty" / noisy QR)
    bilateral: bool = True
    bilateral_d: int = 9
    bilateral_sigma_color: int = 75
    bilateral_sigma_space: int = 75

    # Sharpen
    sharpen: bool = True
    sharpen_amount: float = 0.6  # 0..1

    # Upscale variants (helps when QR is small / far)
    upscale: bool = True
    upscale_factors: Tuple[float, ...] = (2.0,)  # e.g. (1.5, 2.0)

    # Morphology variants (can help close gaps; use carefully)
    morph_close: bool = True
    morph_kernel: Tuple[int, int] = (3, 3)

    # Binarization (useful for OCR/barcode sometimes; QR usually prefers grayscale)
    binarize: bool = False


def _resize_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    if max_side <= 0:
        return img
    h, w = img.shape[:2]
    s = max(h, w)
    if s <= max_side:
        return img
    scale = max_side / float(s)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _unsharp(gray: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return gray
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.2)
    return cv2.addWeighted(gray, 1.0 + amount, blur, -amount, 0)


def _upscale(gray: np.ndarray, factor: float) -> np.ndarray:
    if factor <= 1.0:
        return gray
    return cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)


def preprocess_variants(
    img_bgr: np.ndarray,
    cfg: Optional[PreprocessConfig] = None,
) -> Dict[str, np.ndarray]:
    """
    Returns a set of image variants to try decoders on.
    All returned images are grayscale uint8.
    """
    cfg = cfg or PreprocessConfig()
    img = _resize_max_side(img_bgr, cfg.resize_max_side)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if cfg.clahe:
        clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=cfg.clahe_grid)
        gray = clahe.apply(gray)

    if cfg.denoise:
        gray = cv2.fastNlMeansDenoising(
            gray, None, h=cfg.denoise_h, templateWindowSize=7, searchWindowSize=21
        )

    # Base sharpened
    gray_sharp = _unsharp(gray, cfg.sharpen_amount if cfg.sharpen else 0.0)

    # ✅ Bilateral on (typically) the "contrast-normalized" grayscale
    if cfg.bilateral:
        gray_bilat = cv2.bilateralFilter(
            gray,
            d=cfg.bilateral_d,
            sigmaColor=cfg.bilateral_sigma_color,
            sigmaSpace=cfg.bilateral_sigma_space,
        )
        # Optional sharpen after bilateral (mild)
        gray_bilat_sharp = _unsharp(gray_bilat, 0.35 if cfg.sharpen else 0.0)
    else:
        gray_bilat = None
        gray_bilat_sharp = None

    # ✅ Morphological close (soft) to close small gaps (careful: can hurt some QRs)
    if cfg.morph_close:
        kx, ky = cfg.morph_kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
        morph_close = cv2.morphologyEx(gray_sharp, cv2.MORPH_CLOSE, kernel, iterations=1)
    else:
        morph_close = None

    # Optional binarized version (more aggressive)
    if cfg.binarize:
        bw = cv2.adaptiveThreshold(
            gray_sharp,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            2,
        )
    else:
        bw = None

    out: Dict[str, np.ndarray] = {
        "gray": gray,
        "sharp": gray_sharp,
    }

    if gray_bilat is not None:
        out["bilateral"] = gray_bilat
    if gray_bilat_sharp is not None:
        out["bilateral_sharp"] = gray_bilat_sharp
    if morph_close is not None:
        out["morph_close"] = morph_close
    if bw is not None:
        out["bw"] = bw

    # ✅ Upscale variants (only for selected bases to keep it lightweight)
    if cfg.upscale and cfg.upscale_factors:
        for f in cfg.upscale_factors:
            if f <= 1.0:
                continue
            out[f"sharp_x{f:g}"] = _upscale(gray_sharp, f)

            if gray_bilat is not None:
                out[f"bilateral_x{f:g}"] = _upscale(gray_bilat, f)

            if morph_close is not None:
                out[f"morph_close_x{f:g}"] = _upscale(morph_close, f)

            if bw is not None:
                out[f"bw_x{f:g}"] = _upscale(bw, f)

    return out


def laplacian_sharpness(gray: np.ndarray) -> float:
    """Higher = sharper. Useful for best-frame selection and quality gating."""
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())