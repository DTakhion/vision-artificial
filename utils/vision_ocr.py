# # utils/vision_ocr.py
# from __future__ import annotations

# from typing import Any, Dict, Optional, List, Union, Tuple
# import time
# import re
# import os

# import cv2
# import numpy as np

# from utils.vision_preprocess import preprocess_variants, PreprocessConfig


# DEFAULT_VARIANTS_OCR: List[str] = [
#     "bw", "bw_x2",
#     "sharp", "sharp_x2",
#     "gray",
#     "bilateral", "bilateral_sharp",
#     "morph_close", "morph_close_x2",
# ]


# def _resolve_variants(variants: Optional[Union[List[str], str]], available: List[str]) -> List[str]:
#     if variants is None:
#         return [v for v in DEFAULT_VARIANTS_OCR if v in available]
#     if isinstance(variants, str):
#         if variants.lower() == "all":
#             return available
#         if "," in variants:
#             req = [x.strip() for x in variants.split(",") if x.strip()]
#             return [v for v in req if v in available]
#         return [variants] if variants in available else []
#     return [v for v in variants if v in available]


# def _resize_gray(gray: np.ndarray, factor: float) -> np.ndarray:
#     if factor <= 1.0:
#         return gray
#     return cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)


# def _backend_available() -> Dict[str, bool]:
#     avail = {"tesseract": False}
#     try:
#         import pytesseract  # type: ignore  # noqa: F401
#         avail["tesseract"] = True
#     except Exception:
#         pass
#     return avail


# def _ocr_tesseract(gray: np.ndarray, *, allow: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-_/.:") -> str:
#     import pytesseract  # type: ignore
#     configs = [
#         f'--oem 1 --psm 7 -c tessedit_char_whitelist="{allow}"',
#         f'--oem 1 --psm 8 -c tessedit_char_whitelist="{allow}"',
#         f'--oem 1 --psm 6 -c tessedit_char_whitelist="{allow}"',
#     ]
#     best = ""
#     for cfg in configs:
#         txt = pytesseract.image_to_string(gray, config=cfg)
#         txt = (txt or "").strip()
#         if len(txt) > len(best):
#             best = txt
#     return best


# def _clean_text(txt: str) -> str:
#     txt = (txt or "").strip()
#     txt = re.sub(r"[ \t\r]+", " ", txt)
#     txt = txt.replace("\n", " ").strip()
#     return txt


# def _best_numeric_candidate(txt: str, min_len: int = 4) -> Optional[str]:
#     nums = re.findall(r"\d+", txt or "")
#     nums = [n for n in nums if len(n) >= min_len]
#     if not nums:
#         return None
#     nums.sort(key=len, reverse=True)
#     return nums[0]


# def _downscale_for_detect(img_bgr: np.ndarray, max_side: int = 960) -> Tuple[np.ndarray, float]:
#     h, w = img_bgr.shape[:2]
#     m = max(h, w)
#     if m <= max_side:
#         return img_bgr, 1.0
#     s = max_side / float(m)
#     small = cv2.resize(img_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
#     return small, s


# def _barcode_like_bbox_fast(img_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
#     h, w = img_bgr.shape[:2]
#     gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

#     gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
#     gx = cv2.convertScaleAbs(gx)
#     gx = cv2.GaussianBlur(gx, (0, 0), 2.0)
#     _, bw = cv2.threshold(gx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#     k = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 7))
#     bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=2)
#     bw = cv2.erode(bw, None, iterations=1)
#     bw = cv2.dilate(bw, None, iterations=2)

#     cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not cnts:
#         return None
#     cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:8]

#     for c in cnts:
#         x, y, ww, hh = cv2.boundingRect(c)
#         aspect = ww / float(hh + 1e-6)
#         area = ww * hh
#         if aspect < 2.0:
#             continue
#         if area < 0.006 * (w * h):
#             continue

#         pad = int(0.08 * max(ww, hh))
#         x0 = max(0, x - pad)
#         y0 = max(0, y - pad)
#         x1 = min(w, x + ww + pad)
#         y1 = min(h, y + hh + pad)
#         return (x0, y0, x1 - x0, y1 - y0)

#     return None


# def _find_rois(img_bgr: np.ndarray, roi_mode: str) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
#     """
#     roi_mode: "barcode_like" | "bottom" | "all"
#     """
#     out: List[Tuple[np.ndarray, Dict[str, Any]]] = []
#     h, w = img_bgr.shape[:2]

#     if roi_mode in ("barcode_like", "all"):
#         small, s = _downscale_for_detect(img_bgr, max_side=960)
#         bbox_small = None
#         try:
#             bbox_small = _barcode_like_bbox_fast(small)
#         except Exception:
#             bbox_small = None

#         if bbox_small is not None:
#             x0s, y0s, wws, hhs = bbox_small
#             inv = 1.0 / float(s)
#             x0 = int(round(x0s * inv))
#             y0 = int(round(y0s * inv))
#             ww = int(round(wws * inv))
#             hh = int(round(hhs * inv))
#             x0 = max(0, x0); y0 = max(0, y0)
#             x1 = min(w, x0 + ww); y1 = min(h, y0 + hh)
#             if x1 > x0 and y1 > y0:
#                 out.append((img_bgr[y0:y1, x0:x1].copy(), {"kind": "barcode_like", "bbox": (x0, y0, x1 - x0, y1 - y0)}))

#     if roi_mode in ("bottom", "all"):
#         y0 = int(h * 0.55)
#         out.append((img_bgr[y0:h, 0:w].copy(), {"kind": "bottom_band", "bbox": (0, y0, w, h - y0)}))

#     if not out:
#         out.append((img_bgr, {"kind": "full", "bbox": (0, 0, w, h)}))
#     return out


# def _subrois_for_numbers(roi_bgr: np.ndarray, meta: Dict[str, Any]) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
#     out: List[Tuple[np.ndarray, Dict[str, Any]]] = [(roi_bgr, {**meta, "subkind": "full"})]
#     h, w = roi_bgr.shape[:2]

#     if meta.get("kind") == "barcode_like":
#         y35 = int(h * 0.65)
#         if y35 < h - 5:
#             out.append((roi_bgr[y35:h, 0:w].copy(), {**meta, "subkind": "bottom_35pct"}))
#         y55 = int(h * 0.55)
#         if y55 < h - 5:
#             out.append((roi_bgr[y55:h, 0:w].copy(), {**meta, "subkind": "bottom_45pct"}))

#     return out


# def ocr_serial_best_effort(
#     img_bgr: np.ndarray,
#     *,
#     time_budget_ms: int = 350,
#     no_budget: bool = False,
#     max_tries: int = 120,
#     roi_mode: str = "all",  # "barcode_like" | "bottom" | "all"
#     variants: Optional[Union[List[str], str]] = None,
#     aggressive: bool = False,
#     numeric_only: bool = True,
#     min_numeric_len: int = 4,
#     save_debug_rois: Optional[str] = None,
#     debug: bool = False,
# ) -> Dict[str, Any]:
#     t0 = time.perf_counter()
#     tried: List[str] = []

#     if not _backend_available().get("tesseract", False):
#         return {"status": "not_available", "text": None, "backend": None, "elapsed_ms": 0, "variant": None, "tried": tried}

#     cfg = PreprocessConfig(
#         resize_max_side=1800 if aggressive else 1280,
#         clahe=True,
#         denoise=aggressive,
#         sharpen=True,
#         binarize=aggressive,
#     )

#     def budget_ok() -> bool:
#         if no_budget:
#             return True
#         return ((time.perf_counter() - t0) * 1000) <= time_budget_ms

#     try:
#         rois = _find_rois(img_bgr, roi_mode=roi_mode)

#         if save_debug_rois:
#             os.makedirs(save_debug_rois, exist_ok=True)
#             for i, (r, m) in enumerate(rois):
#                 cv2.imwrite(os.path.join(save_debug_rois, f"roi_{i}_{m.get('kind')}.png"), r)

#         tries = 0
#         for ridx, (roi_bgr, meta) in enumerate(rois):
#             if not budget_ok():
#                 break

#             for roi2, meta2 in _subrois_for_numbers(roi_bgr, meta):
#                 if not budget_ok():
#                     break

#                 ims = preprocess_variants(roi2, cfg=cfg) or {}
#                 if "gray" not in ims or ims.get("gray") is None:
#                     ims["gray"] = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)

#                 available = list(ims.keys())
#                 to_try = _resolve_variants(variants, available)
#                 if not to_try:
#                     to_try = [v for v in DEFAULT_VARIANTS_OCR if v in available] or ["gray"]

#                 for vname in to_try:
#                     base = ims.get(vname)
#                     if base is None:
#                         continue

#                     candidates: List[Tuple[str, np.ndarray]] = [
#                         (vname, base),
#                         (f"{vname}_x3", _resize_gray(base, 3.0)),
#                         (f"{vname}_x4", _resize_gray(base, 4.0)),
#                     ]

#                     for cname, gray in candidates:
#                         if not budget_ok():
#                             break
#                         tried.append(f"roi{ridx}:{meta2.get('kind')}:{meta2.get('subkind')}:{cname}:tesseract")
#                         tries += 1
#                         if tries > max_tries:
#                             elapsed = int((time.perf_counter() - t0) * 1000)
#                             return {
#                                 "status": "not_found",
#                                 "text": None,
#                                 "backend": None,
#                                 "elapsed_ms": elapsed,
#                                 "variant": None,
#                                 "tried": tried,
#                                 "note": "max_tries_reached",
#                                 "max_tries": max_tries,
#                             }

#                         txt = _clean_text(_ocr_tesseract(gray))
#                         if not txt:
#                             continue

#                         if numeric_only:
#                             cand = _best_numeric_candidate(txt, min_len=min_numeric_len)
#                             if not cand:
#                                 continue
#                             elapsed = int((time.perf_counter() - t0) * 1000)
#                             return {
#                                 "status": "success",
#                                 "text": cand,
#                                 "backend": "tesseract",
#                                 "elapsed_ms": elapsed,
#                                 "variant": vname,
#                                 "tried": tried,
#                                 "roi_bbox": meta2.get("bbox"),
#                                 "note": "numeric_candidate",
#                             }
#                         else:
#                             elapsed = int((time.perf_counter() - t0) * 1000)
#                             return {
#                                 "status": "success",
#                                 "text": txt,
#                                 "backend": "tesseract",
#                                 "elapsed_ms": elapsed,
#                                 "variant": vname,
#                                 "tried": tried,
#                                 "roi_bbox": meta2.get("bbox"),
#                                 "note": "raw_text",
#                             }

#         elapsed = int((time.perf_counter() - t0) * 1000)
#         note = None
#         if (not no_budget) and (elapsed > time_budget_ms):
#             note = "budget_exhausted"

#         return {
#             "status": "not_found",
#             "text": None,
#             "backend": None,
#             "elapsed_ms": elapsed,
#             "variant": None,
#             "tried": tried,
#             "note": note,
#         }

#     except Exception as e:
#         elapsed = int((time.perf_counter() - t0) * 1000)
#         return {"status": "error", "text": None, "backend": "tesseract", "elapsed_ms": elapsed, "variant": None, "tried": tried, "error": repr(e)}


# def _cli() -> int:
#     import argparse

#     p = argparse.ArgumentParser(description="OCR serial-like text using ROI + preprocess variants + tesseract.")
#     p.add_argument("image_path", type=str)
#     p.add_argument("--budget", type=int, default=350)
#     p.add_argument("--no_budget", action="store_true", help="Disable time budget (use max_tries as safety).")
#     p.add_argument("--max_tries", type=int, default=120, help="Hard cap of OCR attempts (safety).")
#     p.add_argument("--roi", type=str, default="all", choices=["barcode_like", "bottom", "all"])
#     p.add_argument("--variants", type=str, default="default")
#     p.add_argument("--aggressive", action="store_true")
#     p.add_argument("--text", action="store_true")
#     p.add_argument("--min_digits", type=int, default=4)
#     p.add_argument("--save_debug_rois", type=str, default=None)
#     p.add_argument("--debug", action="store_true")
#     args = p.parse_args()

#     if args.debug:
#         print({"vision_ocr_file": __file__})

#     img = cv2.imread(args.image_path)
#     if img is None:
#         print({"status": "error", "error": "Could not load image", "path": args.image_path})
#         return 2

#     v: Optional[Union[List[str], str]] = None if args.variants == "default" else args.variants

#     res = ocr_serial_best_effort(
#         img,
#         time_budget_ms=args.budget,
#         no_budget=args.no_budget,
#         max_tries=args.max_tries,
#         roi_mode=args.roi,
#         variants=v,
#         aggressive=args.aggressive,
#         numeric_only=(not args.text),
#         min_numeric_len=args.min_digits,
#         save_debug_rois=args.save_debug_rois,
#         debug=args.debug,
#     )
#     print(res)
#     return 0


# if __name__ == "__main__":
#     raise SystemExit(_cli())

# utils/vision_ocr.py
from __future__ import annotations

from typing import Any, Dict, Optional, List, Union, Tuple
import time
import re
import os

import cv2
import numpy as np

from utils.vision_preprocess import preprocess_variants, PreprocessConfig


DEFAULT_VARIANTS_OCR: List[str] = [
    "bw",
    "bw_x2",
    "sharp",
    "sharp_x2",
    "gray",
    "bilateral",
    "bilateral_sharp",
    "morph_close",
    "morph_close_x2",
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _resolve_variants(
    variants: Optional[Union[List[str], str]],
    available: List[str],
) -> List[str]:
    if variants is None:
        return [v for v in DEFAULT_VARIANTS_OCR if v in available]
    if isinstance(variants, str):
        if variants.lower() == "all":
            return available
        if "," in variants:
            req = [x.strip() for x in variants.split(",") if x.strip()]
            return [v for v in req if v in available]
        return [variants] if variants in available else []
    return [v for v in variants if v in available]


def _resize_gray(gray: np.ndarray, factor: float) -> np.ndarray:
    if factor <= 1.0:
        return gray
    return cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)


def _clean_text(txt: str) -> str:
    txt = (txt or "").strip()
    txt = re.sub(r"[ \t\r]+", " ", txt)
    txt = txt.replace("\n", " ").strip()
    return txt


def _normalize_text(txt: str) -> str:
    txt = _clean_text(txt)
    txt = txt.replace("—", "-").replace("–", "-")
    return txt


def _best_numeric_candidate(txt: str, min_len: int = 4) -> Optional[str]:
    nums = re.findall(r"\d+", txt or "")
    nums = [n for n in nums if len(n) >= min_len]
    if not nums:
        return None
    nums.sort(key=len, reverse=True)
    return nums[0]


def _backend_available() -> Dict[str, bool]:
    avail = {"tesseract": False}
    try:
        import pytesseract  # type: ignore  # noqa: F401
        avail["tesseract"] = True
    except Exception:
        pass
    return avail


def _configure_tesseract_from_env() -> None:
    """
    Permite setear Tesseract vía variable de entorno si hiciera falta:
      export TESSERACT_CMD="/opt/homebrew/bin/tesseract"
    """
    try:
        import pytesseract  # type: ignore
    except Exception:
        return

    cmd = os.getenv("TESSERACT_CMD")
    if cmd:
        try:
            pytesseract.pytesseract.tesseract_cmd = cmd
        except Exception:
            pass


def _downscale_for_detect(img_bgr: np.ndarray, max_side: int = 960) -> Tuple[np.ndarray, float]:
    h, w = img_bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img_bgr, 1.0
    s = max_side / float(m)
    small = cv2.resize(img_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return small, s


# ---------------------------------------------------------------------
# OCR core helpers
# ---------------------------------------------------------------------
def _ocr_tesseract_runs(
    gray: np.ndarray,
    *,
    allow: str,
) -> List[Dict[str, Any]]:
    """
    Ejecuta varios PSMs y devuelve candidatos con texto + confianza aproximada.
    """
    try:
        import pytesseract  # type: ignore
    except Exception:
        return []

    configs = [
        ("psm7", f'--oem 1 --psm 7 -c tessedit_char_whitelist="{allow}"'),
        ("psm8", f'--oem 1 --psm 8 -c tessedit_char_whitelist="{allow}"'),
        ("psm6", f'--oem 1 --psm 6 -c tessedit_char_whitelist="{allow}"'),
        ("psm13", f'--oem 1 --psm 13 -c tessedit_char_whitelist="{allow}"'),
    ]

    out: List[Dict[str, Any]] = []
    for psm_name, cfg in configs:
        try:
            txt = pytesseract.image_to_string(gray, config=cfg)
            txt = _normalize_text(txt or "")

            conf_avg = None
            try:
                data = pytesseract.image_to_data(gray, config=cfg, output_type=pytesseract.Output.DICT)
                confs = []
                for c in data.get("conf", []):
                    try:
                        val = float(c)
                        if val >= 0:
                            confs.append(val)
                    except Exception:
                        continue
                if confs:
                    conf_avg = float(sum(confs) / len(confs))
            except Exception:
                conf_avg = None

            out.append(
                {
                    "text": txt,
                    "psm": psm_name,
                    "config": cfg,
                    "confidence": conf_avg,
                }
            )
        except Exception:
            continue

    return out


def _score_ocr_candidate(
    *,
    raw_text: str,
    final_text: Optional[str],
    confidence: Optional[float],
    variant: str,
    candidate: str,
    roi_kind: str,
    numeric_only: bool,
) -> float:
    score = 0.0

    if final_text:
        score += min(len(final_text), 32) * 0.40

    if raw_text:
        score += min(len(raw_text), 32) * 0.06

    if confidence is not None:
        score += max(0.0, min(float(confidence), 100.0)) / 25.0

    if numeric_only and final_text and final_text.isdigit():
        score += 2.5

    if variant.startswith("bw"):
        score += 0.6
    elif variant.startswith("sharp"):
        score += 0.4
    elif variant.startswith("gray"):
        score += 0.2

    if "_x4" in candidate:
        score += 0.5
    elif "_x3" in candidate:
        score += 0.35
    elif "_x2" in candidate:
        score += 0.2

    if roi_kind == "barcode_like":
        score += 0.8
    elif roi_kind == "bottom_band":
        score += 0.35
    elif roi_kind == "full":
        score += 0.1

    return float(score)


def _pick_best_candidate(candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not candidates:
        return None
    candidates = sorted(candidates, key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return candidates[0]


def _extra_ocr_candidates(gray: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    out: List[Tuple[str, np.ndarray]] = []

    g = gray
    out.append(("base_x3", _resize_gray(g, 3.0)))
    out.append(("base_x4", _resize_gray(g, 4.0)))

    try:
        _, otsu = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        out.append(("otsu", otsu))
        out.append(("otsu_x3", _resize_gray(otsu, 3.0)))
        out.append(("otsu_x4", _resize_gray(otsu, 4.0)))
    except Exception:
        pass

    try:
        adap = cv2.adaptiveThreshold(
            g,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            2,
        )
        out.append(("adapt", adap))
        out.append(("adapt_x3", _resize_gray(adap, 3.0)))
        out.append(("adapt_x4", _resize_gray(adap, 4.0)))
    except Exception:
        pass

    try:
        inv = cv2.bitwise_not(g)
        out.append(("invert", inv))
        out.append(("invert_x3", _resize_gray(inv, 3.0)))
    except Exception:
        pass

    return out


# ---------------------------------------------------------------------
# ROI detection
# ---------------------------------------------------------------------
def _barcode_like_bboxes_fast(
    img_bgr: np.ndarray,
    max_rois: int = 4,
) -> List[Tuple[int, int, int, int]]:
    """
    Busca varias regiones tipo barcode/serial en vez de solo una.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    gx = cv2.convertScaleAbs(gx)
    gx = cv2.GaussianBlur(gx, (0, 0), 2.0)
    _, bw = cv2.threshold(gx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 7))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=2)
    bw = cv2.erode(bw, None, iterations=1)
    bw = cv2.dilate(bw, None, iterations=2)

    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:16]
    out: List[Tuple[int, int, int, int]] = []

    for c in cnts:
        x, y, ww, hh = cv2.boundingRect(c)
        aspect = ww / float(hh + 1e-6)
        area = ww * hh

        if aspect < 1.8:
            continue
        if area < 0.0045 * (w * h):
            continue

        pad = int(0.10 * max(ww, hh))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(w, x + ww + pad)
        y1 = min(h, y + hh + pad)
        if x1 <= x0 or y1 <= y0:
            continue

        out.append((x0, y0, x1 - x0, y1 - y0))
        if len(out) >= max_rois:
            break

    return out


def _find_rois(
    img_bgr: np.ndarray,
    roi_mode: str,
    *,
    max_rois: int = 4,
) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
    """
    roi_mode:
      - "barcode_like"
      - "bottom"
      - "full"
      - "all"
    """
    out: List[Tuple[np.ndarray, Dict[str, Any]]] = []
    h, w = img_bgr.shape[:2]

    if roi_mode in ("barcode_like", "all"):
        small, s = _downscale_for_detect(img_bgr, max_side=960)

        bboxes_small: List[Tuple[int, int, int, int]] = []
        try:
            bboxes_small = _barcode_like_bboxes_fast(small, max_rois=max_rois)
        except Exception:
            bboxes_small = []

        inv = 1.0 / float(s)
        for i, bbox_small in enumerate(bboxes_small):
            x0s, y0s, wws, hhs = bbox_small
            x0 = int(round(x0s * inv))
            y0 = int(round(y0s * inv))
            ww = int(round(wws * inv))
            hh = int(round(hhs * inv))

            x0 = max(0, x0)
            y0 = max(0, y0)
            x1 = min(w, x0 + ww)
            y1 = min(h, y0 + hh)
            if x1 > x0 and y1 > y0:
                out.append(
                    (
                        img_bgr[y0:y1, x0:x1].copy(),
                        {
                            "kind": "barcode_like",
                            "bbox": (x0, y0, x1 - x0, y1 - y0),
                            "roi_index": i,
                        },
                    )
                )

    if roi_mode in ("bottom", "all"):
        y0 = int(h * 0.55)
        out.append(
            (
                img_bgr[y0:h, 0:w].copy(),
                {
                    "kind": "bottom_band",
                    "bbox": (0, y0, w, h - y0),
                    "roi_index": len(out),
                },
            )
        )

    if roi_mode in ("full", "all") and not out:
        out.append(
            (
                img_bgr,
                {
                    "kind": "full",
                    "bbox": (0, 0, w, h),
                    "roi_index": 0,
                },
            )
        )

    if not out:
        out.append(
            (
                img_bgr,
                {
                    "kind": "full",
                    "bbox": (0, 0, w, h),
                    "roi_index": 0,
                },
            )
        )

    return out


def _subrois_for_numbers(
    roi_bgr: np.ndarray,
    meta: Dict[str, Any],
) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
    out: List[Tuple[np.ndarray, Dict[str, Any]]] = [(roi_bgr, {**meta, "subkind": "full"})]
    h, w = roi_bgr.shape[:2]

    if h < 12 or w < 12:
        return out

    if meta.get("kind") == "barcode_like":
        y65 = int(h * 0.65)
        y55 = int(h * 0.55)
        y45 = int(h * 0.45)

        if y65 < h - 5:
            out.append((roi_bgr[y65:h, 0:w].copy(), {**meta, "subkind": "bottom_35pct"}))
        if y55 < h - 5:
            out.append((roi_bgr[y55:h, 0:w].copy(), {**meta, "subkind": "bottom_45pct"}))
        if y45 < h - 5:
            out.append((roi_bgr[y45:h, 0:w].copy(), {**meta, "subkind": "bottom_55pct"}))

    elif meta.get("kind") == "bottom_band":
        y0 = int(h * 0.20)
        if y0 < h - 5:
            out.append((roi_bgr[y0:h, 0:w].copy(), {**meta, "subkind": "bottom_80pct"}))

    return out


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def ocr_serial_best_effort(
    img_bgr: np.ndarray,
    *,
    time_budget_ms: int = 350,
    no_budget: bool = False,
    max_tries: int = 120,
    roi_mode: str = "all",  # "barcode_like" | "bottom" | "full" | "all"
    variants: Optional[Union[List[str], str]] = None,
    aggressive: bool = False,
    numeric_only: bool = True,
    min_numeric_len: int = 4,
    save_debug_rois: Optional[str] = None,
    debug: bool = False,
    mode: str = "fast",  # "fast" | "collect"
    max_rois: int = 4,
) -> Dict[str, Any]:
    """
    OCR best effort alineado al nuevo flujo:
      - explora múltiples ROIs
      - explora múltiples variantes
      - prueba varios PSMs
      - rankea y elige el mejor candidato

    mode:
      - fast: corta temprano si ya obtuvo un candidato fuerte
      - collect: explora más y devuelve el mejor al final
    """
    t0 = time.perf_counter()
    tried: List[str] = []
    mode = (mode or "fast").strip().lower()

    _configure_tesseract_from_env()

    if not _backend_available().get("tesseract", False):
        return {
            "status": "not_available",
            "text": None,
            "backend": None,
            "elapsed_ms": 0,
            "variant": None,
            "candidate": None,
            "tried": tried,
        }

    cfg = PreprocessConfig(
        resize_max_side=1800 if aggressive else 1280,
        clahe=True,
        denoise=aggressive,
        sharpen=True,
        binarize=aggressive,
        bilateral=True,
        morph_close=True,
        upscale=True,
        upscale_factors=(2.0,),
    )

    def budget_ok() -> bool:
        if no_budget:
            return True
        return ((time.perf_counter() - t0) * 1000) <= time_budget_ms

    def strong_enough(cand: Dict[str, Any]) -> bool:
        txt = cand.get("text") or ""
        score = float(cand.get("score", 0.0))

        if numeric_only:
            return txt.isdigit() and len(txt) >= max(min_numeric_len + 2, 6) and score >= 6.0
        return len(txt) >= 6 and score >= 5.5

    allow = "0123456789" if numeric_only else "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-_/.:"
    best_candidates: List[Dict[str, Any]] = []

    try:
        rois = _find_rois(img_bgr, roi_mode=roi_mode, max_rois=max_rois)

        if save_debug_rois:
            os.makedirs(save_debug_rois, exist_ok=True)
            for i, (r, m) in enumerate(rois):
                cv2.imwrite(os.path.join(save_debug_rois, f"roi_{i}_{m.get('kind')}.png"), r)

        tries = 0

        for ridx, (roi_bgr, meta) in enumerate(rois):
            if not budget_ok():
                break

            for roi2, meta2 in _subrois_for_numbers(roi_bgr, meta):
                if not budget_ok():
                    break

                ims = preprocess_variants(roi2, cfg=cfg) or {}
                if "gray" not in ims or ims.get("gray") is None:
                    ims["gray"] = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)

                available = list(ims.keys())
                to_try = _resolve_variants(variants, available)
                if not to_try:
                    to_try = [v for v in DEFAULT_VARIANTS_OCR if v in available] or ["gray"]

                for vname in to_try:
                    if not budget_ok():
                        break

                    base = ims.get(vname)
                    if base is None:
                        continue

                    local_candidates: List[Tuple[str, np.ndarray]] = [(vname, base)]
                    local_candidates.extend(_extra_ocr_candidates(base))

                    for cname, gray in local_candidates:
                        if not budget_ok():
                            break

                        tries += 1
                        tried.append(
                            f"roi{ridx}:{meta2.get('kind')}:{meta2.get('subkind')}:{vname}:{cname}:tesseract"
                        )

                        if tries > max_tries:
                            elapsed = int((time.perf_counter() - t0) * 1000)
                            best = _pick_best_candidate(best_candidates)
                            if best is not None:
                                return {
                                    "status": "success",
                                    "text": best.get("text"),
                                    "backend": "tesseract",
                                    "elapsed_ms": elapsed,
                                    "variant": best.get("variant"),
                                    "candidate": best.get("candidate"),
                                    "psm": best.get("psm"),
                                    "confidence": best.get("confidence"),
                                    "score": best.get("score"),
                                    "roi_bbox": best.get("roi_bbox"),
                                    "roi_kind": best.get("roi_kind"),
                                    "roi_index": best.get("roi_index"),
                                    "subkind": best.get("subkind"),
                                    "tried": tried,
                                    "note": "best_before_max_tries",
                                    "mode": mode,
                                }

                            return {
                                "status": "not_found",
                                "text": None,
                                "backend": None,
                                "elapsed_ms": elapsed,
                                "variant": None,
                                "candidate": None,
                                "tried": tried,
                                "note": "max_tries_reached",
                                "max_tries": max_tries,
                                "mode": mode,
                            }

                        runs = _ocr_tesseract_runs(gray, allow=allow)

                        for run in runs:
                            raw_txt = _normalize_text(run.get("text") or "")
                            if not raw_txt:
                                continue

                            if numeric_only:
                                final_txt = _best_numeric_candidate(raw_txt, min_len=min_numeric_len)
                            else:
                                final_txt = raw_txt

                            if not final_txt:
                                continue

                            cand = {
                                "text": final_txt,
                                "raw_text": raw_txt,
                                "backend": "tesseract",
                                "variant": vname,
                                "candidate": cname,
                                "psm": run.get("psm"),
                                "confidence": run.get("confidence"),
                                "roi_bbox": meta2.get("bbox"),
                                "roi_kind": meta2.get("kind"),
                                "roi_index": meta2.get("roi_index"),
                                "subkind": meta2.get("subkind"),
                            }
                            cand["score"] = _score_ocr_candidate(
                                raw_text=raw_txt,
                                final_text=final_txt,
                                confidence=run.get("confidence"),
                                variant=vname,
                                candidate=cname,
                                roi_kind=meta2.get("kind", "unknown"),
                                numeric_only=numeric_only,
                            )

                            best_candidates.append(cand)

                            if mode == "fast" and strong_enough(cand):
                                elapsed = int((time.perf_counter() - t0) * 1000)
                                return {
                                    "status": "success",
                                    "text": cand.get("text"),
                                    "backend": "tesseract",
                                    "elapsed_ms": elapsed,
                                    "variant": cand.get("variant"),
                                    "candidate": cand.get("candidate"),
                                    "psm": cand.get("psm"),
                                    "confidence": cand.get("confidence"),
                                    "score": cand.get("score"),
                                    "roi_bbox": cand.get("roi_bbox"),
                                    "roi_kind": cand.get("roi_kind"),
                                    "roi_index": cand.get("roi_index"),
                                    "subkind": cand.get("subkind"),
                                    "tried": tried,
                                    "note": "strong_candidate_fast",
                                    "mode": mode,
                                }

        elapsed = int((time.perf_counter() - t0) * 1000)
        best = _pick_best_candidate(best_candidates)

        if best is not None:
            return {
                "status": "success",
                "text": best.get("text"),
                "backend": "tesseract",
                "elapsed_ms": elapsed,
                "variant": best.get("variant"),
                "candidate": best.get("candidate"),
                "psm": best.get("psm"),
                "confidence": best.get("confidence"),
                "score": best.get("score"),
                "roi_bbox": best.get("roi_bbox"),
                "roi_kind": best.get("roi_kind"),
                "roi_index": best.get("roi_index"),
                "subkind": best.get("subkind"),
                "tried": tried,
                "note": "best_candidate",
                "mode": mode,
            }

        note = None
        if (not no_budget) and (elapsed > time_budget_ms):
            note = "budget_exhausted"

        return {
            "status": "not_found",
            "text": None,
            "backend": None,
            "elapsed_ms": elapsed,
            "variant": None,
            "candidate": None,
            "tried": tried,
            "note": note,
            "mode": mode,
        }

    except Exception as e:
        elapsed = int((time.perf_counter() - t0) * 1000)
        return {
            "status": "error",
            "text": None,
            "backend": "tesseract",
            "elapsed_ms": elapsed,
            "variant": None,
            "candidate": None,
            "tried": tried,
            "error": repr(e),
            "mode": mode,
        }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def _cli() -> int:
    import argparse

    p = argparse.ArgumentParser(description="OCR serial-like text using ROI + preprocess variants + tesseract.")
    p.add_argument("image_path", type=str)
    p.add_argument("--budget", type=int, default=350)
    p.add_argument("--no_budget", action="store_true", help="Disable time budget (use max_tries as safety).")
    p.add_argument("--max_tries", type=int, default=120, help="Hard cap of OCR attempts (safety).")
    p.add_argument("--roi", type=str, default="all", choices=["barcode_like", "bottom", "full", "all"])
    p.add_argument("--variants", type=str, default="default")
    p.add_argument("--aggressive", action="store_true")
    p.add_argument("--text", action="store_true", help="Disable numeric_only mode")
    p.add_argument("--min_digits", type=int, default=4)
    p.add_argument("--save_debug_rois", type=str, default=None)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--mode", type=str, default="fast", choices=["fast", "collect"])
    p.add_argument("--max_rois", type=int, default=4)
    args = p.parse_args()

    if args.debug:
        print({"vision_ocr_file": __file__})

    img = cv2.imread(args.image_path)
    if img is None:
        print({"status": "error", "error": "Could not load image", "path": args.image_path})
        return 2

    v: Optional[Union[List[str], str]] = None if args.variants == "default" else args.variants

    res = ocr_serial_best_effort(
        img,
        time_budget_ms=args.budget,
        no_budget=args.no_budget,
        max_tries=args.max_tries,
        roi_mode=args.roi,
        variants=v,
        aggressive=args.aggressive,
        numeric_only=(not args.text),
        min_numeric_len=args.min_digits,
        save_debug_rois=args.save_debug_rois,
        debug=args.debug,
        mode=args.mode,
        max_rois=args.max_rois,
    )
    print(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())