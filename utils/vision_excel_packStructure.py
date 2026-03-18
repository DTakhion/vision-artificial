# utils/vision_excel_packStructure.py
from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree as ET


@dataclass
class ExcelPackStructureConfig:
    sheet_name: Optional[str] = "PS"
    header_row: int = 1  # 1-based


NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL_NS = {"r": "http://schemas.openxmlformats.org/package/2006/relationships"}


def _clean_text(txt: Any) -> str:
    s = "" if txt is None else str(txt)
    s = s.replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _norm_key(txt: Any) -> str:
    s = _clean_text(txt).lower()
    s = (
        s.replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
    )
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def _norm_code(txt: Any) -> str:
    s = _clean_text(txt).upper()
    s = s.replace(" ", "").replace(".", "").replace(",", "")
    return s


def _to_int(val: Any) -> Optional[int]:
    if val is None or val == "":
        return None
    try:
        return int(round(float(val)))
    except Exception:
        return None


def _to_float(val: Any) -> Optional[float]:
    if val is None or val == "":
        return None
    try:
        return float(str(val).replace(",", "."))
    except Exception:
        return None


def _col_to_index(col_letters: str) -> int:
    n = 0
    for ch in col_letters:
        if "A" <= ch <= "Z":
            n = n * 26 + (ord(ch) - ord("A") + 1)
    return n - 1


def _split_cell_ref(cell_ref: str) -> tuple[int, int]:
    m = re.fullmatch(r"([A-Z]+)(\d+)", cell_ref)
    if not m:
        return 0, 0
    col_letters, row_str = m.groups()
    return int(row_str), _col_to_index(col_letters)


def _load_shared_strings(zf: zipfile.ZipFile) -> List[str]:
    path = "xl/sharedStrings.xml"
    if path not in zf.namelist():
        return []

    root = ET.fromstring(zf.read(path))
    out: List[str] = []

    for si in root.findall("x:si", NS):
        texts = []
        for t in si.findall(".//x:t", NS):
            texts.append(t.text or "")
        out.append("".join(texts))

    return out


def _load_workbook_sheet_map(zf: zipfile.ZipFile) -> List[Dict[str, str]]:
    wb_root = ET.fromstring(zf.read("xl/workbook.xml"))
    rel_root = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))

    rel_map: Dict[str, str] = {}
    for rel in rel_root.findall("r:Relationship", REL_NS):
        rel_id = rel.attrib.get("Id")
        target = rel.attrib.get("Target")
        if rel_id and target:
            if not target.startswith("xl/"):
                target = f"xl/{target}"
            rel_map[rel_id] = target

    sheets = []
    for sh in wb_root.findall("x:sheets/x:sheet", NS):
        name = sh.attrib.get("name", "")
        rid = sh.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
        if rid and rid in rel_map:
            sheets.append({"name": name, "path": rel_map[rid]})

    return sheets


def _cell_value(cell_elem: ET.Element, shared_strings: List[str]) -> Any:
    cell_type = cell_elem.attrib.get("t")
    v = cell_elem.find("x:v", NS)

    if cell_type == "inlineStr":
        t = cell_elem.find(".//x:t", NS)
        return t.text if t is not None else ""

    if v is None or v.text is None:
        return ""

    raw = v.text

    if cell_type == "s":
        try:
            idx = int(raw)
            return shared_strings[idx]
        except Exception:
            return raw

    return raw


def _sheet_to_matrix(zf: zipfile.ZipFile, sheet_path: str, shared_strings: List[str]) -> List[List[Any]]:
    root = ET.fromstring(zf.read(sheet_path))
    sheet_data = root.find("x:sheetData", NS)
    if sheet_data is None:
        return []

    rows_map: Dict[int, Dict[int, Any]] = {}
    max_col = 0
    max_row = 0

    for row in sheet_data.findall("x:row", NS):
        r_idx = int(row.attrib.get("r", "0"))
        if r_idx <= 0:
            continue
        row_dict: Dict[int, Any] = {}

        for c in row.findall("x:c", NS):
            ref = c.attrib.get("r", "")
            rr, cc = _split_cell_ref(ref)
            if rr <= 0:
                rr = r_idx
            val = _cell_value(c, shared_strings)
            row_dict[cc] = val
            max_col = max(max_col, cc)
            max_row = max(max_row, rr)

        rows_map[r_idx] = row_dict

    matrix: List[List[Any]] = []
    for r in range(1, max_row + 1):
        row_dict = rows_map.get(r, {})
        row_vals = [row_dict.get(c, "") for c in range(max_col + 1)]
        matrix.append(row_vals)

    return matrix


def _matrix_to_records(matrix: List[List[Any]], header_row: int) -> List[Dict[str, Any]]:
    if not matrix or header_row < 1 or header_row > len(matrix):
        return []

    headers = [_clean_text(x) for x in matrix[header_row - 1]]
    keys = [_norm_key(h) for h in headers]

    records: List[Dict[str, Any]] = []
    for i in range(header_row, len(matrix)):
        row = matrix[i]
        rec: Dict[str, Any] = {}
        empty = True
        for j, key in enumerate(keys):
            value = row[j] if j < len(row) else ""
            if value not in ("", None):
                empty = False
            rec[key] = value
        if not empty:
            rec["_row"] = i + 1
            records.append(rec)

    return records


def load_excel_packstructure(
    excel_path: str,
    *,
    cfg: Optional[ExcelPackStructureConfig] = None,
) -> Dict[str, Any]:
    cfg = cfg or ExcelPackStructureConfig()

    with zipfile.ZipFile(excel_path, "r") as zf:
        shared_strings = _load_shared_strings(zf)
        sheets = _load_workbook_sheet_map(zf)

        if not sheets:
            return {
                "status": "error",
                "error": "no_sheets_found",
                "excel_path": excel_path,
            }

        if cfg.sheet_name:
            selected = next((s for s in sheets if s["name"] == cfg.sheet_name), None)
            if selected is None:
                return {
                    "status": "error",
                    "error": "sheet_not_found",
                    "excel_path": excel_path,
                    "sheet_name": cfg.sheet_name,
                    "available_sheets": [s["name"] for s in sheets],
                }
        else:
            selected = sheets[0]

        matrix = _sheet_to_matrix(zf, selected["path"], shared_strings)
        records = _matrix_to_records(matrix, header_row=cfg.header_row)

    return {
        "status": "success",
        "excel_path": excel_path,
        "sheet_name": selected["name"],
        "rows_loaded": len(records),
        "records": records,
        "config": asdict(cfg),
    }


def filter_packstructure_by_skus(
    records: List[Dict[str, Any]],
    *,
    skus: List[str],
) -> List[Dict[str, Any]]:
    sku_set = {_norm_code(x) for x in skus if _norm_code(x)}
    if not sku_set:
        return records

    out = []
    for rec in records:
        sku = _norm_code(rec.get("sku"))
        if sku in sku_set:
            out.append(rec)
    return out


def build_packstructure_index(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}

    for rec in records:
        sku = _norm_code(rec.get("sku"))
        if not sku:
            continue

        idx[sku] = {
            "row": rec.get("_row"),
            "sku": _clean_text(rec.get("sku")),
            "descripcion": _clean_text(rec.get("descripcion")),
            "item_type": _clean_text(rec.get("item_type")),

            "qty_ea": _to_int(rec.get("qty_ea")),
            "largo_ea": _to_float(rec.get("largo_ea")),
            "ancho_ea": _to_float(rec.get("ancho_ea")),
            "alto_ea": _to_float(rec.get("alto_ea")),
            "peso_ea": _to_float(rec.get("peso_ea")),
            "ean": _clean_text(rec.get("ean")),

            "qty_inn": _to_int(rec.get("qty_inn")),
            "largo_inn": _to_float(rec.get("largo_inn")),
            "ancho_inn": _to_float(rec.get("ancho_inn")),
            "alto_inn": _to_float(rec.get("alto_inn")),
            "peso_inn": _to_float(rec.get("peso_inn")),
            "ean_in": _clean_text(rec.get("ean_in")),

            "qty_cs": _to_int(rec.get("qty_cs")),
            "largo_cs": _to_float(rec.get("largo_cs")),
            "ancho_cs": _to_float(rec.get("ancho_cs")),
            "alto_cs": _to_float(rec.get("alto_cs")),
            "peso_cs": _to_float(rec.get("peso_cs")),
            "ean_cs": _clean_text(rec.get("ean_cs")),

            "qty_pal": _to_int(rec.get("qty_pal")),
            "largo_pal": _to_float(rec.get("largo_pal")),
            "ancho_pal": _to_float(rec.get("ancho_pal")),
            "alto_pal": _to_float(rec.get("alto_pal")),
            "peso_pal": _to_float(rec.get("peso_pal")),

            "raw": rec,
        }

    return idx


def match_picking_with_packstructure(
    picking_result: Dict[str, Any],
    packstructure_result: Dict[str, Any],
) -> Dict[str, Any]:
    products = picking_result.get("products", []) or []
    pack_records = packstructure_result.get("records", []) or []

    idx = build_packstructure_index(pack_records)

    matched_products = []
    for prod in products:
        codigo = _norm_code(prod.get("codigo_item"))
        pack_prod = idx.get(codigo)

        if pack_prod is None:
            matched_products.append({
                "codigo_item_ocr": prod.get("codigo_item"),
                "descripcion_ocr": prod.get("descripcion"),
                "unidades_ocr": prod.get("unidades"),
                "match_status": "not_found_in_packstructure",
                "packstructure": None,
            })
            continue

        matched_products.append({
            "codigo_item_ocr": prod.get("codigo_item"),
            "codigo_item_pack": pack_prod.get("sku"),
            "descripcion_ocr": prod.get("descripcion"),
            "descripcion_pack": pack_prod.get("descripcion"),
            "unidades_ocr": prod.get("unidades"),
            "item_type_pack": pack_prod.get("item_type"),

            "qty_ea_pack": pack_prod.get("qty_ea"),
            "largo_ea_pack": pack_prod.get("largo_ea"),
            "ancho_ea_pack": pack_prod.get("ancho_ea"),
            "alto_ea_pack": pack_prod.get("alto_ea"),
            "peso_ea_pack": pack_prod.get("peso_ea"),
            "ean_pack": pack_prod.get("ean"),

            "qty_inn_pack": pack_prod.get("qty_inn"),
            "largo_inn_pack": pack_prod.get("largo_inn"),
            "ancho_inn_pack": pack_prod.get("ancho_inn"),
            "alto_inn_pack": pack_prod.get("alto_inn"),
            "peso_inn_pack": pack_prod.get("peso_inn"),
            "ean_in_pack": pack_prod.get("ean_in"),

            "qty_cs_pack": pack_prod.get("qty_cs"),
            "largo_cs_pack": pack_prod.get("largo_cs"),
            "ancho_cs_pack": pack_prod.get("ancho_cs"),
            "alto_cs_pack": pack_prod.get("alto_cs"),
            "peso_cs_pack": pack_prod.get("peso_cs"),
            "ean_cs_pack": pack_prod.get("ean_cs"),

            "qty_pal_pack": pack_prod.get("qty_pal"),
            "largo_pal_pack": pack_prod.get("largo_pal"),
            "ancho_pal_pack": pack_prod.get("ancho_pal"),
            "alto_pal_pack": pack_prod.get("alto_pal"),
            "peso_pal_pack": pack_prod.get("peso_pal"),

            "match_status": "matched",
        })

    return {
        "status": "success",
        "products": matched_products,
        "counts": {
            "picking_products": len(products),
            "packstructure_rows": len(pack_records),
            "matched_products": sum(1 for p in matched_products if p["match_status"] == "matched"),
        },
    }


def save_json(data: Dict[str, Any], out_path: str) -> str:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return str(p)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load and match PackStructure Excel data.")
    parser.add_argument("excel_path", type=str)
    parser.add_argument("--sheet_name", type=str, default="PS")
    parser.add_argument("--header_row", type=int, default=1)
    args = parser.parse_args()

    cfg = ExcelPackStructureConfig(
        sheet_name=args.sheet_name,
        header_row=args.header_row,
    )

    res = load_excel_packstructure(args.excel_path, cfg=cfg)
    print(json.dumps(res, ensure_ascii=False, indent=2))