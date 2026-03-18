# utils/vision_excel_picking.py
from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree as ET


@dataclass
class ExcelPickingConfig:
    sheet_name: Optional[str] = None
    header_row: int = 2  # 1-based


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
            rec["_row"] = i + 1  # Excel row number
            records.append(rec)

    return records


def load_excel_picking(
    excel_path: str,
    *,
    cfg: Optional[ExcelPickingConfig] = None,
) -> Dict[str, Any]:
    cfg = cfg or ExcelPickingConfig()

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


def filter_excel_by_picking_metadata(
    excel_records: List[Dict[str, Any]],
    *,
    entrega: Optional[str],
    ruta: Optional[str],
) -> List[Dict[str, Any]]:
    entrega_n = _norm_code(entrega)
    ruta_n = _norm_code(ruta)

    out = []
    for rec in excel_records:
        shipping = _norm_code(rec.get("shipping"))
        ruta_excel = _norm_code(rec.get("ruta"))

        ok_entrega = (not entrega_n) or (shipping == entrega_n)
        ok_ruta = (not ruta_n) or (ruta_excel == ruta_n)

        if ok_entrega and ok_ruta:
            out.append(rec)

    return out


def build_excel_product_index(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}

    for rec in records:
        articulo = _norm_code(rec.get("articulo"))
        if not articulo:
            continue

        idx[articulo] = {
            "row": rec.get("_row"),
            "shipping": _clean_text(rec.get("shipping")),
            "orden_compra": _clean_text(rec.get("orden_compra")),
            "cliente": _clean_text(rec.get("cliente")),
            "direccion": _clean_text(rec.get("direccion")),
            "ruta": _clean_text(rec.get("ruta")),
            "articulo": _clean_text(rec.get("articulo")),
            "descripcion": _clean_text(rec.get("descripcion")),
            "estado": _clean_text(rec.get("estado")),
            "cant_original": _to_int(rec.get("cant_original")),
            "cant_trabajada": _to_int(rec.get("cant_trabajada")),
            "diferencia": _to_int(rec.get("diferencia")),
            "kits": rec.get("kits"),
            "raw": rec,
        }

    return idx


def match_picking_with_excel(
    picking_result: Dict[str, Any],
    excel_result: Dict[str, Any],
) -> Dict[str, Any]:
    metadata = picking_result.get("metadata", {}) or {}
    products = picking_result.get("products", []) or []
    excel_records = excel_result.get("records", []) or []

    entrega = metadata.get("entrega")
    ruta = metadata.get("ruta")

    filtered = filter_excel_by_picking_metadata(
        excel_records,
        entrega=entrega,
        ruta=ruta,
    )

    idx = build_excel_product_index(filtered)

    matched_products = []
    for prod in products:
        codigo = _norm_code(prod.get("codigo_item"))
        excel_prod = idx.get(codigo)

        if excel_prod is None:
            matched_products.append({
                "codigo_item_ocr": prod.get("codigo_item"),
                "descripcion_ocr": prod.get("descripcion"),
                "unidades_ocr": prod.get("unidades"),
                "match_status": "not_found_in_excel",
                "excel": None,
            })
            continue

        matched_products.append({
            "codigo_item_ocr": prod.get("codigo_item"),
            "codigo_item_excel": excel_prod.get("articulo"),
            "descripcion_ocr": prod.get("descripcion"),
            "descripcion_excel": excel_prod.get("descripcion"),
            "unidades_ocr": prod.get("unidades"),
            "cant_original_excel": excel_prod.get("cant_original"),
            "cant_trabajada_excel": excel_prod.get("cant_trabajada"),
            "diferencia_excel": excel_prod.get("diferencia"),
            "cliente_excel": excel_prod.get("cliente"),
            "direccion_excel": excel_prod.get("direccion"),
            "orden_compra_excel": excel_prod.get("orden_compra"),
            "shipping_excel": excel_prod.get("shipping"),
            "ruta_excel": excel_prod.get("ruta"),
            "match_status": "matched",
        })

    return {
        "status": "success",
        "metadata_match": {
            "picking_entrega": entrega,
            "picking_ruta": ruta,
            "excel_rows_filtered": len(filtered),
        },
        "products": matched_products,
        "counts": {
            "picking_products": len(products),
            "excel_filtered_rows": len(filtered),
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

    parser = argparse.ArgumentParser(description="Load and match picking Excel data.")
    parser.add_argument("excel_path", type=str)
    args = parser.parse_args()

    res = load_excel_picking(args.excel_path)
    print(json.dumps(res, ensure_ascii=False, indent=2))
