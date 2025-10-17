from typing import Dict, Any
from .engines import extract_blocks_pymupdf, cluster_columns, strip_header_footer, order_blocks
from .utils import dehyphenate, infer_meta_from_name, now_utc_iso
import json, pathlib

PAGE_BREAK = "\n\n<<<PAGE_BREAK>>>\n\n"

def extract_issue(pdf_path: str) -> Dict[str, Any]:
    pdf_path = pathlib.Path(pdf_path)
    blocks = extract_blocks_pymupdf(str(pdf_path))
    blocks = strip_header_footer(blocks, header_pct=0.08, footer_pct=0.08)
    blocks = cluster_columns(blocks, k_candidates=(1,2,3))
    blocks = order_blocks(blocks)

    texts = []
    current_page = None
    for b in blocks:
        if current_page is None: current_page = b["page"]
        if b["page"] != current_page:
            texts.append(PAGE_BREAK)
            current_page = b["page"]
        texts.append(b["text"].strip())

    joined = "\n\n".join([t for t in texts if t])
    cleaned = dehyphenate(joined)

    meta = infer_meta_from_name(pdf_path.name)
    meta.update({
        "pages": (blocks[-1]["page"] + 1) if blocks else 0,
        "extracted_at": now_utc_iso()
    })
    return {"meta": meta, "text": cleaned}

def save_json(obj: Dict[str,Any], out_dir: str, stem: str = None) -> str:
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    name = (stem or "issue") + ".json"
    path = out / name
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return str(path)
