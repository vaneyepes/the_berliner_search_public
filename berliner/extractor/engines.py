from typing import List, Tuple, Dict, Any
import fitz  # PyMuPDF
import numpy as np

def extract_blocks_pymupdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Returns a list of blocks per page with text + bbox:
    [{"page": i, "bbox": (x0,y0,x1,y1), "text": "..."}]
    """
    doc = fitz.open(pdf_path)
    blocks = []
    for i, page in enumerate(doc):
        for b in page.get_text("blocks"):
            # b = (x0, y0, x1, y1, "text", block_no, block_type, block_flags)
            x0, y0, x1, y1, txt, *_ = b
            txt = (txt or "").strip()
            if not txt:
                continue
            blocks.append({"page": i, "bbox": (x0,y0,x1,y1), "text": txt, "size": page.rect})
    return blocks

def cluster_columns(blocks: List[Dict[str,Any]], k_candidates=(1,2,3)) -> List[Dict[str,Any]]:
    """
    Clusters blocks into 1-3 columns by their x-center; assigns 'col' index.
    """
    if not blocks: return blocks
    xs = np.array([ (b["bbox"][0]+b["bbox"][2])/2.0 for b in blocks ]).reshape(-1,1)

    from sklearn.cluster import KMeans
    best_inertia = None; best_k = 1; best_labels = None
    for k in k_candidates:
        if k > len(blocks): break
        km = KMeans(n_clusters=k, n_init="auto", random_state=0).fit(xs)
        if best_inertia is None or km.inertia_ < best_inertia*0.8:
            best_inertia = km.inertia_; best_k = k; best_labels = km.labels_
    for b, lab in zip(blocks, best_labels):
        b["col"] = int(lab)
    # sort columns by leftmost center
    col_lefts = {}
    for c in set(best_labels):
        col_lefts[c] = min([ (b["bbox"][0]+b["bbox"][2])/2.0 for b in blocks if b["col"]==c ])
    order = {c:i for i, c in enumerate(sorted(col_lefts, key=lambda c: col_lefts[c]))}
    for b in blocks:
        b["col"] = order[b["col"]]
    return blocks

def strip_header_footer(blocks: List[Dict[str,Any]], header_pct=0.08, footer_pct=0.08) -> List[Dict[str,Any]]:
    """
    Removes blocks that live inside the top/bottom bands of each page.
    """
    out = []
    by_page = {}
    for b in blocks:
        by_page.setdefault(b["page"], []).append(b)
    for p, arr in by_page.items():
        # page size is same for all blocks in page
        page_h = arr[0]["size"].height
        head_y = page_h*header_pct
        foot_y = page_h*(1-footer_pct)
        for b in arr:
            y0 = b["bbox"][1]; y1 = b["bbox"][3]
            if y1 <= head_y or y0 >= foot_y:
                continue  # drop headers/footers
            out.append(b)
    return out

def order_blocks(blocks: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    """
    Reading order: sort by (page, col, y_top, x_left).
    """
    return sorted(blocks, key=lambda b: (b["page"], b["col"], b["bbox"][1], b["bbox"][0]))
