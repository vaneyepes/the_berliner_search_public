# berliner/chunker/make_chunks.py
from __future__ import annotations
import json, re
from pathlib import Path
from typing import List, Dict, Tuple
import argparse

# === Defaults ===
PAGE_BREAK_TOKEN = "<PAGE_BREAK>"
DEFAULT_CHUNK_WORDS = 900
DEFAULT_OVERLAP_WORDS = 120
MIN_CHUNK_WORDS = 400  # merge tail if smaller


# === Core helpers ===
def _load_issue_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _normalize_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()

def _split_pages(text: str) -> List[str]:
    text = text.replace(PAGE_BREAK_TOKEN, f"\n{PAGE_BREAK_TOKEN}\n")
    parts = [t.strip() for t in text.split(PAGE_BREAK_TOKEN)]
    return [p for p in parts if p.strip()]

def _rough_sentences(paragraph: str) -> List[str]:
    # lightweight sentence splitter
    s = re.split(r"(?<=[\.\!\?])\s+(?=[A-ZÄÖÜ])|[\r\n]+", paragraph.strip())
    return [x.strip() for x in s if x.strip()]

def _words(s: str) -> List[str]:
    return re.findall(r"\b\w[\w\-’']*\b", s)


# === Chunking logic ===
def _take_tail_by_words(sentences: List[Tuple[int, str]], words_needed: int) -> List[Tuple[int, str]]:
    acc, total = [], 0
    for pg, s in reversed(sentences):
        w = len(_words(s))
        acc.append((pg, s))
        total += w
        if total >= words_needed:
            break
    return list(reversed(acc))

def _finalize(sentences: List[Tuple[int, str]]) -> Dict:
    text = " ".join(s for _, s in sentences)
    text = _normalize_whitespace(text)
    return {"_sentences": sentences, "text": text}

def _chunk_sentences(
    sentences: List[Tuple[int, str]],
    chunk_words=DEFAULT_CHUNK_WORDS,
    overlap_words=DEFAULT_OVERLAP_WORDS
) -> List[Dict]:
    chunks = []
    cur, cur_wc = [], 0

    for pg, s in sentences:
        wc = len(_words(s))
        if cur_wc + wc <= chunk_words or not cur:
            cur.append((pg, s)); cur_wc += wc
        else:
            chunks.append(_finalize(cur))
            tail = _take_tail_by_words(cur, overlap_words) if overlap_words > 0 else []
            cur = tail + [(pg, s)]
            cur_wc = sum(len(_words(ts)) for _, ts in tail) + wc

    if cur:
        if chunks and sum(len(_words(s)) for _, s in cur) < MIN_CHUNK_WORDS:
            last = chunks.pop()
            merged = last["_sentences"] + cur
            chunks.append(_finalize(merged))
        else:
            chunks.append(_finalize(cur))

    for i, ch in enumerate(chunks, start=1):
        ch["chunk_index"] = i
        pages = [pg for pg, _ in ch["_sentences"]]
        ch["page_span"] = [min(pages), max(pages)]
        ch["word_count"] = len(_words(ch["text"]))
        del ch["_sentences"]

    return chunks


# === File-level ops ===
def _issue_id_from_meta(meta: Dict, fallback_name: str) -> str:
    for key in ("issue_id", "date", "slug", "name"):
        if key in meta and meta[key]:
            return str(meta[key]).replace(" ", "_")
    return Path(fallback_name).stem

def chunk_issue_file(
    in_path: Path, out_dir: Path,
    chunk_words=DEFAULT_CHUNK_WORDS,
    overlap_words=DEFAULT_OVERLAP_WORDS
) -> Path:
    data = _load_issue_json(in_path)
    meta = data.get("meta", {})
    raw_text = data.get("text", "")
    text = _normalize_whitespace(raw_text)
    pages = _split_pages(text)

    sentences: List[Tuple[int, str]] = []
    for i, page in enumerate(pages, start=1):
        for s in _rough_sentences(page):
            if s:
                sentences.append((i, s))

    chunks = _chunk_sentences(sentences, chunk_words, overlap_words)
    issue_id = _issue_id_from_meta(meta, in_path.name)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{issue_id}.jsonl"

    with out_path.open("w", encoding="utf-8") as f:
        for i, ch in enumerate(chunks, start=1):
            rec = {
                "issue_id": issue_id,
                "chunk_id": f"{issue_id}_{i:04d}",
                "chunk_index": ch["chunk_index"],
                "page_span": ch["page_span"],
                "word_count": ch["word_count"],
                "text": ch["text"],
                "source_file": str(in_path)
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return out_path

def chunk_dir(
    in_dir: Path, out_dir: Path,
    chunk_words=DEFAULT_CHUNK_WORDS,
    overlap_words=DEFAULT_OVERLAP_WORDS
) -> List[Path]:
    outputs = []
    for p in sorted(in_dir.glob("*.json")):
        outputs.append(chunk_issue_file(p, out_dir, chunk_words, overlap_words))
    return outputs


# === CLI entrypoint ===
def main():
    ap = argparse.ArgumentParser(description="Phase 3: Chunk The Berliner issues into JSONL")
    ap.add_argument("input_dir", type=str, help="Folder with issue JSON files (e.g., data/json/)")
    ap.add_argument("-o", "--output_dir", type=str, default="data/chunks/", help="Output folder for .jsonl files")
    ap.add_argument("--chunk_words", type=int, default=DEFAULT_CHUNK_WORDS)
    ap.add_argument("--overlap_words", type=int, default=DEFAULT_OVERLAP_WORDS)
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    outs = chunk_dir(in_dir, out_dir, args.chunk_words, args.overlap_words)

    print(f"✅ Chunked {len(outs)} issues → {out_dir}")

if __name__ == "__main__":
    main()
