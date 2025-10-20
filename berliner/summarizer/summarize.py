from pathlib import Path
import json, time
from typing import Iterable, List, Dict
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Helper functions for reading and writing jsonl files I/O


def _iter_jsonl(p: Path) -> Iterable[Dict]:
    """Stream JSONL records to avoid loading entire file in memory."""
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def _write_jsonl(records: List[Dict], p: Path):
    """Write records as JSONL; ensures parent dir exists."""
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def load_model(model_name: str, device_pref: str = "auto"):
    """Load a seq2seq model and tokenizer from HuggingFace."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    if device_pref == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_pref
    model.to(device)
    return tokenizer, model, device


# Core summarization routine
def summarize_issue(
    issue_path: Path,
    out_dir: Path,
    model_name: str,
    max_input_tokens: int,
    max_new_tokens: int,
    num_beams: int,
    batch_size: int,
    device: str = "auto",
) -> Path:
    issue_id = issue_path.stem
    out_path = out_dir / f"{issue_id}.jsonl"

    tokenizer, model, device = load_model(model_name, device)
    records = list(_iter_jsonl(issue_path))

    summaries = []
    t0 = time.time()

    # simple micro-batching (safe for CPU; increase if on GPU)
    for i in tqdm(range(0, len(records), batch_size), desc=f"summarize:{issue_id}"):
        batch = records[i:i+batch_size]
        texts = [r["text"] for r in batch]
        inputs = tokenizer(
            texts,
            truncation=True,
            max_length=max_input_tokens,
            padding=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            gen_tokens = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=True
            )
        outs = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

        for r, s in zip(batch, outs):
            summaries.append({
                "issue_id": r.get("issue_id"),
                "chunk_id": r.get("chunk_id"),
                "page_span": r.get("page_span"),
                "word_count": r.get("word_count"),
                "summary": s.strip(),
                "model": model_name,
                "gen": {
                    "max_input_tokens": max_input_tokens,
                    "max_new_tokens": max_new_tokens,
                    "num_beams": num_beams
                }
            })

    runtime_s = round(time.time() - t0, 3)
    meta = {
        "_meta": {
            "issue_id": issue_id,
            "n_chunks": len(records),
            "model": model_name,
            "device": device,
            "runtime_s": runtime_s
        }
    }
    # append meta as last line
    _write_jsonl(summaries + [meta], out_path)
    return out_path

def aggregate_issue_summary(issue_summary_file: Path, model_name: str, device: str = "auto",
                            max_input_tokens: int = 512, max_new_tokens: int = 120, num_beams: int = 4) -> str:
    """Optional: turn many chunk summaries into one issue-level summary."""
    tokenizer, model, device = load_model(model_name, device)
    chunk_summaries = []
    for rec in _iter_jsonl(issue_summary_file):
        if "_meta" in rec:  # skip meta line
            continue
        chunk_summaries.append(rec["summary"])
    text = " ".join(chunk_summaries)
    inputs = tokenizer(text, truncation=True, max_length=max_input_tokens, return_tensors="pt").to(device)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=num_beams, early_stopping=True)
    return tokenizer.decode(gen[0], skip_special_tokens=True).strip()
from pathlib import Path
import json, time
from typing import Iterable, List, Dict, Optional
from dataclasses import dataclass
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# -----------------------------
# JSONL I/O helpers
# -----------------------------
def _iter_jsonl(p: Path) -> Iterable[Dict]:
    """Stream JSONL records to avoid loading entire file in memory."""
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def _write_jsonl(records: List[Dict], p: Path):
    """Write records as JSONL; ensures parent dir exists."""
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -----------------------------
# Config (optional typed holder)
# -----------------------------
@dataclass
class GenConfig:
    model_name: str
    max_input_tokens: int
    max_new_tokens: int
    num_beams: int
    batch_size: int
    device: str = "auto"
    skip_if_exists: bool = True
    t5_prefix: str = "summarize: "  # only used for T5 models


# -----------------------------
# Model / device helpers
# -----------------------------
def _pick_device(pref: str = "auto") -> str:
    """Choose best device based on availability and user preference."""
    if pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return pref

def _is_t5(model_name: str) -> bool:
    """Minimal heuristic to decide whether to prepend T5 instruction prefix."""
    return model_name.lower().startswith("t5")

def load_model(model_name: str, device_pref: str = "auto"):
    """Load a seq2seq model and tokenizer from HuggingFace and move to device."""
    device = _pick_device(device_pref)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)
    model.eval()  # inference mode
    return tokenizer, model, device


# -----------------------------
# Core summarization routine
# -----------------------------
def summarize_issue(
    issue_path: Path,
    out_dir: Path,
    model_name: str,
    max_input_tokens: int,
    max_new_tokens: int,
    num_beams: int,
    batch_size: int,
    device: str = "auto",
    skip_if_exists: bool = True,
    t5_prefix: str = "summarize: ",
) -> Optional[Path]:
    """
    Summarize all chunks in one issue JSONL and write to output JSONL.
    Returns the output path or None if skipped (already exists).
    """
    issue_id = issue_path.stem
    out_path = out_dir / f"{issue_id}.jsonl"

    # Idempotency: let long runs be restartable
    if skip_if_exists and out_path.exists():
        return None

    tokenizer, model, device = load_model(model_name, device)
    records = list(_iter_jsonl(issue_path))

    summaries: List[Dict] = []
    t0 = time.time()

    # Precompute flags/config once
    use_t5 = _is_t5(model_name)
    gen_cfg = {
        "max_input_tokens": max_input_tokens,
        "max_new_tokens": max_new_tokens,
        "num_beams": num_beams
    }

    # Micro-batching (safe for CPU; bump if on GPU)
    rng = range(0, len(records), batch_size)
    for i in tqdm(rng, desc=f"summarize:{issue_id}"):
        batch = records[i:i+batch_size]

        # Model-specific input preparation
        texts = [r["text"] for r in batch]
        if use_t5:
            # T5 benefits from the task prefix for better summarization behavior
            texts = [t5_prefix + t for t in texts]

        # Tokenize with truncation to stay within model context window
        inputs = tokenizer(
            texts,
            truncation=True,
            max_length=max_input_tokens,
            padding=True,
            return_tensors="pt"
        ).to(device)

        # Robust inference: one bad batch shouldn't kill the whole run
        try:
            with torch.inference_mode():
                gen_tokens = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    early_stopping=True
                )
            outs = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        except Exception as e:
            # Minimal error record; you can expand with logging if you prefer
            outs = ["[ERROR: generation failed]"] * len(batch)

        # Collect outputs with provenance
        for r, s in zip(batch, outs):
            summaries.append({
                "issue_id": r.get("issue_id", issue_id),
                "chunk_id": r.get("chunk_id"),
                "page_span": r.get("page_span"),
                "word_count": r.get("word_count"),
                "summary": s.strip(),
                "model": model_name,
                "gen": gen_cfg
            })

    runtime_s = round(time.time() - t0, 3)
    meta = {
        "_meta": {
            "issue_id": issue_id,
            "n_chunks": len(records),
            "model": model_name,
            "device": device,
            "runtime_s": runtime_s
        }
    }

    # Append meta as last line
    _write_jsonl(summaries + [meta], out_path)
    return out_path


def aggregate_issue_summary(
    issue_summary_file: Path,
    model_name: str,
    device: str = "auto",
    max_input_tokens: int = 512,
    max_new_tokens: int = 120,
    num_beams: int = 4,
    t5_prefix: str = "summarize: "
) -> str:
    """Combine chunk summaries into one issue-level summary."""
    tokenizer, model, device = load_model(model_name, device)
    use_t5 = _is_t5(model_name)

    chunk_summaries: List[str] = []
    for rec in _iter_jsonl(issue_summary_file):
        if "_meta" in rec:  # skip meta line
            continue
        chunk_summaries.append(rec["summary"])

    text = " ".join(chunk_summaries)
    if use_t5:
        text = t5_prefix + text

    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_input_tokens,
        return_tensors="pt"
    ).to(device)

    with torch.inference_mode():
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True
        )
    return tokenizer.decode(gen[0], skip_special_tokens=True).strip()
