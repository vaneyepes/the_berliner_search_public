import re, datetime, pathlib

SOFT_HYPHEN = "\u00AD"

def now_utc_iso():
    return datetime.datetime.utcnow().isoformat() + "Z"

def clean_soft_hyphens(s: str) -> str:
    return s.replace(SOFT_HYPHEN, "")

def dehyphenate(text: str) -> str:
    # join words split across line breaks: "govern-\nment" -> "government"
    # keep real hyphenated compounds (next token capitalized or digit/uppercase → keep)
    lines = text.splitlines()
    out = []
    for i, ln in enumerate(lines):
        if ln.rstrip().endswith("-") and i + 1 < len(lines):
            nextln = lines[i+1].lstrip()
            # if next starts lower-case letter, join
            if nextln and nextln[:1].islower():
                out.append(ln.rstrip()[:-1] + nextln)
                lines[i+1] = ""  # consume next line
                continue
        out.append(ln)
    s = "\n".join([l for l in out if l != ""])
    # collapse excess whitespace
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return clean_soft_hyphens(s)

def infer_meta_from_name(fname: str) -> dict:
    # Example: TheBerliner243_2025_08_21.pdf or TheBerliner_2017_05.pdf
    import re
    year = month = issue = None
    m = re.search(r"(20\d{2})[^\d]?([01]?\d)?", fname)
    if m:
        year = int(m.group(1))
        if m.group(2):
            try:
                month = int(m.group(2))
            except:
                month = None
    m2 = re.search(r"Berliner(\d+)", fname, re.I)
    if m2:
        issue = int(m2.group(1))
    return {"year": year, "month": month, "issue": issue, "source_file": fname}

def band_mask(y0, y1, page_h, pct=0.08):
    # header/footer band in PDF coordinates
    head = (0, page_h*(1-pct), "foot")  # dummy
    return (page_h*pct, page_h*(1-pct))
