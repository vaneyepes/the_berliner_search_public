# berlin-aware query expansion for keyword branch (BM25)
ALIASES = {
    "berlin brandenburg airport": [
        "berlin brandenburg airport",
        "berlin airport",
        "brandenburg airport",
        "ber airport",
        "flughafen berlin brandenburg",
        "willy brandt flughafen",
        "willy-brandt-flughafen",
        "bbi",  # old project name
    ],
    "delay": ["delay", "delays", "postponed", "postponement", "verschoben", "verzögerung", "verzögerungen"],
}

def expand_for_bm25(query: str) -> list[str]:
    q = query.lower()
    expanded = [query]
    # airport bundle
    if "airport" in q or "brandenburg" in q or "berlin brandenburg" in q:
        expanded.extend(ALIASES["berlin brandenburg airport"])
    # delay bundle
    if "delay" in q or "postpon" in q or "verschob" in q or "verzöger" in q:
        expanded.extend(ALIASES["delay"])
    # de-duplicate and keep originals first
    seen = set(); out=[]
    for t in expanded:
        if t not in seen:
            out.append(t); seen.add(t)
    return out
