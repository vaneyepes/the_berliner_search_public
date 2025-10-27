from pathlib import Path
import csv

# automatically find your latest run
base = Path("eval")
runs = sorted(base.rglob("labels.tsv"))
assert runs, "No labels.tsv found"
labels_path = runs[-1]
print("Fixing:", labels_path)

lines = labels_path.read_text(encoding="utf-8").splitlines()
if not lines:
    raise ValueError("File is empty")

header = lines[0].split("\t")
print("Current header columns:", header)

# create a correct header
correct_header = ["query", "rank", "issue", "chunk", "snippet", "is_relevant", "is_ad"]

# if header differs, rewrite it
if header != correct_header:
    print("⚙️ Rewriting header...")
    lines[0] = "\t".join(correct_header)
    labels_path.write_text("\n".join(lines), encoding="utf-8")
    print("✅ Header fixed.")
else:
    print("✅ Header already correct.")
