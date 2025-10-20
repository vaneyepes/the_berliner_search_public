## Notes: For extractor: How to run the CLI commands from the terminal

"""# make sure click + pyyaml are installed in your venv
source .venv/bin/activate
pip install click pyyaml

# extract one PDF

python -m berliner.cli extract "data/raw_pdfs/TheBerliner243_2025_08_21.pdf" -o data/json/

# or extract a whole folder

python -m berliner.cli extract "data/raw_pdfs/" -o data/json/

# (after adding berliner/chunker.py)

python -m berliner.cli chunk "data/json/TheBerliner243_2025_08_21.json" -o data/chunks/
python -m berliner.cli chunk "data/json/" -o data/chunks/

# For Chunking: Run from terminal

python -m berliner.cli chunk data/json/ -o data/chunks/

# For Summarization:

python -m berliner.cli summarize data/chunks/

# Generate MD files: from summaries

python - <<'PY'
from pathlib import Path
from berliner.summarizer.export_md import write_issue_md
summ_dir = Path("data/summaries")
out_dir = Path("data/summaries_issue")
for f in sorted(summ_dir.glob("\*.jsonl")):
p = write_issue_md(f, out_dir)
print("Wrote", p)
PY
