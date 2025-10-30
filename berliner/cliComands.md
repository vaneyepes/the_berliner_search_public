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

# Metadata Tagging structure (Create Meta schema)

python -m berliner.cli metadata

# Semantic search embeddings and index

# Rebuild with E5-base (instruction-tuned)

python -m berliner.cli search index --index-type flat --model-name "intfloat/e5-base-v2" --batch-size 64

# Query with hybrid on and (optionally) rerank

python -m berliner.cli search query "Brandenburg airport delays" -k 10 --faiss-top 100 --rerank

# Semanting search emb and index FOR SERVER version

Server setting (better quality):

# Option 1 (very strong): MPNet

python -m berliner.cli search index --index-type flat --model-name "sentence-transformers/all-mpnet-base-v2"

# Option 2 (instruction-tuned, stronger than base): E5-large (slower)

python -m berliner.cli search index --index-type flat --model-name "intfloat/e5-large-v2"

# Runing Queries in Model A (MPNet)

# Run your 6 probes quickly

while read -r q; do
echo "===================="
echo "QUERY: $q"
  python -m berliner.cli search "$q" --model "sentence-transformers/multi-qa-mpnet-base-dot-v1" -k 10
echo
done << 'EOF'
Berlin airport delays
BER flughafen verspätung sicherheitskontrolle
Neukölln gentrification rent eviction
Mietendeckel Neukölln Räumung
Tegel closure history
Schönefeld to BER transition delays
EOF

# Runing Queries in Model B (MiniLM-multilingual)

while read -r q; do
echo "===================="
echo "QUERY: $q"
  python -m berliner.cli search "$q" --model "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" -k 10
echo
done << 'EOF'
Berlin airport delays
BER flughafen verspätung sicherheitskontrolle
Neukölln gentrification rent eviction
Mietendeckel Neukölln Räumung
Tegel closure history
Schönefeld to BER transition delays
EOF

# Runing one query in model b (final ersion miniLM)

python -m berliner.cli search "Neukölln gentrification rent eviction" \
 --model "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" \
 -k 10

# Streamlit app run:

streamlit run "$(pwd)/berliner/ui/app_streamlit.py"

TOKENIZERS_PARALLELISM=false PYTORCH_ENABLE_MPS_FALLBACK=1 \
streamlit run berliner/ui/app_streamlit.py
