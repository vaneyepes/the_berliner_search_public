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
