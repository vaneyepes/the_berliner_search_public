import argparse, pathlib, hashlib
from berliner.extractor.parse import extract_issue, save_json

def main():
    ap = argparse.ArgumentParser(prog="berliner")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("extract", help="Extract one PDF to JSON")
    p1.add_argument("pdf", type=str)
    p1.add_argument("-o", "--out", type=str, default="data/json")

    args = ap.parse_args()
    if args.cmd == "extract":
        pdf = pathlib.Path(args.pdf)
        data = extract_issue(str(pdf))
        h = hashlib.sha1(pdf.name.encode()).hexdigest()[:8]
        out = save_json(data, args.out, stem=f"{pdf.stem}.{h}")
        print(f"Saved {out}")

if __name__ == "__main__":
    main()
# manually one file: CLI: python -m berliner.cli extract "data/raw_pdfs/TheBerliner243_2025_08_21.pdf" -o data/json/

#for all files: run in terminal:
"""python - <<'PY'
import pathlib, subprocess, shlex

in_dir = pathlib.Path("data/raw_pdfs")
out_dir = pathlib.Path("data/json"); out_dir.mkdir(exist_ok=True)

pdfs = sorted([p for p in in_dir.iterdir() if p.suffix.lower()==".pdf"])
print(f"Found {len(pdfs)} PDFs.")
ok = 0
for pdf in pdfs:
    cmd = f'python -m berliner.cli extract {shlex.quote(str(pdf))} -o {shlex.quote(str(out_dir))}'
    print(">>", cmd)
    r = subprocess.run(cmd, shell=True)
    ok += (r.returncode == 0)
print(f"Done. {ok}/{len(pdfs)} succeeded.")
PY
"""