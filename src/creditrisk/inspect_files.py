"""Day 1: profile every raw data file and draft a field dictionary.

Usage:
    python inspect_files.py --data-dir data/raw --out docs/field_dictionary_draft.md
    python inspect_files.py --data-dir data/raw --no-header      # if files have no header row
    python inspect_files.py --data-dir data/raw --sep "|"        # force delimiter

Everything is read as strings, so nothing is silently coerced. Nothing here
assumes column names; if there is no header you get col_0, col_1, ...
Then YOU fill in semantic name, units, and availability time from the
vendor's documentation.
"""
import argparse
import csv
from pathlib import Path

import pandas as pd

EXTS = {".csv", ".txt", ".dat", ".psv", ".tsv", ".gz", ".zip"}


def sniff_sep(path: Path, forced):
    if forced:
        return forced
    with open(path, "r", errors="replace") as f:
        sample = f.read(20000)
    try:
        return csv.Sniffer().sniff(sample, delimiters="|,;\t").delimiter
    except csv.Error:
        return "|"


def profile_file(path: Path, sep, header, nrows):
    sep = sniff_sep(path, sep) if path.suffix not in {".gz", ".zip"} else (sep or "|")
    df = pd.read_csv(
        path, sep=sep, header=0 if header else None, dtype=str,
        nrows=nrows, keep_default_na=False, na_values=[""], low_memory=False,
    )
    if not header:
        df.columns = [f"col_{i}" for i in range(df.shape[1])]
    return sep, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--out", default="docs/field_dictionary_draft.md")
    ap.add_argument("--sep", default=None)
    ap.add_argument("--no-header", action="store_true")
    ap.add_argument("--nrows", type=int, default=None, help="limit rows for huge files")
    args = ap.parse_args()

    files = sorted(p for p in Path(args.data_dir).rglob("*") if p.suffix.lower() in EXTS)
    if not files:
        raise SystemExit(f"No data files found under {args.data_dir}")

    lines = ["# Field dictionary (DRAFT)\n",
             "Fill the blank columns from the official documentation. "
             "Do not guess.\n"]
    for p in files:
        sep, df = profile_file(p, args.sep, not args.no_header, args.nrows)
        lines.append(f"\n## {p.name}\n")
        lines.append(f"- delimiter: `{sep}` | rows: {len(df):,} | columns: {df.shape[1]}")
        lines.append("- header present: " + ("yes" if not args.no_header else "no (generic names)"))
        lines.append("\n| raw name | semantic name | type | units | available at (orig / later) | "
                     "missing % | distinct | sample values | special codes |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for c in df.columns:
            s = df[c]
            miss = 100 * s.isna().mean()
            nun = s.nunique(dropna=True)
            sample = ", ".join(map(str, s.dropna().unique()[:5]))
            lines.append(f"| {c} |  |  |  |  | {miss:.1f} | {nun:,} | {sample} |  |")
        print(f"{p.name}: {len(df):,} rows x {df.shape[1]} cols (sep={sep!r})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
