"""Batch scoring: score a large file of loans with an exported model, in chunks.

Usage (from repo root):
  py -m creditrisk.batch --input loans.parquet --output scores.parquet --model-dir deploy/model
Input: CSV or parquet with the model's raw feature columns (and optionally loan_id and, for evaluation, target).
Output: <output> with loan_id, row, pd, score, decision, reason_code_1.. and the model version; <stem>_rejects.csv with
one line per validation failure (rows that fail the range and allowed-value checks are set aside, not scored);
<stem>_summary.json with counts, decision shares, throughput and, if the input has a target column, the observed
default rate in each decision band, which is the evidence for setting the cutoffs in configs/project.yaml.
Scoring uses the same ScoringService as the API. The output appears only when the whole run has finished.
Exit code 2 if the share of rejected rows exceeds --max-reject-rate.
"""
import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from creditrisk.lineage import sha256_file
from creditrisk.serving import ScoringService
from creditrisk.validation import DataValidationError, invalid_rows


def read_chunks(path, chunk_size):
    path = str(path)
    if path.lower().endswith(".csv"):
        yield from pd.read_csv(path, chunksize=chunk_size)
    else:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=chunk_size):
            yield batch.to_pandas()


def result_frame(service, scored, df, offset, stamp):
    k = service.max_reasons
    out = pd.DataFrame({"loan_id": df["loan_id"].astype(str).to_numpy() if "loan_id" in df else "",
                        "row": offset + df.index.to_numpy(), "pd": scored["pd"], "score": scored["score"],
                        "decision": scored["decision"].astype(str)})
    for i in range(k):
        out[f"reason_code_{i + 1}"] = [service.codes[r[i][0]]["code"] if len(r) > i else "" for r in scored["reasons"]]
    out["model_name"] = str(service.metadata.get("registry_name"))
    out["model_version"] = int(service.metadata.get("registry_version") or 0)
    out["scored_at_utc"] = stamp
    return out


class Writer:
    def __init__(self, path):
        self.final, self.tmp, self.writer, self.first = Path(path), Path(f"{path}.partial"), None, True

    def write(self, frame):
        if str(self.final).lower().endswith(".csv"):
            frame.to_csv(self.tmp, mode="w" if self.first else "a", header=self.first, index=False)
        else:
            table = pa.Table.from_pandas(frame, preserve_index=False)
            self.writer = self.writer or pq.ParquetWriter(self.tmp, table.schema)
            self.writer.write_table(table)
        self.first = False

    def close(self, ok=True):
        if self.writer:
            self.writer.close()
        if self.tmp.exists():
            os.replace(self.tmp, self.final) if ok else self.tmp.unlink()


def run(model_dir, input_path, output_path, chunk_size=50_000, max_reject_rate=0.10, log=print):
    started, t0 = datetime.now(UTC), time.perf_counter()
    service = ScoringService(model_dir)
    stamp = started.isoformat(timespec="seconds")
    output_path = Path(output_path)
    stem = output_path.with_suffix("")
    rejects_path, summary_path = Path(f"{stem}_rejects.csv"), Path(f"{stem}_summary.json")
    writer, rejects, pds = Writer(output_path), [], []
    bands, targets = {}, {}
    read = scored_n = 0
    try:
        for df in read_chunks(input_path, chunk_size):
            df = df.reset_index(drop=True)
            bad, failures = invalid_rows(df, service.cfg, service.spec)
            if len(failures):
                ids = df["loan_id"].astype(str) if "loan_id" in df else pd.Series("", index=df.index)
                failures = failures.assign(loan_id=failures.row.map(ids), row=failures.row + read)
                rejects.append(failures[["row", "loan_id", "column", "check", "failure_case"]])
            good = df.drop(index=bad)
            if len(good):
                scored = service.score(good)
                out = result_frame(service, scored, good, read, stamp)
                writer.write(out)
                pds.append(scored["pd"].to_numpy())
                for band, n in out.decision.value_counts().items():
                    bands[band] = bands.get(band, 0) + int(n)
                if "target" in good:
                    t = pd.DataFrame({"decision": out.decision.to_numpy(), "target": good["target"].to_numpy()})
                    for band, g in t.groupby("decision"):
                        d, e = targets.get(band, (0, 0))
                        targets[band] = (d + len(g), e + int(g.target.sum()))
                scored_n += len(good)
            read += len(df)
            log(f"scored {scored_n:,} of {read:,} rows")
    except BaseException:
        writer.close(ok=False)
        raise
    writer.close()
    rej = pd.concat(rejects, ignore_index=True) if rejects else pd.DataFrame(
        columns=["row", "loan_id", "column", "check", "failure_case"])
    rej.to_csv(rejects_path, index=False)
    rejected = int(rej.row.nunique()) if len(rej) else 0
    seconds = time.perf_counter() - t0
    p = np.concatenate(pds) if pds else np.array([])
    model = {**service.describe()["lineage"], "name": service.metadata.get("registry_name"),
             "version": service.metadata.get("registry_version"), "kind": service.kind,
             "scorer_sha256": service.metadata["scorer_sha256"]}
    summary = {
        "status": "ok" if read and rejected / read <= max_reject_rate else "rejects_above_threshold",
        "input": {"file": str(input_path), "sha256": sha256_file(input_path), "rows": read},
        "rows_scored": scored_n, "rows_rejected": rejected,
        "reject_causes": rej.groupby(["column", "check"]).size().rename("count").reset_index().to_dict("records"),
        "model": model,
        "decisions": {b: {"loans": n, "share": round(n / scored_n, 4)} for b, n in sorted(bands.items())},
        "pd": ({"mean": float(p.mean()), **{f"p{q}": float(np.quantile(p, q / 100)) for q in (50, 90, 99)}}
               if len(p) else None),
        "observed_default_rate_by_decision": {b: {"loans": n, "defaults": e, "default_rate": round(e / n, 5)}
                                              for b, (n, e) in sorted(targets.items())} or None,
        "chunk_size": chunk_size, "started_utc": stamp, "seconds": round(seconds, 2),
        "rows_per_second": round(read / seconds, 1) if seconds else None,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", "deploy/model"))
    ap.add_argument("--chunk-size", type=int, default=50_000)
    ap.add_argument("--max-reject-rate", type=float, default=0.10)
    args = ap.parse_args(argv)
    try:
        s = run(args.model_dir, args.input, args.output, args.chunk_size, args.max_reject_rate)
    except DataValidationError as err:
        raise SystemExit(f"batch aborted: {err}") from None
    print(f"Scored {s['rows_scored']:,} loans, rejected {s['rows_rejected']:,}, "
          f"{s['rows_per_second']} rows/s -> {args.output}")
    if s["status"] != "ok":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
