"""Export the per-iteration logs of result pickles to CSV and JSON.

For each run of ``run.PAPER_ITEMS`` (or every run folder with ``--all``) the
latest pickle below ``--results_dir`` is exported to
``<out>/<same relative folder>/<timestamp>.csv`` (one row per log entry, all
per-iteration keys, floats written with full ``repr`` precision) together with
a ``run.json`` holding the run summary (stopping criterion, iterations, time,
final cost and gradient norm, stopping-criteria parameters, optimizer, beta
rule and line searcher). ``analysis_new.py --source csv`` reads this export.

Usage::

    python src/experiments/analysis/export_logs.py --results_dir results --out data/arxiv_v1
"""

import argparse
import csv
import glob
import json
import os
import pickle
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_SRC_DIR))

import jax

# Enable float64 before unpickling: with x64 disabled, JAX silently truncates
# the stored float64 arrays to float32 when they are restored.
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

import numpy as np

from run import PAPER_ITEMS, result_folder


def _scalar(v):
    """Python scalar of a logged value (None for missing values)."""
    if v is None:
        return None
    a = np.asarray(v)
    if a.dtype.kind in "iub":
        return int(a)
    return float(a)


def _csv_value(v):
    v = _scalar(v)
    if v is None:
        return ""
    return repr(v) if isinstance(v, float) else str(v)


def _json_value(v):
    """JSON-safe scalar (non-finite floats as strings)."""
    v = _scalar(v)
    if isinstance(v, float) and not np.isfinite(v):
        return str(v)
    return v


def export_run(pkl_path, results_dir, out_dir):
    """Export one result pickle; return the path of the written CSV."""
    with open(pkl_path, "rb") as f:
        res = pickle.load(f)["res"]
    rel = os.path.relpath(os.path.dirname(pkl_path), results_dir)
    dest = os.path.join(out_dir, rel)
    os.makedirs(dest, exist_ok=True)
    timestamp = os.path.basename(pkl_path)[: -len(".pkl")]

    it = res.log["iterations"]
    keys = list(it.keys())
    n = len(it[keys[0]])
    csv_path = os.path.join(dest, f"{timestamp}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(keys)
        for i in range(n):
            writer.writerow([_csv_value(it[key][i]) for key in keys])

    params = res.log.get("optimizer_parameters") or {}
    line_searcher = params.get("line_searcher")
    info = {
        "timestamp": timestamp,
        "optimizer": res.log.get("optimizer"),
        "beta_rule": params.get("beta_rule"),
        "line_searcher": (
            type(line_searcher).__name__ if line_searcher is not None else None
        ),
        "stopping_criteria": res.log.get("stopping_criteria"),
        "stopping_criterion": res.stopping_criterion,
        "iterations": res.iterations,
        "time": res.time,
        "final_cost": _json_value(res.cost),
        "final_gradient_norm": _json_value(res.gradient_norm),
        "n_log_entries": n,
    }
    with open(os.path.join(dest, "run.json"), "w") as f:
        json.dump(info, f, indent=1)
    return csv_path


def latest_pickles(results_dir, export_all=False):
    """Latest pickle of each run folder (paper runs, or all with export_all)."""
    if export_all:
        folders = sorted(
            {
                os.path.dirname(p)
                for p in glob.glob(f"{results_dir}/**/*.pkl", recursive=True)
            }
        )
    else:
        folders = []
        for runs in PAPER_ITEMS.values():
            for run in runs:
                matches = glob.glob(os.path.join(results_dir, result_folder(run)))
                if len(matches) != 1:
                    raise FileNotFoundError(
                        f"Expected one folder for {result_folder(run)}, found {matches}"
                    )
                if matches[0] not in folders:
                    folders.append(matches[0])
    pkls = []
    for folder in folders:
        files = sorted(glob.glob(os.path.join(folder, "*.pkl")))
        if not files:
            raise FileNotFoundError(f"No pickle in {folder}")
        pkls.append(files[-1])
    return pkls


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results_dir", required=True, help="Results directory")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export every run folder, not only PAPER_ITEMS",
    )
    args = parser.parse_args(argv)
    results_dir = os.path.abspath(args.results_dir)
    pkls = latest_pickles(results_dir, args.all)
    for pkl in pkls:
        print(export_run(pkl, results_dir, os.path.abspath(args.out)))
    print(f"Exported {len(pkls)} runs.")


if __name__ == "__main__":
    main()
