"""Run the numerical experiments of the paper.

Each item below lists the runs behind one figure or table of the paper
(arXiv:2609.37103); runs shared between items are executed once.

=======  ==================================  ==========================================
Item     Paper label                         Runs
=======  ==================================  ==========================================
exp1     Fig. ``fig:exp1_CGvsSD``            harmonic, N=399, S+V, R-NLCG and R-GD;
                                             TT r=5,10,15, full a_u, full H^1
exp2     Tab. ``tab:exp2_precond``           harmonic, N=399, R-NLCG, preconditioner
                                             None/S/S+V; TT r=5,10,15, full a_u
exp3     Fig. ``fig:exp3_fullvsTT``          harmonic, R-NLCG, S+V; N=399: TT r=5,10,15,
                                             full a_u, full H^1; N=799: TT only
exp4     Fig. ``fig:exp4_lattice``           lattice, R-NLCG, S+V; N=399: TT r=5,10,15,
                                             full a_u, full H^1; N=799: TT only
exp5     Fig. ``fig:exp5_anderson``          Anderson (2D), R-NLCG, S+V, N=1599,3199,6399;
                                             TT r=20,40,50,60,80, full a_u, full H^1
exp9     Fig. ``fig:exp9_multicomponent``    multicomponent (2D, p=3), R-NLCG, S+V,
                                             N=1023,2047; TT r=5,10,15,20, full a_u
tab9     Tab. ``tab:exp9_multicomponent``    multicomponent, N=2047, S+V, R-NLCG and
                                             R-GD; TT r=5,10,15,20, full a_u
ref9     reference energy E* of exp9         multicomponent, N=4095, R-NLCG, S+V;
                                             TT r=5,10,15,20, full a_u
=======  ==================================  ==========================================

The code's ``N`` is the paper's ``n - 1`` (e.g. ``N=399`` is ``n=400``); all
runs use spectral elements of degree ``k=4`` in float64. Running every item
takes about 30 hours on an RTX 4090 (dominated by the full-format runs).

Usage::

    python src/run.py                      # all 70 runs (skips existing results)
    python src/run.py --only exp1 exp2     # selected items
    python src/run.py --smoke              # every configuration, small sizes, 5 log entries
    python src/run.py --dry_run            # print the commands only

Results are written to ``--results_dir`` (default: ``$TTGPE_RESULTS_DIR`` if set,
else ``<repo>/results``; ``<repo>/results_smoke`` with ``--smoke``). The figures
and tables are produced by ``src/experiments/analysis/analysis_new.py``.
"""

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
_REPO_DIR = _SRC_DIR.parent

HARMONIC = "experiments/paper/paperexp_harmonic.py"
LATTICE = "experiments/paper/paperexp_lattice.py"
ANDERSON = "experiments/paper/paperexp_anderson.py"
MULTICOMPONENT = "experiments/paper/paperexp_multicomponent_2d.py"

CG = "ConjugateGradient"  # R-NLCG in the paper
SD = "SteepestDescent"  # R-GD in the paper

# Problem parameters per experiment script: (dim, beta, init). init=None uses
# the script's default (Thomas-Fermi); the multicomponent runs start from a
# constant initial guess.
PROBLEMS = {
    HARMONIC: (3, 1000, None),
    LATTICE: (3, 1000, None),
    ANDERSON: (2, 10, None),
    MULTICOMPONENT: (2, 10, "constant"),
}

# Grid sizes used by --smoke (one per script).
SMOKE_N = {HARMONIC: 63, LATTICE: 63, ANDERSON: 159, MULTICOMPONENT: 255}
SMOKE_MAX_ITERATIONS = 5


def _spec(script, method, N, optimizer, precond=None, tt_rank=None, k=4):
    """Return the specification of a single run as a dict."""
    dim, beta, init = PROBLEMS[script]
    return {
        "script": script,
        "method": method,
        "N": N,
        "k": k,
        "dim": dim,
        "beta": beta,
        "optimizer": optimizer,
        "precond": precond,
        "tt_rank": tt_rank,
        "max_tt_rank_u2": tt_rank,
        "init": init,
    }


def _runs(script, N, optimizers, preconds, tt_ranks=(), full_au=False, full_H1=False):
    """Runs for one grid size: TT ranks, full a_u and full H^1 per optimizer.

    full H^1 runs have no preconditioner and are added once per optimizer.
    """
    runs = []
    for precond in preconds:
        for optimizer in optimizers:
            for r in tt_ranks:
                runs.append(_spec(script, "tt", N, optimizer, precond, r))
            if full_au:
                runs.append(_spec(script, "full_au", N, optimizer, precond))
    if full_H1:
        for optimizer in optimizers:
            runs.append(_spec(script, "full_H1", N, optimizer))
    return runs


PAPER_ITEMS = {
    "exp1": _runs(
        HARMONIC, 399, [CG, SD], ["S+V"], [5, 10, 15], full_au=True, full_H1=True
    ),
    "exp2": _runs(HARMONIC, 399, [CG], ["None", "S", "S+V"], [5, 10, 15], full_au=True),
    "exp3": _runs(HARMONIC, 399, [CG], ["S+V"], [5, 10, 15], full_au=True, full_H1=True)
    + _runs(HARMONIC, 799, [CG], ["S+V"], [5, 10, 15]),
    "exp4": _runs(LATTICE, 399, [CG], ["S+V"], [5, 10, 15], full_au=True, full_H1=True)
    + _runs(LATTICE, 799, [CG], ["S+V"], [5, 10, 15]),
    "exp5": [
        run
        for N in [1599, 3199, 6399]
        for run in _runs(
            ANDERSON, N, [CG], ["S+V"], [20, 40, 50, 60, 80], full_au=True, full_H1=True
        )
    ],
    "exp9": [
        run
        for N in [1023, 2047]
        for run in _runs(
            MULTICOMPONENT, N, [CG], ["S+V"], [5, 10, 15, 20], full_au=True
        )
    ],
    "tab9": _runs(
        MULTICOMPONENT, 2047, [CG, SD], ["S+V"], [5, 10, 15, 20], full_au=True
    ),
    # Finer-grid runs that define the reference energy E* of the multicomponent
    # figure (not plotted).
    "ref9": _runs(MULTICOMPONENT, 4095, [CG], ["S+V"], [5, 10, 15, 20], full_au=True),
}


def smoke_runs(runs):
    """Reduced version of a list of runs for --smoke.

    The grid size is replaced by ``SMOKE_N`` of the script and only the
    smallest TT rank of the list is kept; duplicates are removed.
    """
    tt_ranks = [r["tt_rank"] for r in runs if r["method"] == "tt"]
    min_rank = min(tt_ranks) if tt_ranks else None
    out = []
    for r in runs:
        if r["method"] == "tt" and r["tt_rank"] != min_rank:
            continue
        s = dict(r, N=SMOKE_N[r["script"]])
        if s not in out:
            out.append(s)
    return out


def get_items(smoke=False):
    """Return PAPER_ITEMS, or its --smoke reduction."""
    if not smoke:
        return PAPER_ITEMS
    return {item: smoke_runs(runs) for item, runs in PAPER_ITEMS.items()}


def result_folder(run):
    """Glob pattern (relative to the results dir) of the folder of a run.

    Mirrors the layout written by experiments/experiment_run.py:
    ``{name}_{device}_{dtype}_dim{dim}/beta{beta}_N{N}_k{k}/{optimizer}/{method}``.
    """
    name = os.path.basename(run["script"]).replace("paperexp_", "").replace(".py", "")
    precond = run["precond"]
    precond_str = "noprecond" if precond in (None, "None") else "precond" + precond
    if run["method"] == "tt":
        method_folder = f"tt_au/ttrank{run['tt_rank']}_{precond_str}"
        if run["max_tt_rank_u2"] is not None:
            method_folder += f"_maxu2rank{run['max_tt_rank_u2']}"
    elif run["method"] == "full_H1":
        method_folder = "full_H1"
    else:
        method_folder = f"full_au_{precond_str}"
    beta = float(run["beta"])
    beta_str = f"{beta:.1f}" if beta == int(beta) else str(beta)
    return os.path.join(
        f"{name}_*_dim{run['dim']}",
        f"beta{beta_str}_N{run['N']}_k{run['k']}",
        run["optimizer"],
        method_folder,
    )


def results_exist(run, results_dir):
    """True if a result pickle of this run exists below results_dir."""
    pattern = os.path.join(str(results_dir), result_folder(run), "*.pkl")
    return bool(glob.glob(pattern))


def build_command(run, results_dir, max_iterations=None):
    """Command line running one experiment."""
    cmd = [
        sys.executable,
        str(_SRC_DIR / run["script"]),
        "--method",
        run["method"],
        "--N",
        str(run["N"]),
        "--k",
        str(run["k"]),
        "--dim",
        str(run["dim"]),
        "--beta",
        str(run["beta"]),
        "--pymanopt_optimizer",
        run["optimizer"],
    ]
    if run["precond"] is not None:
        cmd += ["--precond", run["precond"]]
    if run["tt_rank"] is not None:
        cmd += ["--tt_rank", str(run["tt_rank"])]
    if run["max_tt_rank_u2"] is not None:
        cmd += ["--max_tt_rank_u2", str(run["max_tt_rank_u2"])]
    if run["init"] is not None:
        cmd += ["--init", run["init"]]
    if max_iterations is not None:
        cmd += ["--max_iterations", str(max_iterations)]
    cmd += ["--results_dir", str(results_dir)]
    return cmd


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the experiments of the paper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=list(PAPER_ITEMS),
        default=None,
        help="Run only these items (default: all)",
    )
    parser.add_argument("--results_dir", default=None, help="Results directory")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Every configuration at small size with 5 log entries",
    )
    parser.add_argument(
        "--max_iterations", type=int, default=None, help="Override max_iterations"
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="Print the commands without running"
    )
    parser.add_argument(
        "--no_skip",
        action="store_true",
        help="Run even if results of a run already exist",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_arguments(argv)
    if args.results_dir is not None:
        results_dir = Path(args.results_dir)
    elif args.smoke:
        results_dir = _REPO_DIR / "results_smoke"
    else:
        results_dir = Path(os.environ.get("TTGPE_RESULTS_DIR", _REPO_DIR / "results"))
    results_dir = results_dir.resolve()
    max_iterations = args.max_iterations
    if args.smoke and max_iterations is None:
        max_iterations = SMOKE_MAX_ITERATIONS

    items = get_items(args.smoke)
    selected = args.only if args.only else list(items)
    runs = []
    for item in selected:
        for run in items[item]:
            if run not in runs:
                runs.append(run)

    failed = []
    for i, run in enumerate(runs, 1):
        cmd = build_command(run, results_dir, max_iterations)
        desc = " ".join(cmd[1:])
        if args.dry_run:
            print(" ".join(cmd))
            continue
        if not args.no_skip and results_exist(run, results_dir):
            print(f"[SKIP {i}/{len(runs)}] {desc}", flush=True)
            continue
        print(f"[RUN {i}/{len(runs)}] {desc}", flush=True)
        ret = subprocess.run(cmd).returncode
        if ret != 0:
            failed.append((ret, desc))

    if args.dry_run:
        print(f"# {len(runs)} runs")
        return 0
    print(f"\n{len(runs)} runs, {len(failed)} failed.")
    for ret, desc in failed:
        print(f"  [exit {ret}] {desc}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
