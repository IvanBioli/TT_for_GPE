"""Figures and tables of the paper from the per-iteration logs of the runs.

Produces the outputs of the numerical experiments section of the paper
(arXiv:2609.37103):

====================================================  ==============================
Output                                                Paper label
====================================================  ==============================
exp1_CGvsSD_iters.pdf, exp1_CGvsSD_time.pdf           Fig. ``fig:exp1_CGvsSD``
exp2_precond_table.tex                                Tab. ``tab:exp2_precond``
exp3_harmonic_N{399,799}_k4_time_shared.pdf           Fig. ``fig:exp3_fullvsTT``
exp4_lattice_N{399,799}_k4_time_shared.pdf            Fig. ``fig:exp4_lattice``
exp5_anderson_k4_N{1599,3199,6399}_k4_time_shared.pdf Fig. ``fig:exp5_anderson``
exp9_multicomponent_k4_N{1023,2047}_k4_time_shared.pdf Fig. ``fig:exp9_multicomponent``
exp9_multicomponent_CGvsSD_table.tex                  Tab. ``tab:exp9_multicomponent``
====================================================  ==============================

Some plotting calls also write companion outputs (iteration plots, plots
without shared y-limits, summary tables) next to these files.

The runs of each item are defined in ``src/run.py`` (``PAPER_ITEMS``). The logs
are read either from the result pickles written by the experiment scripts
(``--source results``, the default) or from the CSV logs of the paper's runs
shipped in ``data/arxiv_v1`` (``--source csv``)::

    python src/experiments/analysis/analysis_new.py --source csv        # data/arxiv_v1
    python src/experiments/analysis/analysis_new.py                     # <repo>/results
    python src/experiments/analysis/analysis_new.py --results_dir DIR
    python src/experiments/analysis/analysis_new.py --smoke             # runs of run.py --smoke

The reference energy E* of the relative energy error |E(u) - E*| / |E*| is the
minimum final energy (high-order quadrature, ``E_exact``) over the R-NLCG runs
of the corresponding experiment family (harmonic: exp1 and exp3; lattice: exp4;
Anderson: exp5); for the multicomponent problem it is the minimum over the
finer-grid runs at N=4095 (item ``ref9`` of ``run.py``). Times exclude the time
spent in logging.
"""

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent.parent
_REPO_ROOT = _SRC_DIR.parent
sys.path.insert(0, str(_SRC_DIR))

from jax import config

config.update("jax_platform_name", "cpu")
config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rc

from experiments.experiment_setup import load_results
from run import ANDERSON, HARMONIC, LATTICE, MULTICOMPONENT, get_items

# Exact figure fonts need a LaTeX installation; otherwise fall back to mathtext.
if shutil.which("latex"):
    rc("font", **{"family": "serif", "serif": ["Computer Modern"]})
    rc("text", usetex=True)
else:
    rc("font", family="serif")
    rc("mathtext", fontset="cm")

pd.set_option("display.max_colwidth", None)
pd.set_option("display.precision", 12)

# ===========================================================================
# Color / style maps
# ===========================================================================
MATLAB_GEM12 = [
    (0, 0.4470, 0.7410),
    (0.8500, 0.3250, 0.0980),
    (0.9290, 0.6940, 0.1250),
    (0.4940, 0.1840, 0.5560),
    (0.4660, 0.6740, 0.1880),
    (0.3010, 0.7450, 0.9330),
    (0.6350, 0.0780, 0.1840),
    (1.0, 0.8390, 0.0390),
    (0.3960, 0.5090, 0.9920),
    (1.0, 0.2700, 0.2270),
    (0, 0.6390, 0.6390),
    (0.7960, 0.5170, 0.3640),
]

METHOD_COLORS = {
    ("tt", 5): MATLAB_GEM12[0],
    ("tt", 10): MATLAB_GEM12[1],
    ("tt", 15): MATLAB_GEM12[2],
    ("tt", 20): MATLAB_GEM12[0],
    ("tt", 40): MATLAB_GEM12[1],
    ("tt", 50): MATLAB_GEM12[2],
    ("tt", 60): MATLAB_GEM12[3],
    ("tt", 80): MATLAB_GEM12[6],
    ("full_au", None): MATLAB_GEM12[4],
    ("full_H1", None): MATLAB_GEM12[5],
}

METHOD_COLORS_MULTICOMPONENT = {
    ("tt", 5): MATLAB_GEM12[0],
    ("tt", 10): MATLAB_GEM12[1],
    ("tt", 15): MATLAB_GEM12[2],
    ("tt", 20): MATLAB_GEM12[3],
    ("full_au", None): MATLAB_GEM12[4],
    ("full_H1", None): MATLAB_GEM12[5],
}

METHOD_ORDER = [
    ("full_au", None),
    ("full_H1", None),
    ("tt", 5),
    ("tt", 10),
    ("tt", 15),
    ("tt", 20),
    ("tt", 40),
    ("tt", 50),
    ("tt", 60),
    ("tt", 80),
]

OPTIMIZER_LINESTYLE = {
    "ConjugateGradient": "-",
    "SteepestDescent": ":",
}

PRECOND_MARKER = {
    "noprecond": "o",
    "precondS": "s",
    "precondS+V": "^",
}

PRECOND_LABEL = {
    "noprecond": "None",
    "precondS": "S",
    "precondS+V": "S+V",
}

# ===========================================================================
# Folder traversal and data loading helpers
# ===========================================================================
# Layout (from experiment_run.py); data/arxiv_v1 (written by export_logs.py)
# mirrors it with a .csv file and a run.json per run instead of the .pkl:
#   {prefix}_{dev}_{dtype}_dim{dim}/
#       beta{beta}_N{N}_k{k}/
#           {optimizer}/
#               tt_au/ttrank{r}_{precond_str}[_maxu2rank{r2}]/  <- .pkl / .csv
#               full_H1/                                         <- .pkl / .csv
#               full_au_{precond_str}/                           <- .pkl / .csv

RUN_EXTENSIONS = (".pkl", ".csv")


def _run_files(folder):
    """Sorted result files of a run folder."""
    return sorted(f for f in os.listdir(folder) if f.endswith(RUN_EXTENSIONS))


def _parse_tt_dir(rank_dir):
    m = re.match(r"ttrank(\d+)_(.*?)(?:_maxu2rank(\d+))?$", rank_dir)
    if not m:
        raise ValueError(f"Cannot parse tt rank dir: {rank_dir}")
    return int(m.group(1)), int(m.group(3)) if m.group(3) else None, m.group(2)


def find_result_dirs(base_dir, exp_prefix, dim, beta, N, k, optimizers=None):
    """
    Return list of (optimizer, method_type, precond_str, tt_rank, max_u2_rank, path).
    method_type: 'tt_au' | 'full_H1' | 'full_au'
    precond_str: folder-name convention, e.g. 'precondS+V', 'noprecond', None for full_H1
    path: latest result file of the run folder
    """
    entries = []
    if not os.path.isdir(base_dir):
        return entries
    beta_str = f"{float(beta):.1f}" if float(beta) == int(beta) else str(beta)
    exp_pat = re.compile(rf"^{re.escape(exp_prefix)}_.*_dim{dim}$")
    nk_pat = re.compile(rf"^beta{re.escape(beta_str)}_N{N}_k{k}$")

    for top in os.listdir(base_dir):
        if not exp_pat.match(top):
            continue
        top_path = os.path.join(base_dir, top)
        if not os.path.isdir(top_path):
            continue
        for nk_dir in os.listdir(top_path):
            if not nk_pat.match(nk_dir):
                continue
            nk_path = os.path.join(top_path, nk_dir)
            for opt_dir in os.listdir(nk_path):
                opt_path = os.path.join(nk_path, opt_dir)
                if not os.path.isdir(opt_path):
                    continue
                if optimizers is not None and opt_dir not in optimizers:
                    continue
                for method_dir in os.listdir(opt_path):
                    method_path = os.path.join(opt_path, method_dir)
                    if not os.path.isdir(method_path):
                        continue
                    if method_dir == "tt_au":
                        for rank_dir in os.listdir(method_path):
                            rank_path = os.path.join(method_path, rank_dir)
                            if not os.path.isdir(rank_path):
                                continue
                            files = _run_files(rank_path)
                            if not files:
                                continue
                            tt_rank, max_u2, prec = _parse_tt_dir(rank_dir)
                            entries.append(
                                (
                                    opt_dir,
                                    "tt_au",
                                    prec,
                                    tt_rank,
                                    max_u2,
                                    os.path.join(rank_path, files[-1]),
                                )
                            )
                    elif method_dir == "full_H1":
                        files = _run_files(method_path)
                        if files:
                            entries.append(
                                (
                                    opt_dir,
                                    "full_H1",
                                    None,
                                    None,
                                    None,
                                    os.path.join(method_path, files[-1]),
                                )
                            )
                    elif method_dir.startswith("full_au_"):
                        files = _run_files(method_path)
                        if files:
                            prec = method_dir[len("full_au_") :]
                            entries.append(
                                (
                                    opt_dir,
                                    "full_au",
                                    prec,
                                    None,
                                    None,
                                    os.path.join(method_path, files[-1]),
                                )
                            )
    return entries


def load_run(path):
    """Load one run from a result pickle or from a CSV log (with its run.json).

    The pickle holds the pymanopt OptimizerResult; for a CSV log an object with
    the same attributes is built. Both expose ``.log["iterations"]`` (dict of per-iteration lists) and
    the scalars ``iterations``, ``time``, ``cost``, ``gradient_norm`` and
    ``stopping_criterion``.
    """
    path = str(path)
    if path.endswith(".csv"):
        return _load_csv_run(path)
    return load_results(os.path.dirname(path), os.path.basename(path)[:-4])["res"]


def _load_csv_run(path):
    """Load a run from a CSV export (data/arxiv_v1) and the run.json next to it.

    Returns an object with the same attributes as the OptimizerResult used by
    this module.
    """
    import json
    from types import SimpleNamespace

    # round_trip: the default C parser may be off by one ulp for repr floats
    df = pd.read_csv(path, float_precision="round_trip")
    with open(os.path.join(os.path.dirname(path), "run.json")) as f:
        info = json.load(f)
    log = {
        "optimizer": info.get("optimizer"),
        "stopping_criteria": info.get("stopping_criteria"),
        "iterations": {c: df[c].tolist() for c in df.columns},
    }
    return SimpleNamespace(
        log=log,
        iterations=info.get("iterations"),
        time=info.get("time"),
        cost=info.get("final_cost"),
        gradient_norm=info.get("final_gradient_norm"),
        stopping_criterion=info.get("stopping_criterion"),
    )


def load_df(path):
    """Per-iteration log of a run as a DataFrame indexed by iteration."""
    res = load_run(path)
    return pd.DataFrame.from_dict(res.log["iterations"]).set_index("iteration")


def compute_time(df):
    """Optimizer wall time at each log entry, excluding logging overhead.

    ``time[k]`` is stamped before the logging of iteration k is performed,
    and ``log_time[k]`` is the time spent logging iteration k. The logging
    time accumulated before entry k is therefore the sum of ``log_time[j]``
    for j < k (the current entry's own logging is not included).
    """
    lt = df["log_time"].fillna(0)
    t = (df["time"] - df["time"].iloc[0]) - (np.cumsum(lt) - lt)
    return t


def get_color(method_type, tt_rank, multicomponent=False):
    key = (
        ("tt", tt_rank)
        if method_type == "tt_au"
        else ("full_au", None) if method_type == "full_au" else ("full_H1", None)
    )
    if multicomponent:
        return METHOD_COLORS_MULTICOMPONENT.get(key, "black")
    else:
        return METHOD_COLORS.get(key, "black")


def get_label(method_type, tt_rank, max_u2_rank, precond_str):
    if precond_str is not None:
        precond_h = (
            PRECOND_LABEL.get(precond_str, str(precond_str)) if precond_str else "-"
        )
    if method_type == "tt_au":
        lbl = rf"TT ($a_u$ SGF), rank={tt_rank}"
        if max_u2_rank is not None:
            lbl += rf" (max {max_u2_rank})"
        if precond_str is not None:
            lbl += f", prec={precond_h}"
    elif method_type == "full_au":
        lbl = rf"Full ($a_u$ SGF)"
        if precond_str is not None:
            lbl += f", prec={precond_h}"
    else:
        lbl = rf"Full ($H^1$ SGF)"
    return lbl


def _entry_sort_key(e):
    """Sort runs: R-GD before R-NLCG, then by METHOD_ORDER."""
    opt, method_type, precond_str, tt_rank, max_u2_rank, path = e
    opt_order = 0 if opt == "SteepestDescent" else 1
    key = (
        ("tt", tt_rank)
        if method_type == "tt_au"
        else ("full_au", None) if method_type == "full_au" else ("full_H1", None)
    )
    method_idx = METHOD_ORDER.index(key) if key in METHOD_ORDER else len(METHOD_ORDER)
    return (opt_order, method_idx)


def build_row(df, label, N=None):
    row = {
        "Label": label,
        "Final E(u)": float(df["E_exact"].iloc[-1]),
        "Final cost(u)": float(df["cost"].iloc[-1]),
        "Time (s)": float(compute_time(df).iloc[-1]),
        "Iterations": int(df.index[-1]),
        "Avg CG iters": (
            float(df["cg_iters"].mean()) if "cg_iters" in df.columns else float("nan")
        ),
    }
    if N is not None:
        row["N"] = N
    return row


def save_latex(df_table, caption, label, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df_table.to_latex(
        filepath,
        escape=False,
        float_format="%.6g",
        bold_rows=True,
        caption=caption,
        label=label,
    )


def compute_E_ref(
    base_dir,
    exp_prefix,
    dim,
    beta,
    N_vec,
    k_vec,
    optimizers=None,
    allowed_preconds=None,
):
    """Return the minimum final E_exact across all (N, k) combinations."""
    if optimizers is None:
        optimizers = ["ConjugateGradient"]
    if allowed_preconds is None:
        allowed_preconds = {"precondS+V", None}
    all_entries = [
        e
        for k in k_vec
        for N in N_vec
        for e in find_result_dirs(base_dir, exp_prefix, dim, beta, N, k, optimizers)
        if e[2] in allowed_preconds
    ]
    if not all_entries:
        raise ValueError(
            f"compute_E_ref: no results found for prefix={exp_prefix!r}, dim={dim}, "
            f"beta={beta}, N_vec={N_vec}, k_vec={k_vec}."
        )
    return min(float(load_df(e[5])["E_exact"].iloc[-1]) for e in all_entries)


# ===========================================================================
# EXP 1 -- CG vs SteepestDescent (Fig. fig:exp1_CGvsSD)
#           Harmonic, k=4, N=399, S+V, tt(5,10,15)+full_au+full_H1
#           Color: method/rank  |  Linestyle: CG=solid, SD=dotted
#           Plots: [iter vs error], [time vs error]
# ===========================================================================


def plot_exp1(base_dir, savedir, E_ref, N=399):
    exp_prefix, dim, beta, k = "harmonic", 3, 1000, 4
    entries = [
        e
        for e in find_result_dirs(
            base_dir,
            exp_prefix,
            dim,
            beta,
            N,
            k,
            ["ConjugateGradient", "SteepestDescent"],
        )
        if e[2] in {"precondS+V", None}
    ]
    if not entries:
        print("[EXP1] No results found.")
        return

    entries.sort(key=_entry_sort_key)

    fig_it, ax_it = plt.subplots(figsize=(5, 4))
    fig_t, ax_t = plt.subplots(figsize=(5, 4))
    rows = []

    for opt, method_type, precond_str, tt_rank, max_u2_rank, path in entries:
        opt_label = "R-NLCG" if opt == "ConjugateGradient" else "R-GD"
        df = load_df(path)
        color = get_color(method_type, tt_rank)
        ls = OPTIMIZER_LINESTYLE[opt]
        lbl = f"[{opt_label}] {get_label(method_type, tt_rank, None, None)}"
        err = np.abs(df["E_exact"].values - E_ref) / np.abs(E_ref)
        kw = dict(color=color, linestyle=ls, marker=".", markersize=3, label=lbl)
        ax_it.plot(df.index.values, err, **kw)
        ax_t.plot(compute_time(df).values, err, **kw)
        rows.append(build_row(df, lbl, N=N))

    for ax, xl in [(ax_it, "Iteration"), (ax_t, "Time (s)")]:
        ax.set_yscale("log")
        ax.set_xlabel(xl)
        ax.set_ylabel(r"$|E(u) - E^*| / |E^*|$")
        ax.legend(fontsize=6, loc="upper right")

    os.makedirs(savedir, exist_ok=True)
    fig_it.tight_layout()
    fig_it.savefig(os.path.join(savedir, "exp1_CGvsSD_iters.pdf"))
    print(f"Saved iteration plot to {os.path.join(savedir, 'exp1_CGvsSD_iters.pdf')}")
    fig_t.tight_layout()
    fig_t.savefig(os.path.join(savedir, "exp1_CGvsSD_time.pdf"))
    print(f"Saved time plot to {os.path.join(savedir, 'exp1_CGvsSD_time.pdf')}")
    plt.close(fig_it)
    plt.close(fig_t)
    table = pd.DataFrame(rows).set_index("Label").drop(columns=["N"])
    save_latex(
        table,
        rf"Steepest Descent (SD) vs Conjugate Gradient (CG) --- Harmonic, $N={N}$, $k={k}$",
        "tab:exp1_CGvsSD",
        os.path.join(savedir, "exp1_CGvsSD_table.tex"),
    )


# ===========================================================================
# EXP 2 -- Preconditioners comparison (Tab. tab:exp2_precond)
#           Harmonic, k=4, N=399, CG, tt(5,10,15)+full_au, prec: None/S/S+V
#           Color: method/rank  |  Marker: preconditioner
#           Plots: [time vs error], [inner CG iters vs iteration]
# ===========================================================================


def plot_exp2(base_dir, savedir, N=399):
    exp_prefix, dim, beta, k = "harmonic", 3, 1000, 4
    entries = [
        e
        for e in find_result_dirs(
            base_dir, exp_prefix, dim, beta, N, k, ["ConjugateGradient"]
        )
        if e[1] != "full_H1"
    ]
    if not entries:
        print("[EXP2] No results found.")
        return

    E_ref = min(float(load_df(e[5])["E_exact"].iloc[-1]) for e in entries)
    fig_t, ax_t = plt.subplots(figsize=(5, 4))
    fig_cg, ax_cg = plt.subplots(figsize=(5, 4))
    rows = []

    for opt, method_type, precond_str, tt_rank, max_u2_rank, path in entries:
        df = load_df(path)
        color = get_color(method_type, tt_rank)
        marker = PRECOND_MARKER.get(precond_str, "x")
        lbl = get_label(method_type, tt_rank, max_u2_rank, precond_str)
        err = np.abs(df["E_exact"].values - E_ref) / np.abs(E_ref)
        kw = dict(
            color=color,
            linestyle="-",
            marker=marker,
            markersize=4,
            label=lbl,
            markevery=10,
        )
        ax_t.plot(compute_time(df).values, err, **kw)
        if "cg_iters" in df.columns:
            ax_cg.plot(df.index.values, df["cg_iters"].values, **kw)
        rows.append(build_row(df, lbl, N=N))

    ax_t.set_yscale("log")
    ax_t.set_xlabel("Time (s)")
    ax_t.set_ylabel(r"$|E(u) - E^*| / |E^*|$")
    ax_t.set_title(rf"Precond. comparison --- Harmonic, $N={N}$, $k={k}$, CG")
    ax_t.legend(fontsize=6, loc="upper right")
    ax_cg.set_xlabel("Iteration")
    ax_cg.set_ylabel("CG iterations (inner solver)")
    ax_cg.set_title(rf"Inner CG iters --- Harmonic, $N={N}$, $k={k}$, CG")
    ax_cg.legend(fontsize=6, loc="upper right")

    os.makedirs(savedir, exist_ok=True)
    fig_t.tight_layout()
    fig_t.savefig(os.path.join(savedir, "exp2_precond_time.pdf"))
    print(f"Saved time plot to {os.path.join(savedir, 'exp2_precond_time.pdf')}")
    fig_cg.tight_layout()
    fig_cg.savefig(os.path.join(savedir, "exp2_precond_cgiters.pdf"))
    print(f"Saved CG iters plot to {os.path.join(savedir, 'exp2_precond_cgiters.pdf')}")
    plt.close(fig_t)
    plt.close(fig_cg)
    table = pd.DataFrame(rows).set_index("Label").drop(columns=["N"])
    save_latex(
        table,
        rf"Precond. comparison --- Harmonic, $N={N}$, $k={k}$, CG.",
        "tab:exp2_precond",
        os.path.join(savedir, "exp2_precond_table.tex"),
    )


# ===========================================================================
# Shared helper: scaling experiments (Exp 3/4/5/9)
#   One [time vs error] figure per N  +  one combined table over all N
# ===========================================================================


def plot_scaling_exp(
    base_dir,
    savedir,
    exp_prefix,
    dim,
    beta,
    k,
    N_vec,
    exp_label,
    exp_tag,
    optimizers=None,
    allowed_preconds=None,
    E_ref=None,
    multicomponent=False,
    shared_N_vec=None,
):
    if optimizers is None:
        optimizers = ["ConjugateGradient"]
    if allowed_preconds is None:
        allowed_preconds = {"precondS+V", None}
    all_rows = []

    if E_ref is None:
        try:
            E_ref = compute_E_ref(
                base_dir,
                exp_prefix,
                dim,
                beta,
                N_vec,
                [k],
                optimizers,
                allowed_preconds,
            )
        except ValueError:
            print(f"[{exp_tag}] No results found for any N.")
            return

    # First pass: load all data to allow computing shared y-axis limits
    all_data = {}  # N -> list of (df, err, color, ls, lbl)
    for N in N_vec:
        entries = [
            e
            for e in find_result_dirs(base_dir, exp_prefix, dim, beta, N, k, optimizers)
            if e[2] in allowed_preconds
        ]
        if not entries:
            print(f"[{exp_tag}] No results for N={N}.")
            continue
        entries.sort(key=_entry_sort_key)
        data_for_N = []
        for opt, method_type, precond_str, tt_rank, max_u2_rank, path in entries:
            df = load_df(path)
            err = np.abs(df["E_exact"].values - E_ref) / np.abs(E_ref)
            lbl = get_label(method_type, tt_rank, None, None)
            data_for_N.append(
                (
                    df,
                    err,
                    get_color(method_type, tt_rank, multicomponent),
                    OPTIMIZER_LINESTYLE.get(opt, "-"),
                    lbl,
                )
            )
            all_rows.append({**build_row(df, lbl), "N": N})
        all_data[N] = data_for_N

    if not all_data:
        return

    # Shared y limits computed only from shared_N_vec subset
    shared_ylim = None
    if shared_N_vec is not None:
        subset_errs = np.concatenate(
            [
                err
                for N, data in all_data.items()
                if N in shared_N_vec
                for (_, err, _, _, _) in data
            ]
        )
        pos = subset_errs[subset_errs > 0]
        shared_ylim = (pos.min() * 0.5, subset_errs.max() * 2) if len(pos) > 0 else None

    os.makedirs(savedir, exist_ok=True)
    for N, data_for_N in all_data.items():
        do_shared = shared_N_vec is not None and N in shared_N_vec
        for shared in [False, True] if do_shared else [False]:
            suffix = "_shared" if shared else ""
            fig_t, ax_t = plt.subplots(figsize=(5, 4))
            fig_it, ax_it = plt.subplots(figsize=(5, 4))

            for df, err, color, ls, lbl in data_for_N:
                kw = dict(
                    color=color, linestyle=ls, marker=".", markersize=3, label=lbl
                )
                ax_t.plot(compute_time(df).values, err, **kw)
                ax_it.plot(df.index.values, err, **kw)

            for ax, xl in [(ax_t, "Time (s)"), (ax_it, "Iteration")]:
                ax.set_yscale("log")
                ax.set_xlabel(xl)
                ax.set_ylabel(r"$|E(u) - E^*| / |E^*|$")
                if shared and shared_ylim is not None:
                    ax.set_ylim(shared_ylim)
                ax.legend(fontsize=7, loc="upper right")

            fig_t.tight_layout()
            fig_t.savefig(
                os.path.join(savedir, f"{exp_tag}_N{N}_k{k}_time{suffix}.pdf")
            )
            print(
                f"Saved time plot to {os.path.join(savedir, f'{exp_tag}_N{N}_k{k}_time{suffix}.pdf')}"
            )
            fig_it.tight_layout()
            fig_it.savefig(
                os.path.join(savedir, f"{exp_tag}_N{N}_k{k}_iters{suffix}.pdf")
            )
            print(
                f"Saved iteration plot to {os.path.join(savedir, f'{exp_tag}_N{N}_k{k}_iters{suffix}.pdf')}"
            )
            plt.close(fig_t)
            plt.close(fig_it)

    if not all_rows:
        return
    table = pd.DataFrame(all_rows).set_index(["N", "Label"])
    save_latex(
        table,
        rf"Summary --- {exp_label}, $k={k}$.",
        f"tab:{exp_tag}_summary",
        os.path.join(savedir, f"{exp_tag}_table.tex"),
    )


# ===========================================================================
# EXP 9 (table) -- Multicomponent 2D, CG vs SteepestDescent
#           (Tab. tab:exp9_multicomponent)
#           N=2047, k=4, S+V, tt(5,10,15,20)+full_au
#           Color: method/rank  |  Linestyle: CG=solid, SD=dotted
#           Plots: [iter vs error], [time vs error]
# ===========================================================================


def plot_exp9_multicomponent_CGvsSD(base_dir, savedir, E_ref=None, N=2047):
    exp_prefix, dim, beta, k = "multicomponent_2d", 2, 10, 4
    entries = [
        e
        for e in find_result_dirs(
            base_dir,
            exp_prefix,
            dim,
            beta,
            N,
            k,
            ["ConjugateGradient", "SteepestDescent"],
        )
        if e[2] in {"precondS+V", None} and e[1] != "full_H1"
    ]
    if not entries:
        print("[EXP9 table] No results found.")
        return

    entries.sort(key=_entry_sort_key)

    if E_ref is None:
        E_ref = min(float(load_df(e[5])["E_exact"].iloc[-1]) for e in entries)
    fig_it, ax_it = plt.subplots(figsize=(5, 4))
    fig_t, ax_t = plt.subplots(figsize=(5, 4))
    rows = []

    for opt, method_type, precond_str, tt_rank, max_u2_rank, path in entries:
        opt_label = "CG" if opt == "ConjugateGradient" else "SD"
        df = load_df(path)
        color = get_color(method_type, tt_rank, multicomponent=True)
        ls = OPTIMIZER_LINESTYLE[opt]
        lbl = f"[{opt_label}] {get_label(method_type, tt_rank, None, None)}"
        err = np.abs(df["E_exact"].values - E_ref) / np.abs(E_ref)
        kw = dict(color=color, linestyle=ls, marker=".", markersize=3, label=lbl)
        ax_it.plot(df.index.values, err, **kw)
        ax_t.plot(compute_time(df).values, err, **kw)
        rows.append(build_row(df, lbl, N=N))

    for ax, xl in [(ax_it, "Iteration"), (ax_t, "Time (s)")]:
        ax.set_yscale("log")
        ax.set_xlabel(xl)
        ax.set_ylabel(r"$|E(u) - E^*| / |E^*|$")
        ax.legend(fontsize=6, loc="upper right")

    os.makedirs(savedir, exist_ok=True)
    fig_it.tight_layout()
    fig_it.savefig(os.path.join(savedir, "exp9_multicomponent_CGvsSD_iters.pdf"))
    print(
        f"Saved iteration plot to {os.path.join(savedir, 'exp9_multicomponent_CGvsSD_iters.pdf')}"
    )
    fig_t.tight_layout()
    fig_t.savefig(os.path.join(savedir, "exp9_multicomponent_CGvsSD_time.pdf"))
    print(
        f"Saved time plot to {os.path.join(savedir, 'exp9_multicomponent_CGvsSD_time.pdf')}"
    )
    plt.close(fig_it)
    plt.close(fig_t)
    table = pd.DataFrame(rows).set_index("Label").drop(columns=["N"])
    save_latex(
        table,
        rf"CG vs SD --- Multicomponent 2D, $N={N}$, $k={k}$, S+V.",
        "tab:exp9_multicomponent_CGvsSD",
        os.path.join(savedir, "exp9_multicomponent_CGvsSD_table.tex"),
    )


# ===========================================================================
# Experiment families, reference energies and driver
# ===========================================================================

# Experiment families: results-folder prefix, dimension, beta, script and the
# items whose R-NLCG runs define the reference energy E* of the family.
FAMILIES = {
    "harmonic": dict(exp_prefix="harmonic", dim=3, beta=1000, script=HARMONIC),
    "lattice": dict(exp_prefix="lattice", dim=3, beta=1000, script=LATTICE),
    "anderson": dict(exp_prefix="anderson", dim=2, beta=10, script=ANDERSON),
    "multicomponent": dict(
        exp_prefix="multicomponent_2d", dim=2, beta=10, script=MULTICOMPONENT
    ),
}
E_REF_ITEMS = {
    "harmonic": ["exp1", "exp3"],
    "lattice": ["exp4"],
    "anderson": ["exp5"],
    "multicomponent": ["ref9"],  # finer-grid reference runs (N=4095)
}
ITEMS = ["exp1", "exp2", "exp3", "exp4", "exp5", "exp9", "tab9"]


def _grid_sizes(items, item, script):
    """Sorted grid sizes N of the runs of an item for a given script."""
    return sorted({r["N"] for r in items[item] if r["script"] == script})


def family_E_ref(base_dir, family, items):
    """Reference energy E* of a family: minimum final E_exact over its R-NLCG runs."""
    fam = FAMILIES[family]
    N_vec = sorted(
        {N for it in E_REF_ITEMS[family] for N in _grid_sizes(items, it, fam["script"])}
    )
    return compute_E_ref(
        base_dir,
        fam["exp_prefix"],
        fam["dim"],
        fam["beta"],
        N_vec=N_vec,
        k_vec=[4],
        optimizers=["ConjugateGradient"],
        allowed_preconds={"precondS+V", None},
    )


def paper_outputs(smoke=False):
    """File names of the paper's figures and tables (for the chosen run sizes)."""
    items = get_items(smoke)
    out = ["exp1_CGvsSD_iters.pdf", "exp1_CGvsSD_time.pdf", "exp2_precond_table.tex"]
    for tag, item, script in [
        ("exp3_harmonic", "exp3", HARMONIC),
        ("exp4_lattice", "exp4", LATTICE),
        ("exp5_anderson_k4", "exp5", ANDERSON),
        ("exp9_multicomponent_k4", "exp9", MULTICOMPONENT),
    ]:
        out += [
            f"{tag}_N{N}_k4_time_shared.pdf" for N in _grid_sizes(items, item, script)
        ]
    out.append("exp9_multicomponent_CGvsSD_table.tex")
    return out


def make_outputs(base_dir, savedir, only=None, smoke=False, E_refs=None):
    """Produce the figures and tables of the selected items.

    Args:
        base_dir: Folder with the runs (results directory or data/arxiv_v1).
        savedir: Output folder.
        only: Items to produce (default: all of ITEMS).
        smoke: Use the grid sizes of ``run.py --smoke``.
        E_refs: Optional dict family -> E* overriding the computed values.
    """
    items = get_items(smoke)
    only = ITEMS if only is None else only
    E_refs = dict(E_refs or {})

    def E_ref(family):
        if family not in E_refs:
            E_refs[family] = family_E_ref(base_dir, family, items)
            print(f"[{family}] E* = {E_refs[family]!r}")
        return E_refs[family]

    def scaling(item, family, exp_tag, exp_label, multicomponent=False):
        fam = FAMILIES[family]
        N_vec = _grid_sizes(items, item, fam["script"])
        plot_scaling_exp(
            base_dir=base_dir,
            savedir=savedir,
            exp_prefix=fam["exp_prefix"],
            dim=fam["dim"],
            beta=fam["beta"],
            k=4,
            N_vec=N_vec,
            exp_label=exp_label,
            exp_tag=exp_tag,
            E_ref=E_ref(family),
            multicomponent=multicomponent,
            shared_N_vec=N_vec,
        )

    if "exp1" in only:
        plot_exp1(
            base_dir,
            savedir,
            E_ref=E_ref("harmonic"),
            N=_grid_sizes(items, "exp1", HARMONIC)[0],
        )
    if "exp2" in only:
        plot_exp2(base_dir, savedir, N=_grid_sizes(items, "exp2", HARMONIC)[0])
    if "exp3" in only:
        scaling("exp3", "harmonic", "exp3_harmonic", r"Harmonic potential ($d=3$)")
    if "exp4" in only:
        scaling("exp4", "lattice", "exp4_lattice", r"Lattice potential ($d=3$)")
    if "exp5" in only:
        scaling("exp5", "anderson", "exp5_anderson_k4", r"Anderson potential ($d=2$)")
    if "exp9" in only:
        scaling(
            "exp9",
            "multicomponent",
            "exp9_multicomponent_k4",
            r"Multicomponent 2D ($p=3$)",
            multicomponent=True,
        )
    if "tab9" in only:
        try:
            E_mc = E_ref("multicomponent")
        except ValueError:
            E_mc = None  # exp9 runs missing: fall back to the table's own runs
        plot_exp9_multicomponent_CGvsSD(
            base_dir,
            savedir,
            E_ref=E_mc,
            N=_grid_sizes(items, "tab9", MULTICOMPONENT)[0],
        )
    return E_refs


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Figures and tables of the paper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results_dir",
        default=None,
        help="Results directory (default: $TTGPE_RESULTS_DIR if set, else "
        "<repo>/results; <repo>/results_smoke with --smoke)",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output folder (default: <results_dir>/paper_figures, or "
        "<repo>/figures/arxiv_v1 with --source csv)",
    )
    parser.add_argument(
        "--source",
        choices=["results", "csv"],
        default="results",
        help="Read result pickles (results) or the CSV export (csv)",
    )
    parser.add_argument(
        "--data_dir",
        default=str(_REPO_ROOT / "data" / "arxiv_v1"),
        help="CSV export for --source csv (default: <repo>/data/arxiv_v1; "
        "output default <repo>/figures/arxiv_v1)",
    )
    parser.add_argument("--only", nargs="+", choices=ITEMS, default=None)
    parser.add_argument(
        "--smoke", action="store_true", help="Analyse the runs of run.py --smoke"
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_arguments(argv)
    if args.results_dir is not None:
        base_dir = Path(args.results_dir)
    elif args.smoke:
        base_dir = _REPO_ROOT / "results_smoke"
    else:
        base_dir = Path(os.environ.get("TTGPE_RESULTS_DIR", _REPO_ROOT / "results"))
    out_dir = args.out_dir or base_dir / "paper_figures"
    if args.source == "csv":
        base_dir = Path(args.data_dir)
        out_dir = args.out_dir or _REPO_ROOT / "figures" / "arxiv_v1"
    make_outputs(str(base_dir.resolve()), str(out_dir), args.only, args.smoke)


if __name__ == "__main__":
    main()
