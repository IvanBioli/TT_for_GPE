"""Command-line driver shared by the experiment scripts in experiments/paper.

Each paper script defines ``get_experiment_config`` (domain, potential, ...)
and calls ``main_exp``, which parses the command line (or a JSON config file),
builds the problem with ``experiments.experiment_setup`` and runs one of:

- ``tt``: a_u Sobolev gradient flow on the manifold of unit-norm fixed TT-rank
  tensors (Section 'Riemannian optimization on the manifold of unit-norm fixed
  TT-rank functions');
- ``full_au``: the same flow on full (dense) tensors;
- ``full_H1``: the H1 Sobolev gradient flow on full tensors.

The optimizer is pymanopt's SteepestDescent (R-GD) or ConjugateGradient
(R-NLCG), see paragraphs 'Riemannian Gradient Descent (R-GD)' and 'Riemannian
Nonlinear Conjugate Gradient (R-NLCG)'. The result (with the per-iteration
log) is pickled under ``RESULTS_DIR/<name>_dim<d>/beta<beta>_N<N>_k<k>/...``.

The values of 'Software, parameters, and hardware.' (Section 'Numerical
experiments', subsection 'Setup') are the defaults below. Note that
``--dim``, ``--beta`` and ``--L`` (defaults 3, 1000, 6) are always passed to
``get_experiment_config`` and override the defaults of its signature, and that
``--precond`` defaults to no preconditioner.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax

import experiments.experiment_setup as experiment_setup
from experiments.experiment_setup import create_setup_functions, save_results

#### CONFIGURATIONS ####
# Stopping criteria (paragraph 'Software, parameters, and hardware.'): the paper
# states the gradient-norm tolerance 1e-6 and the 2000 iterations. In addition,
# runs stop when the step size of the line search falls below
# min_step_size=1e-10 or after max_time = 4 h of wall-clock time. These values
# override those of experiment_setup.DEFAULT_OPTIMIZER_KWARGS (only
# max_cost_evaluations is taken from there). paperexp_anderson.py overrides
# min_gradient_norm with 1e-3.
DEFAULT_OPTIMIZER_KWARGS = {
    "verbosity": 2,
    "log_verbosity": 1,
    "max_time": 3600 * 4,
    "max_iterations": 2000,
    "min_gradient_norm": 1e-6,
    "min_step_size": 1e-10,
}

# Riemannian gradient in the TT format: the linear systems are solved by
# preconditioned CG (core.tangent) with relative residual tolerance 1e-10,
# measured in the Euclidean (Frobenius) norm of the unpreconditioned residual,
# and at most 200 iterations. The TT preconditioner (Section 'Preconditioning in
# the TT format') uses an exponential sum with rank_expsum=10 terms. "precond" is
# overwritten by --precond when the command line (not --config) is used.
DEFAULT_RGRAD_KWARGS_TT = {
    "maxiter": 200,
    "tol": 1e-10,
    "precond": "S+V",
    "precond_kwargs": {"rank_expsum": 10, "batch_expsum": None},
}

# Same CG settings for the full a_u flow (Section 'Preconditioning in the full
# tensor format' for the preconditioner).
DEFAULT_RGRAD_KWARGS_FULL = {"maxiter": 200, "tol": 1e-10, "precond": "S+V"}


def merge_configs(default_config, json_config):
    """Merge JSON configuration with default configuration.

    Args:
        default_config: Default configuration dictionary.
        json_config: JSON configuration dictionary (overrides defaults).

    Returns:
        Merged configuration dictionary.
    """
    if json_config is None:
        return default_config.copy()

    merged_config = default_config.copy()

    for key, value in json_config.items():
        if (
            isinstance(value, dict)
            and key in merged_config
            and isinstance(merged_config[key], dict)
        ):
            # Recursively merge nested dictionaries
            merged_config[key] = merge_configs(merged_config[key], value)
        else:
            # Override with JSON value
            merged_config[key] = value

    return merged_config


def load_config_file(config_path):
    """Load configuration from JSON file.

    Args:
        config_path: Path to JSON configuration file.

    Returns:
        Configuration dictionary.
    """
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Configuration file not found: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error parsing configuration file: {e}")
        sys.exit(1)


def parse_arguments():
    """Parse command line arguments for experiment configuration.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run Gross-Pitaevskii equation experiments ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Method selection
    parser.add_argument(
        "--method",
        choices=["tt", "full_H1", "full_au"],
        help="Experiment method to run",
    )

    # Optimizer selection
    parser.add_argument(
        "--pymanopt_optimizer",
        type=str,
        default="SteepestDescent",
        choices=[
            "SteepestDescent",
            "ConjugateGradient",
            "TrustRegions",
            "NelderMead",
            "ParticleSwarm",
        ],
        help="Pymanopt optimizer to use (default: SteepestDescent)",
    )

    # Parameters
    parser.add_argument("--N", type=int, help="Grid size parameter N")
    parser.add_argument("--k", type=int, help="Parameter k")
    parser.add_argument("--tt_rank", type=int, help="TT rank (required for tt method)")
    parser.add_argument(
        "--max_tt_rank_u2",
        type=int,
        default=None,
        help="Maximum TT rank for u^2 (only for tt method)",
    )

    # Configuration options
    parser.add_argument("--config", help="Path to JSON configuration file")
    parser.add_argument(
        "--dim",
        type=int,
        default=3,
        choices=[2, 3],
        help="Problem dimension (default: 3)",
    )
    parser.add_argument(
        "--beta", type=float, default=1000, help="Beta parameter (default: 1000)"
    )
    parser.add_argument(
        "--L", type=float, default=6, help="Domain parameter L (default: 6)"
    )

    # Preconditioner
    parser.add_argument(
        "--precond",
        type=str,
        default="None",
        choices=["S+V", "S", "None"],
        help="Preconditioner (default: None)",
    )

    # Initialization
    parser.add_argument(
        "--init",
        type=str,
        default="thomas_fermi",
        choices=["thomas_fermi", "random", "constant"],
        help="Initialization type (default: thomas_fermi)",
    )

    # Run control
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=None,
        help="Override the optimizer's max_iterations (also with --config)",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="Directory where results are saved (default: $TTGPE_RESULTS_DIR "
        "if set, else <repo>/results)",
    )

    args = parser.parse_args()
    if args.precond == "None":
        args.precond = None

    # Validate arguments
    if not args.config and not args.method:
        parser.error("Either --config or --method must be specified")

    if args.method and (args.N is None or args.k is None):
        parser.error("--N and --k are required when --method is specified")

    if args.method == "tt" and args.tt_rank is None:
        parser.error("--tt_rank is required when method is 'tt'")

    if args.method == "tt" and args.max_tt_rank_u2 is not None:
        if args.max_tt_rank_u2 <= 0:
            parser.error("--max_tt_rank_u2 must be a positive integer")
    if args.method != "tt" and args.max_tt_rank_u2 is not None:
        parser.error("--max_tt_rank_u2 is only applicable for the 'tt' method")

    if args.method == "full_H1" and args.precond is not None:
        parser.error("full_H1 does not support any preconditioner")

    return args


#### RUNNING EXPERIMENTS ####
def run_tt_experiment(
    N,
    k,
    tt_rank,
    exp_config,
    pymanopt_optimizer,
    rng=None,
    optimizer_kwargs=None,
    rgrad_kwargs_tt=None,
    max_tt_rank_u2=None,
    init="thomas_fermi",
):
    """Run tensor-train experiment for Gross-Pitaevskii equation.

    Builds the problem with ``setup_tt_problem``, runs the optimizer from the
    initial guess and saves the ``OptimizerResult`` (final point and
    per-iteration log) with ``save_results``. Exceptions are caught and
    reported.

    Args:
        N: Number of interior grid points per dimension.
        k: Polynomial degree.
        tt_rank: TT rank.
        exp_config: Experiment configuration dictionary.
        pymanopt_optimizer: Name of the optimizer ("SteepestDescent" or
            "ConjugateGradient"; see experiment_setup.OPTIMIZER_FROM_STRING).
        rng: Random number generator key (default: None).
        optimizer_kwargs: Optimizer configuration (default: None).
        rgrad_kwargs_tt: Riemannian gradient configuration (default: None).
        max_tt_rank_u2: If not None, TT rank to which ``u * u`` is rounded in
            the a_u metric and the Riemannian gradient; the cost is evaluated
            without this rounding (default: None).
        init: Initialization type ('thomas_fermi', 'random', or 'constant').

    Returns:
        True if successful, False otherwise.
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)

    # Use provided configurations or defaults
    if optimizer_kwargs is None or rgrad_kwargs_tt is None:
        default_optimizer_kwargs = DEFAULT_OPTIMIZER_KWARGS.copy()
        default_rgrad_kwargs_tt = DEFAULT_RGRAD_KWARGS_TT.copy()
        if optimizer_kwargs is None:
            optimizer_kwargs = default_optimizer_kwargs
        if rgrad_kwargs_tt is None:
            rgrad_kwargs_tt = default_rgrad_kwargs_tt

    setup_tt_problem, _ = create_setup_functions(exp_config)

    dim = exp_config["dim"]
    beta = exp_config["beta"]
    exp_name = f"{exp_config['name']}_dim{dim}/beta{beta}_N{N}_k{k}"

    try:
        # Get name of preconditioner for saving
        precond = rgrad_kwargs_tt.get("precond", "noprecond")
        if precond is None:
            precond = "noprecond"
        else:
            precond = "precond" + precond
        # Setup tt problem
        problem_tt, optimizer_tt, u0_tt, E_tt, E_highorder_tt, rng = setup_tt_problem(
            N,
            k,
            tt_rank,
            rng,
            optimizer_kwargs=optimizer_kwargs,
            rgrad_kwargs=rgrad_kwargs_tt,
            setup_highorder=True,
            max_tt_rank_u2=max_tt_rank_u2,
            pymanopt_optimizer=pymanopt_optimizer,
            init=init,
        )
        folder = f"{exp_name}/{pymanopt_optimizer}/tt_au/ttrank{tt_rank}_{precond}"
        # Print experiment to run
        print(
            f"Running TT experiment: \n\texp={exp_name} \n\tN={N} \n\tk={k} \n\ttt_rank={tt_rank} \n\tmax_tt_rank_u2={max_tt_rank_u2} \n\tprecond={precond} \n\t{pymanopt_optimizer}"
        )

        res_tt = optimizer_tt.run(problem_tt, initial_point=u0_tt)
        if max_tt_rank_u2 is not None:
            folder += f"_maxu2rank{max_tt_rank_u2}"
        save_results(
            {"res": res_tt},
            folder=folder,
        )

        # Cleanup
        del res_tt, problem_tt, optimizer_tt, u0_tt, E_tt, E_highorder_tt
        jax.clear_caches()

        print(
            f"TT experiment completed successfully: N={N}, k={k}, tt_rank={tt_rank}, max_tt_rank_u2={max_tt_rank_u2}"
        )
        return True

    except Exception as e:
        print(
            f"Error in TT experiment with N={N}, k={k}, tt_rank={tt_rank}, max_tt_rank_u2={max_tt_rank_u2}: {e}"
        )
        return False


def run_full_h1_experiment(
    N,
    k,
    exp_config,
    pymanopt_optimizer,
    rng=None,
    optimizer_kwargs=None,
    init="thomas_fermi",
):
    """Run full tensor H1 Sobolev gradient flow experiment.

    Args:
        N: Number of interior grid points per dimension.
        k: Polynomial degree.
        exp_config: Experiment configuration dictionary.
        pymanopt_optimizer: Name of the optimizer.
        rng: Random number generator key (default: None).
        optimizer_kwargs: Optimizer configuration (default: None).
        init: Initialization type ('thomas_fermi', 'random', or 'constant').

    Returns:
        True if successful, False otherwise.
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)

    # Use provided configuration or default
    if optimizer_kwargs is None:
        optimizer_kwargs = DEFAULT_OPTIMIZER_KWARGS.copy()

    _, setup_full_problem = create_setup_functions(exp_config)

    dim = exp_config["dim"]
    beta = exp_config["beta"]
    exp_name = f"{exp_config['name']}_dim{dim}/beta{beta}_N{N}_k{k}"

    try:
        (
            problem_fullH1,
            optimizer_fullH1,
            u0_fullH1,
            E_fullH1,
            E_highorder_fullH1,
            rng,
        ) = setup_full_problem(
            N,
            k,
            "H1",
            rng,
            optimizer_kwargs=optimizer_kwargs,
            setup_highorder=True,
            pymanopt_optimizer=pymanopt_optimizer,
            init=init,
        )
        folder = f"{exp_name}/{pymanopt_optimizer}/full_H1"
        # Print experiment to run
        print(
            f"Running Full H1 experiment: \n\texp={exp_name} \n\tN={N} \n\tk={k} \n\t{pymanopt_optimizer}"
        )

        res_fullH1 = optimizer_fullH1.run(problem_fullH1, initial_point=u0_fullH1)
        save_results({"res": res_fullH1}, folder=folder)

        # Cleanup
        del (
            res_fullH1,
            problem_fullH1,
            optimizer_fullH1,
            u0_fullH1,
            E_fullH1,
            E_highorder_fullH1,
        )
        jax.clear_caches()

        print(f"Full H1 experiment completed successfully: N={N}, k={k}")
        return True

    except Exception as e:
        print(f"Error in H1 full experiment with N={N}, k={k}: {e}")
        return False


def run_full_au_experiment(
    N,
    k,
    exp_config,
    pymanopt_optimizer,
    rng=None,
    optimizer_kwargs=None,
    rgrad_kwargs_full=None,
    init="thomas_fermi",
):
    """Run full tensor a_u-metric gradient flow experiment.

    Args:
        N: Number of interior grid points per dimension.
        k: Polynomial degree.
        exp_config: Experiment configuration dictionary.
        pymanopt_optimizer: Name of the optimizer.
        rng: Random number generator key (default: None).
        optimizer_kwargs: Optimizer configuration (default: None).
        rgrad_kwargs_full: Riemannian gradient configuration (default: None).
        init: Initialization type ('thomas_fermi', 'random', or 'constant').

    Returns:
        True if successful, False otherwise.
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)

    # Use provided configurations or defaults
    if optimizer_kwargs is None or rgrad_kwargs_full is None:
        default_optimizer_kwargs = DEFAULT_OPTIMIZER_KWARGS.copy()
        default_rgrad_kwargs_full = DEFAULT_RGRAD_KWARGS_FULL.copy()

        if optimizer_kwargs is None:
            optimizer_kwargs = default_optimizer_kwargs
        if rgrad_kwargs_full is None:
            rgrad_kwargs_full = default_rgrad_kwargs_full

    _, setup_full_problem = create_setup_functions(exp_config)

    dim = exp_config["dim"]
    beta = exp_config["beta"]
    exp_name = f"{exp_config['name']}_dim{dim}/beta{beta}_N{N}_k{k}"

    try:
        # Get name of preconditioner for saving
        precond = rgrad_kwargs_full.get("precond", "noprecond")
        if precond is None:
            precond = "noprecond"
        else:
            precond = "precond" + precond
        # Setup functions
        (
            problem_fullau,
            optimizer_fullau,
            u0_fullau,
            E_fullau,
            E_highorder_fullau,
            rng,
        ) = setup_full_problem(
            N,
            k,
            "au",
            rng,
            optimizer_kwargs=optimizer_kwargs,
            rgrad_kwargs=rgrad_kwargs_full,
            setup_highorder=True,
            pymanopt_optimizer=pymanopt_optimizer,
            init=init,
        )
        folder = f"{exp_name}/{pymanopt_optimizer}/full_au_{precond}"
        # Print experiment to run
        print(
            f"Running Full au experiment: \n\texp={exp_name} \n\tN={N} \n\tk={k} \n\tprecond={precond} \n\t{pymanopt_optimizer}"
        )

        res_fullau = optimizer_fullau.run(problem_fullau, initial_point=u0_fullau)
        save_results(
            {"res": res_fullau},
            folder=folder,
        )

        # Cleanup
        del (
            res_fullau,
            problem_fullau,
            optimizer_fullau,
            u0_fullau,
            E_fullau,
            E_highorder_fullau,
        )
        jax.clear_caches()

        print(f"Full au experiment completed successfully: N={N}, k={k}")
        return True

    except Exception as e:
        print(f"Error in au full experiment with N={N}, k={k}: {e}")
        return False


#### MAIN EXPERIMENT FUNCTION ####
def main_exp(
    get_experiment_config,
    default_optimizer_kwargs=None,
    default_rgrad_kwargs_tt=None,
    default_rgrad_kwargs_full=None,
    default_init="thomas_fermi",
):
    """Main function to run experiments based on configuration.

    Parses the command line, merges the configurations, calls
    ``get_experiment_config(dim=..., beta=..., L=..., N=..., k=...)`` with the
    values from the command line or the JSON file (so the CLI defaults
    dim=3, beta=1000, L=6 override the defaults of ``get_experiment_config``)
    and runs the selected method. Exits with status 1 on failure.

    Args:
        get_experiment_config: Function that returns experiment configuration.
        default_optimizer_kwargs: Custom default optimizer kwargs to merge with
            DEFAULT_OPTIMIZER_KWARGS.
        default_rgrad_kwargs_tt: Custom default rgrad kwargs for TT to merge with
            DEFAULT_RGRAD_KWARGS_TT.
        default_rgrad_kwargs_full: Custom default rgrad kwargs for full to merge
            with DEFAULT_RGRAD_KWARGS_FULL.
        default_init: Default initialization type ('thomas_fermi', 'random', or
            'constant'). Used only with --config, when the file has no "init"
            key; on the command line, --init (default 'thomas_fermi') is used.
    """
    args = parse_arguments()
    rng = jax.random.PRNGKey(0)

    # Results directory: --results_dir, else $TTGPE_RESULTS_DIR, else default
    results_dir = args.results_dir or os.environ.get("TTGPE_RESULTS_DIR")
    if results_dir:
        experiment_setup.RESULTS_DIR = Path(results_dir).resolve()

    # Get default configurations (merge provided kwargs with global defaults)
    default_optimizer_kwargs = merge_configs(
        DEFAULT_OPTIMIZER_KWARGS.copy(), default_optimizer_kwargs
    )
    default_rgrad_kwargs_tt = merge_configs(
        DEFAULT_RGRAD_KWARGS_TT.copy(), default_rgrad_kwargs_tt
    )
    default_rgrad_kwargs_full = merge_configs(
        DEFAULT_RGRAD_KWARGS_FULL.copy(), default_rgrad_kwargs_full
    )

    if args.config:
        # Load configuration from file
        config = load_config_file(args.config)

        # Extract parameters from config
        method = config.get("method")
        N = config.get("N")
        k = config.get("k")
        tt_rank = config.get("tt_rank")
        dim = config.get("dim", 3)
        beta = config.get("beta", 1000)
        L = config.get("L", 6)
        max_tt_rank_u2 = config.get("max_tt_rank_u2", None)
        pymanopt_optimizer = config.get("pymanopt_optimizer", "SteepestDescent")

        # Merge optimizer and rgrad configurations with defaults
        optimizer_kwargs = merge_configs(
            default_optimizer_kwargs, config.get("optimizer_kwargs")
        )
        rgrad_kwargs_tt = merge_configs(
            default_rgrad_kwargs_tt, config.get("rgrad_kwargs_tt")
        )
        rgrad_kwargs_full = merge_configs(
            default_rgrad_kwargs_full, config.get("rgrad_kwargs_full")
        )
        init = config.get("init", default_init)

        # Validate config
        if not method or N is None or k is None:
            print("Configuration file must contain 'method', 'N', and 'k'")
            sys.exit(1)

        if method == "tt" and tt_rank is None:
            print("Configuration file must contain 'tt_rank' when method is 'tt'")
            sys.exit(1)

    else:
        # Use command line arguments
        method = args.method
        N = args.N
        k = args.k
        tt_rank = args.tt_rank
        dim = args.dim
        beta = args.beta
        L = args.L
        precond = args.precond
        max_tt_rank_u2 = args.max_tt_rank_u2
        pymanopt_optimizer = args.pymanopt_optimizer
        init = args.init

        # Use default configurations
        optimizer_kwargs = default_optimizer_kwargs
        rgrad_kwargs_tt = default_rgrad_kwargs_tt
        rgrad_kwargs_full = default_rgrad_kwargs_full
        rgrad_kwargs_tt["precond"] = precond
        rgrad_kwargs_full["precond"] = precond

    if args.max_iterations is not None:
        optimizer_kwargs = {**optimizer_kwargs, "max_iterations": args.max_iterations}

    # Get experiment configuration
    experiment_config_kwargs = {
        "dim": dim,
        "beta": beta,
        "L": L,
        "N": N,
        "k": k,
    }
    exp_config = get_experiment_config(**experiment_config_kwargs)

    # Run the specified experiment
    success = False
    if method == "tt":
        success = run_tt_experiment(
            N,
            k,
            tt_rank,
            exp_config,
            pymanopt_optimizer,
            rng,
            optimizer_kwargs,
            rgrad_kwargs_tt,
            max_tt_rank_u2,
            init=init,
        )
    elif method == "full_H1":
        success = run_full_h1_experiment(
            N, k, exp_config, pymanopt_optimizer, rng, optimizer_kwargs, init=init
        )
    elif method == "full_au":
        success = run_full_au_experiment(
            N,
            k,
            exp_config,
            pymanopt_optimizer,
            rng,
            optimizer_kwargs,
            rgrad_kwargs_full,
            init=init,
        )
    else:
        print(f"Unknown method: {method}")
        sys.exit(1)

    if success:
        print("Experiment completed successfully!")
    else:
        print("Experiment failed!")
        sys.exit(1)
