"""Exponential-sum approximations of 1/x for the TT preconditioner.

The "S+V" (and "S") preconditioner in the TT format (Section 'Preconditioning',
subsection 'Preconditioning in the TT format') replaces the entry-wise
reciprocal of the eigenvalue tensor K by an exponential sum
``1/mu ~ sum_i omega_i exp(-alpha_i mu)`` on an interval ``[1, R]``. The
coefficients ``(omega_i, alpha_i)`` are read from the precomputed tables of
Hackbusch, shipped in ``src/external/EXP_SUM/1_x_rel/Rel1_x_n``: each file
``Rel1_x_n{k}.{a}E{b}`` contains the best approximation with ``k`` terms on
``[1, a * 10**b]`` (relative error). This module only locates and parses those
files; the preconditioner itself is assembled in ``core.energy``.
"""

import math

import numpy as np




def to_filename(x, k):
    """Return the names of the coefficient files covering the interval [1, x].

    The tables are indexed by the right end ``R = a * 10**b`` of the interval
    ``[1, R]``, with a one-digit mantissa ``a``. The first candidate rounds the
    mantissa of ``x`` up, so that ``[1, x]`` is contained in the tabulated
    interval; the second candidate is the next power of ten, used by the caller
    when the first file does not exist (e.g. when the rounded mantissa is 10).

    Args:
        x: Right end ``R = max(K) / min(K)`` of the approximation interval.
        k: Number of terms of the exponential sum.

    Returns:
        Tuple ``(file1, file2)`` of candidate file names, e.g.
        ``("Rel1_x_n10.4E2", "Rel1_x_n10.1E3")`` for ``x=345, k=10``. For
        ``x == 0`` the string ``"0E0"`` is returned instead.
    """
    if x == 0:
        return "0E0"
    b = math.floor(math.log10(abs(x)))  # Find exponent
    a = math.ceil(x / (10**b))  # Calculate ceiling of significant
    file1 = f"Rel1_x_n{k}.{a}E{b}"
    if a == 1 and b != 0:
        file2 = file1
    else:
        file2 = f"Rel1_x_n{k}.1E{b+1}"
    return file1, file2


def extract_coefficients(filepath):
    """Read the coefficients of an exponential sum from a table file.

    The file stores the approximation ``1/x ~ sum_k a[k] exp(-b[k] x)`` on
    ``[1, R]``; in the notation of Section 'Preconditioning in the TT format'
    the weights are ``omega_k = a[k]`` and the exponents ``alpha_k = b[k]``.

    Args:
        filepath: Path to a ``Rel1_x_n*`` coefficient file.

    Returns:
        Tuple ``(omega, alpha, xi)`` of lists: weights ``a[k]``, exponents
        ``b[k]`` and the interpolation points ``xi[k]`` of the best
        approximation.
    """
    omega = []  # a[k]
    alpha = []  # b[k]
    xi = []  # xi[k]

    with open(filepath, "r") as file:
        for line in file:
            # Strip and skip empty or short lines
            line = line.strip()
            if not line:
                continue

            # Match patterns like: "value   coefficient a[1]"
            if "coefficient a[" in line:
                value = float(line.split()[0])
                omega.append(value)

            elif "coefficient b[" in line:
                value = float(line.split()[0])
                alpha.append(value)

            elif "interpolation point xi[" in line:
                value = float(line.split()[0])
                xi.append(value)

    return omega, alpha, xi
