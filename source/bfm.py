"""BFM (2022) — minimal Python implementation (math-first)."""
import re
import warnings
import tempfile
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

# UTA in CLP (approx, Dec 31): commercial year -> CLP per UTA
UTA_CLP = {2017: 560000, 2022: 700000}  # 12*UTM avg, used to convert UTA brackets to CLP


def _tramo_lower_uta(tramo: str) -> float:
    """Parse lower bound in UTA from SII label like 'Tramo 2 - 13,5 a 30 UTA (Tasa 4%)' or 'Tramo 8 - Más de 150 UTA'."""
    # Extract first number after '-' or 'Más de'
    m = re.search(r"Más de\s+([\d,]+)", tramo)
    if m:
        return float(m.group(1).replace(",", "."))
    m = re.search(r"-\s*([\d,]+)\s+a", tramo)
    if m:
        return float(m.group(1).replace(",", "."))
    return 0.0


def _fractal_percentiles():
    """Fractal percentiles for BFM: 0-99 (1pp), 99-99.9 (0.1pp), 99.9-99.99 (0.01pp), 99.99-99.999 (0.001pp)."""
    pcts = list(range(0, 100))  # 0..99
    pcts += [99 + 0.1*i for i in range(1, 10)]  # 99.1..99.9
    pcts += [99.9 + 0.01*i for i in range(1, 10)]  # 99.91..99.99
    pcts += [99.99 + 0.001*i for i in range(1, 10)]  # 99.991..99.999
    return np.array(pcts) / 100  # to 0-1


def fractal_percentiles_from(p_start):
    """Fractal grid restricted to the fiscal support [p_start, 1).

    Args:
        p_start (float): F_t(y_2) in population space (start of the fiscal CDF).

    Returns:
        np.ndarray: percentiles in [p_start, 1) as fractions in 0-1, p_start prepended
        when the fractal grid does not contain it.
    """
    pcts = _fractal_percentiles()
    p_global = pcts[pcts >= p_start]  # fractal grid on [p_start, 1)
    if len(p_global) == 0 or p_global[0] > p_start:
        p_global = np.concatenate([[p_start], p_global])
    return p_global


def _linear_interpolation(lower_bounds, tax_n, percentiles):
    """Fallback linear interpolation of thresholds at requested percentiles."""
    tax_share = tax_n / tax_n.sum()
    yk = np.maximum.accumulate(lower_bounds)
    cum = np.concatenate([[0], np.cumsum(tax_share)[:-1]])
    return np.interp(percentiles, cum, yk, left=0, right=yk[-1]*1.5)


def interpolate_tax_tabulation(lower_bounds, tax_n, tax_renta, percentiles=None):
    """Interpolate tax tabulation via gpinter (R) and return quantiles at percentiles.

    Args:
        lower_bounds (np.ndarray): lower bounds of tax brackets in CLP.
        tax_n (np.ndarray): number of taxpayers per bracket.
        tax_renta (np.ndarray): total income per bracket (same units as lower_bounds).
        percentiles (np.ndarray | None): percentiles at which to evaluate quantiles.
            Defaults to the BFM fractal grid (127 points).

    Returns:
        np.ndarray: quantiles at the requested percentiles.
    """
    if percentiles is None:
        percentiles = _fractal_percentiles()

    df = pd.DataFrame({
        "lower_bound_clp": lower_bounds,
        "n_k": tax_n,
        "total_income_k": tax_renta,
    })

    repo_root = Path(__file__).resolve().parent.parent
    r_script = repo_root / "source" / "gpinter_bridge.R"

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.csv"
        output_path = Path(tmpdir) / "output.csv"
        pcts_path = Path(tmpdir) / "percentiles.csv"
        df.to_csv(input_path, index=False)
        pd.DataFrame({"percentile": percentiles}).to_csv(pcts_path, index=False)

        cmd = ["Rscript", str(r_script), str(input_path), str(output_path), str(pcts_path)]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            warnings.warn(
                f"gpinter/R unavailable or failed ({type(exc).__name__}: {exc}); "
                "falling back to linear interpolation of tax tabulation.",
                stacklevel=2,
            )
            return _linear_interpolation(lower_bounds, tax_n, percentiles)

        try:
            out = pd.read_csv(output_path)
            quantiles = out["quantile"].values[:len(percentiles)]
            if len(quantiles) != len(percentiles):
                raise ValueError(f"gpinter output length mismatch: {len(quantiles)} vs {len(percentiles)}")
            return quantiles
        except Exception as exc:
            warnings.warn(
                f"Failed to read gpinter output ({type(exc).__name__}: {exc}); "
                "falling back to linear interpolation of tax tabulation.",
                stacklevel=2,
            )
            return _linear_interpolation(lower_bounds, tax_n, percentiles)


def _fiscal_p_start(tax_n, n_pop):
    """F_t(y2) in population space: tramo 1 is exempt, so the tabulation has no mass below y2."""
    kappa = tax_n.sum() / n_pop  # taxpayer coverage of the population
    return 1.0 - kappa + kappa * (tax_n[0] / tax_n.sum())


def tax_quantiles(tax_n, tax_renta, tax_tramos, at_comercial, n_pop, pcts):
    """Fiscal quantiles at arbitrary percentiles p in [p_start, 1).

    Only brackets with fiscal data (tramo 2 onwards) enter the fit, so probabilities are
    rescaled from population space onto the fitted support: p_fit = (p - p_start)/(1 - p_start).

    Args:
        tax_n (np.ndarray): cons_n per bracket.
        tax_renta (np.ndarray): cons_renta per bracket, in millions of CLP.
        tax_tramos (list[str]): SII labels to get y_k in CLP.
        at_comercial (int): 2017 or 2022 to pick the UTA value.
        n_pop (float): population total (sum of survey weights) behind kappa.
        pcts (np.ndarray): percentiles in [p_start, 1).

    Returns:
        (quantiles, p_start): quantiles in CLP (strictly increasing) and F_t(y2).
    """
    p_start = _fiscal_p_start(tax_n, n_pop)
    lowers_uta = np.array([_tramo_lower_uta(t) for t in tax_tramos], dtype=float)
    y_coarse = np.maximum.accumulate(lowers_uta * UTA_CLP[at_comercial])

    p = np.asarray(pcts, dtype=float)
    p_fit = np.clip((p - p_start) / (1.0 - p_start), 0.0, 1.0 - 1e-12)  # grid rescaled to [0, 1) for gpinter
    # cons_renta is in millions of CLP; gpinter needs income in the same units as the thresholds
    q = interpolate_tax_tabulation(y_coarse[1:], tax_n[1:], tax_renta[1:] * 1e6, percentiles=p_fit)

    # Make strictly increasing to avoid zero-width bins.
    q = np.maximum.accumulate(q)
    eps = np.maximum(1.0, np.abs(q).max() * 1e-12)
    q = np.maximum.accumulate(q + np.arange(len(q)) * eps * 1e-6)
    return q, float(p_start)


def _fractal_thresholds(tax_n, tax_renta, tax_tramos, at_comercial, n_pop):
    """Fractal bins in CLP from the fiscal CDF above y2 (SII has no data below).

    Tramo 1 is exempt (0 to 13.5 UTA), so the tabulation carries no information
    below y2: the fiscal CDF is fitted and evaluated only on [y2, inf), anchored
    at F_t(y2) = p_start in population space.

    Returns:
        yk_edges (np.ndarray): fractal thresholds (quantiles) in CLP, first = y2.
        tax_share (np.ndarray): bin masses in population space (sum 1 - p_start).
        p_start (float): F_t(y2) in population space.
    """
    p_start = _fiscal_p_start(tax_n, n_pop)  # F_t(y2), population space

    p_global = fractal_percentiles_from(p_start)  # fractal grid on [p_start, 1)

    yk_edges, p_start = tax_quantiles(tax_n, tax_renta, tax_tramos, at_comercial, n_pop, p_global)

    tax_share = np.diff(p_global)  # population-space mass per bin; kept unnormalized so F_t stays in [p_start, 1)
    return yk_edges, tax_share, float(p_start)


def estimate_merging_point(survey_y, survey_w, tax_n, tax_renta=None, tax_tramos=None, at_comercial=None, fractal=False):
    """Estimate y* via theta-hat vs Theta-hat.

    Args:
        survey_y (np.ndarray): household income (y_i).
        survey_w (np.ndarray): weights (d_i).
        tax_n (np.ndarray): cons_n per bracket.
        tax_renta (np.ndarray | None): cons_renta per bracket (required for fractal=True).
        tax_tramos (list[str] | None): SII labels to get y_k in CLP.
        at_comercial (int | None): 2017 or 2022 to pick UTA value.
        fractal (bool): use fractal bins on [y2, inf) derived from gpinter interpolation.

    Returns:
        y_star, tilde, theta_hat, Theta_hat, yk
    """
    tax_share = tax_n / tax_n.sum()  # w_k = tax share of bracket k
    # y_k: fiscal thresholds in CLP — use UTA brackets if provided, else fallback to survey quantiles
    if tax_tramos is not None and at_comercial in UTA_CLP:
        lowers_uta = np.array([_tramo_lower_uta(t) for t in tax_tramos], dtype=float)
        yk = lowers_uta * UTA_CLP[at_comercial]  # y_k in CLP
        yk = np.maximum.accumulate(yk)
    else:
        cum_tax = np.cumsum(tax_share)  # F_t(y_k)
        order = np.argsort(survey_y)  # sort for empirical CDF
        ys, ws = survey_y[order], survey_w[order]  # y_(i), d_(i) sorted
        cum_s = np.cumsum(ws) / ws.sum()  # \hat F_s(y)
        yk = np.interp(cum_tax, cum_s, ys)  # y_k = \hat F_s^{-1}(F_t) (fallback, approximate)

    # Fractal bins: override yk/tax_share with fine bins on [y2, inf) if requested
    f_t_start = 0.0  # F_t(y_0^-) — 0 unless the grid starts at y2 (fractal branch)
    f_s_start = 0.0  # \hat F_s(y_0^-)
    if fractal and tax_tramos is not None and at_comercial in UTA_CLP:
        if tax_renta is None:
            raise ValueError("tax_renta is required when fractal=True")
        yk_edges, tax_share, p_start = _fractal_thresholds(
            tax_n, tax_renta, tax_tramos, at_comercial, float(np.sum(survey_w))
        )
        yk = yk_edges[:-1]  # lower bounds; bin 0 starts at y2 (no fiscal data below)
        f_t_start = p_start  # F_t(y2) anchored in population space
        f_s_start = survey_w[survey_y < yk[0]].sum() / survey_w.sum()  # \hat F_s(y2)

    # yk are lower bounds; upper bounds are next lower + inf for last bracket
    yk_upper = np.concatenate([yk[1:], [np.inf]])
    idx = np.digitize(survey_y, yk_upper, right=False)  # bin k: y in [yk[k], yk_upper[k])
    idx = np.clip(idx, 0, len(tax_share) - 1)  # safety using current grid length
    if f_t_start > 0.0:
        idx[survey_y < yk[0]] = -1  # survey mass below y2 has no fiscal counterpart
    share_s = np.array([survey_w[idx == k].sum() / survey_w.sum() for k in range(len(tax_share))])  # share_{s,k}
    F_s_before = np.concatenate([[f_s_start], f_s_start + np.cumsum(share_s)[:-1]])  # \hat F_s(y_k^-)
    F_t_before = np.concatenate([[f_t_start], f_t_start + np.cumsum(tax_share)[:-1]])  # F_t(y_k^-)
    Theta_hat = (1 - F_s_before) / (1 - F_t_before)  # \hat\Theta_k = (1-\hat F_s)/(1-F_t)
    tilde = np.divide(share_s, tax_share, out=np.ones_like(share_s), where=tax_share > 0)  # \tilde\theta_k = share_{s,k}/w_k
    ir = IsotonicRegression(increasing=False, out_of_bounds='clip')  # PAVA
    theta_hat = ir.fit_transform(np.arange(len(tilde)), tilde, sample_weight=tax_share)  # \hat\theta_k
    below = np.where(theta_hat < Theta_hat)[0]  # {k: \hat\theta_k < \hat\Theta_k}
    k_star = below[0] if len(below) else len(tilde) - 1  # k^* = min{k: \hat\theta_k < \hat\Theta_k}
    y_star = yk[k_star] if k_star < len(yk) else yk[-1]  # y^* = y_{k^*}
    return y_star, tilde, theta_hat, Theta_hat, yk


def calibrate_weights(survey_y, survey_w, y_star, tax_n, alpha=5, yk=None, tax_share=None, f_t_start=0.0):
    """Uniform rescaling to match tax mass above y* (truncated).

    Args:
        survey_y, survey_w, y_star, tax_n, alpha
        yk (np.ndarray | None): fiscal lower-bound thresholds. If provided, used
            instead of interpolating from the survey CDF.
        tax_share (np.ndarray | None): fiscal bin shares matching yk. If provided
            with yk and lengths match, used instead of the coarse tax_n shares.
        f_t_start (float): F_t(yk[0]) offset (p_start in the fractal branch), i.e.
            fiscal mass below the first threshold where SII has no information.

    Returns:
        w_cal
    """
    coarse_share = tax_n / tax_n.sum()  # w_k from coarse brackets
    if tax_share is not None and yk is not None and len(tax_share) == len(yk):
        pass  # use provided fine fiscal grid
    elif yk is not None:
        pcts = _fractal_percentiles()
        derived = np.diff(pcts)
        derived = derived / derived.sum()
        if len(derived) == len(yk):
            tax_share = derived
        else:
            warnings.warn(
                f"yk length {len(yk)} does not match fractal grid {len(derived)}; "
                "falling back to tax_n.",
                stacklevel=2,
            )
            tax_share = coarse_share
            yk = None
    else:
        tax_share = coarse_share

    if yk is None:
        order = np.argsort(survey_y)  # for y_k
        ys, ws = survey_y[order], survey_w[order]
        cum_s = np.cumsum(ws) / ws.sum()  # \hat F_s
        yk = np.interp(np.cumsum(tax_share), cum_s, ys)  # y_k thresholds (fallback)

    F_t_before = f_t_start + np.concatenate([[0.0], np.cumsum(tax_share)[:-1]])  # F_t(y_k^-)
    k_star = int(np.searchsorted(yk, y_star))  # bracket of y^*
    k_star = max(0, min(k_star, len(tax_share) - 1))
    tax_share_above = 1.0 - F_t_before[k_star]  # 1 - F_t(y^*)
    share_s_above = survey_w[survey_y >= y_star].sum() / survey_w.sum() if survey_w.sum() > 0 else 1.0  # 1 - \hat F_s(y^*)
    factor_above = (tax_share_above / share_s_above) if share_s_above > 0 else 1.0  # w_i/d_i above y^*
    factor_above = float(np.clip(factor_above, 1 / alpha, alpha))  # truncated to [1/\alpha,\alpha]
    N_s_total = survey_w.sum()  # N_s
    above_mask = survey_y >= y_star  # \mathbf{1}\{y_i \ge y^*\}
    N_s_above = survey_w[above_mask].sum()  # N_{s,\ge y^*}
    factor_below = (N_s_total - factor_above * N_s_above) / (N_s_total - N_s_above) if N_s_total != N_s_above else 1.0  # keep total population
    factor_below = float(np.clip(factor_below, 1 / alpha, alpha))
    w_cal = survey_w.copy().astype(float)  # w_i^{cal}
    w_cal[above_mask] *= factor_above  # reweight above y^*
    w_cal[~above_mask] *= factor_below  # reweight below y^*
    return w_cal


def expand_support(survey_y, survey_w, y_star, tax_n, tax_renta, tax_tramos, at_comercial):
    """Rank-preserving quantile mapping of the survey tail onto the fiscal CDF above y*.

    Observations at or above y* are re-anchored on the fiscal distribution: the mid-rank
    u_i inside the tail is mapped to the population percentile p_i = F_s(y*) + (1-F_s(y*))*u_i
    and replaced by max(y_i, q_i) with q_i the fiscal quantile at p_i. Weights are unchanged,
    so only positions move and the ordering within the tail is preserved.

    Args:
        survey_y (np.ndarray): household income (y_i), calibrated or raw.
        survey_w (np.ndarray): weights (w_i); their sum defines the population behind kappa.
        y_star (float): merging point.
        tax_n, tax_renta, tax_tramos, at_comercial: SII tabulation (cons_renta in MM CLP).

    Returns:
        np.ndarray: y with the support extended to the fiscal top.
    """
    y = np.asarray(survey_y, dtype=float)
    w = np.asarray(survey_w, dtype=float)
    F_s_star = w[y < y_star].sum() / w.sum()  # \hat F_s(y^*)

    mask = y >= y_star
    if not np.any(mask):
        return y.copy()

    idx = np.flatnonzero(mask)
    order = idx[np.argsort(y[idx], kind="mergesort")]  # tail sorted by income
    w_ord = w[order]
    u = (np.cumsum(w_ord) - w_ord / 2) / w_ord.sum()  # mid-rank within the tail
    p = np.minimum(F_s_star + (1.0 - F_s_star) * u, 1.0 - 1e-6)  # population percentiles

    q, _ = tax_quantiles(tax_n, tax_renta, tax_tramos, at_comercial, w.sum(), p)

    # Tied survey incomes keep a single assigned income, so y -> y_new stays monotone.
    y_sorted = y[order]
    first = np.ones(len(q), dtype=bool)
    first[1:] = y_sorted[1:] != y_sorted[:-1]
    q = np.maximum.reduceat(q, np.flatnonzero(first))[np.cumsum(first) - 1]

    y_new = y.copy()
    y_new[order] = np.maximum(y_sorted, q)  # same weights, only positions move
    return y_new
