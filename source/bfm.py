"""BFM (2022) — minimal Python implementation (math-first)."""
import re
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


def _fractal_thresholds(tax_n, tax_tramos, at_comercial):
    """126 fractal thresholds in CLP via linear interpolation of SII brackets."""
    import numpy as np
    lowers_uta = np.array([_tramo_lower_uta(t) for t in tax_tramos], dtype=float)
    yk_coarse = lowers_uta * UTA_CLP[at_comercial]  # lower bounds coarse
    yk_coarse = np.maximum.accumulate(yk_coarse)
    tax_share = tax_n / tax_n.sum()
    cum_coarse = np.concatenate([[0], np.cumsum(tax_share)[:-1]])  # F_t at yk_coarse
    # Desired fractal percentiles (BFM): 0-99 (1pp), 99-99.9 (0.1pp), 99.9-99.99 (0.01pp), 99.99-99.999 (0.001pp)
    pcts = list(range(0, 100))  # 0..99
    pcts += [99 + 0.1*i for i in range(1, 10)]  # 99.1..99.9
    pcts += [99.9 + 0.01*i for i in range(1, 10)]  # 99.91..99.99
    pcts += [99.99 + 0.001*i for i in range(1, 10)]  # 99.991..99.999
    pcts = np.array(pcts) / 100  # to 0-1
    # Interpolate y at pcts via coarse points (linear, top tail linear in levels)
    # Ensure cum_coarse sorted and yk_coarse increasing; append final point (1.0, max yk)
    # For pct beyond max cum, extrapolate linearly using last segment
    y_fractal = np.interp(pcts, cum_coarse, yk_coarse, left=0, right=yk_coarse[-1]*1.5)
    return y_fractal, pcts


def interpolate_tax_tabulation(*args, **kwargs):
    """Placeholder — gpinter interpolation is done via yk quantiles in estimate_merging_point."""
    raise NotImplementedError("Use estimate_merging_point directly (gpinter approximated via survey quantiles).")


def estimate_merging_point(survey_y, survey_w, tax_n, tax_tramos=None, at_comercial=None, fractal=False):
    """Estimate y* via theta-hat vs Theta-hat.

    Args:
        survey_y (np.ndarray): household income (y_i).
        survey_w (np.ndarray): weights (d_i).
        tax_n (np.ndarray): cons_n per bracket.
        tax_tramos (list[str] | None): SII labels like 'Tramo 2 - 13,5 a 30 UTA...' to get y_k in CLP.
        at_comercial (int | None): 2017 or 2022 to pick UTA value.

    Returns:
        y_star, tilde, theta_hat, Theta_hat, yk
    """
    tax_share = tax_n / tax_n.sum()  # w_k = tax share of bracket k
    # y_k: fiscal thresholds in CLP — use UTA brackets if provided, else fallback to survey quantiles (forces tilde~1)
    if tax_tramos is not None and at_comercial in UTA_CLP:
        lowers_uta = np.array([_tramo_lower_uta(t) for t in tax_tramos], dtype=float)
        yk = lowers_uta * UTA_CLP[at_comercial]  # y_k in CLP
        # Ensure strictly increasing and drop duplicate 0
        # yk[0] is 0, keep as is; ensure monotonic
        yk = np.maximum.accumulate(yk)
    else:
        cum_tax = np.cumsum(tax_share)  # F_t(y_k)
        order = np.argsort(survey_y)  # sort for empirical CDF
        ys, ws = survey_y[order], survey_w[order]  # y_(i), d_(i) sorted
        cum_s = np.cumsum(ws) / ws.sum()  # \hat F_s(y)
        yk = np.interp(cum_tax, cum_s, ys)  # y_k = \hat F_s^{-1}(F_t) (fallback, approximate)
    # Fractal bins: override yk/tax_share with 126 fine bins if requested
    if fractal and tax_tramos is not None and at_comercial in UTA_CLP:
        yk, pcts = _fractal_thresholds(tax_n, tax_tramos, at_comercial)
        # tax_share for fractal is by construction: 1pp, 0.1pp, 0.01pp, 0.001pp
        tax_share = np.concatenate([np.full(99, 0.01), np.full(9, 0.001), np.full(9, 0.0001), np.full(9, 0.00001)])
        # yk are lower bounds fractal (127 thresholds -> 126 bins, drop last)
        yk = yk[:-1]  # 126 lowers
    # yk are lower bounds; upper bounds are next lower + inf for last bracket
    yk_upper = np.concatenate([yk[1:], [np.inf]])
    idx = np.digitize(survey_y, yk_upper, right=False)  # bin k: y in [yk[k], yk_upper[k])
    idx = np.clip(idx, 0, len(tax_n) - 1)  # safety
    share_s = np.array([survey_w[idx == k].sum() / survey_w.sum() for k in range(len(tax_share))])  # share_{s,k}
    F_s_before = np.concatenate([[0], np.cumsum(share_s)[:-1]])  # \hat F_s(y_k^-)
    F_t_before = np.concatenate([[0], np.cumsum(tax_share)[:-1]])  # F_t(y_k^-)
    Theta_hat = (1 - F_s_before) / (1 - F_t_before)  # \hat\Theta_k = (1-\hat F_s)/(1-F_t)
    Theta_hat[0] = 1.0  # by definition at y_0
    tilde = np.divide(share_s, tax_share, out=np.ones_like(share_s), where=tax_share > 0)  # \tilde\theta_k = share_{s,k}/w_k
    ir = IsotonicRegression(increasing=False, out_of_bounds='clip')  # PAVA
    theta_hat = ir.fit_transform(np.arange(len(tilde)), tilde, sample_weight=tax_share)  # \hat\theta_k
    below = np.where(theta_hat < Theta_hat)[0]  # {k: \hat\theta_k < \hat\Theta_k}
    k_star = below[0] if len(below) else len(tilde) - 1  # k^* = min{k: \hat\theta_k < \hat\Theta_k}
    y_star = yk[k_star] if k_star < len(yk) else yk[-1]  # y^* = y_{k^*}
    return y_star, tilde, theta_hat, Theta_hat, yk


def calibrate_weights(survey_y, survey_w, y_star, tax_n, alpha=5):
    """Uniform rescaling to match tax mass above y* (truncated).

    Args:
        survey_y, survey_w, y_star, tax_n, alpha

    Returns:
        w_cal
    """
    tax_share = tax_n / tax_n.sum()  # w_k
    order = np.argsort(survey_y)  # for y_k
    ys, ws = survey_y[order], survey_w[order]
    cum_s = np.cumsum(ws) / ws.sum()  # \hat F_s
    yk = np.interp(np.cumsum(tax_share), cum_s, ys)  # y_k thresholds (fallback)
    k_star = int(np.searchsorted(yk, y_star))  # bracket of y^*
    k_star = max(0, min(k_star, len(tax_share) - 1))
    tax_share_above = tax_share[k_star:].sum()  # 1 - F_t(y^*)
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
