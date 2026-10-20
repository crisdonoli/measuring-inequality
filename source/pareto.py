import pandas as pd
import numpy as np


def estimate_alpha_pareto(wealth, weights, w_min, method="mle"):
    """Estimate Pareto tail index alpha above w_min.

    Args:
        wealth (pd.Series): Wealth values.
        weights (pd.Series): Survey weights.
        w_min (float): Tail threshold.
        method (str): 'mle' (weighted MLE) or 'reg' (Gabaix-Ibragimov).

    Returns:
        float: Estimated alpha.
    """
    mask = wealth >= w_min
    w = wealth[mask]
    wt = weights[mask]
    if method == "mle":
        # pseudo-MLE with survey weights: alpha = 1 / weighted mean(log(w/w_min))
        log_ratio = np.log(w / w_min)
        wt_norm = wt / wt.sum()
        alpha = 1.0 / (log_ratio * wt_norm).sum()
        return float(alpha)
    # regression Gabaix-Ibragimov: log((rank-0.5)*mean_weight) ~ -alpha log(w)
    w_sorted = np.sort(w.values)[::-1]
    wt_sorted = wt.iloc[np.argsort(w.values)[::-1]].values
    # rank with weights
    rank = np.cumsum(wt_sorted)
    N = wt.sum()
    y = np.log((rank - 0.5 * wt_sorted) / N)
    x = np.log(w_sorted)
    # OLS slope
    alpha = -np.cov(x, y, bias=True)[0, 1] / np.var(x)
    return float(alpha)


def generate_pareto(n, w_min, alpha, w_max=None, seed=0, w_low=None):
    """Draw n Pareto samples, optionally truncated.

    Args:
        n (int): Number of draws.
        w_min (float): Scale parameter.
        alpha (float): Tail index.
        w_max (float|None): Upper truncation.
        seed (int): RNG seed.
        w_low (float|None): Lower truncation.

    Returns:
        np.ndarray: Pareto draws.
    """
    rng = np.random.default_rng(seed)
    if w_low is not None and w_max is not None:
        # conditional Pareto between w_low and w_max
        Fa = 1 - (w_min / w_low) ** alpha if w_low >= w_min else 0
        Fb = 1 - (w_min / w_max) ** alpha if w_max >= w_min else 1
        u = rng.random(n) * (Fb - Fa) + Fa
        w = w_min * (1 - u) ** (-1 / alpha)
        return w
    u = rng.random(n)
    w = w_min * (1 - u) ** (-1 / alpha)
    if w_max is not None:
        w = w[w <= w_max]
        while len(w) < n:
            extra = w_min * (1 - rng.random(n - len(w))) ** (-1 / alpha)
            if w_max is not None:
                extra = extra[extra <= w_max]
            w = np.concatenate([w, extra])
        w = w[:n]
    if w_low is not None:
        w = w[w >= w_low]
        while len(w) < n:
            extra = w_min * (1 - rng.random(n - len(w))) ** (-1 / alpha)
            extra = extra[extra >= w_low]
            if w_max is not None:
                extra = extra[extra <= w_max]
            w = np.concatenate([w, extra])
        w = w[:n]
    return w


def uprate(df, totals, wealth_col="net_wealth"):
    """Rescale wealth to match macro totals.

    Args:
        df (pd.DataFrame): Data with 'factor' weights.
        totals (float|dict): Scalar total or {col: total} dict.
        wealth_col (str): Column to scale if totals is scalar.

    Returns:
        pd.DataFrame: DataFrame with rescaled values (in place).
    """
    # totals: dict {component: macro_total} or single total for net wealth
    # simplified: scale net wealth to macro total
    if isinstance(totals, dict):
        for col, tot in totals.items():
            if col in df.columns:
                cur = (df[col] * df["factor"]).sum()
                df[col] = df[col] * (tot / cur) if cur != 0 else df[col]
        return df
    cur = (df[wealth_col] * df["factor"]).sum()
    factor = totals / cur if cur != 0 else 1.0
    df[wealth_col] = df[wealth_col] * factor
    return df


def add_forbes(df, forbes_df, wealth_col="net_wealth", weight_col="factor", forbes_wealth_col="networth-clp"):
    """Append Forbes billionaires to survey data.

    Args:
        df (pd.DataFrame): Survey data.
        forbes_df (pd.DataFrame): Forbes data.
        wealth_col (str): Target wealth column in df.
        weight_col (str): Weight column (Forbes gets 1).
        forbes_wealth_col (str): Source wealth column in forbes_df.

    Returns:
        pd.DataFrame: Combined data.
    """
    forbes_obs = pd.DataFrame(
        {
            wealth_col: forbes_df[forbes_wealth_col].values,
            weight_col: 1,
            "id": ["forbes_" + str(i) for i in range(len(forbes_df))],
        }
    )
    return pd.concat([df, forbes_obs], ignore_index=True)


def _wmin_from_percentile(df, wealth_col="net_wealth", weight_col="factor", p=0.99):
    """Wealth threshold at weighted percentile p.

    Args:
        df (pd.DataFrame): Data.
        wealth_col (str): Wealth column.
        weight_col (str): Weight column.
        p (float): Percentile (0-1).

    Returns:
        float: Threshold value.
    """
    df_sorted = df.sort_values(wealth_col)
    cum = df_sorted[weight_col].cumsum() / df_sorted[weight_col].sum()
    return df_sorted.loc[cum >= p, wealth_col].iloc[0]


def vermeulen_correction(df, forbes_df, w_min=None, wealth_col="net_wealth", weight_col="factor", forbes_wealth_col="networth-clp", alpha_method="reg", seed=0, p_wmin=0.99):
    """Vermeulen tail correction: estimate alpha, fill gap, append Forbes.

    Args:
        df (pd.DataFrame): Survey data.
        forbes_df (pd.DataFrame): Forbes data.
        w_min (float|None): Tail threshold; if None, derived from p_wmin.
        wealth_col (str): Wealth column.
        weight_col (str): Weight column.
        forbes_wealth_col (str): Forbes wealth column.
        alpha_method (str): 'mle' or 'reg' for alpha estimation.
        seed (int): RNG seed for synthetic draws.
        p_wmin (float): Percentile for w_min if not given.

    Returns:
        (pd.DataFrame, float): Corrected data and estimated alpha.
    """
    if w_min is None:
        w_min = _wmin_from_percentile(df, wealth_col, weight_col, p=p_wmin)
    # estimate alpha on tail including Forbes (tail = >= w_min)
    combined_wealth = pd.concat([df[wealth_col], forbes_df[forbes_wealth_col]], ignore_index=True)
    combined_weights = pd.concat([df[weight_col], pd.Series([1] * len(forbes_df))], ignore_index=True)
    alpha = estimate_alpha_pareto(combined_wealth, combined_weights, w_min, method=alpha_method)

    w_max_forbes = forbes_df[forbes_wealth_col].min()
    w_max_efh = df[wealth_col].max()
    # fill gap [w_max_efh, w_max_forbes] with conditional Pareto — Stata ft logic approx
    n_synth = max(300, int(len(forbes_df) * 150))
    synth = generate_pareto(n_synth, w_min, alpha, w_max=w_max_forbes, w_low=w_max_efh, seed=seed)
    synth_df = pd.DataFrame(
        {
            wealth_col: synth,
            weight_col: 1,
            "id": ["synth_" + str(i) for i in range(len(synth))],
        }
    )
    corrected = pd.concat([df, synth_df], ignore_index=True)
    # add forbes as well
    corrected = add_forbes(corrected, forbes_df, wealth_col, weight_col, forbes_wealth_col)
    return corrected, alpha
