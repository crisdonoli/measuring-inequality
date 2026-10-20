import matplotlib.pyplot as plt
import re
import numpy as np

from pathlib import Path


def fmt(v):
    """Format wealth in billions of CLP.

    Args:
        v (float): Value in CLP.

    Returns:
        str: Formatted string like '1.2 B'.
    """
    return f"{v/1e12:.1f} B"


def tex_escape(s):
    """Escape LaTeX special characters.

    Args:
        s (str): Raw string.

    Returns:
        str: Escaped string.
    """
    return re.sub(r"([&%$_#{}])", lambda m: "\\" + m.group(1), s)


def plot_bar_billionaires(df, fig, ax):
    """Horizontal bar chart of Forbes billionaires.

    Args:
        df (pd.DataFrame): Must have 'Name' and 'networth-clp'.
        fig (Figure): Matplotlib figure.
        ax (Axes): Target axes.

    Returns:
        (Figure, Axes): Updated figure and axes.
    """
    names = df["Name"].apply(tex_escape)
    ax.barh(names, df["networth-clp"], color="darkblue")
    for i, value in enumerate(df["networth-clp"]):
        ax.text(value * 1.02, i, fmt(value), va="center")

    ax.set_ylabel("Name")
    ax.set_xlabel("Net Worth (CLP)")
    ax.set_xlim(0, df["networth-clp"].max() + 0.3e13)
    # ax.set_title("Chilean Billionaires 2021 --- Net Worth (Forbes)")
    return fig, ax


def plot_wealth_weighted_vs_unweighted(df, fig, ax, wealth_col="net_wealth", weight_col="factor", bins=50):
    """Overlay weighted vs unweighted density histograms.

    Args:
        df (pd.DataFrame): Household data.
        fig (Figure): Figure.
        ax (Axes): Axes.
        wealth_col (str): Wealth column.
        weight_col (str): Weight column.
        bins (int): Number of bins.

    Returns:
        (Figure, Axes): Updated figure and axes.
    """
    values = df[wealth_col].dropna()
    w = df.loc[values.index, weight_col].astype(float)
    edges = np.histogram_bin_edges(values, bins=bins)
    # outline only, no fill
    ax.hist(values, bins=edges, density=True, histtype="step", fill=False, linewidth=1.5, linestyle="-", color="darkblue", label="Unweighted")
    ax.hist(values, bins=edges, weights=w, density=True, histtype="step", fill=False, linewidth=1.5, linestyle="--", color="darkred", label="Weighted")
    ax.set_xlabel("Net Wealth (CLP)")
    ax.set_ylabel("Density")
    ax.legend()
    return fig, ax


def plot_wealth_distribution(df, fig, ax, wealth_col="net_wealth", weight_col="factor", bins=50, normalize=True, overlay_unweighted=False):
    """Histogram of wealth distribution.

    Args:
        df (pd.DataFrame): Household data.
        fig (Figure): Figure.
        ax (Axes): Axes.
        wealth_col (str): Wealth column.
        weight_col (str): Weight column.
        bins (int): Number of bins.
        normalize (bool): If True, y is share of total wealth.
        overlay_unweighted (bool): Also plot unweighted histogram.

    Returns:
        (Figure, Axes): Updated figure and axes.
    """
    values = df[wealth_col].dropna()
    w = df.loc[values.index, weight_col].astype(float)
    if normalize:
        # share of total wealth per bin
        wealth_weights = values * w
        total = wealth_weights.sum()
        weights = wealth_weights / total if total != 0 else w
        ylabel = "Share of Total Wealth"
    else:
        weights = w
        ylabel = "Weighted Frequency"

    ax.hist(values, bins=bins, weights=weights, color="darkblue", edgecolor="white", label="Weighted")
    if overlay_unweighted:
        edges = np.histogram_bin_edges(values, bins=bins)
        if normalize:
            uw = np.ones(len(values)) / len(values)
            ax.hist(values, bins=edges, weights=uw, color="darkred", edgecolor="white", label="Unweighted")
        else:
            # rescale counts to weighted scale for visual overlap
            scale = w.sum() / len(values)
            ax.hist(values, bins=edges, color="darkred", edgecolor="white", weights=np.ones(len(values)) * scale, label="Unweighted")
        ax.legend()
    ax.set_xlabel("Net Wealth (CLP)")
    ax.set_ylabel(ylabel)

    return fig, ax


def save_fig(path, fig=None, dpi=300, bbox_inches="tight", **kwargs):
    """Save figure to disk.

    Args:
        path (str|Path): Output path.
        fig (Figure|None): Figure to save; current figure if None.
        dpi (int): Resolution.
        bbox_inches (str): Bounding box.
        **kwargs: Forwarded to savefig.

    Returns:
        Path: Saved path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fig is None:
        fig = plt.gcf()
    fig.savefig(str(path), dpi=dpi, bbox_inches=bbox_inches, **kwargs)
    print(f"saved figure to {path}")
    return path


def plot_shares(shares, fig, ax):
    """Bar chart of wealth shares.

    Args:
        shares (dict): {label: share} fractions.
        fig (Figure): Figure.
        ax (Axes): Axes.

    Returns:
        (Figure, Axes): Updated figure and axes.
    """
    labels = list(shares.keys())
    values = list(shares.values())
    ax.bar(labels, values, color="darkblue")
    for i, v in enumerate(values):
        ax.text(i, v * 1.02, f"{v:.2f}", ha="center", va="bottom")
    ax.set_ylabel("Share of Total Wealth")
    ax.set_ylim(0, max(values) * 1.3)
    # ax.set_title("Wealth Shares")
    return fig, ax


def plot_shares_comparison(shares_list, labels, fig, ax):
    """Grouped bar chart comparing multiple share dicts.

    Args:
        shares_list (list[dict]): List of {label: share} dicts.
        labels (list[str]): Group labels.
        fig (Figure): Figure.
        ax (Axes): Axes.

    Returns:
        (Figure, Axes): Updated figure and axes.
    """
    keys = list(shares_list[0].keys())
    colors = ["darkblue", "darkred"]
    x = np.arange(len(keys))
    width = 0.8 / len(shares_list)
    for j, shares in enumerate(shares_list):
        values = [shares[k] for k in keys]
        ax.bar(x + j * width, values, width=width, label=labels[j], color=colors[j])
        for i, v in enumerate(values):
            ax.text(x[i] + j * width, v * 1.02, f"{v:.1%}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x + width * (len(shares_list) - 1) / 2)
    ax.set_xticklabels(keys)
    ax.set_ylabel("Share of Total Wealth")
    ax.legend()
    ax.set_title("Wealth Shares Comparison")
    ax.set_ylim(0, 0.6)
    return fig, ax


def plot_pareto_qq(wealth, weights, w_min, alpha, fig, ax, alphas=None):
    """Pareto QQ plot: log tail prob vs log(w/w_min).

    Args:
        wealth (pd.Series): Wealth values.
        weights (pd.Series): Survey weights.
        w_min (float): Tail threshold.
        alpha (float): Fitted Pareto alpha.
        fig (Figure): Figure.
        ax (Axes): Axes.
        alphas (dict|None): Extra {label: alpha} lines to overlay.

    Returns:
        (Figure, Axes): Updated figure and axes.
    """
    # alphas: optional dict {label: value} to overlay sensitivity (e.g. {"MLE": a1, "Reg": a2})
    overlays = dict(alphas) if alphas else {"Pareto alpha": alpha}

    mask = wealth >= w_min
    if mask.sum() == 0:
        ax.text(0.5, 0.5, "no tail obs >= w_min", ha="center", transform=ax.transAxes)
        return fig, ax
    w = np.sort(wealth[mask].values)
    wt = weights[mask].iloc[np.argsort(wealth[mask].values)].values
    rank = np.cumsum(wt[::-1])[::-1]
    N = wt.sum()
    y = np.log(rank / N)
    x = np.log(w / w_min)
    ax.scatter(x, y, s=10, alpha=0.6, color="darkblue")
    xs = np.linspace(0, x.max(), 100) if len(x) else np.linspace(0, 1, 100)
    styles = [("red", "-", 1.5), ("darkorange", "--", 1.5), ("green", ":", 1.5)]
    for i, (label, a) in enumerate(overlays.items()):
        c, ls, lw = styles[i % len(styles)]
        lbl = label if "=" in label else f"{label}={a:.2f}"
        ax.plot(xs, -a * xs, color=c, linestyle=ls, lw=lw, label=lbl)
    ax.set_xlabel("log(w / w_min)")
    ax.set_ylabel("log(P(W>w))")
    ax.legend()
    ax.set_title("Pareto QQ — Tail Fit")
    return fig, ax


def plot_vermeulen_comparison(df_raw, df_corr, forbes_df, fig, ax, wealth_col="net_wealth", weight_col="factor"):
    """Log-log density comparison: EFH raw vs Vermeulen-corrected.

    Args:
        df_raw (pd.DataFrame): Original survey data.
        df_corr (pd.DataFrame): Corrected data (with synthetic + Forbes).
        forbes_df (pd.DataFrame): Forbes data with 'networth-clp'.
        fig (Figure): Figure.
        ax (Axes): Axes.
        wealth_col (str): Wealth column.
        weight_col (str): Weight column.

    Returns:
        (Figure, Axes): Updated figure and axes.
    """
    colors = {"EFH raw": "darkblue", "Vermeulen": "darkred"}
    bins = np.geomspace(max(1e6, df_raw[wealth_col].quantile(0.5)), forbes_df["networth-clp"].max(), 40)
    for df, label in [(df_corr, "Vermeulen"), (df_raw, "EFH raw")]:
        vals = df[wealth_col].dropna()
        wts = df.loc[vals.index, weight_col]
        hist, edges = np.histogram(vals, bins=bins, weights=wts * vals)
        # share of total wealth per bin
        total = (vals * wts).sum()
        hist = hist / total if total != 0 else hist
        centers = 0.5 * (edges[:-1] + edges[1:])
        # mask empty bins so log scale does not compress to zero
        hist = np.ma.masked_where(hist <= 0, hist)
        if label == "EFH raw":
            ax.plot(centers, hist, color=colors[label], label=label, lw=2.5, marker="o", ms=4, zorder=3, linestyle="--")
        else:
            ax.plot(centers, hist, color=colors[label], label=label, lw=2, marker="s", ms=3, alpha=0.85, zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Net Wealth (CLP, log scale)")
    ax.set_ylabel("Share of Total Wealth per bin \n(log scale)")
    # mark filled gap and Forbes entry to explain the lift
    try:
        ax.axvline(df_raw[wealth_col].max(), color="gray", ls=":", lw=1)
        ax.axvline(forbes_df["networth-clp"].min(), color="gray", ls="--", lw=1)
        ax.text(df_raw[wealth_col].max(), ax.get_ylim()[0] if len(ax.get_ylim()) else 1e-6, "max EFH", rotation=90, va="bottom", ha="right", fontsize=7, color="gray")
    except Exception:
        pass
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    ax.set_title("Vermeulen — Tail Correction (log-log)")
    return fig, ax


def plot_alpha_sensitivity(wmins, alphas_mle, alphas_reg, fig, ax):
    """Line plot of alpha vs w_min for MLE and regression.

    Args:
        wmins (list[float]): Threshold values in CLP.
        alphas_mle (list[float]): MLE alphas.
        alphas_reg (list[float]): Regression alphas.
        fig (Figure): Figure.
        ax (Axes): Axes.

    Returns:
        (Figure, Axes): Updated figure and axes.
    """
    x = np.array(wmins, dtype=float) / 1e6
    ax.plot(x, alphas_mle, marker="o", linestyle="--", color="darkblue", label="MLE")
    ax.plot(x, alphas_reg, marker="s", linestyle="-", color="darkred", label="Reg")
    for xi, a in zip(x, alphas_mle):
        ax.text(xi, a + 0.03, f"{a:.2f}", ha="center", va="bottom", fontsize=8)
    for xi, a in zip(x, alphas_reg):
        ax.text(xi, a * 0.98, f"{a:.2f}", ha="center", va="top", fontsize=8)
    ax.set_xlabel("Cutoff w_min (Million CLP)")
    ax.set_ylabel("Alpha")
    ax.set_ylim(1.85, 2.7)
    ax.legend()
    return fig, ax
