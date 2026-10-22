import matplotlib.pyplot as plt
import re
import numpy as np

from pathlib import Path
from source.sii import sii_for_at


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


def plot_bfm_fractal_bins(fig, axes=None):
    """Fractal binning used by BFM (2022) for tax tabulations.

    Bins are fractiles: 0-99% (99 percentiles), then 99-99.9%
    (9 bins of 0.1pp), 99.9-99.99% (9 bins of 0.01pp), 99.99-99.999%
    (9 bins of 0.001pp). Visualized as nested zooms with inset
    for the top 1% tail.

    Args:
        fig (Figure): Matplotlib figure.
        axes (list[Axes]|None): 4 axes for each zoom level. Created if None.

    Returns:
        (Figure, list[Axes]): Updated figure and axes.
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

    if axes is None:
        axes = fig.subplots(4, 1, sharex=False, gridspec_kw={"hspace": 1.0})
        fig.subplots_adjust(left=0.32, right=0.96, top=0.92, bottom=0.08)

    levels = [
        ("0-99%  (99 bins × 1 pp)", 0, 99, 99, "#d0d0d0"),
        ("99-99.9%  (9 bins × 0.1 pp)", 99, 99.9, 9, "#7fb3d5"),
        ("99.9-99.99%  (9 bins × 0.01 pp)", 99.9, 99.99, 9, "#2e86c1"),
        ("99.99-99.999%  (9 bins × 0.001 pp)", 99.99, 99.999, 9, "#0b2c4d"),
    ]

    for idx, (ax, (label, start, end, n_bins, color)) in enumerate(zip(axes, levels)):
        width = (end - start) / n_bins
        for k in range(n_bins):
            left = start + k * width
            ax.add_patch(plt.Rectangle((left, 0), width, 1, facecolor=color, edgecolor="white", linewidth=0.7))
            if k == 0:
                ax.text(left + width / 2, 0.5, f"{left:g}%", ha="center", va="center", fontsize=6, color="black" if n_bins >= 50 else "white")
            if k == n_bins - 1:
                ax.text(left + width / 2, 0.5, f"{end:g}%", ha="center", va="center", fontsize=6, color="white")

        if end < 100:
            ax.axvspan(end, 100, color="red", alpha=0.08, hatch="///")
            ax.text(end + (100 - end) / 2, 0.5, "tail →", ha="center", va="center", fontsize=7, color="darkred", style="italic")

        # --- per-level x-limits to make zoom appreciable ---
        if idx == 0:
            # full 0-99 but emphasize tail via inset
            ax.set_xlim(0, 100)
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.tick_params(labelsize=6)
            # inset zoom on 97-100 — absolute position within parent axes
            # tune bbox_to_anchor=(x0, y0, width, height) in axes fraction (0-1)
            # x0=0.58 leaves 58% from left, y0=0.15 leaves 15% from bottom
            ax_in = inset_axes(
                ax,
                width="100%", height="100%",
                loc="upper left",
                bbox_to_anchor=(0.48, 0.92, 0.38, 0.78),
                bbox_transform=ax.transAxes,
                borderpad=0,
            )
            for k in range(n_bins):
                left = start + k * width
                if left < 97:
                    continue
                ax_in.add_patch(plt.Rectangle((left, 0), width, 1, facecolor=color, edgecolor="white", linewidth=0.6))
            ax_in.set_facecolor("white")
            ax_in.patch.set_alpha(1.0)
            for spine in ax_in.spines.values():
                spine.set_edgecolor("0.4")
                spine.set_linewidth(0.8)
            ax_in.axvspan(end, 100, color="red", alpha=0.12, hatch="///")
            ax_in.set_xlim(97, 100)
            ax_in.set_ylim(0, 1)
            ax_in.set_xticks([97, 99, 100])
            ax_in.set_xticklabels(["97%", "99%", "100%"], fontsize=6)
            ax_in.tick_params(labelsize=6)
            ax_in.set_yticks([])
            ax_in.set_title("zoom 97-100%", fontsize=6, pad=2, backgroundcolor="white")
            mark_inset(ax, ax_in, loc1=2, loc2=4, fc="none", ec="0.5", ls="--", lw=0.7)
        elif idx == 1:
            ax.set_xlim(99, 100)
            ax.set_xticks([99, 99.5, 99.9, 100])
            ax.set_xticklabels(["99%", "99.5%", "99.9%", "100%"], fontsize=6)
        elif idx == 2:
            ax.set_xlim(99.9, 100)
            ax.set_xticks([99.9, 99.95, 99.99, 100])
            ax.set_xticklabels(["99.9%", "99.95%", "99.99%", "100%"], fontsize=6)
        else:
            ax.set_xlim(99.99, 100)
            ax.set_xticks([99.99, 99.995, 99.999, 100])
            ax.set_xticklabels(["99.99%", "99.995%", "99.999%", "100%"], fontsize=6)

        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.tick_params(labelsize=6)
        ax.set_ylabel(label, fontsize=7, rotation=0, ha="right", va="center", labelpad=12)
        ax.set_xlabel("Percentile" if idx == len(axes) - 1 else "")

    # fig.suptitle("BFM (2022) fractal bins — tax tabulations (nested zooms)", fontsize=10, y=0.98)
    return fig, axes


def plot_income_distribution(df, fig, ax, income_col="ytotcor", weight_col="expr", bins=80, log_x=False):
    """Histogram of income (CASEN) — weighted.

    Args:
        df (pd.DataFrame): CASEN (person or household).
        fig (Figure): Figure.
        ax (Axes): Axes.
        income_col (str): Income column (e.g. ytotcorh).
        weight_col (str): Weight column (expr).
        bins (int): Bins.
        log_x (bool): Log x-axis.

    Returns:
        (Figure, Axes)
    """
    sub = df[[income_col, weight_col]].dropna(subset=[income_col])
    # keep positive incomes for log
    if log_x:
        sub = sub[sub[income_col] > 0]
    x = sub[income_col].values
    w = sub[weight_col].values.astype(float)
    if log_x:
        bins = np.geomspace(max(1, np.percentile(x, 1)), np.percentile(x, 99.9), bins)
    ax.hist(x, bins=bins, weights=w, color="#2e86c1", edgecolor="white", linewidth=0.4, alpha=0.9)
    ax.set_xlabel("Household Income (CLP)" + (" — log scale" if log_x else ""))
    ax.set_ylabel("Weighted Frequency")
    if log_x:
        ax.set_xscale("log")
    # annotate median/mean
    wmean = np.average(x, weights=w)
    order = np.argsort(x)
    cum = np.cumsum(w[order]) / w.sum()
    wmed = x[order][np.searchsorted(cum, 0.5)]
    ax.axvline(wmed, color="darkred", ls="--", lw=1.2, label=f"median {wmed:,.0f}")
    ax.axvline(wmean, color="darkblue", ls=":", lw=1.2, label=f"mean {wmean:,.0f}")
    ax.legend(fontsize=7)
    return fig, ax


def plot_income_lorenz(df, fig, ax, income_col="ytotcor", weight_col="expr", label="Lorenz", color="#2e86c1"):
    """Lorenz curve for income (single series).

    Args:
        df (pd.DataFrame): CASEN.
        fig (Figure): Figure.
        ax (Axes): Axes.
        income_col (str): Income col.
        weight_col (str): Weight col.
        label (str): Legend label.
        color (str): Line color.

    Returns:
        (Figure, Axes)
    """
    sub = df[[income_col, weight_col]].dropna(subset=[income_col])
    sub = sub[sub[income_col] >= 0]
    x = sub[income_col].values
    w = sub[weight_col].values.astype(float)
    order = np.argsort(x)
    xs, ws = x[order], w[order]
    cum_w = np.cumsum(ws) / ws.sum()
    cum_y = np.cumsum(xs * ws) / np.sum(xs * ws)
    ax.plot(np.append([0], cum_w), np.append([0], cum_y), color=color, lw=1.8, label=label)
    ax.set_xlabel("Cumulative population")
    ax.set_ylabel("Cumulative income")
    return fig, ax


def plot_income_lorenz_comparison(dfs, labels, fig, ax, income_col="ytotcorh", weight_col="expr", fontsize=14):
    """Overlay Lorenz curves for 2017 vs 2022 on same axes.

    Args:
        dfs (list[pd.DataFrame]): List of CASEN household dataframes.
        labels (list[str]): Labels e.g. ["2017","2022"].
        fig (Figure): Figure.
        ax (Axes): Axes.
        income_col (str): Income col.
        weight_col (str): Weight col.

    Returns:
        (Figure, Axes)
    """
    
    colors = ["darkblue", "darkred"]
    for df, lab, col in zip(dfs, labels, colors):
        plot_income_lorenz(df, fig, ax, income_col=income_col, weight_col=weight_col, label=lab, color=col)
    ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1.2, label="Equality")
    ax.tick_params(labelsize=fontsize)
    ax.set_xlabel("Cumulative Population", fontsize=fontsize)
    ax.set_ylabel("Cumulative Income", fontsize=fontsize)
    ax.legend(fontsize=fontsize, framealpha=0.9)
    # ax.set_title("Lorenz Curve — CASEN 2017 vs 2022", fontsize=fontsize)
    return fig, ax


def plot_sii_brackets(cleaned, fig, ax, at_comercial=2017, value="cons_n"):
    """Bar plot of SII tabulation (Consolidado) for one commercial year.

    Args:
        cleaned (pd.DataFrame): from load_sii_pub_total (tidy).
        fig (Figure): Figure.
        ax (Axes): Axes.
        at_comercial (int): e.g. 2017 -> AT 2018.
        value (str): 'cons_n' (taxpayers) or 'cons_renta' (MM$).

    Returns:
        (Figure, Axes)
    """
    from source.sii import sii_for_at
    df = sii_for_at(cleaned, at_comercial)
    if df.empty:
        ax.text(0.5, 0.5, f"No data for {at_comercial}", ha="center", va="center", transform=ax.transAxes, fontsize=9, color="gray")
        ax.set_axis_off()
        return fig, ax
    tramo = df["tramo"].astype(str).str.replace("Tramo", "Bracket", regex=False).str.replace(" a ", " to ", regex=False).str.replace("Más de", "More than", regex=False).str.replace("Exento", "Exempt", regex=False).str.replace("Tasa", "Rate", regex=False).str.replace("y más", "and above", regex=False).str.replace(",", ".", regex=False).str.replace("%", r"\%", regex=False)
    vals = df[value].astype(float)
    ax.barh(tramo, vals, color="#2e86c1" if value == "cons_n" else "#c0392b")
    ax.set_xlabel("Number of Taxpayers" if value == "cons_n" else "Total Income (MM CLP)")
    ax.set_ylabel("Income Bracket (UTA)")
    # ax.set_title(f"SII Consolidated — AT {at_comercial+1} (FY {at_comercial})", fontsize=9)
    # annotate top value
    if len(vals):
        ax.text(vals.max()*1.02, len(vals)-1, f"{vals.iloc[-1]:,.0f}", va="center", fontsize=7)
    return fig, ax


def plot_sii_brackets_comparison(cleaned, fig, axes, at_list=(2017, 2022), value="cons_n"):
    """Grouped horizontal bars: 2017 vs 2022 on same brackets.

    Handles different bracket counts (8 vs 7 after 2018 reform) by
    outer-joining on tramo labels.

    Args:
        cleaned (pd.DataFrame): from load_sii_pub_total.
        fig (Figure): Figure.
        axes (list[Axes]|Axes): Single ax or [ax_n, ax_renta] if value is None.
        at_list (tuple): e.g. (2017, 2022) commercial years.
        value (str|None): 'cons_n' or 'cons_renta' or None for both panels.

    Returns:
        (Figure, Axes)
    """

    def _prep(at_list, value):
        dfs = [sii_for_at(cleaned, at) for at in at_list]
        # Build unified tramo order (preserve first year order, append new ones)
        tramo_raw = []
        seen = set()
        for df in dfs:
            for t in df["tramo"].astype(str).tolist():
                if t not in seen:
                    seen.add(t)
                    tramo_raw.append(t)
        # Map each df to unified order
        tramo_disp = [t.replace("Tramo", "Bracket").replace(" a ", " to ").replace("Más de", "More than").replace("Exento", "Exempt").replace("Tasa", "Rate").replace("y más", "and above").replace(",", ".").replace("%", r"\%") for t in tramo_raw]
        y = np.arange(len(tramo_raw))
        vals_list = []
        for df in dfs:
            m = dict(zip(df["tramo"].astype(str).tolist(), df[value].astype(float).tolist()))
            vals = np.array([m.get(t, 0) for t in tramo_raw], dtype=float)
            vals_list.append(vals)
        return tramo_disp, y, vals_list

    # If single value, single ax comparison
    if value is not None:
        ax = axes if not isinstance(axes, (list, np.ndarray)) else axes
        if isinstance(axes, (list, np.ndarray)):
            ax = axes[0] if len(axes) == 1 else axes
        tramo_disp, y, vals_list = _prep(at_list, value)
        h = 0.35
        colors = ["#2e86c1", "#c0392b"]
        for i, (vals, at) in enumerate(zip(vals_list, at_list)):
            ax.barh(y + (i - 0.5) * h, vals, height=h, label=f"FY {at} (AT {at+1})", color=colors[i], edgecolor="white", linewidth=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(tramo_disp, fontsize=7)
        ax.set_xlabel("Number of Taxpayers" if value == "cons_n" else "Total Income (MM CLP)")
        ax.set_ylabel("Income Bracket (UTA)")
        # ax.set_title(f"SII Consolidated — Bracket Comparison ({' vs '.join(str(a) for a in at_list)})", fontsize=9)
        ax.legend(fontsize=7)
        return fig, ax

    # Both panels side-by-side
    for ax, val in zip(axes, ["cons_n", "cons_renta"]):
        tramo_disp, y, vals_list = _prep(at_list, val)
        h = 0.35
        colors = ["#2e86c1", "#c0392b"]
        for i, (vals, at) in enumerate(zip(vals_list, at_list)):
            ax.barh(y + (i - 0.5) * h, vals, height=h, label=f"FY {at}", color=colors[i], edgecolor="white", linewidth=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(tramo_disp, fontsize=6.5)
        for lbl in ax.get_yticklabels():
            lbl.set_clip_on(False)
        ax.tick_params(pad=4)
        ax.set_xlabel("Taxpayers" if val == "cons_n" else "Total Income (MM CLP)")
        if ax == axes[0]:
            ax.set_ylabel("Income Bracket (UTA)")
        ax.legend(fontsize=7)
    axes[0].set_title("Taxpayers by Bracket", fontsize=9)
    axes[1].set_title("Total Income by Bracket", fontsize=9)
    # fig.suptitle(f"SII Consolidated — FY {at_list[0]} vs {at_list[1]}", fontsize=10, y=0.98)
    fig.subplots_adjust(left=0.36, right=0.97, wspace=0.35)
    return fig, axes


def plot_rank_sector(rr, fig, axes):
    """Rank-rank lines by sector (Fig. 1): total, formal, informal.

    Args:
        rr (pd.DataFrame): rank_rank output with columns grupo, modelo, sector, constante, beta.
        fig (Figure): Figure.
        axes (list[Axes]): Two axes: Education, Education + occupation.

    Returns:
        (Figure, Axes)
    """
    colors = ["#244D66", "#3B836B", "#B04D3A"]
    for ax, tipo in zip(axes, ["Education", "Education and occupation"]):
        z = rr[(rr["grupo"] == "25-40") & (rr["modelo"] == tipo)].set_index("sector")
        x = np.linspace(10, 90, 100)
        for j, s in enumerate(["Total", "Formal", "Informal"]):
            ax.plot(x, z.loc[s, "constante"] + z.loc[s, "beta"] * x, color=colors[j], lw=2.0, ls=["-", "--", ":"][j], label=s)
        ax.set_xlim(10, 90)
        ax.set_xlabel("Predicted Paternal Percentile", fontsize=10)
        ax.set_ylabel("Expected Child Percentile", fontsize=10)
        ax.set_title("Education" if tipo == "Education" else "Education and Occupation", fontsize=11)
        ax.tick_params(labelsize=9)
        ax.legend(fontsize=9, framealpha=0.9)
    return fig, axes


def plot_mi_sensitivity(mires, fig, ax):
    """MI combined slopes vs assumed residual correlation (Fig. 2).

    Args:
        mires (pd.DataFrame): mi_resumen with rho, criterio_hijo, combinacion, beta.
        fig (Figure): Figure.
        ax (Axes): Axes.

    Returns:
        (Figure, Axes)
    """
    colors = ["#244D66", "#B04D3A", "#958041", "#3B836B"]
    sub = mires[mires["criterio_hijo"] == "Cotizacion"]
    for j, g in enumerate(["F-F", "F-I", "I-F", "I-I"]):
        z = sub[sub["combinacion"] == g].sort_values("rho")
        ax.plot(z["rho"], z["beta"], color=colors[j], lw=2.2, marker="osD^"[j], ms=5, label=g)
    ax.set_xlim(-0.02, 0.62)
    ax.set_xlabel("Assumed Residual Correlation", fontsize=11)
    ax.set_ylabel("Combined Rank-Rank Slope", fontsize=11)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=10, framealpha=0.9, title="Father-Son", title_fontsize=10)
    ax.grid(True, alpha=0.2)
    return fig, ax


def plot_bfm_theta_Theta_real(fig, ax, tilde, theta_hat, Theta_hat, yk, tax_share, pct=None, merging_pct=None, title="(e) Chile — BFM diagnostic"):
    """Real BFM diagnostic: theta vs Theta from estimate_merging_point.

    Args:
        fig, ax: Figure/Axes.
        tilde, theta_hat, Theta_hat: arrays length m (bins).
        yk: thresholds in CLP.
        tax_share: w_k per bin; only used to derive the x-axis percentiles when pct is None.
        pct: optional array with the lower-bound percentile (in %) of each bin, length m.
            Needed whenever the bins are not the coarse SII brackets (e.g. the fractal grid),
            where cumsum(tax_share) no longer lines up with theta_hat.
        merging_pct: override percentile for vertical line (else from k*).
        title: panel title.

    Returns:
        (Figure, Axes)
    """
    if pct is None:
        if len(tax_share) != len(theta_hat):
            raise ValueError(
                f"tax_share length {len(tax_share)} != theta_hat length {len(theta_hat)}; pass pct="
            )
        cum_tax = np.cumsum(tax_share)
        pct = np.concatenate([[0], cum_tax[:-1]]) * 100  # lower bound percentile
    pct = np.asarray(pct, dtype=float)
    # Find k* where theta_hat < Theta_hat
    below = np.where(theta_hat < Theta_hat)[0]
    k_star = below[0] if len(below) else len(tilde)-1
    if merging_pct is None:
        merging_pct = float(pct[k_star]) if k_star < len(pct) else 83.5
    # Plot
    ax.axvline(merging_pct, color="gray", ls="--", lw=0.9, alpha=0.7)
    ax.text(merging_pct, 2.95, "merging point", ha="center", va="bottom", fontsize=7, color="gray", style="italic")
    # Red Theta (cumulative)
    ax.plot(pct, Theta_hat, color="#c0392b", lw=1.8, marker="o", ms=4, label=r"$\Theta(y)$")
    # Blue antitonic
    ax.plot(pct, theta_hat, color="#2e86c1", lw=1.8, marker="s", ms=4, label=r"$\theta(y)$ (antitonic)")
    # Grey dots raw theta
    ax.scatter(pct, tilde, s=28, color="gray", alpha=0.6, edgecolors="white", linewidths=0.4, label=r"$\theta(y)$", zorder=3)
    ax.set_xlim(75, 100)
    ax.set_ylim(0, 3)
    # Ticks similar to paper: P60, P80, P90, P99, P99.9...
    # Map pct to labels: show nearest
    # ax.set_xticks([60, 70, 80, 83.5, 90, 95, 99, 99.9])
    # ax.set_xticklabels(["P60", "P70", "P80", "P83.5", "P90", "P95", "P99", "P99.9"], fontsize=7, rotation=15, ha="right")
    ax.set_yticks([0, 1, 2, 3])
    ax.set_xlabel("Percentile", fontsize=9)
    ax.set_ylabel(r"$\theta(y)$ / $\Theta(y)$", fontsize=9)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9, ncol=2)
    ax.grid(True, axis="y", alpha=0.15)
    return fig, ax


def plot_bfm_fig6e_emulation(fig, ax, merging_pct=83.5):
    """Emulate BFM (2022) Figure 6e — Chile 2015: theta(y) vs Theta(y).

    Synthetic data mimicking the description: grey dots (raw theta),
    blue solid (antitonic), red solid (Theta), green moving avg, blue
    dashed extrapolation. X = percentiles P0..P99.999, Y = 0..3.

    Args:
        fig (Figure): Figure.
        ax (Axes): Axes.
        merging_pct (float): merging point percentile (e.g. 83.5).

    Returns:
        (Figure, Axes)
    """

    # X positions: map percentile -> 0..100, visible from P60
    pct = np.array([60, 70, 80, 83.5, 90, 95, 99, 99.9, 99.99, 99.999])
    x = pct  # use percentile as x

    # Theta (red): 0.8 at 60 -> 1.0 at merging, flat after
    x_red = np.array([60, 70, 80, merging_pct, 90, 99.999])
    y_red = np.array([0.82, 0.90, 0.98, 1.0, 1.0, 1.0])

    # Theta antitonic blue solid: 1.2 at merging -> 0 at top
    x_blue = np.linspace(merging_pct, 99.999, 80)
    y_blue = 1.2 * np.exp(-0.045 * (x_blue - merging_pct))  # decay to ~0
    y_blue = np.clip(y_blue, 0, 3)

    # Grey dots: noisy around blue solid from merging to top
    rng = np.random.default_rng(0)
    x_grey = np.linspace(merging_pct + 0.5, 99.999, 55)
    y_grey = np.interp(x_grey, x_blue, y_blue) + rng.normal(0, 0.07, len(x_grey))
    y_grey = np.clip(y_grey, 0, 1.3)

    # Green moving avg: smoothed version of grey (running mean)
    y_green = np.convolve(y_grey, np.ones(5)/5, mode='same')
    x_green = x_grey

    # Blue dashed extrapolation: from 60 at 2.8 down to merging at ~1.1
    x_extrap = np.linspace(60, merging_pct, 40)
    y_extrap = 2.8 - 1.7 * (x_extrap - 60) / (merging_pct - 60)  # linear 2.8->1.1
    # also continue slightly after merging as dashed (as in paper)
    x_extrap2 = np.linspace(merging_pct, 99.999, 20)
    y_extrap2 = np.interp(x_extrap2, x_blue, y_blue) * 1.05

    # Plot
    ax.axvline(merging_pct, color="gray", ls="--", lw=0.9, alpha=0.7)
    ax.text(merging_pct, 2.95, "merging point", ha="center", va="bottom", fontsize=7, color="gray", style="italic")

    # Red Theta
    ax.plot(x_red, y_red, color="#c0392b", lw=1.8, label=r"$\Theta(y)$")
    # Blue antitonic
    ax.plot(x_blue, y_blue, color="#2e86c1", lw=1.8, label=r"$\theta(y)$ (antitonic)")
    # Blue extrapolation dashed
    ax.plot(x_extrap, y_extrap, color="#2e86c1", lw=1.2, ls="--", alpha=0.9, label=r"$\theta(y)$ (extrapolation)")
    ax.plot(x_extrap2, y_extrap2, color="#2e86c1", lw=1.0, ls=":", alpha=0.6)
    # Green moving avg
    ax.plot(x_green, y_green, color="#27ae60", lw=1.4, alpha=0.9, label=r"$\theta(y)$ (moving avg.)")
    # Grey dots
    ax.scatter(x_grey, y_grey, s=12, color="gray", alpha=0.55, edgecolors="white", linewidths=0.3, label=r"$\theta(y)$", zorder=3)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 3)
    ax.set_xticks([60, 70, 80, 90, 99, 99.9, 99.99, 99.999])
    ax.set_xticklabels(["P60", "P70", "P80", "P90", "P99", "P99.9", "P99.99", "P99.999"], fontsize=7, rotation=20, ha="right")
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["0", "1", "2", "3"], fontsize=8)
    ax.set_xlabel("Percentile", fontsize=9)
    ax.set_ylabel(r"$\theta(y)$ / $\Theta(y)$", fontsize=9)
    ax.set_title("(e) Chile 2015", fontsize=9, pad=8)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9, ncol=2)
    ax.grid(True, axis="y", alpha=0.15)
    return fig, ax
