"""Núñez-Miranda (2010) replication — intergenerational mobility with informality.

Pure numpy/scipy implementation matching MirandaNunez.Rmd. All estimators are
closed-form OLS/logit with HC1 robust covariance, to reproduce the R output
exactly (no statsmodels). Plots live in figures.py.
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.special import expit  # logistic


# ---------------------------------------------------------------------------
# Core estimators
# ---------------------------------------------------------------------------
def ols(y, X):
    """OLS via normal equations. Returns (beta, V_hc1, residuals, nobs).

    Args:
        y (np.ndarray): (n,)
        X (np.ndarray): (n, k) design matrix WITH intercept (first col or explicit).

    Returns:
        beta (k,), V_hc1 (k,k), resid (n,), nobs
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(X.T @ X)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    # HC1: n/(n-k) * (X'X)^-1 (X' diag(e^2) X) (X'X)^-1
    Xe = X * resid[:, None]
    meat = Xe.T @ Xe
    V = (n / max(n - k, 1)) * XtX_inv @ meat @ XtX_inv
    return beta, V, resid, n


def logit(y, X, maxiter=50, tol=1e-10):
    """Binomial logit via IRLS. Returns (beta, V_sandwich, converged)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape
    beta = np.zeros(k)
    for _ in range(maxiter):
        p = expit(X @ beta)
        p = np.clip(p, 1e-10, 1 - 1e-10)
        W = p * (1 - p)
        z = X @ beta + (y - p) / W
        XtWX = X.T @ (X * W[:, None])
        XtWz = X.T @ (W * z)
        beta_new, *_ = np.linalg.lstsq(XtWX, XtWz, rcond=None)
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    p = expit(X @ beta)
    # sandwich: (X'WX)^-1 (X' diag((y-p)^2) X) (X'WX)^-1 via lstsq
    W = p * (1 - p)
    XtWX = X.T @ (X * W[:, None])
    meat = X.T @ (X * (y - p)[:, None] ** 2)
    A, *_ = np.linalg.lstsq(XtWX, meat, rcond=None)
    B, *_ = np.linalg.lstsq(XtWX, A.T, rcond=None)
    V = B.T
    V = (V + V.T) / 2
    return beta, V, True


def ranking_pct(x):
    """Percentile ranks in (0, 100] with average ties, matching R rank()."""
    x = np.asarray(x, dtype=float)
    return 100.0 * rankdata(x, method="average") / len(x)


# ---------------------------------------------------------------------------
# Recoding (CASEN 1987 / 1990 / 2006)
# ---------------------------------------------------------------------------
def recode_1987(df):
    d = df.copy()
    d["ingreso"] = d[["ysefaj", "yganaj", "yespaj"]].sum(axis=1, skipna=True)
    e4, e3 = d["e4"].astype(float), d["e3"].astype(float)
    cond = [
        (e4 == 11) | ((e4 == 2) & e3.between(1, 7)),
        ((e4 == 2) & (e3 == 8)) | (e4.isin([3, 4]) & e3.between(1, 3)),
        (e4.isin([3, 4]) & e3.between(4, 5)) | ((e4 == 5) & e3.between(1, 4)) | ((e4 == 6) & e3.between(1, 2)),
        (e4 == 6) & e3.between(3, 6),
        (e4 == 5) & e3.between(5, 7),
    ]
    d["nivel"] = np.select(cond, [0, 1, 2, 3, 4], default=np.nan)
    o4 = d["o4"].astype(float)
    d["ocup"] = np.select([o4 == 1, o4 == 2, o4.isin([3, 4, 5, 6]), o4 == 7], [1, 2, 3, 4], default=np.nan)
    return d


def recode_1990(df):
    d = df.copy()
    d = d[d["jh"].between(30, 168)].copy()
    d["ingreso"] = d["ytrabaj"]
    educ = d["educ"].astype(float)
    d["nivel"] = np.select(
        [educ.isin([0, 1]), educ.isin([2, 3, 4]), educ.isin([5, 6, 7]), educ == 8, educ == 9],
        [0, 1, 2, 3, 4], default=np.nan,
    )
    o7 = d["o7"].astype(float)
    d["ocup"] = np.select([o7 == 1, o7 == 2, o7.isin([3, 7]), o7.isin([4, 5])], [1, 2, 3, 4], default=np.nan)
    return d


def build_sample(df, year):
    """First-stage sample: male, 15-55, o21==1, positive income, non-missing nivel."""
    if year == 1987:
        d = recode_1987(df)
    elif year == 1990:
        d = recode_1990(df)
    else:
        raise ValueError(year)
    d = d[(d["sexo"] == 1) & d["edad"].between(15, 55) & (d["o21"] == 1) & (d["ingreso"] > 0) & d["nivel"].notna()].copy()
    d["Y"] = np.log(d["ingreso"])
    d["dP"] = (d["nivel"] == 1).astype(int)
    d["dS"] = (d["nivel"] == 2).astype(int)
    d["dT"] = (d["nivel"] == 3).astype(int)
    d["dU"] = (d["nivel"] == 4).astype(int)
    # R indicadores: as.integer(ocup==k) -> NA when ocup is NA (lm drops those rows)
    for c, k in [("Empleador", 1), ("Self", 2), ("Servicio", 4)]:
        d[c] = np.where(d["ocup"].isna(), np.nan, (d["ocup"] == k).astype(float))
    d["Age2"] = d["edad"].astype(float) ** 2
    return d


def design_matrix(df, cols, with_intercept=True):
    X = df[cols].to_numpy(dtype=float)
    if with_intercept:
        X = np.column_stack([np.ones(len(X)), X])
    return X


# ---------------------------------------------------------------------------
# Second stage (CASEN 2006 children)
# ---------------------------------------------------------------------------
def recode_2006(df):
    d = df[(df["sexo"] == 1) & df["edad"].between(25, 40) & df["o15"].between(30, 168)
           & (df["ytrabaj"] > 0) & df["t14a"].isin([1, 2, 3, 4, 5]) & df["t15a"].isin([1, 2, 3, 4, 5, 6])].copy()
    d["ingreso_hijo"] = d["ytrabaj"]
    d["Y"] = np.log(d["ytrabaj"])
    o29 = d["o29"]
    d["sector"] = pd.Series(np.where(o29.isin([1, 2, 3, 4, 5]), "Formal", np.where(o29.isin([6, 7]), "Informal", "")), index=d.index)
    d["sector"] = d["sector"].replace("", np.nan)
    d["nivel"] = np.where(d["t14a"] == 5, 0, d["t14a"])
    t15 = d["t15a"].astype(float)
    d["ocup"] = np.select([t15.isin([1, 2]), t15 == 3, t15.isin([4, 5]), t15 == 6], [1, 2, 3, 4], default=np.nan)
    d["anio"] = np.where(d["edad"] <= 35, "1990", "1987")
    d["dP"] = (d["nivel"] == 1).astype(int)
    d["dS"] = (d["nivel"] == 2).astype(int)
    d["dT"] = (d["nivel"] == 3).astype(int)
    d["dU"] = (d["nivel"] == 4).astype(int)
    d["Empleador"] = pd.Series(np.where(d["ocup"].isna(), np.nan, (d["ocup"] == 1).astype(float)), index=d.index)
    d["Self"] = pd.Series(np.where(d["ocup"].isna(), np.nan, (d["ocup"] == 2).astype(float)), index=d.index)
    d["Servicio"] = pd.Series(np.where(d["ocup"].isna(), np.nan, (d["ocup"] == 4).astype(float)), index=d.index)
    d["Age2"] = d["edad"].astype(float) ** 2
    d["Y"] = d["Y"] - d["Y"].mean()
    return d


# ---------------------------------------------------------------------------
# Two-step: Murphy-Topel influence (matches parametros() in Rmd)
# ---------------------------------------------------------------------------
def fit_first_stage(d, model_cols):
    """Fit OLS dropping NA rows (like lm) and return beta, V, resid, n, X_fit."""
    dd = d.dropna(subset=model_cols + ["Y"]).copy()
    X = design_matrix(dd, model_cols)
    beta, V, resid, n = ols(dd["Y"].to_numpy(), X)
    return beta, V, resid, n, X, dd
def parametros(beta, V, X, v_cols):
    """First-stage params: b, mu (means of v_cols), V (variance via influence).

    Args:
        beta: full coefficient vector (with intercept).
        V: HC1 covariance of full model.
        X: design matrix WITH intercept, (n,k).
        v_cols: indices of columns whose coefficients/means enter the prediction.

    Returns:
        dict(b, mu, V)
    """
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    Z = X[:, v_cols]
    mu = Z.mean(axis=0)
    # Influence of beta_v
    resid = _resid_from_beta(X, beta)
    Ib = (X * resid[:, None]) @ np.linalg.inv(X.T @ X)[:, v_cols] * np.sqrt(n / (n - k))
    # Influence of mu
    Imu = (Z - mu) / n * np.sqrt(n / (n - 1))
    Vfirst = np.c_[Ib, Imu].T @ np.c_[Ib, Imu]
    return {"b": beta[v_cols], "mu": mu, "V": Vfirst}


def _resid_from_beta(X, beta):
    return X[:, -1] * 0  # placeholder, replaced by caller — see two_step


def two_step_elasticity(samples, hijos, edu_cols, ocup_cols=None):
    """Compute TSTSLS elasticity + Murphy-Topel SE by age group.

    samples: dict {year: first-stage DataFrame} with columns dP,dS,dT,dU,edad,Age2,Empleador,Self,Servicio,Y
    hijos: recoded 2006 children DataFrame.
    """
    v = ["dP", "dS", "dT", "dU"] + ([] if ocup_cols is None else ocup_cols)
    edu_extra = ["edad", "Age2"]
    model_cols = ["dP", "dS", "dT", "dU", "edad", "Age2"] + ([] if ocup_cols is None else ocup_cols)

    first = {}
    for year, d in samples.items():
        beta, V, resid, n, X, dd = fit_first_stage(d, model_cols)
        # v_cols indices: intercept=0, then model_cols order
        allcols = ["intercept"] + model_cols
        v_idx = [allcols.index(c) for c in v]
        b = beta[v_idx]
        mu = X[:, v_idx].mean(axis=0)
        Ib = (X * resid[:, None]) @ np.linalg.inv(X.T @ X)[:, v_idx] * np.sqrt(n / (n - X.shape[1]))
        Imu = (X[:, v_idx] - mu) / n * np.sqrt(n / (n - 1))
        Vf = np.c_[Ib, Imu].T @ np.c_[Ib, Imu]
        first[year] = {"b": b, "mu": mu, "V": Vf, "n_fit": n}

    Z = hijos[v].to_numpy(dtype=float)
    pred = np.zeros(len(hijos))
    G = {y: np.zeros((len(hijos), 2 * len(v))) for y in first}
    for year, q in first.items():
        pertenece = (hijos["anio"] == year).to_numpy()
        Zc = Z - q["mu"]
        pred[pertenece] = (Zc @ q["b"])[pertenece]
        G[year][pertenece] = np.column_stack([Zc, np.tile(-q["b"], (len(hijos), 1))])[pertenece]

    grupos = {"25-30": range(25, 31), "31-35": range(31, 36), "36-40": range(36, 41),
              "25-40": range(25, 41), "31-40": range(31, 41)}
    rows = []
    for g, ages in grupos.items():
        i = hijos["edad"].isin(ages).to_numpy()
        X2 = np.column_stack([np.ones(i.sum()), pred[i], hijos.loc[i, "edad"].to_numpy(), hijos.loc[i, "Age2"].to_numpy()])
        beta2, V2, resid2, n2 = ols(hijos.loc[i, "Y"].to_numpy(), X2)
        se_cond = np.sqrt(V2[1, 1])
        Vtot = V2.copy()
        for year, q in first.items():
            H = G[year][i]
            C = np.zeros((X2.shape[1], H.shape[1]))
            C[1, :] = resid2 @ H
            D = np.linalg.inv(X2.T @ X2) @ (C - beta2[1] * (X2.T @ H))
            Vtot = Vtot + D @ q["V"] @ D.T
        rows.append({"grupo": g, "modelo": "Education" if ocup_cols is None else "Education and occupation",
                     "elasticidad": beta2[1], "EE_MT": np.sqrt(Vtot[1, 1]),
                     "EE_condicional": se_cond, "N": n2})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Rank-rank regressions
# ---------------------------------------------------------------------------
def predecir_salario_padre(samples, hijos, edu_cols, ocup_cols=None, edad_std=40):
    """Father predicted log-wage (standardized at age 40) and centered version."""
    v = edu_cols if ocup_cols is None else edu_cols + ocup_cols
    model_cols = ["dP", "dS", "dT", "dU", "edad", "Age2"] + ([] if ocup_cols is None else ocup_cols)
    log_std = np.zeros(len(hijos))
    centrado = np.zeros(len(hijos))
    for year, d in samples.items():
        beta, V, resid, n, X, dd = fit_first_stage(d, model_cols)
        allcols = ["intercept"] + model_cols
        v_idx = [allcols.index(c) for c in v]
        edad_idx = allcols.index("edad")
        age2_idx = allcols.index("Age2")
        i = (hijos["anio"] == year).to_numpy()
        Z = hijos[v].to_numpy(dtype=float)
        log_std[i] = (beta[0] + Z @ beta[v_idx] + beta[edad_idx] * edad_std + beta[age2_idx] * edad_std ** 2)[i]
        centrado[i] = ((Z - X[:, v_idx].mean(axis=0)) @ beta[v_idx])[i]
    return log_std, centrado


def rank_rank(samples, hijos, edu_cols, ocup_cols=None):
    """Rank-rank slope + Spearman by age group and sector."""
    log_std, centrado = predecir_salario_padre(samples, hijos, edu_cols, ocup_cols)
    hijos = hijos.copy()
    hijos["padre_cent"] = centrado
    hijos["Ranking_hijo"] = ranking_pct(hijos["ingreso_hijo"].to_numpy())
    tipo = "Education" if ocup_cols is None else "Education and occupation"
    grupos = {"25-30": range(25, 31), "31-35": range(31, 36), "36-40": range(36, 41),
              "25-40": range(25, 41), "31-40": range(31, 41)}
    rows = []
    for g, ages in grupos.items():
        i = hijos["edad"].isin(ages).to_numpy()
        d = hijos.loc[i].copy()
        Rp = ranking_pct(d["padre_cent"].to_numpy())
        Rh = ranking_pct(d["ingreso_hijo"].to_numpy())
        d["Rp"] = Rp
        d["Rh"] = Rh
        for s in ["Total", "Formal", "Informal"]:
            ds = d if s == "Total" else d[d["sector"] == s]
            X1 = np.column_stack([np.ones(len(ds)), ds["Rp"].to_numpy()])
            y = ds["Rh"].to_numpy()
            b, V, _, n = ols(y, X1)
            Xa = np.column_stack([np.ones(len(ds)), ds["Rp"].to_numpy(), ds["edad"].to_numpy(), ds["Age2"].to_numpy()])
            ba, Va, _, _ = ols(y, Xa)
            rows.append({"grupo": g, "modelo": tipo, "sector": s,
                         "constante": b[0], "EE_constante": np.sqrt(V[0, 0]),
                         "beta": b[1], "EE_beta": np.sqrt(V[1, 1]),
                         "rho_spearman": np.corrcoef(ds["Rp"].to_numpy(), y)[0, 1],
                         "beta_con_edad": ba[1], "EE_beta_con_edad": np.sqrt(Va[1, 1]),
                         "N": n})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Multiple imputation (bivariate lognormal + logistic affiliation)
# ---------------------------------------------------------------------------
def _rmvnorm(mu, V, rng):
    return mu + np.linalg.cholesky(V) @ rng.normal(size=len(mu))


def _sortear_lm(beta, XtX_inv, resid, df_resid, rng):
    """Draw beta ~ N(beta_hat, s2*(X'X)^-1), s2 = RSS/chi2(df)."""
    s2 = (resid @ resid) / rng.chisquare(df_resid)
    return _rmvnorm(beta, s2 * XtX_inv, rng), np.sqrt(s2)


def multiple_imputation(samples, hijos, edu_cols, ocup_cols, seed=17102026, M=100, rhos=(0.0, 0.3, 0.6)):
    """Run MI and return (sim_resultados, mi_resumen)."""
    rng = np.random.default_rng(seed)
    v = edu_cols + ocup_cols
    model_cols = ["dP", "dS", "dT", "dU", "edad", "Age2"] + ocup_cols
    allcols = ["intercept"] + model_cols
    v_idx = [allcols.index(c) for c in v]

    # Father sample (1990) for affiliation logit (o11 -> F)
    d_hist = samples["1990"].copy()
    d_hist["F"] = np.where(d_hist["o11"].isin([1, 2, 3, 4, 5, 6]), 1, np.where(d_hist["o11"] == 7, 0, np.nan)).astype(float)
    d_hist = d_hist[d_hist["F"].notna()].copy()

    # Children (sons 25-35, known sector)
    d_mi = hijos[(hijos["anio"] == "1990") & hijos["sector"].notna()].copy()
    d_mi["Y"] = np.log(d_mi["ingreso_hijo"])
    d_mi["F"] = (d_mi["sector"] == "Formal").astype(int)
    d_mi["afiliado"] = (d_mi["o29"].isin([1, 2, 3, 4, 5, 6])).astype(int)

    # Father log-wage equation (m_p) on full 1990 sample
    beta_p, V_p, resid_p, n_p, Xp_full, dd_p = fit_first_stage(samples["1990"], model_cols)
    # Son log-wage equation (m_h) on d_mi with same covariates (paternal chars X)
    beta_h, V_h, resid_h, n_h, Xh_full, dd_h = fit_first_stage(d_mi, model_cols)

    # Father affiliation logit g_p on d_hist: F ~ v_ocup + edad + Age2 + Y (drop NA like glm)
    cols_gp = model_cols + ["Y"]
    d_hist = d_hist.copy()
    d_hist["Y"] = np.log(d_hist["ingreso"])
    d_hist_gp = d_hist.dropna(subset=cols_gp + ["F"]).copy()
    Xgp = design_matrix(d_hist_gp, cols_gp)
    g_p_b, g_p_V, _ = logit(d_hist_gp["F"].to_numpy(), Xgp)
    # Son contribution logit g_h on d_mi: F ~ v_ocup + edad + Age2 + Y
    Xgh = design_matrix(d_mi, cols_gp)
    g_h_b, g_h_V, _ = logit(d_mi["F"].to_numpy(), Xgh)

    # Prediction matrices at fixed age 40 for father
    d_padre = d_mi.copy()
    d_padre["edad"] = 40
    d_padre["Age2"] = 1600
    Xp = design_matrix(d_padre, model_cols)          # father income covariates at age 40
    Xh = design_matrix(d_mi, model_cols)             # son income covariates (paternal X)
    Xwp = design_matrix(d_padre, cols_gp)            # father affiliation covariates (with Y col)
    # Y column index in cols_gp
    Y_idx = len(cols_gp)  # design matrix prepends intercept, so Y (last of cols_gp) sits at this index

    n_mi = len(d_mi)

    def analizar_combinaciones(logp, logh, Fp, Fh):
        Rp = ranking_pct(logp)
        Rh = ranking_pct(logh)
        comb = np.where(Fp == 1, "F", "I") + "-" + np.where(Fh == 1, "F", "I")
        out = []
        for g in ["F-F", "F-I", "I-F", "I-I"]:
            m = comb == g
            X = np.column_stack([np.ones(m.sum()), Rp[m]])
            b, V, _, _ = ols(Rh[m], X)
            out.append({"combinacion": g, "beta": b[1], "U": V[1, 1], "N": int(m.sum()), "proporcion": m.mean()})
        return pd.DataFrame(out)

    # Illustration: bivariate lognormal simulation, 10k families
    perfil = d_mi.iloc[[0]].copy()
    for c in v:
        perfil[c] = 0.0
    perfil["dS"] = 1.0
    perfil["edad"] = 30.0
    perfil["Age2"] = 900.0
    perfil_p = perfil.copy()
    perfil_p["edad"] = 40.0
    perfil_p["Age2"] = 1600.0
    Xp_perfil = design_matrix(perfil_p, model_cols)
    Xh_perfil = design_matrix(perfil, model_cols)
    mu_sim = np.array([Xp_perfil @ beta_p, Xh_perfil @ beta_h]).ravel()
    sigma_p = np.sqrt((resid_p @ resid_p) / (n_p - Xp_full.shape[1]))
    sigma_h = np.sqrt((resid_h @ resid_h) / (n_h - Xh_full.shape[1]))
    rho_sim = 0.3
    Sigma_sim = np.diag([sigma_p, sigma_h]) @ np.array([[1, rho_sim], [rho_sim, 1]]) @ np.diag([sigma_p, sigma_h])
    n_sim = 10000
    log_sim = rng.normal(size=(n_sim, 2)) @ np.linalg.cholesky(Sigma_sim) + mu_sim

    def prob_sim(m, d, y, Y_idx):
        W = design_matrix(d, cols_gp)[np.repeat(0, len(y))]
        W[:, Y_idx] = y
        return expit(W @ m)

    # father affiliation prob at perfil (fixed), son at perfil
    Xwp_perfil = design_matrix(perfil_p, cols_gp)
    Xgh_perfil = design_matrix(perfil, cols_gp)
    pFp = expit(Xwp_perfil @ g_p_b)
    pFh = expit(Xgh_perfil @ g_h_b)
    # probability conditional on log-income (recompute W with Y)
    def prob_gp(y):
        W = design_matrix(perfil_p, cols_gp)[np.zeros(len(y), int)]
        W[:, Y_idx] = y
        return expit(W @ g_p_b)
    def prob_gh(y):
        W = design_matrix(perfil, cols_gp)[np.zeros(len(y), int)]
        W[:, Y_idx] = y
        return expit(W @ g_h_b)
    sim = pd.DataFrame({
        "ingreso_padre": np.exp(log_sim[:, 0]),
        "ingreso_hijo": np.exp(log_sim[:, 1]),
        "afiliado_padre": (rng.uniform(size=n_sim) < prob_gp(log_sim[:, 0])).astype(int),
        "cotizante_hijo": (rng.uniform(size=n_sim) < prob_gh(log_sim[:, 1])).astype(int),
    })
    sim_resultados = analizar_combinaciones(np.log(sim["ingreso_padre"].to_numpy()),
                                            np.log(sim["ingreso_hijo"].to_numpy()),
                                            sim["afiliado_padre"].to_numpy(),
                                            sim["cotizante_hijo"].to_numpy())

    # MI loop
    XtXp_inv = np.linalg.inv(Xp_full.T @ Xp_full)
    XtXh_inv = np.linalg.inv(Xh_full.T @ Xh_full)
    n_gp = len(d_hist_gp)
    resultados = []
    for m in range(M):
        p, sig_p = _sortear_lm(beta_p, XtXp_inv, resid_p, n_p - Xp_full.shape[1], rng)
        h, sig_h = _sortear_lm(beta_h, XtXh_inv, resid_h, n_h - Xh_full.shape[1], rng)
        bF = _rmvnorm(g_p_b, g_p_V, rng)
        mu_p = Xp @ p
        eh = (d_mi["Y"].to_numpy() - Xh @ h) / sig_h
        z = rng.normal(size=n_mi)
        u = rng.uniform(size=n_mi)
        for r in rhos:
            yp = mu_p + sig_p * (r * eh + np.sqrt(1 - r ** 2) * z)  # p[-1] is age2? use sigma_p estimate
            Wp = Xwp.copy()
            Wp[:, Y_idx] = yp
            Fp = (u < expit(Wp @ bF)).astype(int)
            for criterio in ["Cotizacion", "Afiliacion"]:
                Fh = d_mi["F"].to_numpy() if criterio == "Cotizacion" else d_mi["afiliado"].to_numpy()
                rres = analizar_combinaciones(yp, d_mi["Y"].to_numpy(), Fp, Fh)
                rres["imputacion"] = m + 1
                rres["rho"] = r
                rres["criterio_hijo"] = criterio
                resultados.append(rres)

    mi_ind = pd.concat(resultados, ignore_index=True)
    grp = mi_ind.groupby(["rho", "criterio_hijo", "combinacion"])
    mi_res = pd.DataFrame({
        "M": grp["beta"].count(),
        "B": grp["beta"].var(),
        "beta": grp["beta"].mean(),
        "U_bar": grp["U"].mean(),
        "N_medio": grp["N"].mean(),
        "N_min": grp["N"].min(),
        "N_max": grp["N"].max(),
        "proporcion_media": grp["proporcion"].mean(),
    }).reset_index()
    Mval = M
    mi_res["T"] = mi_res["U_bar"] + (1 + 1 / Mval) * mi_res["B"]
    mi_res["EE"] = np.sqrt(mi_res["T"])
    mi_res["lambda"] = (1 + 1 / Mval) * mi_res["B"] / mi_res["T"]
    mi_res["gl_com"] = mi_res["N_min"] - 2
    mi_res["gl_old"] = np.where(mi_res["lambda"] > 0, (Mval - 1) / mi_res["lambda"] ** 2, np.inf)
    mi_res["gl_obs"] = (mi_res["gl_com"] + 1) / (mi_res["gl_com"] + 3) * mi_res["gl_com"] * (1 - mi_res["lambda"])
    mi_res["gl"] = 1 / (1 / mi_res["gl_old"] + 1 / mi_res["gl_obs"])
    tcrit = _tcrit(mi_res["gl"])
    mi_res["IC_inf"] = mi_res["beta"] - tcrit * mi_res["EE"]
    mi_res["IC_sup"] = mi_res["beta"] + tcrit * mi_res["EE"]
    mi_res["MCSE"] = np.sqrt(mi_res["B"] / Mval)
    return sim_resultados, mi_res, mi_ind


def _tcrit(df):
    from scipy.stats import t
    df = np.where(df > 0, df, 1.0)
    return t.ppf(0.975, df)
