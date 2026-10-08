"""Fungsi evaluasi statistik untuk model regresi taksiran nilai properti.

Semua fungsi menerima target & prediksi dalam skala LOG (log harga), lalu mengonversi
ke rupiah bila perlu. Metrik standar AVM: MdAPE, PPE10/PPE20 (porsi taksiran dalam ±10%/±20%).
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression


# ============================================================================ metrik
def regression_metrics(y_log, p_log) -> dict:
    y_log, p_log = np.asarray(y_log, float), np.asarray(p_log, float)
    e = p_log - y_log
    y, p = np.exp(y_log), np.exp(p_log)
    ape = np.abs(p / y - 1)
    return {
        "rmse_log": float(np.sqrt(np.mean(e**2))),
        "mae_log": float(np.mean(np.abs(e))),
        "r2_log": float(1 - np.sum(e**2) / np.sum((y_log - y_log.mean()) ** 2)),
        "bias_pct": float(np.exp(np.mean(e)) - 1),          # >0 = overvaluasi rata-rata
        "mae_juta": float(np.mean(np.abs(p - y))),
        "rmse_juta": float(np.sqrt(np.mean((p - y) ** 2))),
        "mape": float(np.mean(ape)),
        "mdape": float(np.median(ape)),
        "ppe10": float(np.mean(ape <= 0.10)),
        "ppe20": float(np.mean(ape <= 0.20)),
    }


def bootstrap_ci(y_log, p_log, n_boot: int = 1000, alpha: float = 0.05, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y_log, p_log = np.asarray(y_log), np.asarray(p_log)
    point = regression_metrics(y_log, p_log)
    boot = pd.DataFrame([regression_metrics(y_log[i], p_log[i])
                         for i in (rng.integers(0, len(y_log), len(y_log)) for _ in range(n_boot))])
    return pd.DataFrame({"estimate": pd.Series(point), "ci_lower": boot.quantile(alpha / 2),
                         "ci_upper": boot.quantile(1 - alpha / 2), "std_error": boot.std()})


# ============================================================================ perbandingan model
def corrected_resampled_ttest(scores_a, scores_b, n_train: int, n_test: int) -> dict:
    """Nadeau & Bengio (2003): t-test untuk skor repeated K-fold berpasangan."""
    d = np.asarray(scores_a) - np.asarray(scores_b)
    k = len(d)
    t = d.mean() / np.sqrt((1 / k + n_test / n_train) * d.var(ddof=1))
    return {"mean_diff": float(d.mean()), "t": float(t), "df": k - 1, "p_value": float(2 * stats.t.sf(abs(t), k - 1))}


def corrected_mean_ci(values, n_train: int, n_test: int, alpha: float = 0.05):
    v = np.asarray(values)
    se = np.sqrt((1 / len(v) + n_test / n_train) * v.var(ddof=1))
    half = stats.t.ppf(1 - alpha / 2, len(v) - 1) * se
    return float(v.mean()), float(v.mean() - half), float(v.mean() + half)


def paired_error_tests(y_log, p_a, p_b) -> pd.DataFrame:
    """Uji berpasangan pada data test yang sama.
    - Paired t-test kuadrat error (versi cross-section dari uji Diebold-Mariano)
    - Wilcoxon signed-rank pada selisih |error| (non-parametrik)
    Selisih negatif = model A lebih akurat.
    """
    e_a, e_b = np.asarray(p_a) - y_log, np.asarray(p_b) - y_log
    d_sq = e_a**2 - e_b**2
    t = stats.ttest_1samp(d_sq, 0)
    w = stats.wilcoxon(np.abs(e_a), np.abs(e_b))
    return pd.DataFrame({
        "uji": ["Paired t-test kuadrat error (Diebold-Mariano)", "Wilcoxon signed-rank |error|"],
        "selisih_rata2": [d_sq.mean(), np.mean(np.abs(e_a) - np.abs(e_b))],
        "statistik": [t.statistic, w.statistic],
        "p_value": [t.pvalue, w.pvalue],
        "A_lebih_akurat_pada_%": [np.mean(np.abs(e_a) < np.abs(e_b))] * 2,
    }).set_index("uji")


# ============================================================================ diagnostik residual
def mincer_zarnowitz(y_log, p_log) -> dict:
    """Regresi aktual = a + b·prediksi. Ideal a = 0, b = 1 (prediksi tidak bias & skalanya tepat).
    b < 1: prediksi terlalu “menyebar” (properti mahal ditaksir terlalu tinggi); b > 1: terlalu menyusut ke rata-rata.
    Uji F gabungan H0: a = 0 dan b = 1.
    """
    y, p = np.asarray(y_log), np.asarray(p_log)
    n = len(y)
    X = np.column_stack([np.ones(n), p])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    s2 = resid @ resid / (n - 2)
    cov = s2 * np.linalg.inv(X.T @ X)
    diff = beta - np.array([0.0, 1.0])
    f = float(diff @ np.linalg.inv(cov) @ diff / 2)
    return {"intercept": float(beta[0]), "slope": float(beta[1]), "slope_se": float(np.sqrt(cov[1, 1])),
            "F": f, "p_value": float(stats.f.sf(f, 2, n - 2))}


def breusch_pagan(residuals, fitted) -> dict:
    """Heteroskedastisitas: regresi residual² pada [prediksi, prediksi²]. LM = n·R² ~ χ²(2).
    H0: variance error konstan."""
    r2 = np.asarray(residuals) ** 2
    Z = np.column_stack([fitted, np.asarray(fitted) ** 2])
    rsq = LinearRegression().fit(Z, r2).score(Z, r2)
    lm = len(r2) * rsq
    return {"LM": float(lm), "df": 2, "p_value": float(stats.chi2.sf(lm, 2))}


def residual_normality(residuals) -> dict:
    jb = stats.jarque_bera(residuals)
    return {"skewness": float(stats.skew(residuals)), "excess_kurtosis": float(stats.kurtosis(residuals)),
            "jarque_bera": float(jb.statistic), "p_value": float(jb.pvalue)}


def segment_bias(y_log, p_log, groups) -> pd.DataFrame:
    """Bias per segmen: rata-rata error log (≈ % over/under-valuasi) + one-sample t-test (H0: bias = 0)."""
    df = pd.DataFrame({"e": np.asarray(p_log) - np.asarray(y_log), "g": np.asarray(groups)})
    rows = []
    for g, part in df.groupby("g", observed=True):
        t = stats.ttest_1samp(part["e"], 0)
        ape = np.abs(np.exp(part["e"]) - 1)
        rows.append({"segmen": g, "n": len(part), "bias_pct": np.exp(part["e"].mean()) - 1,
                     "mdape": ape.median(), "ppe10": (ape <= 0.1).mean(), "p_value": t.pvalue})
    out = pd.DataFrame(rows).set_index("segmen")
    out["p_holm"] = holm_correction(out["p_value"])
    return out


# ============================================================================ interval prediksi
def conformal_quantiles(oof_residuals_log, coverage: float = 0.90) -> tuple[float, float]:
    """Split-conformal (versi CV+): kuantil residual out-of-fold (aktual − prediksi, skala log).
    Interval harga = prediksi × [exp(q_low), exp(q_high)]."""
    r = np.asarray(oof_residuals_log)
    a = (1 - coverage) / 2
    n = len(r)
    lo = np.quantile(r, max(0.0, np.floor(a * (n + 1)) / n))
    hi = np.quantile(r, min(1.0, np.ceil((1 - a) * (n + 1)) / n))
    return float(lo), float(hi)


def interval_coverage(y_log, p_log, q_low: float, q_high: float, nominal: float = 0.90) -> dict:
    r = np.asarray(y_log) - np.asarray(p_log)
    inside = (r >= q_low) & (r <= q_high)
    test = stats.binomtest(int(inside.sum()), len(inside), nominal)
    ci = test.proportion_ci(method="wilson")
    return {"coverage": float(inside.mean()), "ci_lower": ci.low, "ci_upper": ci.high, "n": len(inside),
            "p_value": float(test.pvalue), "below": float((r < q_low).mean()), "above": float((r > q_high).mean())}


# ============================================================================ uji fitur & stabilitas
def holm_correction(p_values: pd.Series) -> pd.Series:
    p = p_values.sort_values()
    m = len(p)
    adj = np.maximum.accumulate([min(1, (m - i) * v) for i, v in enumerate(p.values)])
    return pd.Series(adj, index=p.index).reindex(p_values.index)


def kruskal_epsilon(values, groups) -> tuple[float, float, float]:
    df = pd.DataFrame({"v": np.asarray(values), "g": np.asarray(groups)}).dropna()
    h, p = stats.kruskal(*[g["v"].to_numpy() for _, g in df.groupby("g")])
    n = len(df)
    return float(h), float(p), float(h / ((n**2 - 1) / (n + 1)))


def vif(df_numeric: pd.DataFrame) -> pd.Series:
    X = df_numeric.fillna(df_numeric.median())
    out = {}
    for col in X.columns:
        others = X.drop(columns=col)
        r2 = LinearRegression().fit(others, X[col]).score(others, X[col])
        out[col] = 1 / (1 - r2) if r2 < 1 else np.inf
    return pd.Series(out, name="VIF").sort_values(ascending=False)


def psi_numeric(expected, actual, bins: int = 10) -> float:
    e_arr, a_arr = np.asarray(expected, float), np.asarray(actual, float)
    e_arr, a_arr = e_arr[~np.isnan(e_arr)], a_arr[~np.isnan(a_arr)]
    edges = np.unique(np.quantile(e_arr, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.clip(np.histogram(e_arr, edges)[0] / len(e_arr), 1e-6, None)
    a = np.clip(np.histogram(a_arr, edges)[0] / len(a_arr), 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_categorical(expected, actual) -> float:
    e = pd.Series(expected).fillna("missing").value_counts(normalize=True)
    a = pd.Series(actual).fillna("missing").value_counts(normalize=True)
    cats = e.index.union(a.index)
    e, a = e.reindex(cats, fill_value=1e-6).clip(lower=1e-6), a.reindex(cats, fill_value=1e-6).clip(lower=1e-6)
    return float(np.sum((a - e) * np.log(a / e)))


def stability_label(v: float) -> str:
    return "stabil" if v < 0.1 else ("perlu perhatian" if v < 0.25 else "bergeser signifikan")
