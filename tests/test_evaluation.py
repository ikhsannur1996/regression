import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from src.evaluation import (
    breusch_pagan,
    conformal_quantiles,
    interval_coverage,
    mincer_zarnowitz,
    paired_error_tests,
    regression_metrics,
)
from src.features import FEATURES, MODEL_INPUT, TIME_COL, build_preprocessor
from src.json_model import JsonValuationModel, export_model
from src.model import TrendAdjustedRegressor

rng = np.random.default_rng(0)


def test_regression_metrics_perfect_and_known_bias():
    y = np.log(rng.uniform(100, 5000, 1000))
    perfect = regression_metrics(y, y)
    assert perfect["rmse_log"] == 0 and perfect["ppe10"] == 1 and perfect["mdape"] == 0
    over = regression_metrics(y, y + np.log(1.05))   # semua dinilai 5% terlalu tinggi
    assert over["bias_pct"] == pytest.approx(0.05)
    assert over["ppe10"] == 1 and over["ppe20"] == 1


def test_mincer_zarnowitz_unbiased_vs_shrunk():
    p = rng.normal(7, 0.5, 5000)
    y = p + rng.normal(0, 0.1, 5000)
    assert mincer_zarnowitz(y, p)["slope"] == pytest.approx(1, abs=0.02)
    shrunk = 7 + 0.7 * (p - 7)                       # prediksi terlalu menyusut ke rata-rata
    assert mincer_zarnowitz(y, shrunk)["slope"] > 1.3
    assert mincer_zarnowitz(y, shrunk)["p_value"] < 1e-6


def test_breusch_pagan_detects_heteroskedasticity():
    fitted = rng.uniform(6, 9, 4000)
    assert breusch_pagan(rng.normal(0, 0.1, 4000), fitted)["p_value"] > 0.01
    assert breusch_pagan(rng.normal(0, 1, 4000) * (fitted - 5) * 0.05, fitted)["p_value"] < 1e-6


def test_conformal_interval_has_nominal_coverage():
    r_train = rng.normal(0, 0.12, 5000)
    lo, hi = conformal_quantiles(r_train, 0.9)
    y_new = rng.normal(0, 0.12, 5000)
    cov = interval_coverage(y_new, np.zeros(5000), lo, hi, 0.9)
    assert cov["coverage"] == pytest.approx(0.9, abs=0.02)


def test_paired_tests_prefer_better_model():
    y = rng.normal(7, 0.5, 2000)
    good, bad = y + rng.normal(0, 0.05, 2000), y + rng.normal(0, 0.2, 2000)
    res = paired_error_tests(y, good, bad)
    assert (res["p_value"] < 1e-10).all() and (res["selisih_rata2"] < 0).all()


def _toy_data(n=1500):
    from scripts.generate_data import generate
    from src.features import add_time
    df = add_time(generate(n, seed=1))
    return df[MODEL_INPUT], np.log(df["price"])


def test_trend_adjustment_recovers_growth_and_extrapolates():
    X, y = _toy_data(3000)
    m = TrendAdjustedRegressor(Pipeline([("preprocess", build_preprocessor()), ("model", Ridge(1.0))])).fit(X, y)
    assert 0.04 < m.annual_growth_ < 0.08          # data dibangkitkan dengan apresiasi 4,5–8%/tahun
    later = X.assign(**{TIME_COL: X[TIME_COL] + 12})
    assert np.mean(m.predict(later) - m.predict(X)) == pytest.approx(np.log(1 + m.annual_growth_), abs=1e-9)


@pytest.mark.parametrize("estimator", [Ridge(1.0), GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=0)])
def test_json_model_matches_sklearn(estimator):
    X, y = _toy_data()
    m = TrendAdjustedRegressor(Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])).fit(X, y)
    jm = JsonValuationModel(export_model(m, {"features": FEATURES}, {"q_low": -0.2, "q_high": 0.2}))
    edge = X.head(30).copy()
    edge.loc[edge.index[:10], ["building_age", "road_width_m"]] = np.nan
    edge.loc[edge.index[10:20], "city"] = "medan"
    edge.loc[edge.index[20:], "condition"] = "rusak_berat"
    for data in (X, edge):
        assert np.abs(jm.predict_log(data, data[TIME_COL]) - m.predict(data)).max() < 1e-9
