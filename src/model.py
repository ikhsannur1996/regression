"""Regressor dengan penyesuaian tren harga (indexing), standar pada Automated Valuation Model.

Masalah: harga properti naik dari waktu ke waktu. Model berbasis pohon (Gradient Boosting,
Random Forest) TIDAK bisa mengekstrapolasi: untuk tanggal di luar data latih, prediksinya
“membeku” di level harga bulan terakhir.

Solusi dua tahap:
1. Tren pasar g (log-harga per bulan) diestimasi dengan regresi hedonik linear:
   log(harga) ~ fitur + g · waktu  (Ridge, hanya koefisien waktu yang dipakai).
2. Target disesuaikan ke level harga bulan acuan t_ref (bulan terakhir data latih):
   y_adj = log(harga) − g · (t − t_ref), lalu model utama dilatih pada y_adj TANPA fitur waktu.

Prediksi:  log(harga) = model(fitur) + g · (t − t_ref)
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import Ridge

from src.features import TIME_COL


class TrendAdjustedRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, pipeline, time_col: str = TIME_COL, trend_alpha: float = 1.0):
        self.pipeline = pipeline
        self.time_col = time_col
        self.trend_alpha = trend_alpha

    def _split(self, X):
        return X.drop(columns=self.time_col), X[self.time_col].to_numpy(dtype=float)

    def fit(self, X, y):
        Xf, t = self._split(X)
        y = np.asarray(y, dtype=float)
        pre = clone(self.pipeline.named_steps["preprocess"]).fit(Xf)
        hedonic = Ridge(alpha=self.trend_alpha).fit(np.column_stack([pre.transform(Xf), t]), y)
        self.trend_per_month_ = float(hedonic.coef_[-1])
        self.t_ref_ = float(t.max())
        self.pipeline_ = clone(self.pipeline).fit(Xf, y - self.trend_per_month_ * (t - self.t_ref_))
        return self

    def predict(self, X):
        Xf, t = self._split(X)
        return self.pipeline_.predict(Xf) + self.trend_per_month_ * (t - self.t_ref_)

    @property
    def annual_growth_(self) -> float:
        return float(np.exp(12 * self.trend_per_month_) - 1)
