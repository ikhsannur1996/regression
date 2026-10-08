"""Fitur & preprocessing untuk model taksiran nilai properti (AVM).

Target model: log(harga). Waktu transaksi TIDAK masuk preprocessing; waktu ditangani
terpisah oleh `TrendAdjustedRegressor` (src/model.py) agar model bisa “maju ke masa depan”.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

DATE_COL, ID_COL, TARGET = "transaction_date", "property_id", "price"
TIME_COL = "months_since_start"          # dihitung dari tanggal transaksi / tanggal valuasi
TIME_START = pd.Timestamp("2022-01-01")

LOG_COLS = ["land_area", "building_area", "distance_to_cbd_km"]
NUM_COLS = ["floors", "bedrooms", "bathrooms", "carport"]
BINARY_COLS = ["near_toll", "near_transit", "flood_prone"]
ORDINAL_COLS = {"condition": ["perlu_renovasi", "sedang", "baik"]}
ONEHOT_COLS = ["city", "property_type", "certificate"]

# binning berbasis aturan appraisal (bukan dari data)
DOMAIN_BINS = {
    "building_age": {"edges": [-np.inf, 5, 15, 30, np.inf], "labels": ["baru_0-5", "6-15", "16-30", "tua_>30"]},
    "road_width_m": {"edges": [-np.inf, 3, 6, np.inf], "labels": ["gang_<3m", "sedang_3-6m", "lebar_>6m"]},
}

FEATURES = list(DOMAIN_BINS) + LOG_COLS + NUM_COLS + BINARY_COLS + list(ORDINAL_COLS) + ONEHOT_COLS
MODEL_INPUT = FEATURES + [TIME_COL]


def months_since_start(dates) -> np.ndarray:
    """Bulan (pecahan) sejak 1 Jan 2022, misalnya 15 Mar 2023 → 14,5."""
    d = pd.to_datetime(pd.Series(dates))
    return ((d - TIME_START).dt.days / 30.4375).to_numpy()


def add_time(df: pd.DataFrame, date_col: str = DATE_COL) -> pd.DataFrame:
    return df.assign(**{TIME_COL: months_since_start(df[date_col])})


class DomainBinner(BaseEstimator, TransformerMixin):
    """Angka → label bin berdasarkan batas tetap; NaN → 'missing'."""

    def __init__(self, bins: dict):
        self.bins = bins

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(list(self.bins))
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.feature_names_in_)
        out = pd.DataFrame(index=X.index)
        for col, spec in self.bins.items():
            b = pd.cut(X[col].astype(float), bins=spec["edges"], labels=spec["labels"])
            out[f"{col}_bin"] = b.astype(object).where(b.notna(), "missing")
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"{c}_bin" for c in self.bins])


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("domain_bin", Pipeline([("bin", DomainBinner(DOMAIN_BINS)),
                                 ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), list(DOMAIN_BINS)),
        ("log_num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                              ("scaler", StandardScaler())]), LOG_COLS),
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), NUM_COLS),
        ("binary", SimpleImputer(strategy="most_frequent"), BINARY_COLS),
        ("ordinal", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("encode", OrdinalEncoder(categories=list(ORDINAL_COLS.values()),
                                                        handle_unknown="use_encoded_value", unknown_value=np.nan)),
                              ("impute_unknown", SimpleImputer(strategy="median"))]), list(ORDINAL_COLS)),
        ("onehot", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                             ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), ONEHOT_COLS),
    ], remainder="drop")
