"""Serialisasi model taksiran properti ke JSON + inferensi tanpa scikit-learn.

Isi JSON: preprocessing (bin, imputasi, log, scaler, ordinal, one-hot), model (Ridge atau
GradientBoostingRegressor), tren harga (indexing), dan kuantil interval prediksi (conformal).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

FORMAT_NAME = "property-valuation-json-model"
FORMAT_VERSION = 1


def _l(a):
    return np.asarray(a).tolist()


def _edges(e):
    return [None if np.isinf(v) else float(v) for v in e]


def _edges_back(e):
    out = np.array([np.nan if v is None else v for v in e], float)
    out[0] = -np.inf if np.isnan(out[0]) else out[0]
    out[-1] = np.inf if np.isnan(out[-1]) else out[-1]
    return out


# ----------------------------------------------------------------------------- export
def _export_preprocessor(ct) -> list[dict]:
    steps = []
    for name, tr, cols in ct.transformers_:
        cols = list(cols)
        if name == "remainder":
            continue
        if name == "domain_bin":
            binner, oh = tr.named_steps["bin"], tr.named_steps["onehot"]
            steps.append({"name": name, "type": "domain_bin_onehot", "columns": cols,
                          "bins": {c: {"edges": _edges(binner.bins[c]["edges"]), "labels": binner.bins[c]["labels"]} for c in cols},
                          "categories": [_l(c) for c in oh.categories_]})
        elif name in ("log_num", "num"):
            steps.append({"name": name, "type": "impute_scale", "columns": cols, "log1p": name == "log_num",
                          "impute_values": _l(tr.named_steps["imputer"].statistics_),
                          "mean": _l(tr.named_steps["scaler"].mean_), "scale": _l(tr.named_steps["scaler"].scale_)})
        elif name == "binary":
            steps.append({"name": name, "type": "impute_passthrough", "columns": cols, "impute_values": _l(tr.statistics_)})
        elif name == "ordinal":
            steps.append({"name": name, "type": "impute_ordinal", "columns": cols,
                          "impute_values": _l(tr.named_steps["imputer"].statistics_),
                          "categories": [_l(c) for c in tr.named_steps["encode"].categories_],
                          "unknown_values": _l(tr.named_steps["impute_unknown"].statistics_)})
        elif name == "onehot":
            steps.append({"name": name, "type": "impute_onehot", "columns": cols,
                          "impute_values": _l(tr.named_steps["imputer"].statistics_),
                          "categories": [_l(c) for c in tr.named_steps["onehot"].categories_]})
        else:
            raise ValueError(f"Transformer '{name}' belum didukung")
    return steps


def _export_model(m) -> dict:
    kind = type(m).__name__
    if kind == "Ridge":
        return {"type": "ridge", "coef": _l(m.coef_), "intercept": float(m.intercept_)}
    if kind == "GradientBoostingRegressor":
        if m.loss != "squared_error":
            raise ValueError("Hanya loss squared_error yang didukung")
        trees = []
        for est in m.estimators_[:, 0]:
            t = est.tree_
            trees.append({"left": _l(t.children_left), "right": _l(t.children_right), "feature": _l(t.feature),
                          "threshold": _l(t.threshold), "value": _l(t.value[:, 0, 0])})
        return {"type": "gbr", "init": float(m.init_.constant_.ravel()[0]), "learning_rate": float(m.learning_rate), "trees": trees}
    raise ValueError(f"Model '{kind}' belum didukung")


def export_model(trend_model, metadata: dict, interval: dict) -> dict:
    pipe = trend_model.pipeline_
    ct = pipe.named_steps["preprocess"]
    return {
        "format": FORMAT_NAME, "format_version": FORMAT_VERSION, "metadata": metadata,
        "preprocessing": _export_preprocessor(ct), "feature_names_out": _l(ct.get_feature_names_out()),
        "model": _export_model(pipe.named_steps["model"]),
        "trend": {"per_month_log": trend_model.trend_per_month_, "t_ref": trend_model.t_ref_,
                  "time_start": "2022-01-01", "annual_growth": trend_model.annual_growth_},
        "interval": interval,
    }


def save_json(spec: dict, path) -> None:
    Path(path).write_text(json.dumps(spec, allow_nan=False, ensure_ascii=False, separators=(",", ":")))


# ----------------------------------------------------------------------------- inference
def _onehot(values, cats):
    return (np.asarray(values, dtype=object)[:, None] == np.asarray(cats, dtype=object)[None, :]).astype(float)


class JsonValuationModel:
    def __init__(self, spec: dict):
        if spec.get("format") != FORMAT_NAME or spec.get("format_version") != FORMAT_VERSION:
            raise ValueError("Format model JSON tidak dikenali")
        self.spec, self.metadata, self.model = spec, spec["metadata"], spec["model"]
        self.features, self.trend, self.interval = self.metadata["features"], spec["trend"], spec["interval"]
        if self.model["type"] == "gbr":
            self._trees = [{k: np.asarray(v) for k, v in t.items()} for t in self.model["trees"]]

    @classmethod
    def load(cls, path) -> "JsonValuationModel":
        return cls(json.loads(Path(path).read_text()))

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        blocks = []
        for s in self.spec["preprocessing"]:
            cols = s["columns"]
            if s["type"] == "domain_bin_onehot":
                for col, cats in zip(cols, s["categories"]):
                    spec = s["bins"][col]
                    b = pd.cut(X[col].astype(float), _edges_back(spec["edges"]), labels=spec["labels"])
                    blocks.append(_onehot(b.astype(object).where(b.notna(), "missing"), cats))
            elif s["type"] == "impute_scale":
                x = X[cols].astype(float).to_numpy()
                x = np.where(np.isnan(x), np.asarray(s["impute_values"], float), x)
                if s["log1p"]:
                    x = np.log1p(x)
                blocks.append((x - np.asarray(s["mean"])) / np.asarray(s["scale"]))
            elif s["type"] == "impute_passthrough":
                x = X[cols].astype(float).to_numpy()
                blocks.append(np.where(np.isnan(x), np.asarray(s["impute_values"], float), x))
            elif s["type"] == "impute_ordinal":
                for col, fill, cats, unk in zip(cols, s["impute_values"], s["categories"], s["unknown_values"]):
                    v = X[col].astype(object).where(X[col].notna(), fill)
                    idx = {c: i for i, c in enumerate(cats)}
                    blocks.append(np.array([idx.get(x, unk) for x in v], float)[:, None])
            elif s["type"] == "impute_onehot":
                for col, fill, cats in zip(cols, s["impute_values"], s["categories"]):
                    blocks.append(_onehot(X[col].astype(object).where(X[col].notna(), fill), cats))
            else:
                raise ValueError(s["type"])
        out = np.hstack(blocks)
        assert out.shape[1] == len(self.spec["feature_names_out"])
        return out

    def _tree(self, t, Z):
        node = np.zeros(len(Z), dtype=np.int64)
        rows = np.arange(len(Z))
        while True:
            active = t["left"][node] != -1
            if not active.any():
                return t["value"][node]
            r, n = rows[active], node[active]
            go_left = Z[r, t["feature"][n]] <= t["threshold"][n]
            node[r] = np.where(go_left, t["left"][n], t["right"][n])

    def predict_log_at_reference(self, X: pd.DataFrame) -> np.ndarray:
        """log(harga) pada level harga bulan acuan t_ref (tanpa penyesuaian waktu)."""
        Z = self.transform(X[self.features])
        if self.model["type"] == "ridge":
            return Z @ np.asarray(self.model["coef"]) + self.model["intercept"]
        Z32 = Z.astype(np.float32).astype(np.float64)  # pohon sklearn membandingkan dalam float32
        return self.model["init"] + self.model["learning_rate"] * np.sum([self._tree(t, Z32) for t in self._trees], axis=0)

    def predict_log(self, X: pd.DataFrame, months_since_start) -> np.ndarray:
        t = np.asarray(months_since_start, float)
        return self.predict_log_at_reference(X) + self.trend["per_month_log"] * (t - self.trend["t_ref"])

    def predict_value(self, X: pd.DataFrame, months_since_start) -> pd.DataFrame:
        """Taksiran harga (juta Rp) + interval prediksi."""
        p = self.predict_log(X, months_since_start)
        return pd.DataFrame({"estimate": np.exp(p), "lower": np.exp(p + self.interval["q_low"]),
                             "upper": np.exp(p + self.interval["q_high"])})
