"""FastAPI service: Automated Valuation Model (AVM) untuk taksiran nilai agunan properti."""

import logging
import os
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import BatchRequest, BatchResponse, HealthResponse, LoanCheck, PropertyInput, ValuationResult
from src.json_model import JsonValuationModel

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "models" / "model.json"))
MAX_EXTRAPOLATION_MONTHS = float(os.getenv("MAX_EXTRAPOLATION_MONTHS", 6))
TIME_START = pd.Timestamp("2022-01-01")
BINARY = ("near_toll", "near_transit", "flood_prone")

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("valuation-api")
state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    model = JsonValuationModel.load(MODEL_PATH)
    allowed = set(PropertyInput.model_fields) - {"property_id", "valuation_date", "loan_amount"}
    if allowed != set(model.metadata["features"]):
        raise RuntimeError("Field di schemas.py tidak sama dengan fitur di models/model.json")
    state["model"], state["metadata"] = model, model.metadata
    logger.info("Model AVM '%s' versi %s dimuat (tren %.2f%%/tahun, data s.d. bulan ke-%.1f)",
                model.metadata["model_name"], model.metadata["model_version"],
                model.trend["annual_growth"] * 100, model.trend["t_ref"])
    yield
    state.clear()


app = FastAPI(
    title="Property Valuation API (AVM)",
    description="Taksiran nilai agunan properti + interval prediksi 90% + plafon maksimum berdasarkan LTV.",
    version="1.0.0",
    lifespan=lifespan,
)


def _months_since_start(d: date) -> float:
    return (pd.Timestamp(d) - TIME_START).days / 30.4375


def _warnings(row: PropertyInput, t: float) -> list[str]:
    meta, trend = state["metadata"], state["model"].trend
    out = []
    ahead = t - trend["t_ref"]
    if ahead > MAX_EXTRAPOLATION_MONTHS:
        out.append(f"Tanggal valuasi {ahead:.0f} bulan setelah data terakhir; taksiran memakai proyeksi tren "
                   f"{trend['annual_growth']:.1%}/tahun. Disarankan refit model.")
    for col, (lo, hi) in meta["numeric_ranges_p1_p99"].items():
        v = getattr(row, col)
        if v is not None and (v < lo or v > hi):
            out.append(f"{col} = {v} di luar rentang umum data latih ({lo:g}–{hi:g}); ketidakpastian lebih besar.")
    if row.building_age is None or row.road_width_m is None:
        out.append("Sebagian data (umur bangunan/lebar jalan) kosong; diperlakukan sebagai kategori 'missing'.")
    return out


def _value(rows: list[PropertyInput]) -> list[ValuationResult]:
    model, meta = state["model"], state["metadata"]
    dates = [r.valuation_date or date.today() for r in rows]
    t = np.array([_months_since_start(d) for d in dates])
    X = pd.DataFrame([r.model_dump(include=set(meta["features"])) for r in rows])[meta["features"]]
    for b in BINARY:
        X[b] = X[b].astype(int)
    X = X.fillna(value=np.nan)
    try:
        values = model.predict_value(X, t)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Inferensi gagal")
        raise HTTPException(status_code=500, detail="Inferensi model gagal") from exc

    results = []
    for row, d, ti, (_, v) in zip(rows, dates, t, values.iterrows()):
        ltv = meta["ltv_policy"][row.property_type]
        loan = None
        if row.loan_amount is not None:
            loan = LoanCheck(loan_amount=row.loan_amount,
                             ltv_on_estimate=round(row.loan_amount / v["estimate"], 4),
                             ltv_on_lower_bound=round(row.loan_amount / v["lower"], 4),
                             max_ltv_policy=ltv,
                             within_policy=row.loan_amount / v["estimate"] <= ltv,
                             within_policy_conservative=row.loan_amount / v["lower"] <= ltv)
        results.append(ValuationResult(
            property_id=row.property_id, valuation_date=d,
            estimated_value=round(float(v["estimate"]), 1), lower_90=round(float(v["lower"]), 1),
            upper_90=round(float(v["upper"]), 1), price_per_m2_land=round(float(v["estimate"]) / row.land_area, 3),
            max_ltv_policy=ltv, max_loan=round(ltv * float(v["estimate"]), 1),
            conservative_max_loan=round(ltv * float(v["lower"]), 1), loan_check=loan, warnings=_warnings(row, ti),
        ))
    return results


@app.get("/", include_in_schema=False)
def root():
    return {"message": "Property Valuation API. Buka /docs untuk dokumentasi."}


@app.get("/health", response_model=HealthResponse)
def health():
    meta = state.get("metadata")
    return HealthResponse(status="ok", model_loaded="model" in state, model_version=meta["model_version"] if meta else None)


@app.get("/model-info")
def model_info():
    model, meta = state["model"], state["metadata"]
    return {**{k: meta[k] for k in ("model_name", "model_version", "trained_at", "training_period", "n_training_rows",
                                    "target", "features", "categorical_values", "ltv_policy", "evaluation")},
            "trend": model.trend, "interval": model.interval}


@app.post("/valuation", response_model=ValuationResult)
def valuation(payload: PropertyInput):
    return _value([payload])[0]


@app.post("/valuation/batch", response_model=BatchResponse)
def valuation_batch(payload: BatchRequest):
    results = _value(payload.instances)
    return BatchResponse(model_version=state["metadata"]["model_version"], results=results,
                         total_estimated_value=round(sum(r.estimated_value for r in results), 1))
