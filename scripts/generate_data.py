"""Generate dataset sintetis transaksi/appraisal properti (agunan KPR) -> data/properties.csv.

Harga = nilai tanah + nilai bangunan, dengan faktor lokasi, akses, banjir, legalitas,
kondisi, dan apresiasi harga per kota sepanjang 2022–2025. Struktur "tanah + bangunan"
membuat hubungan fitur → harga TIDAK linear (interaksi luas × lokasi), sehingga ada
alasan nyata membandingkan model linear vs model berbasis pohon.

    python scripts/generate_data.py --n 10000 --seed 42
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "properties.csv"

# kota: (porsi, harga tanah juta/m² di pusat, jarak ke CBD (min, max) km, apresiasi per tahun)
CITIES = {
    "jakarta_selatan": (0.12, 32.0, (2, 18), 0.055),
    "jakarta_barat": (0.10, 20.0, (3, 20), 0.045),
    "tangerang_selatan": (0.13, 16.0, (3, 25), 0.075),
    "bekasi": (0.12, 10.0, (3, 30), 0.060),
    "depok": (0.10, 11.0, (3, 25), 0.065),
    "bogor": (0.08, 7.5, (3, 30), 0.050),
    "bandung": (0.11, 11.0, (2, 22), 0.055),
    "surabaya": (0.11, 14.0, (2, 25), 0.050),
    "semarang": (0.06, 7.0, (2, 20), 0.045),
    "yogyakarta": (0.07, 8.5, (2, 20), 0.080),
}
START = pd.Timestamp("2022-01-01")
N_MONTHS = 48  # Jan 2022 – Des 2025


def generate(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = list(CITIES)
    city = rng.choice(names, n, p=[CITIES[c][0] for c in names])
    land_price_cbd = np.array([CITIES[c][1] for c in city])
    growth = np.array([CITIES[c][3] for c in city])
    dist = np.array([rng.uniform(*CITIES[c][2]) for c in city])

    property_type = rng.choice(["rumah", "townhouse", "ruko"], n, p=[0.72, 0.13, 0.15])
    land_area = np.where(
        property_type == "rumah", rng.lognormal(np.log(120), 0.45, n),
        np.where(property_type == "townhouse", rng.lognormal(np.log(80), 0.25, n), rng.lognormal(np.log(75), 0.30, n)),
    )
    land_area = np.clip(np.round(land_area), 30, 1500)
    floors = np.where(property_type == "ruko", rng.choice([2, 3, 4], n, p=[0.3, 0.5, 0.2]),
                      np.where(property_type == "townhouse", rng.choice([2, 3], n, p=[0.8, 0.2]),
                               rng.choice([1, 2, 3], n, p=[0.45, 0.48, 0.07])))
    coverage = rng.uniform(0.45, 0.95, n)
    building_area = np.clip(np.round(land_area * coverage * floors * rng.uniform(0.7, 1.0, n)), 21, 2500)
    bedrooms = np.clip(np.round(building_area / rng.uniform(28, 45, n)), 1, 10).astype(int)
    bedrooms = np.where(property_type == "ruko", np.clip(bedrooms - 2, 0, 6), bedrooms)
    bathrooms = np.clip(np.round(bedrooms * rng.uniform(0.6, 1.1, n)), 1, 8).astype(int)
    carport = np.clip(np.round(land_area / 90 + rng.normal(0, 0.6, n)), 0, 6).astype(int)
    building_age = np.clip(np.round(rng.gamma(2.0, 7.0, n)), 0, 60)
    building_age = np.where(property_type == "townhouse", np.clip(building_age, 0, 15), building_age)
    condition = np.where(building_age > 30, rng.choice(["perlu_renovasi", "sedang", "baik"], n, p=[0.45, 0.4, 0.15]),
                         np.where(building_age > 12, rng.choice(["perlu_renovasi", "sedang", "baik"], n, p=[0.15, 0.5, 0.35]),
                                  rng.choice(["perlu_renovasi", "sedang", "baik"], n, p=[0.03, 0.27, 0.70])))
    certificate = rng.choice(["SHM", "HGB", "girik_ajb"], n, p=[0.68, 0.27, 0.05])
    road_width = np.round(np.clip(np.where(property_type == "ruko", rng.normal(9, 3, n), rng.lognormal(np.log(5), 0.4, n)), 1.5, 20), 1)
    near_toll = (rng.uniform(0, 1, n) < np.clip(0.55 - dist / 60, 0.1, 0.6)).astype(int)
    near_transit = (rng.uniform(0, 1, n) < np.clip(0.5 - dist / 40, 0.05, 0.5)).astype(int)
    flood_prone = (rng.uniform(0, 1, n) < 0.14).astype(int)

    # --- nilai tanah ------------------------------------------------------------------
    access = np.where(road_width < 3, 0.82, np.where(road_width < 6, 1.0, 1.12))
    land_unit = (land_price_cbd * np.exp(-0.045 * dist) * access
                 * np.where(flood_prone == 1, 0.85, 1.0)
                 * np.where(near_transit == 1, 1.10, 1.0) * np.where(near_toll == 1, 1.06, 1.0)
                 * pd.Series(certificate).map({"SHM": 1.0, "HGB": 0.94, "girik_ajb": 0.75}).to_numpy()
                 * np.where(property_type == "ruko", 1.30, 1.0))
    land_value = land_area ** 0.95 * land_unit  # tanah sangat luas sedikit lebih murah per m²

    # --- nilai bangunan ---------------------------------------------------------------
    construction_cost = np.where(property_type == "ruko", 4.2, np.where(property_type == "townhouse", 5.5, 4.8))
    depreciation = np.clip(1 - building_age / 55, 0.25, 1.0)
    cond_mult = pd.Series(condition).map({"baik": 1.0, "sedang": 0.85, "perlu_renovasi": 0.6}).to_numpy()
    building_value = building_area * construction_cost * depreciation * cond_mult

    # --- waktu & apresiasi --------------------------------------------------------------
    month = rng.integers(0, N_MONTHS, n)
    day = rng.integers(1, 29, n)
    tx_date = START + pd.to_timedelta(month * 30.44 + day, unit="D")
    years = (tx_date - START).days.to_numpy() / 365.25
    appreciation = (1 + growth) ** years

    price = (land_value + building_value) * appreciation * rng.lognormal(0, 0.10, n)

    df = pd.DataFrame({
        "property_id": [f"P{i + 1:06d}" for i in range(n)],
        "transaction_date": tx_date.strftime("%Y-%m-%d"),
        "city": city,
        "property_type": property_type,
        "land_area": land_area,
        "building_area": building_area,
        "floors": floors,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "carport": carport,
        "building_age": building_age,
        "condition": condition,
        "certificate": certificate,
        "road_width_m": road_width,
        "distance_to_cbd_km": np.round(dist, 1),
        "near_toll": near_toll,
        "near_transit": near_transit,
        "flood_prone": flood_prone,
        "price": np.round(price, 1),  # juta Rp
    })
    # data appraisal tidak lengkap
    df.loc[rng.uniform(0, 1, n) < 0.05, "building_age"] = np.nan
    df.loc[rng.uniform(0, 1, n) < 0.03, "road_width_m"] = np.nan
    return df.sort_values("transaction_date").reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    data = generate(args.n, args.seed)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT_PATH, index=False)
    print(f"{data.shape} -> {OUT_PATH}")
    print(data["price"].describe().round(1).to_dict())
