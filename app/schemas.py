"""Skema request/response API taksiran nilai properti. Nama fitur = `metadata.features` di models/model.json."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CITIES = Literal["jakarta_selatan", "jakarta_barat", "tangerang_selatan", "bekasi", "depok",
                 "bogor", "bandung", "surabaya", "semarang", "yogyakarta"]

EXAMPLE = {
    "property_id": "AGN-0001",
    "city": "tangerang_selatan",
    "property_type": "rumah",
    "land_area": 120,
    "building_area": 150,
    "floors": 2,
    "bedrooms": 4,
    "bathrooms": 3,
    "carport": 1,
    "building_age": 8,
    "condition": "baik",
    "certificate": "SHM",
    "road_width_m": 6,
    "distance_to_cbd_km": 12,
    "near_toll": True,
    "near_transit": False,
    "flood_prone": False,
    "valuation_date": "2026-01-15",
    "loan_amount": 1200,
}


class PropertyInput(BaseModel):
    """Data agunan properti. Luas dalam m², nominal dalam juta Rupiah."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": EXAMPLE})

    property_id: str | None = Field(None, max_length=64, description="Opsional, dikembalikan di response")
    city: CITIES
    property_type: Literal["rumah", "townhouse", "ruko"]
    land_area: float = Field(..., ge=15, le=10_000, description="Luas tanah (m²)")
    building_area: float = Field(..., ge=10, le=20_000, description="Luas bangunan (m²)")
    floors: int = Field(..., ge=1, le=8)
    bedrooms: int = Field(..., ge=0, le=20)
    bathrooms: int = Field(..., ge=0, le=20)
    carport: int = Field(..., ge=0, le=20)
    building_age: float | None = Field(None, ge=0, le=150, description="Umur bangunan (tahun); null jika tidak diketahui")
    condition: Literal["baik", "sedang", "perlu_renovasi"]
    certificate: Literal["SHM", "HGB", "girik_ajb"]
    road_width_m: float | None = Field(None, ge=0.5, le=60, description="Lebar jalan akses (m); null jika tidak diketahui")
    distance_to_cbd_km: float = Field(..., ge=0, le=150)
    near_toll: bool
    near_transit: bool
    flood_prone: bool
    valuation_date: date | None = Field(None, description="Tanggal valuasi; default = hari ini")
    loan_amount: float | None = Field(None, gt=0, description="Plafon yang diajukan (juta Rp), untuk cek LTV")

    @model_validator(mode="after")
    def check_consistency(self):
        if self.building_area > self.land_area * self.floors * 1.1:
            raise ValueError("building_area tidak masuk akal: melebihi land_area × floors × 1,1")
        if self.valuation_date is not None and self.valuation_date < date(2022, 1, 1):
            raise ValueError("valuation_date harus ≥ 2022-01-01 (awal data latih)")
        return self


class BatchRequest(BaseModel):
    instances: list[PropertyInput] = Field(..., min_length=1, max_length=2000)


class LoanCheck(BaseModel):
    loan_amount: float
    ltv_on_estimate: float = Field(..., description="loan_amount / taksiran")
    ltv_on_lower_bound: float = Field(..., description="loan_amount / batas bawah interval (konservatif)")
    max_ltv_policy: float
    within_policy: bool = Field(..., description="LTV terhadap taksiran ≤ kebijakan")
    within_policy_conservative: bool = Field(..., description="LTV terhadap batas bawah ≤ kebijakan")


class ValuationResult(BaseModel):
    property_id: str | None
    valuation_date: date
    estimated_value: float = Field(..., description="Taksiran nilai (juta Rp), median prediksi")
    lower_90: float
    upper_90: float
    price_per_m2_land: float
    max_ltv_policy: float
    max_loan: float = Field(..., description="LTV kebijakan × taksiran")
    conservative_max_loan: float = Field(..., description="LTV kebijakan × batas bawah interval")
    loan_check: LoanCheck | None = None
    warnings: list[str]


class BatchResponse(BaseModel):
    model_version: str
    results: list[ValuationResult]
    total_estimated_value: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None
