# Taksiran Nilai Agunan Properti (AVM): End-to-End Regresi

Proyek regresi perbankan dari notebook sampai API: **Automated Valuation Model (AVM)** yang menaksir nilai wajar rumah/townhouse/ruko yang dijadikan agunan KPR, lengkap dengan **interval prediksi 90%** dan **plafon maksimum berdasarkan LTV**. Dua model dibandingkan (**Ridge** vs **Gradient Boosting**). Model terbaik disimpan sebagai **JSON**, disajikan lewat **FastAPI**, dan dijalankan di **Docker lokal**.

**Highlight:**
- 10.000 transaksi properti (Jan 2022 – Des 2025) di 10 kota, 16 fitur. Target: **log(harga)**, sehingga error dibaca dalam persen
- **Binning** umur bangunan (baru/6–15/16–30/tua) & lebar jalan (gang/sedang/lebar) dan **one-hot** kota/tipe/sertifikat di dalam pipeline
- **Penyesuaian tren harga** (`TrendAdjustedRegressor`): mencegah undervaluasi sistematis karena model pohon tidak bisa mengekstrapolasi waktu
- **Interval prediksi conformal 90%**, dengan cakupan yang diuji di test, OOT, dan setiap kuartal backtest
- Evaluasi statistik lengkap: repeated CV + corrected t-test, bootstrap CI, uji Diebold-Mariano & Wilcoxon, Mincer-Zarnowitz, Breusch-Pagan, bias per segmen
- **Backtesting**: out-of-time (Jul–Des 2025) dan walk-forward 12 kuartal
- Model JSON (±370 KB) → API **tanpa scikit-learn**

## Hasil

**Model terpilih: Gradient Boosting** (600 pohon, depth 3, learning rate 0,05) + penyesuaian tren (apresiasi terestimasi **±6%/tahun**).

| Metrik (test, 1.753 properti) | Gradient Boosting | Ridge | Patokan AVM |
|---|---|---|---|
| **MdAPE** (median error %) | **7,7%** (CI 7,2–8,3%) | 9,0% | ≤ 10% |
| **PPE10** (taksiran dalam ±10%) | **60,7%** (CI 58,5–63,0%) | 54,7% | ≥ 50% |
| **PPE20** (dalam ±20%) | **90,6%** | 86,0% | ≥ 80% |
| R² (log) | 0,960 | 0,946 | |
| Bias rata-rata | −0,1% (CI mencakup 0) | −0,4% | ±2% |

- GBR lebih akurat secara signifikan: repeated CV 5×5 (menang 25/25 fold, corrected t-test p < 10⁻¹⁵) dan uji Diebold-Mariano di test (p < 10⁻¹⁵).
- **Interval 90%** = taksiran × [0,82; 1,21]. Cakupan di test **89,7%** dan di OOT **89,6%** (uji binomial vs 90%: tidak berbeda).
- **Out-of-time Jul–Des 2025**: MdAPE 8,0%, bias −0,03%. Tanpa penyesuaian tren (waktu sebagai fitur biasa), bias menjadi **−4,4%** (p < 10⁻³⁵): agunan ditaksir terlalu rendah.
- **Walk-forward 12 kuartal** (2023Q1–2025Q4): MdAPE 7,4–8,7%, tidak ada kuartal dengan bias signifikan, cakupan interval rata-rata 90,8%.
- **Mini-backtest**: tanpa penyesuaian tren, bias Gradient Boosting membesar seiring horizon: −4,8% (1–6 bulan), −8,1% (7–12 bulan), −10,2% (13–18 bulan).
- ⚠️ Catatan: 2 dari 26 segmen punya bias kecil tetapi signifikan (Surabaya −3,7%; kuintil termurah +1,9%). Detail ada di notebook Bagian 13.

## Struktur Proyek

```
regression-e2e/
├── data/properties.csv                   # 10.000 transaksi (dibuat oleh scripts/generate_data.py)
├── scripts/generate_data.py              # generator: nilai tanah + bangunan, faktor lokasi/akses/banjir/legalitas, apresiasi per kota
├── notebooks/property_valuation_end_to_end.ipynb
├── src/
│   ├── features.py                       # fitur, binning domain, preprocessing, waktu (months_since_start)
│   ├── model.py                          # TrendAdjustedRegressor (indexing harga)
│   ├── evaluation.py                     # metrik AVM, bootstrap, Diebold-Mariano, Mincer-Zarnowitz, Breusch-Pagan, conformal, PSI
│   └── json_model.py                     # ekspor ke JSON + inferensi (numpy/pandas)
├── models/model.json                     # preprocessing + 600 pohon + tren + interval + metadata evaluasi
├── app/
│   ├── main.py                           # FastAPI
│   └── schemas.py                        # validasi input (Pydantic)
├── tests/
│   ├── test_api.py
│   └── test_evaluation.py                # metrik, uji statistik, tren, dan kesamaan JSON vs scikit-learn
├── sample_request.json / sample_batch_request.json
├── requirements.txt                      # API (tanpa scikit-learn)
├── requirements-dev.txt                  # + scikit-learn, scipy, notebook, testing
├── Dockerfile
├── docker-compose.yml
└── .dockerignore
```

---

## 1. Data & Notebook

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# (opsional) buat ulang dataset
python scripts/generate_data.py --n 10000 --seed 42

jupyter lab notebooks/property_valuation_end_to_end.ipynb
```

Jalankan semua cell (*Run All*, ±3 menit di laptop 10 core). Bagian terberat adalah walk-forward backtest, yang melatih ulang Gradient Boosting setiap kuartal. Hasilnya menimpa `models/model.json` dan `sample_*.json`.

Isi notebook (20 bagian; setiap cell kode didahului kotak **Alur data: Input → Proses → Output → Berikutnya**):

| Bagian | Isi |
|---|---|
| 0–3 | Peta alur, import, load & kamus data, EDA (distribusi harga, harga/m² per kota, indeks harga, efek binning) |
| 4 | **Uji statistik fitur**: Spearman, Kruskal-Wallis + ε², VIF (koreksi Holm) |
| 5–6 | Split waktu (dev/OOT) + split acak (train/test), pipeline binning + one-hot + log + scaling, telusuri 1 properti |
| 7 | 2 model + **penyesuaian tren**, dengan mini-backtest yang membuktikan bias ekstrapolasi |
| 8–10 | GridSearchCV, **repeated K-fold 5×5** + corrected t-test + Wilcoxon, learning curve, pemilihan model |
| 11 | **Interval prediksi conformal** dari residual out-of-fold |
| 12 | Evaluasi test: 10 metrik, **bootstrap CI**, **Diebold-Mariano & Wilcoxon**, cakupan interval, grafik akurasi |
| 13 | **Diagnostik residual**: Mincer-Zarnowitz, Breusch-Pagan, Jarque-Bera, **bias per segmen** (kota, tipe, rentang harga, umur, sertifikat) |
| 14 | Permutation importance & partial dependence |
| 15 | **Backtesting**: indeks harga, out-of-time (tren vs naif), walk-forward 12 kuartal, PSI/CSI |
| 16 | **Ringkasan evaluasi** (22 pengujian ✅/⚠️/❌) & kesimpulan |
| 17–19 | Refit semua data → `model.json`, validasi identik dengan scikit-learn, alur data di API |

## 2. Menjalankan API Lokal (tanpa Docker)

```bash
uvicorn app.main:app --reload --port 8002
```

Buka **http://localhost:8002/docs**. Untuk test:

```bash
pytest -q
```

## 3. Deploy ke Docker Lokal

### Prasyarat
- Docker Desktop terpasang dan **berjalan** (cek: `docker info`)
- `models/model.json` sudah ada (hasil notebook)

### Opsi A: Docker CLI

```bash
docker build -t property-valuation-api:latest .
docker run -d --name property-valuation-api -p 8002:8000 property-valuation-api:latest
docker ps                                   # tunggu STATUS = healthy
docker logs -f property-valuation-api
```

### Opsi B: Docker Compose

```bash
docker compose up -d --build
docker compose ps
docker compose logs -f api
```

> Port host **8002** dipakai agar bisa berjalan bersamaan dengan API credit default (8000) dan segmentasi nasabah (8001).

### Uji API

```bash
curl http://localhost:8002/health
curl http://localhost:8002/model-info       # tren harga, interval, kebijakan LTV, ringkasan evaluasi

curl -X POST http://localhost:8002/valuation \
  -H "Content-Type: application/json" \
  -d @sample_request.json

curl -X POST http://localhost:8002/valuation/batch \
  -H "Content-Type: application/json" \
  -d @sample_batch_request.json
```

Contoh request (luas dalam m², nominal dalam **juta Rp**):

```json
{
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
  "near_toll": true,
  "near_transit": false,
  "flood_prone": false,
  "valuation_date": "2026-01-15",
  "loan_amount": 1200
}
```

Response (rumah Tangerang Selatan, dinilai Jan 2026; LTV 59% → sesuai kebijakan):

```json
{
  "property_id": "AGN-0001",
  "valuation_date": "2026-01-15",
  "estimated_value": 2037.4,
  "lower_90": 1671.2,
  "upper_90": 2465.7,
  "price_per_m2_land": 16.978,
  "max_ltv_policy": 0.8,
  "max_loan": 1629.9,
  "conservative_max_loan": 1336.9,
  "loan_check": {
    "loan_amount": 1200.0,
    "ltv_on_estimate": 0.589,
    "ltv_on_lower_bound": 0.7181,
    "max_ltv_policy": 0.8,
    "within_policy": true,
    "within_policy_conservative": true
  },
  "warnings": []
}
```

| Field | Arti |
|---|---|
| `estimated_value` | Taksiran nilai (median prediksi), juta Rp |
| `lower_90`, `upper_90` | Interval prediksi 90%: harga sebenarnya diperkirakan ada di rentang ini 9 dari 10 kali |
| `max_loan` | LTV kebijakan × taksiran |
| `conservative_max_loan` | LTV kebijakan × **batas bawah** interval. Lebih aman untuk keputusan kredit |
| `loan_check` | Jika `loan_amount` diisi: LTV aktual terhadap taksiran & batas bawah, dan apakah sesuai kebijakan |
| `warnings` | Mis. tanggal valuasi > 6 bulan setelah data terakhir, ukuran di luar rentang data latih, data kosong |

> Kebijakan LTV di proyek ini (rumah/townhouse 80%, ruko 70%) hanyalah **ilustrasi**. Sesuaikan dengan kebijakan kredit internal dan ketentuan Bank Indonesia yang berlaku.

### Stop & Bersihkan

```bash
docker stop property-valuation-api && docker rm property-valuation-api   # Opsi A
docker compose down                                                       # Opsi B
docker rmi property-valuation-api:latest                                  # opsional
```

### Update Model (Refit)

Harga properti terus bergerak, jadi **refit bulanan** dianjurkan:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/property_valuation_end_to_end.ipynb
docker compose up -d --build
curl http://localhost:8002/health        # model_version berubah
```

---

## Referensi API

| Method | Endpoint | Keterangan |
|---|---|---|
| GET | `/health` | Status & versi model |
| GET | `/model-info` | Fitur, nilai kategori, tren harga, interval, kebijakan LTV, ringkasan evaluasi |
| POST | `/valuation` | Taksiran 1 properti |
| POST | `/valuation/batch` | `{"instances": [...]}` (maks. 2.000) + total nilai portofolio |
| GET | `/docs` | Swagger UI |

**Validasi input** (HTTP 422 jika gagal):
- Kota harus salah satu dari 10 kota yang dicakup model. Properti di kota lain **ditolak**, karena model belum pernah melihat harga tanah kota tersebut.
- Rentang angka harus wajar.
- `building_area` ≤ `land_area` × `floors` × 1,1.
- `valuation_date` ≥ 2022-01-01.
- Field yang tidak dikenal ditolak.
- `building_age` dan `road_width_m` boleh `null`.

| Env var | Default | Keterangan |
|---|---|---|
| `MODEL_PATH` | `models/model.json` | Lokasi model |
| `MAX_EXTRAPOLATION_MONTHS` | `6` | Batas peringatan proyeksi tren ke depan |
| `LOG_LEVEL` | `INFO` | Level logging |

## Monitoring yang Disarankan

| Indikator | Patokan | Tindakan |
|---|---|---|
| MdAPE & PPE10 pada transaksi/appraisal baru (bulanan) | MdAPE ≤ 10%, PPE10 ≥ 50% | Turun → investigasi segmen, lalu refit |
| Bias rata-rata & per kota | \|bias\| < 2%, tidak signifikan | Bias sistematis → refit / tambah fitur lokasi |
| Cakupan interval 90% | 87–93% | Di luar rentang → hitung ulang kuantil conformal |
| PSI taksiran & CSI fitur | < 0,1 | > 0,25 → tinjau ulang model |
| Jarak tanggal valuasi ke data terakhir | ≤ 6 bulan | Refit bulanan |

## Troubleshooting

| Masalah | Solusi |
|---|---|
| `Cannot connect to the Docker daemon` | Buka Docker Desktop dan tunggu sampai *running* |
| `port is already allocated` | Ganti port host, mis. `-p 8003:8000` |
| `ModuleNotFoundError: No module named 'src'` | Jalankan uvicorn dari root proyek; di Docker pastikan `src/json_model.py` ikut di-COPY |
| Response 422 untuk kota tertentu | Kota di luar cakupan model; tambahkan datanya lalu refit |
| Taksiran terlihat “membeku” untuk tanggal masa depan | Tidak akan terjadi selama model dibuat lewat `TrendAdjustedRegressor`; cek `trend.annual_growth` di `/model-info` |
| Notebook lama dijalankan | Bagian terberat adalah walk-forward (Bagian 15). Kurangi kuartal di `wf_quarters` atau `n_estimators` untuk eksperimen |

> ⚠️ Data bersifat sintetis untuk pembelajaran. AVM di produksi perlu data transaksi/appraisal riil, validasi independen, dan tetap didampingi appraisal fisik untuk agunan bernilai besar atau properti atipikal.
