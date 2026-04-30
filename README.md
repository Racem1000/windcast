# 🌬️ WindCast — Wind Turbine Power Forecasting Dashboard

[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-ff4b4b?logo=streamlit)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Course](https://img.shields.io/badge/Course-CS434-green)](.)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

**WindCast** is an end-to-end machine learning dashboard for wind turbine power forecasting, developed as the final project for **CS434 — Data Analytics**. The system automatically preprocesses SCADA/weather data, engineers physics-informed features, trains and compares multiple regression models, and provides an interactive interface for real-time power forecasting.

> 🏆 **Best model: CatBoost (Tuned)** — MAE: 0.1152 · RMSE: 0.1499 · R²: 0.7488

---

## 👥 Team

| Name | Name |
|---|---|
| Linda Mkaouar | Dorra Ben Salah |
| Dorra Houas | Racem Kamel |

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | [Kaggle](https://www.kaggle.com) |
| Time range | 2017 – 2021 |
| Samples | 43,800 hourly records |
| Features | 9 (meteorological + power) |
| Target | Normalized power output [0, 1] |

### Columns

| Column | Description |
|---|---|
| `Time` | Timestamp (hourly) |
| `Power` | Normalized power output (0–1, where 1 = rated power) |
| `windspeed_10m` | Wind speed at 10 m (m/s) |
| `windspeed_100m` | Wind speed at 100 m hub height (m/s) |
| `windgusts_10m` | Wind gust speed at 10 m (m/s) |
| `winddirection_100m` | Wind direction at 100 m (degrees, 0–360) |
| `temperature_2m` | Air temperature at 2 m (°F) |
| `relativehumidity_2m` | Relative humidity at 2 m (%) |
| `dewpoint_2m` | Dew point temperature at 2 m (°F) |

---

## ✨ Features

- **Automated ML pipeline** — preprocessing, feature engineering, and multi-model training triggered by a single CSV upload
- **7 trained models** including tuned boosting ensembles (see Results below)
- **6 interactive dashboard pages:**
  - 🏠 **Overview** — KPI cards, monthly energy production, seasonal distribution
  - 🔎 **Data Quality** — missing values report, outlier detection, preprocessing log
  - 📈 **Energy Production** — interactive time-series explorer and hourly heatmap
  - 🔍 **Wind & Climate Analysis** — correlation matrix, feature distributions, wind rose
  - ⚡ **Turbine Performance** — empirical vs. IEC theoretical power curve, curtailment detection
  - 🤖 **Forecast Accuracy** — predicted vs. actual, residuals, feature importances
  - 🎯 **Power Forecasting** — real-time slider-based forecast with Best Model or Ensemble Average
- **Dark / Light theme** toggle
- **Ensemble prediction** — average across all available trained models
- **Wind speed scenario comparison** — side-by-side predictions at Low / Medium / High / Custom wind

---

## 🔬 Key EDA Findings

- **Wind speed** at 100 m is the strongest predictor of power (r = 0.78)
- **Diurnal pattern:** power peaks at ~5 AM and reaches a minimum around 6 PM
- **Seasonal pattern:** Spring and Winter are most productive; Summer is the low point
- The empirical power curve aligns closely with the theoretical IEC cubic curve, confirming data quality

---

## 🤖 Model Results

All models evaluated on a chronological 80/20 train/test split (no data leakage):

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **CatBoost (Tuned)** 🏆 | **0.1152** | **0.1499** | **0.7488** |
| LightGBM (Tuned) | 0.1155 | 0.1503 | 0.7474 |
| CatBoost | 0.1148 | 0.1503 | 0.7472 |
| LightGBM | 0.1157 | 0.1517 | 0.7424 |
| XGBoost (Tuned) | 0.1179 | 0.1520 | 0.7416 |
| XGBoost | 0.1166 | 0.1523 | 0.7405 |
| Random Forest | 0.1184 | 0.1534 | 0.7366 |
| Linear Regression | 0.1347 | 0.1759 | 0.6539 |
| Ridge | 0.1347 | 0.1759 | 0.6539 |
| Lasso | 0.1351 | 0.1762 | 0.6525 |

**Top features (CatBoost Tuned):** `windspeed100_roll3` dominates, followed by `Power_lag12`, `windspeed100_lag6`, and `wind_dir_cos`.

---

## 🏗️ ML Pipeline

### Preprocessing

| Step | Method |
|---|---|
| Missing values | Forward-fill → Backward-fill → Column median |
| Duplicates | Drop duplicate timestamps (keep first) |
| Outliers | IQR × 3 capping (features only, not the target) |
| Train/test split | 80/20 chronological (time order preserved) |
| Scaling | `StandardScaler` for linear models only |
| Feature selection | Drop one from any pair with correlation > 0.85; keep the one more correlated with target |

### Feature Engineering

| Feature | Description |
|---|---|
| `windspeed_100m_cubed` | Captures the physical cubic P ∝ v³ relationship |
| `turbulence_intensity` | (gusts − wind) / wind, categorised Low / Moderate / High |
| `air_density` | Derived from temperature via ideal gas law |
| `wind_dir_sin`, `wind_dir_cos` | Cyclic encoding of wind direction |
| Lag features | Power and wind speed at 1/2/3/6/12/24 h lags |
| Rolling features | 3-hour rolling mean of power and wind speed |
| `season`, `hour`, `day`, `month` | Temporal context features |

### Hyperparameter Tuning

`RandomizedSearchCV` (10 iterations) with `TimeSeriesSplit` (5 folds) on: `n_estimators`, `max_depth`, `learning_rate`, `max_features`, `subsample`, `colsample_bytree`.

---

## 🚀 Quickstart

### Run Locally

```bash
git clone https://github.com/<your-username>/windcast.git
cd windcast

python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Open **http://localhost:8501**, then upload your CSV from the sidebar to train all models.

### Deploy on Streamlit Community Cloud (Free)

1. Push this repository to GitHub (public or private).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app** → select your repo → set main file to `app.py` → **Deploy**.

Streamlit Cloud installs `requirements.txt` automatically. No server setup needed. Training runs on upload per session.

---

## 🗂️ Project Structure

```
windcast/
├── app.py                  # Main Streamlit application (~1 850 lines)
├── requirements.txt        # Python dependencies
├── .gitignore              # Excludes models/, *.pkl, venv/, *.csv
├── .streamlit/
│   └── config.toml         # Dark theme + 200 MB upload limit
├── LICENSE
└── README.md
```

The `models/` directory is generated at runtime and is **not** tracked by Git.

---

## 📦 Dependencies

`streamlit` · `pandas` · `numpy` · `plotly` · `scikit-learn` · `xgboost` · `lightgbm` · `catboost` · `joblib` · `scipy`

XGBoost, LightGBM, and CatBoost are optional — the pipeline skips any that are not installed and logs a warning.

---

## 🔭 Future Work

- Integrate a real-time weather API (e.g., Open-Meteo) for live forecasts without manual input
- Model turbine degradation and planned downtime events
- Experiment with deep learning (LSTM, Transformer) for multi-step ahead forecasting

---

## 📄 License

Released under the [MIT License](LICENSE).
