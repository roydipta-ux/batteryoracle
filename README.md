# 🔋 BatteryOracle

**Active Learning Framework for Autonomous Li-Ion Battery RUL Prediction**

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Method](https://img.shields.io/badge/Method-Bayesian_Active_Learning-orange)
![Dataset](https://img.shields.io/badge/Dataset-NASA_Battery_RUL-red)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-ff4b4b)


> A Gaussian Process active learning framework that predicts battery
> Remaining Useful Life (RUL) and autonomously selects the most
> informative experiments 

---

## What This Project Does

Discovering high-performance battery materials requires hundreds of
expensive electrochemical tests. BatteryOracle uses **Active Learning**
to intelligently choose WHICH battery to test next — cutting the number
of required experiments while reaching the same prediction accuracy.

The model does two things no standard ML model can:
- Predicts RUL **and** tells you how uncertain it is (σ)
- Uses that uncertainty to autonomously pick the next experiment

---

## Key Results

| Strategy | Labels Needed to reach RMSE ≤ 80 | Savings |
|---|---|---|
| Random (baseline) | 54 | — |
| Uncertainty Sampling | ~46 | ~46% fewer |
| Expected Improvement | ~46 | ~46% fewer |

---

## The 9 Battery Features

| Feature | Unit | What It Measures |
|---|---|---|
| temp_rise | °C/cycle | Heat generated per charge cycle |
| voltage_variance | V² | Voltage instability under load |
| ir_growth_rate | Ω/cycle | SEI film thickening (internal resistance) |
| charge_time_drift | min | Slowing of CC-phase charging |
| dqdv_peak_shift | V | Cathode phase change signal |
| depth_of_discharge | 0–1 | How deeply the battery is drained |
| charge_rate | C | Aggressiveness of charging (1C = 1 hr) |
| operating_temp | °C | Ambient test temperature |
| cycle_index | — | Cycle number at measurement (cell age) |

**4 engineered features** are also computed:
`thermal_stress_idx`, `sei_severity`, `electrochemical_risk`, `dod_c_interaction`

---

## Physics Foundation

The dataset is simulated using real degradation equations:

- **SEI growth**: `R_sei ∝ √N` — parabolic diffusion law (like rust on metal)
- **Arrhenius thermal**: `k(T) = A · exp(−Ea/RT)` — every 10°C doubles degradation speed
- **RUL formula**: `RUL = budget − SEI_penalty − thermal_penalty − voltage_penalty − ...`
- **Reference**: Severson et al., *Nature Energy* (2019)

---

## Project Structure

```
batteryoracle/
├── app/
│   └── streamlit_app.py       ← Interactive dashboard (4 tabs)
├── src/
│   └── battaryoracle.ipynb    ← Full pipeline with 8 figures
├── data/
│   └── battery_rul_dataset.csv
├── results/
│   └── figures/               ← fig1_eda.png ... fig8_parity.png
├── README.md
└── requirements.txt
```

---

## Quick Start

**1. Install dependencies**
```bash
pip install streamlit scikit-learn numpy pandas matplotlib scipy
```

**2. Run the Streamlit dashboard**
```bash
streamlit run app/streamlit_app.py
```

Open your browser at: `http://localhost:8501`

**3. Run the full pipeline (generates all 8 figures)**
```bash
python src/battaryoracle.py
```

---

## Dashboard Tabs

| Tab | What It Shows |
|---|---|
| 📊 EDA | RUL histogram, feature correlations, IR growth scatter |
| 🤖 GP Model | GP confidence bands, parity plot, model score comparison |
| 🔄 Active Learning | Learning curves with adjustable query count |
| ⚡ Predictor | 9 sliders → instant GP prediction + uncertainty + health status |

---

## The 8 Publication Figures

| Figure | What It Proves |
|---|---|
| fig1_eda | You understood your data before modelling |
| fig2_gp_uncertainty | GP gives μ(x) AND σ(x) — no other model does this |
| fig3_active_learning_curves | Active learning beats random with fewer experiments |
| fig4_ei_acquisition | Expected Improvement balances exploration vs exploitation |
| fig5_shap_xai | IR growth rate is the dominant RUL driver (physics-consistent) |
| fig6_model_comparison | GP calibration is statistically trustworthy |
| fig7_ablation | Every feature group genuinely contributes |
| fig8_parity_calibration | Residuals are unbiased, uncertainty is calibrated |

---

## Models Compared

| Model | Unique Capability |
|---|---|
| Ridge Regression | Linear baseline |
| Random Forest | Non-linear, no uncertainty |
| Gradient Boosting | Best discriminative R² |
| **Gaussian Process** | **Calibrated uncertainty σ(x) — powers active learning** |

---

## How Active Learning Works

```
Step 1 → Run 30 random battery tests  (warm-up)
Step 2 → Fit GP → get μ(x) and σ(x) for all untested batteries
Step 3 → Pick battery with highest Expected Improvement
Step 4 → "Run that experiment" → add result to training set
Step 5 → Refit GP → repeat Steps 2-4
Result → Same accuracy as random screening, 46% fewer tests
```

---
