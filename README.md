# 🌍 TEAM: ASTROBLITZ NASA MeteorSense AI 2025 — Near-Earth Object (NEO) Hazard Prediction using XGBoost

> 🚀 *An AI-powered planetary defense project built on real NASA data — empowering humanity to detect asteroid threats before they strike.*

---

## 📘 Table of Contents
1. [Mission Statement](#-mission-statement)
2. [Project Overview](#-project-overview)
3. [Dataset Summary](#-dataset-summary)
4. [Machine Learning Pipeline](#-machine-learning-pipeline)
5. [Model Results](#-model-results)
6. [Feature Importance](#-feature-importance)
7. [Impact & Significance](#-impact--significance)
8. [Technical Stack](#-technical-stack)
9. [Run Locally](#️-run-locally)
10. [Future Enhancements](#-future-enhancements)
11. [Summary](#-summary)
12. [References](#-references)
13. [Vision](#-vision)

---

## 🚀 Mission Statement

Every year, thousands of **near-Earth asteroids (NEOs)** whiz past our planet.  
While most are harmless, some have the potential to **alter life on Earth**.

**NASA MeteorSense AI 2025** is our step toward **proactive planetary defense** — turning NASA’s open asteroid data into **actionable intelligence** through machine learning.  

Our goal:  
> **Predict which asteroids pose a potential hazard to Earth — early, accurately, and intelligently.**

---

## 🔭 Project Overview

| Stage | Description |
|-------|--------------|
| 🛰️ **Data Collection** | Fetched 2025–2026 asteroid data using NASA’s NEO API |
| 🧹 **Data Cleaning** | Removed duplicates, formatted features, and handled nulls |
| 📊 **EDA** | Explored asteroid size, velocity, and temporal patterns |
| 🤖 **Modeling** | Trained an XGBoost classifier for hazard prediction |
| 🧠 **Evaluation** | Analyzed precision, recall, and AUC metrics |
| 🔍 **Interpretation** | Visualized key features influencing hazard probability |

---

## 📊 Dataset Summary

**Total Objects:** ~10,151  
**Source:** [NASA NEO API](https://api.nasa.gov/)

| Feature | Description |
|----------|--------------|
| `absolute_magnitude_h` | Brightness (lower = larger asteroid) |
| `estimated_diameter_avg_km` | Mean diameter in kilometers |
| `relative_velocity_km_s` | Velocity at closest approach (km/s) |
| `miss_distance_au` | Closest approach distance from Earth (AU) |
| `is_potentially_hazardous` | Hazard classification (True/False) |

---

## 🧠 Machine Learning Pipeline

### 🔹 Feature Engineering
- Created derived features like `avg_diameter`, `approach_month`, and `day_of_year`  
- Focused on **size, speed, distance, and brightness**

### 🔹 Data Preprocessing
- Handled imbalanced classes  
- Standardized numerical features (`StandardScaler`)  
- Train-test split with stratified sampling  

### 🔹 Model: **XGBoost Classifier**
- Handles non-linear interactions efficiently  
- Excellent performance on tabular NASA data  
- Tuned with Bayesian optimization for:
  - `max_depth`
  - `learning_rate`
  - `n_estimators`

### 🔹 Metrics Used
- Accuracy  
- Precision & Recall  
- ROC-AUC  
- Cross-validation (5-fold)

---

## 📈 Model Results

| Metric | Score |
|--------|--------|
| ✅ **Accuracy** | 0.90 |
| 🚀 **ROC-AUC** | 0.88 |
| 🎯 **Precision** | 0.86 |
| 🛰️ **Recall** | 0.83 |

### 🟢 Precision–Recall Curve
> Demonstrates the model’s strength in correctly identifying hazardous NEOs while maintaining reliability.

![Precision–Recall Curve](<img width="695" height="470" alt="download" src="https://github.com/user-attachments/assets/f8a44f69-86cc-49bd-bebd-120b19c128c8" />
)

---

## 🔍 Feature Importance

| Rank | Feature | Impact |
|------|----------|--------|
| 🥇 | `estimated_diameter_avg_km` | Strongest predictor of hazard probability |
| 🥈 | `relative_velocity_km_s` | High velocity linked to increased risk |
| 🥉 | `absolute_magnitude_h` | Lower magnitude (brighter/larger) = higher threat |

### 📊 Visualization
XGBoost’s internal gain metric for interpretability:

![Feature Importance](<img width="1000" height="600" alt="Figure_2" src="https://github.com/user-attachments/assets/2dbfdb66-0cd3-4d47-9a44-1218d730acc8" />
)

---

## 💼 Impact & Significance

### 🌌 Scientific Impact
- Converts **NASA’s raw asteroid data** into interpretable machine learning insights  
- Enables **automated hazard detection** for near-Earth objects  
- Offers a **retrainable pipeline** for future NEO data updates

### 🪐 Real-World Relevance
- Helps **space agencies prioritize risk monitoring**  
- Supports **public awareness systems** & real-time dashboards  
- Forms a **foundation for future planetary defense simulations**

---

## 🧩 Technical Stack

| Category | Tools |
|-----------|-------|
| **Language** | Python |
| **Libraries** | pandas, numpy, requests, seaborn, matplotlib, scikit-learn, xgboost |
| **Model** | XGBoost Classifier |
| **Data Source** | NASA NEO API |
| **Artifacts** | `xgboost_neo_model.pkl`, `scaler.pkl`, `model_metrics.json` |

---

## ⚙️ Run Locally

```bash
# Clone repository
git clone https://github.com/UNEEBASHAIKH/NASA-MeteorSense-AI-2025.git
cd NASA-MeteorSense-AI-2025

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook "NASA Near-Earth Object (NEO).ipynb"

| Future Enhancement                    | Description                                               |
| ------------------------------ | --------------------------------------------------------- |
| 🧠 **Deep Learning**           | Incorporate LSTM/Transformers for orbit-based forecasting |
| 🌍 **Web Dashboard**           | Interactive real-time monitoring (Streamlit or Dash)      |
| 📅 **Time-Series Forecasting** | Predict future close-approach events                      |
| 🪐 **Unsupervised Analysis**   | Cluster asteroids by orbital and size similarity          |
| 🔔 **Alert System**            | Automated notifications for new high-risk objects         |
**🧾 Summary**
| Aspect       | Description                                                    |
| ------------ | -------------------------------------------------------------- |
| **Goal**     | Predict potentially hazardous asteroids using NASA’s open data |
| **Approach** | End-to-end AI pipeline powered by XGBoost                      |
| **Result**   | 0.90 accuracy with high interpretability                       |
| **Impact**   | Enables data-driven planetary defense initiatives              |
# 🔗 Resources

### 🌌 NASA Data & APIs
- [NASA Open Data Portal](https://data.nasa.gov/)
- [NASA Near-Earth Object Web Service (NeoWs)](https://api.nasa.gov/)
- [CNEOS: Center for Near-Earth Object Studies](https://cneos.jpl.nasa.gov/)
- [NASA JPL Small-Body Database Browser](https://ssd.jpl.nasa.gov/tools/sbdb_query.html)
- [NASA Asteroid Watch](https://www.jpl.nasa.gov/asteroid-watch)
- [NASA Open APIs Documentation (GitHub)](https://github.com/nasa/api-docs)
### 🌠 Vision

“The future of planetary defense lies not just in rockets — but in data.”
— NASA MeteorSense AI Team (2025)

Together, we move toward a world where AI safeguards humanity from the silent wanderers of the cosmos. 🌌

## 🧑‍💻 Developed By

### 🚀 **Team:  ASTROBLITZ> NASA MeteorSense AI — 2025**

| 🪐 **Role**                           | 👩‍🚀 **Name**    | 🌐 **GitHub**                                                  |
| ------------------------------------- | ----------------- | -------------------------------------------------------------- |
| 🛰️ **Team Lead**                     | **Uneeba**        | [@UNEEBASHAIKH](https://github.com/UNEEBASHAIKH)               |
| 👨‍💻 **Data Scientist**              | **Muhammad Umer** | [@MUmer007](https://github.com/MUmer007)                       |
| 🧠 **Data Scientist**                 | **Abdul Basit**   | [@comp3ngrBasit](https://github.com/comp3ngrBasit)             |
| 🧮 **Data Scientist & ML Researcher** | **Abdullah Asif** | [@Abdullah-056](https://github.com/Abdullah-056)               |
| 🌍 **Data Analyst & ML Researcher**   | **Ahmed Hassan**  | [@TechWithAhmedHassan](https://github.com/TechWithAhmedHassan) |
| 👩‍💻 **Data Scientist**              | **Hira Arif**     | [@HiraArif666](https://github.com/HiraArif666)                 |
