
# 🌍 TEAM: ASTROBLITZ NASA MeteorSense AI 2025 — Near-Earth Object (NEO) Hazard Prediction using XGBoost

> 🚀 *An AI-powered planetary defense project built on real NASA data — empowering humanity to detect asteroid threats before they strike.*

---

## 📋 Table of Contents
- [Mission Statement](#-mission-statement)
- [Project Overview](#-project-overview)
- [Dataset Summary](#-dataset-summary)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [Model Results](#-model-results)
- [Feature Importance](#-feature-importance)
- [Interactive Prediction](#-interactive-prediction)
- [Impact & Significance](#-impact--significance)
- [Technical Stack](#-technical-stack)
- [Run Locally](#️-run-locally)
- [Future Enhancements](#-future-enhancements)
- [Summary](#-summary)
- [References](#-references)
- [Vision](#-vision)
- [Team](#-team)

---

## 🚀 Mission Statement

Every year, thousands of **near-Earth asteroids (NEOs)** whiz past our planet.  
While most are harmless, some have the potential to **alter life on Earth**.

**NASA MeteorSense AI 2025** is our step toward **proactive planetary defense** — turning NASA's open asteroid data into **actionable intelligence** through machine learning.  

Our goal:  
> **Predict which asteroids pose a potential hazard to Earth — early, accurately, and intelligently.**

---

## 🔭 Project Overview

| Stage | Description |
|-------|-------------|
| 🛰️ **Data Collection** | Fetched 2025–2026 asteroid data using NASA's NEO API |
| 🧹 **Data Cleaning** | Removed duplicates, formatted features, and handled nulls |
| 📊 **EDA** | Explored asteroid size, velocity, and temporal patterns |
| 🤖 **Modeling** | Trained an XGBoost classifier for hazard prediction |
| 🧠 **Evaluation** | Analyzed precision, recall, and AUC metrics |
| 🔍 **Interpretation** | Visualized key features influencing hazard probability |

---

## 📊 Dataset Summary

**Total Objects:** 10,151 asteroids  
**Source:** [NASA NEO API](https://api.nasa.gov/)

![Dataset Overview]()

| Feature | Description |
|---------|-------------|
| `absolute_magnitude_h` | Brightness (lower = larger asteroid) |
| `estimated_diameter_min_km` | Minimum diameter in kilometers |
| `estimated_diameter_max_km` | Maximum diameter in kilometers |
| `relative_velocity_km_s` | Velocity at closest approach (km/s) |
| `miss_distance_au` | Closest approach distance from Earth (AU) |
| `orbiting_body` | Celestial body the asteroid orbits |
| `is_potentially_hazardous` | Hazard classification (True/False) |

---

## 🧠 Machine Learning Pipeline

### 🔹 Feature Engineering
- Created derived features like `avg_diameter`, `approach_month`, and `day_of_year`
- Focused on **size, speed, distance, and brightness**
- Encoded categorical variables like `orbiting_body`

### 🔹 Data Preprocessing
- Handled imbalanced classes using stratified sampling
- Standardized numerical features (`StandardScaler`)
- Train-test split (80-20) with stratification

### 🔹 Model: **XGBoost Classifier**
- Handles non-linear interactions efficiently
- Excellent performance on tabular NASA data
- Tuned with Bayesian optimization for:
  - `max_depth`
  - `learning_rate`
  - `n_estimators`

### 🔹 Features Used:
```python
[
    "absolute_magnitude_h",
    "estimated_diameter_min_km", 
    "estimated_diameter_max_km",
    "relative_velocity_km_s",
    "miss_distance_au",
    "orbiting_body_encoded"
]
```

### 🔹 Metrics Used
- Accuracy
- Precision & Recall
- ROC-AUC
- Cross-validation (5-fold)
- Confusion Matrix Analysis

---

## 📈 Model Results

###  Performance Metrics
![Model Accuracy](<img width="1000" height="600" alt="Figure_1" src="https://github.com/user-attachments/assets/98659846-947c-4ec7-8cb6-f537e585df34" />
)

| Metric | Score |
|--------|-------|
| ✅ **Accuracy** | 96.11% |
| 🚀 **ROC-AUC** | 0.88 |
| 🎯 **Precision** | 0.86 |
| 🛰️ **Recall** | 0.83 |

**Key Insights:**
- **96.11% overall accuracy** in hazard classification
- Excellent performance in identifying **safe asteroids** (1890 correct)
- Strong capability to **minimize false negatives** in dangerous asteroid detection

---

## 🔍 Feature Importance

| Rank | Feature | Impact |
|------|---------|--------|
| 🥇 | `estimated_diameter_avg_km` | Strongest predictor of hazard probability |
| 🥈 | `relative_velocity_km_s` | High velocity linked to increased risk |
| 🥉 | `absolute_magnitude_h` | Lower magnitude (brighter/larger) = higher threat |

---

## 🎮 Interactive Prediction

### 🔬 Try a Quick Prediction
![Interactive Prediction](C:\Users\EXTECH\Downloads\image (4).webp)

**Input Parameters:**
```
absolute_magnitude_h: 24.80
estimated_diameter_min_km: 0.03  
estimated_diameter_max_km: 0.07
relative_velocity_km_s: 11.94
miss_distance_au: 0.21
orbiting_body_encoded: 0.00
```

**Prediction Result:**
```diff
+ ✅ Safe asteroid (Confidence 100.0%)
```
**DEPLOYMENT LINK**....https://huggingface.co/spaces/Abdullah-31/AstroBlitz

**Interpretation:**
- Small diameter (0.03-0.07 km) ✅
- Moderate velocity (11.94 km/s) ✅  
- Safe miss distance (0.21 AU) ✅
- **Result: NON-HAZARDOUS** with high confidence

---

## 💼 Impact & Significance

### 🌌 Scientific Impact
- Converts **NASA's raw asteroid data** into interpretable machine learning insights
- Enables **automated hazard detection** for near-Earth objects
- Offers a **retrainable pipeline** for future NEO data updates

### 🪐 Real-World Relevance
- Helps **space agencies prioritize risk monitoring**
- Supports **public awareness systems** & real-time dashboards
- Forms a **foundation for future planetary defense simulations**

---

## 🛠️ Technical Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.8+ |
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

# Or run the prediction script
python predict_hazard.py
```

---

## 🚀 Future Enhancements

| Enhancement | Description |
|-------------|-------------|
| 🧠 **Deep Learning** | Incorporate LSTM/Transformers for orbit-based forecasting |
| 🌍 **Web Dashboard** | Interactive real-time monitoring (Streamlit or Dash) |
| 📅 **Time-Series Forecasting** | Predict future close-approach events |
| 🪐 **Unsupervised Analysis** | Cluster asteroids by orbital and size similarity |
| 🔔 **Alert System** | Automated notifications for new high-risk objects |
| 🌐 **Real-time API** | REST API for instant hazard predictions |

---

## 🏆 Final Challenge Outcome

### ✅ **SUCCESSFULLY COMPLETED**

**Achievement:** Developed a high-performance asteroid hazard prediction system with **96.11% accuracy**

**Key Accomplishments:**
- ✅ Processed and analyzed **10,151 NASA asteroid records**
- ✅ Engineered relevant features from raw orbital data
- ✅ Built and optimized **XGBoost classification model**
- ✅ Achieved **excellent predictive performance** (96.11% accuracy)
- ✅ Implemented **interactive prediction capability**
- ✅ Delivered **interpretable results** with feature importance analysis

**Model Performance Summary:**
```
🎯 ACCURACY: 96.11%
🛡️ RELIABILITY: High confidence in safe/dangerous classification  
🔍 INTERPRETABILITY: Clear feature importance insights
🚀 SCALABILITY: Ready for real-time NASA data integration
```

---

## 📋 Summary

| Aspect | Description |
|--------|-------------|
| **Goal** | Predict potentially hazardous asteroids using NASA's open data |
| **Approach** | End-to-end AI pipeline powered by XGBoost |
| **Result** | 96.11% accuracy with high interpretability |
| **Impact** | Enables data-driven planetary defense initiatives |
| **Status** | ✅ **Challenge Successfully Completed** |

---

## 🔗 References

### 🌌 NASA Data & APIs
- [NASA Open Data Portal](https://data.nasa.gov/)
- [NASA Near-Earth Object Web Service (NeoWs)](https://api.nasa.gov/)
- [CNEOS: Center for Near-Earth Object Studies](https://cneos.jpl.nasa.gov/)
- [NASA JPL Small-Body Database Browser](https://ssd.jpl.nasa.gov/tools/sbdb_query.html)
- [NASA Asteroid Watch](https://www.jpl.nasa.gov/asteroid-watch)
- [NASA Open APIs Documentation (GitHub)](https://github.com/nasa/api-docs)

---

## 🌠 Vision

> "The future of planetary defense lies not just in rockets — but in data."
> — NASA MeteorSense AI Team (2025)

Together, we move toward a world where AI safeguards humanity from the silent wanderers of the cosmos. 🌌

---

## 👥 Team

### 🚀 **Team: ASTROBLITZ NASA MeteorSense AI — 2025**

| Role | Name | GitHub |
|------|------|--------|
| 🛰️ **Team Lead** | **Uneeba** | [@UNEEBASHAIKH](https://github.com/UNEEBASHAIKH) |
| 👨‍💻 **Data Scientist** | **Muhammad Umer** | [@MUmer007](https://github.com/MUmer007) |
| 🧠 **Data Scientist** | **Abdul Basit** | [@comp3ngrBasit](https://github.com/comp3ngrBasit) |
| 🧮 **Data Scientist & ML Researcher** | **Abdullah Asif** | [@Abdullah-056](https://github.com/Abdullah-056) |
| 🌍 **Data Analyst & ML Researcher** | **Ahmed Hassan** | [@TechWithAhmedHassan](https://github.com/TechWithAhmedHassan) |
| 👩‍💻 **Data Scientist** | **Hira Arif** | [@HiraArif666](https://github.com/HiraArif666) |

---

<div align="center">

## 🏅 **CHALLENGE COMPLETED SUCCESSFULLY**

**⭐ If you find this project useful, don't forget to give it a star!**

*Contributing to planetary defense through artificial intelligence*

</div>
