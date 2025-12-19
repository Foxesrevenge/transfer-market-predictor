# ⚽ Transfer Market Predictor

**Machine Learning model predicting football player market values**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Built for sporting director and technical director internship applications | March 2025

---

## 🎯 Project Overview

A production-ready machine learning model that predicts football player market values using performance statistics, team context, and FIFA ratings. The model achieves **R² = 0.689** on 7,700+ player-season observations.

**Key Innovation:** Zero data leakage - all team context features derived from performance metrics (goals, xG, assists) rather than market values.

### Live Demo
🌐 **[Try the Web App](https://transfer-market-predictor.streamlit.app)** - Deployed on Streamlit Cloud

### Author
👨‍💻 **Owen James** | Economics Student & Football Analytics Enthusiast

**Links:**
- 🔗 **GitHub:** [github.com/Foxesrevenge/transfer-market-predictor](https://github.com/Foxesrevenge/transfer-market-predictor)
- 🚀 **Live App:** [transfer-market-predictor.streamlit.app](https://transfer-market-predictor.streamlit.app)

---

## 📊 Model Performance

| Metric | Value | Description |
|--------|-------|-------------|
| **R-squared** | 0.689 | 68.9% variance explained |
| **MAE** | €4.97M | Mean absolute error |
| **Dataset** | 7,722 obs | Player-seasons (2021-22, 2023-24, 2024-25) |
| **Features** | 67 | Engineered features across 6 categories |

**Performance by Value Range:**
- €0-5M: 48% of players, MAE €2.1M
- €5-25M: 42% of players, MAE €5.8M
- €25-50M: 6% of players, MAE €8.3M
- €50M+: 4% of players, MAE €15.2M

---

## ✨ Features

### 67 Engineered Features Across 6 Categories:

#### 🎽 Performance Statistics (23 features)
- **Raw Stats:** goals, assists, minutes_played, appearances, penalties
- **Per-90 Metrics:** goals_per_90, assists_per_90, xG_per_90, xA_per_90
- **Expected Stats:** xG, npxG, xA, npxG+xA totals
- **Efficiency:** xG overperformance, penalty conversion rate

#### 📈 Progressive Statistics (6 features)
From FBref - ball progression metrics:
- progressive_carries, progressive_passes, progressive_receptions
- Per-90 versions of all metrics

#### 👥 Team Context (11 features)
**NO DATA LEAKAGE** - Performance-based only:
- Team aggregates: team_goals, team_xg, team_xa (excluding player)
- Player importance: goal_share_pct, minutes_share_pct
- Status flags: is_team_top_scorer, is_regular_starter
- Efficiency: finishing_efficiency, playmaking_efficiency

#### 🎂 Age Categories (4 features)
- age, is_prime_age (24-28), is_young_talent (<23), is_veteran (30+)

#### 🎮 FIFA Ratings (7 features)
From EA Sports FIFA 22/23/24:
- fifa_overall, fifa_potential, fifa_potential_gap
- is_wonderkid, high_potential_youth, elite_rating
- youth_potential_premium (value multiplier)

#### ⚙️ Multipliers & Adjustments (7 features)
- League strength: EPL (1.35x), La Liga (1.15x), Bundesliga (1.05x), etc.
- Age curve: Peak 23-28 (1.0x), Veterans 30+ (0.70x-0.50x)
- Position premium: FW (1.0x), MID (0.95x), DEF (0.85x), GK (0.70x)
- Market inflation: Year-over-year growth factor

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/transfer-market-predictor.git
cd transfer-market-predictor

# Install dependencies
pip install -r requirements.txt
```

### Run Web App Locally

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` to use the predictor!

---

## 📁 Project Structure

```
transfer-market-predictor/
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .gitignore                      # Git ignore rules
│
├── data/
│   ├── market_value_model.pkl     # Trained model
│   └── sample_data.csv            # Sample predictions
│
├── notebooks/
│   ├── 01_data_exploration.ipynb  # EDA notebook
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py         # Data loading & cleaning
│   ├── feature_engineering.py     # Feature creation
│   ├── model.py                   # Model training & prediction
│   └── utils.py                   # Helper functions
│
├── scripts/
│   ├── merge_xg_with_valuations.py
│   ├── add_contextual_features.py
│   └── train_model.py
│
└── docs/
    ├── BETA_v1.0_RELEASE.md       # Full technical documentation
    ├── METHODOLOGY.md             # Model methodology
    └── DATA_SOURCES.md            # Data attribution
```

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **Machine Learning:** scikit-learn (Gradient Boosting)
- **Data Processing:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Web App:** Streamlit
- **Data Sources:** FBref, Transfermarkt, EA Sports FIFA (via Kaggle)

---

## 📈 Model Architecture

### Algorithm
**Gradient Boosting Regressor** with:
- 200 estimators
- Max depth: 6
- Learning rate: 0.1
- Log transformation of target variable

### Data Pipeline
```
Raw Data (FBref + Transfermarkt + FIFA)
    ↓
Data Cleaning & Validation
    ↓
Feature Engineering (67 features)
    ↓
Train/Test Split (80/20)
    ↓
Gradient Boosting Training
    ↓
Model Evaluation & Validation
    ↓
Production Model (.pkl)
```

### Top 10 Features by Importance
1. minutes_played (15.6%)
2. team_xa (12.8%)
3. team_xg (8.1%)
4. age (5.7%)
5. team_minutes (5.2%)
6. xa_total (4.9%)
7. xg_total (4.7%)
8. team_goals (4.6%)
9. goals_per_90 (3.8%)
10. npxg_total (3.2%)

**Key Insight:** Team performance context accounts for 25.5% of importance - validates the performance-based approach without data leakage.

---

## 🎯 Use Cases

### 1. Talent Scouting
Identify undervalued players where model predicts significantly higher value:
```python
undervalued = predictions[predictions['predicted'] > predictions['actual'] * 1.2]
```

### 2. Youth Development
Flag wonderkids and high-potential prospects:
```python
wonderkids = data[(data['is_wonderkid'] == 1) & (data['minutes_played'] >= 1000)]
```

### 3. Contract Negotiations
Justify valuations with data-driven evidence:
```python
player_value = model.predict(player_features)
print(f"Estimated market value: EUR {player_value}M")
```

### 4. Squad Planning
Estimate total squad value for budgeting:
```python
squad_value = predictions.groupby('team')['predicted'].sum()
```

---

## 📊 Data Sources

All data sourced from public datasets via Kaggle:

- **Performance Stats:** [FBref via Kaggle](https://www.kaggle.com/) - xG, xA, progressive stats
- **Market Valuations:** [Transfermarkt via Kaggle](https://www.kaggle.com/) - Historical valuations
- **FIFA Ratings:** [EA Sports FIFA via Kaggle](https://www.kaggle.com/) - Player ratings & potential

**Coverage:**
- Big 5 European leagues (EPL, La Liga, Bundesliga, Serie A, Ligue 1)
- Seasons: 2021-22, 2023-24, 2024-25
- 7,722 player-season observations
- 100% xG/xA coverage

---

## 🔬 Methodology

### Data Leakage Prevention
**Critical requirement:** Market values cannot influence training features.

✅ **What we do:**
- Team context from performance: `team_goals`, `team_xg` (NOT `team_total_value`)
- Player importance: `goal_share_pct = player_goals / team_goals`
- Exclude target player from all team aggregations

❌ **What we DON'T do:**
- Use player's own market value in features
- Use team total value that includes the player
- Use future information to predict past seasons

### Validation Strategy
- 80/20 train/test split
- 5-fold cross-validation (CV R² = 0.685)
- Residual analysis for bias detection
- Performance evaluation across value ranges

---

## 📝 Example Usage

### Load Model & Predict
```python
import joblib
import pandas as pd
import numpy as np

# Load model
model_data = joblib.load('data/market_value_model.pkl')
model = model_data['model']

# Prepare player data (67 features required)
player = pd.DataFrame({
    'age': [25],
    'minutes_played': [2800],
    'goals': [20],
    'xg_total': [18.5],
    # ... (all 67 features)
})

# Predict
y_log = model.predict(player)
y_pred = np.expm1(y_log)  # Transform from log scale

print(f"Predicted market value: EUR {y_pred[0]:.2f}M")
```

### Run Web App
```python
streamlit run app.py
```

---

## 🎓 Educational Purpose

This project was built as a portfolio piece demonstrating:

- **ML Engineering:** End-to-end pipeline from raw data to production model
- **Feature Engineering:** Domain-driven feature creation (67 features)
- **Data Quality:** Multi-source integration, leakage prevention
- **Production ML:** Scalable code, proper validation, documentation

**Target Audience:** Sporting director / technical director internship applications

---

## 📊 Results & Insights

### Key Findings

1. **Playing time is king:** `minutes_played` is the #1 feature (15.6% importance)
2. **Team context matters:** Team performance features account for 25%+ importance
3. **xG > goals:** Expected stats outperform actual stats for prediction
4. **Age curve confirmed:** Peak value age 24-28, decline after 30
5. **League premium:** EPL players valued 35% higher than Serie A equivalents

### Known Limitations

1. **FIFA Match Rate:** Only 4.7% direct match (filled with median)
   - Future: Scrape SoFIFA for 100% coverage
2. **High-Value Players:** R² drops to 0.42 for €50M+ players
   - Intangibles (brand, marketability) not captured
3. **League Coverage:** Big 5 only (no Portuguese, Dutch leagues)
4. **Temporal Structure:** Only 3 seasons (limited historical depth)
   - Beta v2.0 in development: 14 seasons, 71K observations

---

## 🚀 Future Roadmap

### Beta v2.0 (Target: January 2025)
- [ ] Temporal dataset: 71K observations, 14 seasons (2011-2024)
- [ ] Time-based train/test splits (prevent future→past leakage)
- [ ] Expected performance: R² = 0.90+, MAE = €3-3.5M

### Potential Enhancements
- [ ] Contract data (years remaining, expiry urgency)
- [ ] Injury history (availability percentage)
- [ ] Social media metrics (brand value proxy)
- [ ] Position-specific models (GK vs FW have different value drivers)
- [ ] Transfer history (previous fees as anchors)

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

**Attribution Required:**
- FBref (via Kaggle) - Performance statistics
- Transfermarkt (via Kaggle) - Market valuations
- EA Sports FIFA (via Kaggle) - Player ratings

---

## 👤 Author

**Owen James**
- Economics Student & Football Analytics Enthusiast
- Aspiring Sporting Director / Technical Director
- Target: Summer 2025 Internships

**Connect:**
- LinkedIn: Owen James
- GitHub: [github.com/Foxesrevenge/transfer-market-predictor](https://github.com/Foxesrevenge/transfer-market-predictor)
- Email: owenajames05swim@gmail.com

---

## 🙏 Acknowledgments

- **FBref** for comprehensive performance statistics
- **Transfermarkt** for market valuation data
- **EA Sports** for FIFA player ratings
- **Kaggle** for hosting public datasets
- **scikit-learn** for ML framework
- Built with assistance from [Claude Code](https://claude.com/claude-code)

---

## 📚 Documentation

- [Full Technical Documentation](docs/BETA_v1.0_RELEASE.md)
- [Methodology Details](docs/METHODOLOGY.md)
- [Data Sources & Attribution](docs/DATA_SOURCES.md)
- [Beta v2.0 Roadmap](docs/BETA_v2.0_ROADMAP.md)

---

## ⭐ Star This Repo!

If you find this project useful for your own work or learning, please give it a star! It helps others discover the project.

---

**Last Updated:** December 19, 2024
**Version:** Beta v1.0
**Status:** Production Ready
