# 🏏 Cricket Analytics Dashboard & Wicket Prediction System

A **machine learning–powered cricket analytics dashboard** built using **Streamlit**, providing ball-by-ball intelligence, wicket probability prediction, player analytics, head-to-head matchups, venue insights, and full T20 match simulations.

---

## 🚀 Features

### 🎯 Wicket Probability Prediction

* Predicts the **likelihood of a wicket on the next ball**
* Uses a **Random Forest ML model**
* Factors considered:

  * Match phase (Powerplay / Middle / Death)
  * Pressure index
  * Batter & bowler form
  * Batting position
  * Venue run tendency
  * Historical batter vs bowler (H2H) data
* Risk categorization:

  * ✅ Low Risk
  * ⚠️ Medium Risk
  * 🔥 High Risk

---

### 🔍 Player Search & Analytics

* Detailed **batter & bowler performance dashboards**
* Key performance indicators (KPIs)
* Interactive visualizations:

  * Bar charts
  * Radar charts
* Recent matchup history
* Phase-wise insights (where available)

---

### ⚔ Batter vs Bowler (H2H Analysis)

* Head-to-head statistics:

  * Strike rate
  * Dismissals
  * Balls faced
  * Runs scored
* Automated matchup analysis:

  * Batter dominance
  * Bowler dominance
  * Balanced contests

---

### 🏟 Venue Analysis

* Top run scorers
* Highest strike rates
* Aggregate statistical overview:

  * Total runs
  * Average strike rate
  * Player counts

---

### 📊 T20 Match Simulation

* Full **ball-by-ball 20-over match simulation**
* Uses ML-predicted wicket probability per ball
* Generates:

  * Final scorecard
  * Over-by-over runs & wickets
  * Run rate trends
  * Commentary-style timeline

---

## 🧠 Machine Learning Model

* **Algorithm**: Random Forest Classifier
* **Objective**: Predict wicket occurrence (binary classification)
* **Model File**:

  ```
  models/wicket_prediction_rf.pkl
  ```
* Loaded using `joblib`
* Cached with `st.cache_resource` for performance

---

## 📁 Project Structure

```
cricket-analytics-dashboard/
│
├── app.py                         # Main Streamlit application
│
├── data/
│   ├── batter_stats.csv           # Batter statistics
│   ├── bowler_stats.csv           # Bowler statistics
│   └── batter_bowler_matchups.csv # Head-to-head data
│
├── models/
│   └── wicket_prediction_rf.pkl   # Trained ML model
│
├── requirements.txt               # Python dependencies
│
└── README.md                      # Project documentation
```

---

## 📦 Installation & Setup

### 1️⃣ Create Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Required Libraries

* streamlit
* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* joblib

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

The app will open automatically in your browser at:

```
http://localhost:8501
```

---

## 🎨 UI & Design

* Light-mode professional UI
* Custom CSS for:

  * Metrics
  * Buttons
  * Tables
  * Charts
* Responsive wide-layout dashboard
* Sidebar-based navigation

---

## ⚠️ Data Requirements

Ensure the following columns exist:

### `batter_stats.csv`

* batter
* matches_played
* runs
* strike_rate
* average
* boundaries *(optional but recommended)*

### `bowler_stats.csv`

* bowler
* matches_played
* wickets
* economy
* average
* strike_rate *(optional)*

### `batter_bowler_matchups.csv`

* batter
* bowler
* strike_rate
* dismissals
* balls_faced *(optional)*

---

## 🛡 Error Handling & Caching

* Graceful handling of missing files
* Column validation with warnings
* Cached data loading for speed
* Default fallbacks for prediction failures

---

## 📌 Future Enhancements

* Live ball-by-ball data integration
* Team-level analytics
* Player form trends over seasons
* Bowling variation classification
* Win probability modeling
* IPL / ODI / Test format support
  ---
