# Smart-City-AQI
Prediction AQI with Ridge Regression

##  Project Story 

### Situation
Air pollution is a critical challenge in smart cities. Traditional monitoring often lags behind real-time needs, making it difficult to respond quickly to rising AQI levels.  

### Task
We set out to build a **predictive model** that can estimate AQI using smart mobility data and temporal features, enabling proactive decision-making for urban health and sustainability.  

### Action
- **Feature Engineering**:
  - Rolling averages (7-day, 30-day AQI)
  - Lag features (previous day, 3-day, 7-day AQI)
  - Seasonality (month, day of week)
  - Smart mobility indicators (road occupancy, emissions, sentiment, energy consumption)
- **Modeling**:
  - Ridge Regression with hyperparameter tuning (`alpha=10`)
  - Scaling inputs with `StandardScaler`
  - Cross-validation to ensure robustness
- **Deployment**:
  - Streamlit app (`app.py`) for interactive predictions
  - Model and scaler saved as `.joblib` for reproducibility

### Result
- **R² ≈ 0.71** → explains 71% of AQI variance  
- **MSE ≈ 810** → significantly reduced error compared to baseline linear regression  
- **Deployment-ready** → judges and users can run the app with one command and see predictions live  

---

##  Results Summary
| Model              | MSE   | R²   | Notes |
|--------------------|-------|------|-------|
| Linear Regression  | 2591  | 0.08 | Weak baseline, underfitting |
| Ridge Regression   | 810   | 0.71 | Strong baseline, interpretable |

---

##  How to Run
1. Clone the repo:
   ```bash
   git clone https://github.com/SHEisSONALI/Smart-City-AQI.git
   cd aqi-prediction
2. Installing Dependencies
   ```bash
   pip install -r requirements.txt
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
4. Open the local URL (usually http://localhost:8501) to interact with the app.

