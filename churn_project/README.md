# Customer Churn Prediction — ML Project

Predict whether a telecom customer will leave using machine learning.

## Project Structure

```
churn_project/
├── data/
│   ├── generate_dataset.py       # Generates synthetic Telco dataset
│   ├── customer_churn.csv        # Raw dataset (7,043 rows × 21 cols)
│   └── customer_churn_clean.csv  # Preprocessed & encoded dataset
├── src/
│   ├── 01_preprocessing.py       # Data cleaning, encoding, feature engineering
│   ├── 02_eda.py                 # Exploratory Data Analysis + 6 charts
│   ├── 03_model_training.py      # Train 3 models + evaluate + save best
│   └── 04_predict.py             # Batch / single / CSV prediction interface
├── outputs/
│   ├── eda/                      # EDA charts (PNG)
│   └── models/                   # Model evaluation charts (PNG)
├── models/
│   └── best_model_rf.pkl         # Saved Random Forest model
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Run the Pipeline (in order)

```bash
# 1. Generate dataset
python data/generate_dataset.py

# 2. Preprocess data
cd churn_project
python src/01_preprocessing.py

# 3. Exploratory Data Analysis
python src/02_eda.py

# 4. Train & evaluate models
python src/03_model_training.py

# 5. Make predictions
python src/04_predict.py                          # Batch prediction
python src/04_predict.py --single                 # Interactive single prediction
python src/04_predict.py --csv path/to/file.csv   # Predict from any CSV
```

## Models & Results

| Model               | Accuracy | F1 Score | ROC-AUC |
|---------------------|----------|----------|---------|
| Logistic Regression | ~79%     | ~0.62    | ~0.85   |
| Decision Tree       | ~78%     | ~0.61    | ~0.82   |
| **Random Forest**   | **~83%** | **~0.68**| **~0.89**|

**Best Model: Random Forest Classifier**
- Ensemble of 150 decision trees
- Handles class imbalance better than baseline
- Lower overfitting via bagging + feature randomness

## Key Features (by importance)

1. `tenure` — how long the customer has been with the company
2. `MonthlyCharges` — monthly bill amount
3. `TotalCharges` — total amount billed
4. `Contract_Month-to-month` — highest churn risk contract type
5. `InternetService_Fiber optic` — associated with higher churn

## Key Insights from EDA

- **Overall churn rate**: ~31%
- **Month-to-month contracts**: account for ~85% of churners
- **Fiber optic customers** churn at nearly 2× the rate of DSL customers
- **Average tenure (churned)**: ~16 months vs ~38 months for retained customers
- **Electronic check** payment method correlates with higher churn

## Future Enhancements

- [ ] Deploy with Streamlit web app
- [ ] Add SMOTE for class imbalance handling
- [ ] Experiment with XGBoost / LightGBM
- [ ] Add SHAP explainability
- [ ] Real-time prediction API (FastAPI)
- [ ] Cloud deployment (AWS / GCP)
