from pathlib import Path
import joblib, pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "telco_churn.csv"
MODEL = ROOT / "models" / "churn_model.joblib"
METRICS = ROOT / "models" / "metrics.json"
PREDS = ROOT / "models" / "test_predictions.csv"

NUM = ["tenure", "MonthlyCharges", "TotalCharges"]
CAT = ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
       "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
       "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"]
FEATURES = CAT + NUM

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"])
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes", "0": "No", "1": "Yes"}).fillna(df["SeniorCitizen"])
    return df

def load_data() -> pd.DataFrame:
    return clean(pd.read_csv(DATA))

def load_model():
    return joblib.load(MODEL)
