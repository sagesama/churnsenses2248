import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from common import *

st.set_page_config(page_title="ChurnSense", page_icon="📉", layout="wide")
sns.set_theme(style="whitegrid")

@st.cache_data
def get_data(): return load_data()
@st.cache_resource
def get_model(): return load_model()
@st.cache_data
def get_metrics(): return json.loads(METRICS.read_text())

def risk_band(p): return "High" if p >= .6 else "Medium" if p >= .3 else "Low"

df, model = get_data(), get_model()
df["ChurnFlag"] = (df["Churn"] == "Yes").astype(int)

page = st.sidebar.radio("ChurnSense", ["Dashboard", "Individual Prediction",
                                       "Batch Prediction", "Model Analytics"])
st.sidebar.caption("Customer Data → Model → Churn Probability → Retention Action")

# ---------------------------------------------------------------- Dashboard
if page == "Dashboard":
    st.title("📉 Customer Churn Dashboard")
    with st.sidebar.expander("Filters", expanded=True):
        contracts = st.multiselect("Contract", df.Contract.unique(), df.Contract.unique())
        internet = st.multiselect("Internet service", df.InternetService.unique(), df.InternetService.unique())
        t_min, t_max = st.slider("Tenure (months)", 0, int(df.tenure.max()), (0, int(df.tenure.max())))
    d = df[df.Contract.isin(contracts) & df.InternetService.isin(internet) & df.tenure.between(t_min, t_max)]
    if d.empty:
        st.warning("No customers match the filters."); st.stop()

    churned = d[d.ChurnFlag == 1]
    k = st.columns(5)
    k[0].metric("Customers", f"{len(d):,}")
    k[1].metric("Churn rate", f"{d.ChurnFlag.mean():.1%}")
    k[2].metric("Churned customers", f"{len(churned):,}")
    k[3].metric("Avg monthly charge", f"${d.MonthlyCharges.mean():.2f}")
    k[4].metric("Monthly revenue at risk", f"${churned.MonthlyCharges.sum():,.0f}")

    def bar(col, title, ax):
        r = d.groupby(col).ChurnFlag.mean().sort_values(ascending=False) * 100
        sns.barplot(x=r.values, y=r.index, ax=ax, color="#d9534f")
        ax.set(title=title, xlabel="Churn rate (%)", ylabel="")
        for i, v in enumerate(r.values): ax.text(v + .5, i, f"{v:.1f}%", va="center")

    c1, c2 = st.columns(2)
    fig, ax = plt.subplots(figsize=(6, 3.5)); bar("Contract", "Churn by contract", ax); c1.pyplot(fig)
    fig, ax = plt.subplots(figsize=(6, 3.5)); bar("PaymentMethod", "Churn by payment method", ax); c2.pyplot(fig)
    c3, c4 = st.columns(2)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    sns.histplot(data=d, x="tenure", hue="Churn", multiple="stack", bins=24, ax=ax,
                 palette={"No": "#5cb85c", "Yes": "#d9534f"})
    ax.set_title("Tenure distribution"); c3.pyplot(fig)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    sns.kdeplot(data=d, x="MonthlyCharges", hue="Churn", fill=True, common_norm=False, ax=ax,
                palette={"No": "#5cb85c", "Yes": "#d9534f"})
    ax.set_title("Monthly charges by churn"); c4.pyplot(fig)

    st.subheader("Explore a segment")
    dim = st.selectbox("Group churn by", ["InternetService", "TechSupport", "OnlineSecurity", "SeniorCitizen",
                                          "Partner", "Dependents", "PaperlessBilling", "StreamingTV", "gender"])
    summ = d.groupby(dim).agg(customers=("ChurnFlag", "size"), churn_rate=("ChurnFlag", "mean"),
                              avg_tenure=("tenure", "mean"), avg_monthly=("MonthlyCharges", "mean")).round(3)
    st.dataframe(summ.style.format({"churn_rate": "{:.1%}", "avg_tenure": "{:.1f}", "avg_monthly": "${:.2f}"}),
                 use_container_width=True)

# ------------------------------------------------------ Individual Prediction
elif page == "Individual Prediction":
    st.title("🔮 Individual Customer Prediction")
    opts = {c: sorted(df[c].unique()) for c in CAT}
    with st.form("single"):
        a, b, c = st.columns(3)
        row = {}
        with a:
            st.markdown("**Demographics**")
            row["gender"] = st.selectbox("Gender", opts["gender"])
            row["SeniorCitizen"] = st.selectbox("Senior citizen", opts["SeniorCitizen"])
            row["Partner"] = st.selectbox("Partner", opts["Partner"])
            row["Dependents"] = st.selectbox("Dependents", opts["Dependents"])
            row["tenure"] = st.slider("Tenure (months)", 0, 72, 12)
        with b:
            st.markdown("**Services**")
            for col in ["PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
                        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
                row[col] = st.selectbox(col, opts[col])
        with c:
            st.markdown("**Account**")
            row["Contract"] = st.selectbox("Contract", opts["Contract"])
            row["PaperlessBilling"] = st.selectbox("Paperless billing", opts["PaperlessBilling"])
            row["PaymentMethod"] = st.selectbox("Payment method", opts["PaymentMethod"])
            row["MonthlyCharges"] = st.number_input("Monthly charges ($)", 0.0, 200.0, 70.0, 0.05)
            row["TotalCharges"] = st.number_input("Total charges ($)", 0.0, 10000.0,
                                                  float(round(row["MonthlyCharges"] * row["tenure"], 2)))
        go = st.form_submit_button("Predict churn", type="primary")
    if go:
        p = float(model.predict_proba(pd.DataFrame([row])[FEATURES])[0, 1])
        band = risk_band(p)
        m1, m2 = st.columns(2)
        m1.metric("Churn probability", f"{p:.1%}")
        m2.metric("Risk level", band)
        st.progress(p)
        {"High": st.error, "Medium": st.warning, "Low": st.success}[band](
            {"High": "High risk: prioritise a retention offer (e.g. contract upgrade or discount).",
             "Medium": "Moderate risk: monitor and consider a proactive check-in.",
             "Low": "Low risk: no immediate action needed."}[band])

# ---------------------------------------------------------- Batch Prediction
elif page == "Batch Prediction":
    st.title("📦 Batch Prediction")
    st.write("Upload a CSV with the same columns as the Telco dataset (`Churn` is optional).")
    st.download_button("Download sample template", df[FEATURES].head(5).to_csv(index=False),
                       "template.csv", "text/csv")
    up = st.file_uploader("CSV file", type="csv")
    if up:
        raw = pd.read_csv(up)
        missing = [c for c in FEATURES if c not in raw.columns]
        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}"); st.stop()
        X = clean(raw.assign(TotalCharges=raw["TotalCharges"] if "TotalCharges" in raw else None))[FEATURES]
        out = raw.copy()
        out["churn_probability"] = model.predict_proba(X)[:, 1].round(4)
        out["predicted_churn"] = (out.churn_probability >= .5).map({True: "Yes", False: "No"})
        out["risk_level"] = out.churn_probability.map(risk_band)
        k = st.columns(3)
        k[0].metric("Customers scored", f"{len(out):,}")
        k[1].metric("Predicted to churn", f"{(out.predicted_churn == 'Yes').sum():,}")
        k[2].metric("High-risk", f"{(out.risk_level == 'High').sum():,}")
        fig, ax = plt.subplots(figsize=(7, 3))
        sns.histplot(out.churn_probability, bins=30, ax=ax, color="#d9534f")
        ax.set_title("Distribution of churn probabilities"); st.pyplot(fig)
        st.dataframe(out.sort_values("churn_probability", ascending=False), use_container_width=True)
        st.download_button("Download predictions", out.to_csv(index=False), "churn_predictions.csv", "text/csv")

# ----------------------------------------------------------- Model Analytics
else:
    st.title("🧪 Model Analytics")
    m = get_metrics()
    st.caption(f"Selected model: **{m['best_model']}** · trained on {m['n_train']:,} / tested on {m['n_test']:,} customers")
    k = st.columns(5)
    for col, key in zip(k, ["accuracy", "precision", "recall", "f1", "roc_auc"]):
        col.metric(key.replace("_", " ").title(), f"{m[key]:.3f}")

    c1, c2 = st.columns(2)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(m["confusion_matrix"], annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    ax.set(xlabel="Predicted", ylabel="Actual", title="Confusion matrix"); c1.pyplot(fig)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(m["roc_curve"]["fpr"], m["roc_curve"]["tpr"], label=f"AUC = {m['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve"); ax.legend(); c2.pyplot(fig)

    c3, c4 = st.columns(2)
    imp = pd.DataFrame(m["importance"]).head(12)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(data=imp, x="importance", y="feature", ax=ax, color="#337ab7")
    ax.set(title="Feature importance (permutation, AUC drop)", ylabel=""); c3.pyplot(fig)
    cmp = pd.DataFrame(m["comparison"]).T
    c4.markdown("**Model comparison (test set)**"); c4.dataframe(cmp.style.format("{:.3f}"))

    preds = pd.read_csv(PREDS)
    st.subheader("Actual vs predicted")
    fig, ax = plt.subplots(figsize=(7, 3))
    sns.histplot(data=preds.assign(actual=preds.actual.map({0: "Stayed", 1: "Churned"})),
                 x="churn_probability", hue="actual", bins=30, ax=ax, element="step")
    ax.set_title("Predicted probability by actual outcome"); st.pyplot(fig)
    st.dataframe(preds.assign(correct=preds.actual == preds.predicted).head(200), use_container_width=True)
