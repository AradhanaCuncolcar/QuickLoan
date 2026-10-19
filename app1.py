# app.py — QuickLoan Risk Dashboard
# Run locally:  streamlit run app.py
# Needs (same folder): loan_model.joblib, loan_model_costaware.joblib, loans.csv, leaderboard.csv, leaderboard_chart.png
# Model files are saved by the notebook like this:
#   joblib.dump({'model': clf, 'columns': list(X_train.columns)}, 'loan_model.joblib')
from pathlib import Path
import streamlit as st, pandas as pd, numpy as np, joblib
from sklearn.model_selection import train_test_split

HERE = Path(__file__).parent          # so the app works no matter which folder it is launched from
st.set_page_config(page_title="QuickLoan Risk Dashboard", page_icon="💰", layout="wide")

MODEL_FILES = {
    "Gradient Boosting — recommended (cost-matched 1/6 rule)": "loan_model_costaware.joblib",
    "Gradient Boosting — bake-off champion (default 0.5 rule)": "loan_model.joblib",
}

@st.cache_resource
def load(fname):
    M = joblib.load(HERE / fname)
    return M["model"], M["columns"]

st.sidebar.header("⚙️ Model")
choice = st.sidebar.radio("Which model should decide?", list(MODEL_FILES), index=0)
st.sidebar.caption(
    "**Cost-matched 1/6 rule:** reject when P(default) ≥ 1/6, because losing a loan costs 5× the "
    "interest lost by turning away a good customer (0.20 / 1.20 = 1/6).\n\n"
    "**Default 0.5 rule:** reject only when P(default) ≥ 50 % — the setting scored on the leaderboard.")
model, COLUMNS = load(MODEL_FILES[choice])

def encode(df):
    return pd.get_dummies(df, columns=["home_ownership"]).reindex(columns=COLUMNS, fill_value=0)

def money_saved(y_true, y_pred, loan, interest=0.20):
    y_true, y_pred, loan = np.array(y_true), np.array(y_pred), np.array(loan)
    caught = (y_pred==1)&(y_true==1)        # defaulter rejected -> loan saved
    wrong  = (y_pred==1)&(y_true==0)        # good borrower rejected -> interest lost
    return loan[caught].sum() - interest*loan[wrong].sum()

st.title("💰 QuickLoan — Risk Dashboard")
st.caption(f"Active model: **{choice}**")

# ---------- CARD 1: Decision ----------
st.header("1 · Loan Decision")
c1, c2 = st.columns(2)
with c1:
    age = st.slider("Age", 21, 64, 35)
    annual_income = st.number_input("Annual income (AED)", 24000, 400000, 90000, step=1000)
    loan_amount = st.number_input("Loan amount (AED)", 5000, 150000, 40000, step=1000)
    loan_term_months = st.selectbox("Loan term (months)", [12,24,36,48,60], index=2)
    employment_years = st.slider("Employment years", 0.0, 35.0, 4.0, 0.5)
with c2:
    credit_history_months = st.slider("Credit history (months)", 0, 300, 60)
    num_existing_loans = st.slider("Existing loans", 0, 8, 1)
    past_defaults = st.slider("Past defaults", 0, 6, 0)
    home_ownership = st.selectbox("Home ownership", ["rent","own","mortgage"])
debt_to_income = round(loan_amount/(annual_income+1), 2)
st.caption(f"Debt-to-income (auto): {debt_to_income}")

applicant = dict(age=age, annual_income=annual_income, loan_amount=loan_amount,
    loan_term_months=loan_term_months, employment_years=employment_years,
    credit_history_months=credit_history_months, num_existing_loans=num_existing_loans,
    past_defaults=past_defaults, debt_to_income=debt_to_income, home_ownership=home_ownership)
X1 = encode(pd.DataFrame([applicant]))
pred = int(model.predict(X1)[0])
prob = float(model.predict_proba(X1)[0][1]) if hasattr(model,"predict_proba") else float(pred)
if pred==1:
    st.error(f"❌ REJECT — high default risk ({prob*100:.0f}%)")
else:
    st.success(f"✅ APPROVE — low default risk ({prob*100:.0f}%)")

# ---------- CARD 2: Money saved on the portfolio ----------
st.header("2 · Money Saved (portfolio view)")
try:
    df = pd.read_csv(HERE / "loans.csv")
    X = encode(df.drop(columns="default")); y = df["default"]
    Xtr,Xte,ytr,yte,lo_tr,lo_te = train_test_split(X,y,df["loan_amount"],test_size=0.25,random_state=1,stratify=y)
    pred_te = model.predict(Xte)
    saved = money_saved(yte, pred_te, lo_te)
    caught = int(((pred_te==1)&(yte.values==1)).sum()); total_def = int((yte.values==1).sum())
    wrong  = int(((pred_te==1)&(yte.values==0)).sum())
    m1,m2,m3 = st.columns(3)
    m1.metric("💵 Money saved vs approve-all", f"AED {saved:,.0f}")
    m2.metric("🎯 Defaulters caught", f"{caught}/{total_def}")
    m3.metric("😕 Good borrowers rejected", f"{wrong}")
    st.caption("Evaluated on a held-out test split of loans.csv (25 %, 1,500 loans) using the model you trained.")
except FileNotFoundError:
    st.info("Put loans.csv in this folder to see the money-saved view.")

# ---------- CARD 3: Leaderboard ----------
st.header("3 · The Bake-Off Leaderboard")
try:
    lb = pd.read_csv(HERE / "leaderboard.csv")
    show = lb[["Model","Family","Accuracy","Recall","Money_saved","Defaulters_caught","Good_rejected"]].rename(
        columns={"Money_saved": "Money saved (AED)", "Defaulters_caught": "Defaulters caught", "Good_rejected": "Good rejected"})
    st.dataframe(show.style.format({"Accuracy": "{:.2%}", "Recall": "{:.2%}", "Money saved (AED)": "{:,.0f}"}),
                 hide_index=True, width="stretch")
    st.caption("All 10 models scored on the same test set with the default 0.5 rule. "
               "Accuracy spans 1.5 points; money saved varies 2.4×.")
    img = HERE / "leaderboard_chart.png"
    if img.exists():
        st.image(str(img), width="stretch")
except FileNotFoundError:
    st.info("Put leaderboard.csv in this folder to see the leaderboard.")
