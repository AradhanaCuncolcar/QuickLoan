# app.py — QuickLoan Risk Dashboard
# Run locally:  streamlit run app.py
# Needs (same folder): loan_model.joblib, loan_model_costaware.joblib, loans.csv, leaderboard.csv, leaderboard_chart.png
# Model files are saved by the notebook like this:
#   joblib.dump({'model': clf, 'columns': list(X_train.columns)}, 'loan_model.joblib')
from html import escape
from pathlib import Path
import streamlit as st, pandas as pd, numpy as np, joblib
from sklearn.model_selection import train_test_split

HERE = Path(__file__).parent          # so the app works no matter which folder it is launched from
st.set_page_config(page_title="QuickLoan Risk Dashboard", page_icon="💰", layout="wide")

CHAMPION_FILE = "loan_model.joblib"
MODEL_FILES = {
    "Gradient Boosting — recommended (cost-matched 1/6 rule)": "loan_model_costaware.joblib",
    "Gradient Boosting — bake-off champion (default 0.5 rule)": CHAMPION_FILE,
}


# ───────────────────────── number formatting ─────────────────────────
# (divisor, suffix, smallest value that uses it) — K only starts at 10,000 so "AED 40K" and "AED 100K" read alike
_UNITS = ((1e9, "B", 1e9), (1e6, "M", 1e6), (1e3, "K", 1e4))


def _fmt(x, keep_zeros=False):
    """2 decimals under 10, 1 decimal under 100, none above. Trailing zeros are dropped (5.00 -> 5, 4.50 -> 4.5)
    unless keep_zeros=True (used in tables, where a column should line up: 4.50M, 4.58M)."""
    d = 2 if x < 10 else 1 if x < 100 else 0
    out = f"{x:.{d}f}"
    return out if (keep_zeros or not d) else out.rstrip("0").rstrip(".")


def short(value, keep_zeros=False):
    """5_815_160 -> '5.82M' · 1_046_920 -> '1.05M' · 2.4e9 -> '2.4B' · 530_000 -> '530K' · 143 -> '143'."""
    v = float(value)
    sign, a = ("-" if v < 0 else ""), abs(v)
    for i, (div, suffix, start) in enumerate(_UNITS):
        if a >= start:
            s = _fmt(a / div, keep_zeros)
            if float(s) >= 1000 and i > 0:            # 999,700 would print '1000K' -> promote to '1M'
                return sign + _fmt(a / _UNITS[i - 1][0], keep_zeros) + _UNITS[i - 1][1]
            return sign + s + suffix
    return sign + f"{a:,.0f}"


def aed(value, signed=False, keep_zeros=False):
    """'AED 5.82M'  (signed=True -> '+AED 1.05M' / '-AED 3K')."""
    sign = "-" if value < 0 else ("+" if signed and value > 0 else "")
    return f"{sign}AED {short(abs(value), keep_zeros)}"


# ───────────────────────── model, data, metrics ─────────────────────────
@st.cache_resource
def load(fname):
    M = joblib.load(HERE / fname)
    return M["model"], M["columns"]


def encode(df, columns):
    return pd.get_dummies(df, columns=["home_ownership"]).reindex(columns=columns, fill_value=0)


def money_saved(y_true, y_pred, loan, interest=0.20):
    y_true, y_pred, loan = np.array(y_true), np.array(y_pred), np.array(loan)
    caught = (y_pred == 1) & (y_true == 1)        # defaulter rejected -> loan saved
    wrong = (y_pred == 1) & (y_true == 0)         # good borrower rejected -> interest lost
    return loan[caught].sum() - interest * loan[wrong].sum()


@st.cache_data
def portfolio_stats(fname):
    """Score one saved model on the held-out test split of loans.csv (same split as the notebook)."""
    model, cols = load(fname)
    df = pd.read_csv(HERE / "loans.csv")
    X, y = encode(df.drop(columns="default"), cols), df["default"]
    _, Xte, _, yte, _, lo_te = train_test_split(X, y, df["loan_amount"], test_size=0.25, random_state=1, stratify=y)
    pred, yt, ln = model.predict(Xte), yte.values, lo_te.values
    n, approved = len(yt), int((pred == 0).sum())
    caught_loans = float(ln[(pred == 1) & (yt == 1)].sum())
    lost_interest = float(0.20 * ln[(pred == 1) & (yt == 0)].sum())
    return dict(
        n=n, total_def=int((yt == 1).sum()), n_good=int((yt == 0).sum()),
        saved=float(money_saved(yt, pred, ln)), caught_loans=caught_loans, lost_interest=lost_interest,
        caught=int(((pred == 1) & (yt == 1)).sum()), wrong=int(((pred == 1) & (yt == 0)).sum()),
        approved=approved, approval_rate=approved / n,
        missed=int(((pred == 0) & (yt == 1)).sum()),
        default_rate_approved=float(((pred == 0) & (yt == 1)).sum() / max(approved, 1)),
    )


# ───────────────────────── look & feel (one CSS block) ─────────────────────────
CSS = """
<style>
@media (prefers-color-scheme: light){ .stApp{background:linear-gradient(135deg,#eef2ff 0%,#f0f9ff 55%,#f0fdf4 100%)} }

.qd-hero{background:linear-gradient(120deg,#4f46e5 0%,#7c3aed 55%,#db2777 100%);color:#fff;border-radius:20px;
  padding:1.3rem 1.6rem;margin:0 0 1rem;box-shadow:0 10px 28px rgba(79,70,229,.25)}
.qd-hero h1{margin:0;padding:0;color:#fff;font-size:clamp(1.5rem,4vw,2.2rem);line-height:1.15}
.qd-hero p{margin:.4rem 0 0;opacity:.93;font-size:clamp(.9rem,2.2vw,1.05rem);line-height:1.35}
.qd-chip{display:inline-block;margin-top:.8rem;max-width:100%;overflow-wrap:anywhere;background:rgba(255,255,255,.18);
  border:1px solid rgba(255,255,255,.4);border-radius:999px;padding:.2rem .85rem;font-size:.82rem}

.qd-banner{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:.8rem 1.5rem;color:#fff;
  border-radius:20px;padding:1.2rem 1.5rem;margin:.5rem 0 1rem;box-shadow:0 10px 26px rgba(15,23,42,.18)}
.qd-banner.reject{background:linear-gradient(110deg,#dc2626 0%,#f43f5e 45%,#f97316 100%)}
.qd-banner.approve{background:linear-gradient(110deg,#059669 0%,#10b981 55%,#14b8a6 100%)}
.qd-banner .main{flex:1 1 16rem;min-width:0}
.qd-banner .verdict{font-size:clamp(1.35rem,4vw,2rem);font-weight:800;line-height:1.1}
.qd-banner .sub{margin-top:.4rem;font-size:.92rem;line-height:1.35;opacity:.95;overflow-wrap:anywhere}
.qd-banner .prob{flex:0 0 auto;text-align:right}
.qd-banner .prob b{display:block;font-size:clamp(1.8rem,5vw,2.6rem);line-height:1;white-space:nowrap}
.qd-banner .prob span{font-size:.8rem;opacity:.92}

/* the grid adapts to the width of the content area (sidebar open or closed), not the screen */
.qd-wrap{container-type:inline-size;width:100%}
.qd-grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));grid-auto-rows:1fr;gap:1rem;margin:.4rem 0 1rem}
@container (max-width: 860px){ .qd-grid{grid-template-columns:repeat(3,minmax(0,1fr))} }
@container (max-width: 540px){ .qd-grid{grid-template-columns:repeat(2,minmax(0,1fr))} }
@container (max-width: 330px){ .qd-grid{grid-template-columns:minmax(0,1fr)} }

.qc{container-type:inline-size;position:relative;box-sizing:border-box;display:flex;flex-direction:column;gap:.55rem;
  min-height:9.5rem;padding:1rem 1.1rem;border-radius:20px;color:#fff;box-shadow:0 8px 22px rgba(15,23,42,.14)}
.qc:hover,.qc:focus-within{z-index:20}                 /* lift the hovered card so its tooltip is never hidden */
.qc-green{background:linear-gradient(135deg,#10b981,#14b8a6)}
.qc-purple{background:linear-gradient(135deg,#6366f1,#8b5cf6)}
.qc-orange{background:linear-gradient(135deg,#f97316,#f59e0b)}
.qc-blue{background:linear-gradient(135deg,#0ea5e9,#3b82f6)}
.qc-pink{background:linear-gradient(135deg,#ec4899,#f43f5e)}

.qc-top{display:flex;align-items:flex-start;justify-content:space-between;gap:.6rem}
.qc-label{flex:1 1 0;min-width:0;min-height:2.5em;font-size:clamp(.76rem,6.2cqw,.95rem);line-height:1.25;font-weight:600;
  overflow-wrap:break-word}                              /* reserved 2-line height keeps every value on the same baseline */
.qc-help{flex:0 0 auto;display:inline-flex;align-items:center;justify-content:center;width:1.3rem;height:1.3rem;
  border:1.5px solid rgba(255,255,255,.9);border-radius:50%;font-size:.72rem;font-weight:700;line-height:1;cursor:help;outline:none}
/* tooltip = an overlay that covers its own card, so it can never poke outside the card or the page edge */
.qc-help::after{content:attr(data-tip);position:absolute;top:0;left:0;right:0;min-height:100%;z-index:30;box-sizing:border-box;
  display:flex;align-items:center;padding:.8rem 1rem;border-radius:20px;background:#0f172a;color:#fff;
  font-size:.76rem;font-weight:400;line-height:1.4;text-align:left;white-space:normal;
  opacity:0;visibility:hidden;pointer-events:none}
.qc-help:hover::after,.qc-help:focus::after{opacity:1;visibility:visible}

.qc-value{display:flex;align-items:baseline;gap:.25em;white-space:nowrap;font-size:clamp(1.3rem,13cqw,2.3rem);font-weight:700;
  line-height:1.05;font-variant-numeric:tabular-nums}
.qc-unit{font-size:.5em;font-weight:600;letter-spacing:.04em;opacity:.92}
.qc-pill{margin-top:auto;align-self:flex-start;box-sizing:border-box;max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
  padding:.18rem .65rem;border-radius:999px;background:#fff;font-size:.8rem;font-weight:700}
.qc-pill.good{color:#047857}.qc-pill.bad{color:#b91c1c}.qc-pill.neutral{color:#475569}
</style>
"""


def metric_card(color, label, value_html, pill_text, pill_kind, tip):
    return (f'<div class="qc qc-{color}"><div class="qc-top"><span class="qc-label">{escape(label)}</span>'
            f'<span class="qc-help" tabindex="0" data-tip="{escape(tip, quote=True)}">?</span></div>'
            f'<div class="qc-value">{value_html}</div>'
            f'<div class="qc-pill {pill_kind}">{escape(pill_text)}</div></div>')


def delta_pill(delta, fmt, good_when, unit=""):
    """Return (text, kind). good_when: 'up' | 'down' | None (neutral)."""
    if abs(delta) < 1e-9:
        return "no change", "neutral"
    arrow = "↑" if delta > 0 else "↓"
    kind = "neutral" if good_when is None else ("good" if (delta > 0) == (good_when == "up") else "bad")
    return f"{arrow} {fmt(delta)}{unit}", kind


# ───────────────────────── sidebar: pick the model ─────────────────────────
st.sidebar.header("⚙️ Model")
choice = st.sidebar.radio("Which model should decide?", list(MODEL_FILES), index=0)
st.sidebar.caption(
    "**Cost-matched 1/6 rule:** reject when P(default) ≥ 1/6, because losing a loan costs 5× the "
    "interest lost by turning away a good customer (0.20 / 1.20 = 1/6).\n\n"
    "**Default 0.5 rule:** reject only when P(default) ≥ 50 % — the setting scored on the leaderboard.")
fname = MODEL_FILES[choice]
model, COLUMNS = load(fname)
threshold = float(getattr(model, "threshold", 0.5))

st.html(CSS + f"""
<div class="qd-hero"><h1>💰 QuickLoan — Risk Dashboard</h1>
<p>Approve or reject loans by <b>money saved</b>, not just accuracy.</p>
<span class="qd-chip">Active model: {escape(choice)}</span></div>""")

# ───────────────────────── CARD 1: decision ─────────────────────────
st.header("1 · Loan Decision")
c1, c2 = st.columns(2)
with c1:
    age = st.slider("Age", 21, 64, 35)
    annual_income = st.number_input("Annual income (AED)", 24000, 400000, 90000, step=1000)
    loan_amount = st.number_input("Loan amount (AED)", 5000, 150000, 40000, step=1000)
    loan_term_months = st.selectbox("Loan term (months)", [12, 24, 36, 48, 60], index=2)
    employment_years = st.slider("Employment years", 0.0, 35.0, 4.0, 0.5)
with c2:
    credit_history_months = st.slider("Credit history (months)", 0, 300, 60)
    num_existing_loans = st.slider("Existing loans", 0, 8, 1)
    past_defaults = st.slider("Past defaults", 0, 6, 0)
    home_ownership = st.selectbox("Home ownership", ["rent", "own", "mortgage"])
debt_to_income = round(loan_amount / (annual_income + 1), 2)
st.caption(f"Debt-to-income (auto): {debt_to_income}")

applicant = dict(age=age, annual_income=annual_income, loan_amount=loan_amount,
                 loan_term_months=loan_term_months, employment_years=employment_years,
                 credit_history_months=credit_history_months, num_existing_loans=num_existing_loans,
                 past_defaults=past_defaults, debt_to_income=debt_to_income, home_ownership=home_ownership)
X1 = encode(pd.DataFrame([applicant]), COLUMNS)
pred = int(model.predict(X1)[0])
prob = float(model.predict_proba(X1)[0][1]) if hasattr(model, "predict_proba") else float(pred)
verdict, kind = ("❌ REJECT — high default risk", "reject") if pred == 1 else ("✅ APPROVE — low default risk", "approve")
st.html(CSS + f"""
<div class="qd-banner {kind}">
  <div class="main"><div class="verdict">{verdict}</div>
  <div class="sub">Loan {aed(loan_amount)} · Income {aed(annual_income)} · Debt-to-income {debt_to_income} ·
  Rule: reject if P(default) ≥ {threshold*100:.1f}%</div></div>
  <div class="prob"><b>{prob*100:.0f}%</b><span>default probability</span></div>
</div>""")

# ───────────────────────── CARD 2: money saved on the portfolio ─────────────────────────
st.header("2 · Money Saved (portfolio view)")
try:
    s = portfolio_stats(fname)
    base = portfolio_stats(CHAMPION_FILE)                 # comparison point: the bake-off champion (default 0.5 rule)
    is_base = fname == CHAMPION_FILE
    n, td = s["n"], s["total_def"]

    def pill(delta, fmt, good_when, unit=""):
        return ("baseline (0.5 rule)", "neutral") if is_base else delta_pill(delta, fmt, good_when, unit)

    cards = [
        metric_card("green", "💵 Money saved vs approve-all",
                    f'<span class="qc-unit">AED</span>{short(s["saved"])}',
                    *pill(s["saved"] - base["saved"], lambda d: aed(d, signed=True), "up"),
                    f'Net AED saved vs approving everyone: loans of defaulters rejected minus 20% interest lost on good customers rejected. '
                    f'Exact: AED {s["saved"]:,.0f}.'),
        metric_card("purple", "🎯 Defaulters caught",
                    f'{s["caught"]}/{td}',
                    *pill(100 * (s["caught"] / td - base["caught"] / td), lambda d: f"{d:+.1f}", "up", " pts"),
                    f'Defaulters rejected, out of {td} in the test split (recall {100*s["caught"]/td:.1f}%).'),
        metric_card("orange", "😕 Good borrowers rejected",
                    f'{s["wrong"]:,}',
                    *pill(s["wrong"] - base["wrong"], lambda d: f"{d:+,.0f}", "down"),
                    f'Repaying customers turned away, out of {s["n_good"]:,}. Each costs ~20% of the loan in lost interest.'),
        metric_card("blue", "✅ Approval rate",
                    f'{100*s["approval_rate"]:.1f}%',
                    *pill(100 * (s["approval_rate"] - base["approval_rate"]), lambda d: f"{d:+.1f}", None, " pts"),
                    f'Share of the {n:,} test applicants approved ({s["approved"]:,} loans).'),
        metric_card("pink", "⚠️ Default rate of approved",
                    f'{100*s["default_rate_approved"]:.2f}%',
                    *pill(100 * (s["default_rate_approved"] - base["default_rate_approved"]), lambda d: f"{d:+.2f}", "down", " pts"),
                    f'Of the {s["approved"]:,} approved loans, {s["missed"]} still default.'),
    ]
    st.html(CSS + f'<div class="qd-wrap"><div class="qd-grid">{"".join(cards)}</div></div>')
    st.caption(f'Loans saved {aed(s["caught_loans"])} − interest lost {aed(s["lost_interest"])} = net {aed(s["saved"])}. '
               f'Evaluated on a held-out test split of loans.csv (25 %, {n:,} loans). '
               + ("" if is_base else "Pills compare with the bake-off champion (default 0.5 rule). ")
               + "Hover the ? for exact figures.")
except FileNotFoundError:
    st.info("Put loans.csv in this folder to see the money-saved view.")

# ───────────────────────── CARD 3: leaderboard ─────────────────────────
st.header("3 · The Bake-Off Leaderboard")
try:
    lb = pd.read_csv(HERE / "leaderboard.csv")
    show = lb[["Model", "Family", "Accuracy", "Recall", "Money_saved", "Defaulters_caught", "Good_rejected"]].copy()
    show["Money_saved"] = show["Money_saved"].map(lambda v: aed(v, keep_zeros=True))
    show = show.rename(columns={"Money_saved": "Money saved", "Defaulters_caught": "Defaulters caught", "Good_rejected": "Good rejected"})
    st.dataframe(show.style.format({"Accuracy": "{:.2%}", "Recall": "{:.2%}"}), hide_index=True, width="stretch")
    st.caption("All 10 models scored on the same test set with the default 0.5 rule. "
               "Accuracy spans 1.5 points; money saved varies 2.4×.")
    img = HERE / "leaderboard_chart.png"
    if img.exists():
        st.image(str(img), width="stretch")
except FileNotFoundError:
    st.info("Put leaderboard.csv in this folder to see the leaderboard.")
