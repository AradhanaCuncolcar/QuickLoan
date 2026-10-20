# app.py — QuickLoan Risk Dashboard  ·  the whole bake-off, step by step
# Run locally:  streamlit run app.py
# Needs (same folder): loans.csv, loan_model.joblib, loan_model_costaware.joblib, leaderboard.csv,
#                      leaderboard_cost_aware_threshold.csv, cv_robustness_table.csv, leaderboard_chart.png, cv_money_chart.png
# Model files are saved by the notebook like this:
#   joblib.dump({'model': clf, 'columns': list(X_train.columns)}, 'loan_model.joblib')
from html import escape
from pathlib import Path
import altair as alt
import streamlit as st, pandas as pd, numpy as np, joblib
from sklearn.model_selection import train_test_split

HERE = Path(__file__).parent          # so the app works no matter which folder it is launched from
st.set_page_config(page_title="QuickLoan Risk Dashboard", page_icon="💰", layout="wide")

CHAMPION_FILE, COSTAWARE_FILE = "loan_model.joblib", "loan_model_costaware.joblib"
MODEL_FILES = {
    "Gradient Boosting — recommended (cost-matched 1/6 rule)": COSTAWARE_FILE,
    "Gradient Boosting — bake-off champion (default 0.5 rule)": CHAMPION_FILE,
}
FAM_COL = {"Single": "#64748b", "Voting": "#8b5cf6", "Bagging": "#0ea5e9", "Boosting": "#f59e0b", "Stacking": "#ec4899"}


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


# ───────────────────────── models, data, metrics ─────────────────────────
@st.cache_resource
def load(fname):
    M = joblib.load(HERE / fname)
    return M["model"], M["columns"]


@st.cache_data
def load_csv(name):
    p = HERE / name
    return pd.read_csv(p) if p.exists() else None


def encode(df, columns):
    return pd.get_dummies(df, columns=["home_ownership"]).reindex(columns=columns, fill_value=0)


def money_saved(y_true, y_pred, loan, interest=0.20):
    """The brief's exact metric: AED saved versus approving everyone."""
    y_true, y_pred, loan = np.array(y_true), np.array(y_pred), np.array(loan)
    caught = (y_pred == 1) & (y_true == 1)        # defaulter rejected -> loan saved
    wrong = (y_pred == 1) & (y_true == 0)         # good borrower rejected -> interest lost
    return loan[caught].sum() - interest * loan[wrong].sum()


@st.cache_data
def data_overview():
    """Step 1 — everything about loading, encoding and splitting, computed live from loans.csv."""
    df = pd.read_csv(HERE / "loans.csv")
    X = pd.get_dummies(df.drop(columns="default"), columns=["home_ownership"], dtype=int)
    y = df["default"]
    Xtr, Xte, ytr, yte, ltr, lte = train_test_split(X, y, df["loan_amount"], test_size=0.25, random_state=1, stratify=y)
    pd_bucket = df["past_defaults"].clip(upper=3).map(lambda v: "3+" if v >= 3 else str(int(v)))
    return dict(
        df_head=df.head(6),
        encode_example=pd.concat([df[["home_ownership"]].head(5), X.filter(like="home_ownership_").head(5)], axis=1),
        describe=df.drop(columns="default").describe().T.round(2),
        balance=pd.DataFrame({"Outcome": ["Repaid (0)", "Defaulted (1)"], "Loans": [int((y == 0).sum()), int((y == 1).sum())],
                              "Share": [float((y == 0).mean()), float((y == 1).mean())]}),
        by_past=df.groupby(pd_bucket)["default"].agg(["mean", "size"]).reset_index().rename(
            columns={"past_defaults": "Past defaults", "mean": "Default rate", "size": "Loans"}),
        rows=len(df), n_features=df.shape[1] - 1, n_numeric=df.shape[1] - 2, missing=int(df.isna().sum().sum()),
        dupes=int(df.duplicated().sum()), default_rate=float(y.mean()),
        n_train=len(Xtr), n_test=len(Xte), dr_train=float(ytr.mean()), dr_test=float(yte.mean()),
        test_def=int(yte.sum()), test_loans=float(lte.sum()), test_def_principal=float(lte[yte == 1].sum()),
        leak_target="default" not in X.columns,
        leak_disjoint=set(Xtr.index).isdisjoint(Xte.index),
        leak_aligned=bool((Xtr.index == ltr.index).all() and (Xte.index == lte.index).all()),
        approve_acc=float(1 - yte.mean()), reject_acc=float(yte.mean()),
        reject_money=float(money_saved(yte, np.ones(len(yte), dtype=int), lte)),
        feature_cols=list(X.columns),
    )


@st.cache_data
def portfolio_stats(fname):
    """Score one saved model on the held-out test split of loans.csv (same split as the notebook)."""
    model, cols = load(fname)
    df = pd.read_csv(HERE / "loans.csv")
    X, y = encode(df.drop(columns="default"), cols), df["default"]
    _, Xte, _, yte, _, lo_te = train_test_split(X, y, df["loan_amount"], test_size=0.25, random_state=1, stratify=y)
    pred, yt, ln = model.predict(Xte), yte.values, lo_te.values
    n, approved, td = len(yt), int((pred == 0).sum()), int((yt == 1).sum())
    caught = int(((pred == 1) & (yt == 1)).sum())
    return dict(
        n=n, total_def=td, n_good=int((yt == 0).sum()), acc=float((pred == yt).mean()), recall=caught / td,
        saved=float(money_saved(yt, pred, ln)),
        caught_loans=float(ln[(pred == 1) & (yt == 1)].sum()), lost_interest=float(0.20 * ln[(pred == 1) & (yt == 0)].sum()),
        missed_principal=float(ln[(pred == 0) & (yt == 1)].sum()),
        caught=caught, wrong=int(((pred == 1) & (yt == 0)).sum()),
        approved=approved, approval_rate=approved / n, missed=int(((pred == 0) & (yt == 1)).sum()),
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
.qd-chip{display:inline-block;margin:.8rem .4rem 0 0;max-width:100%;overflow-wrap:anywhere;background:rgba(255,255,255,.18);
  border:1px solid rgba(255,255,255,.4);border-radius:999px;padding:.2rem .85rem;font-size:.82rem}

.qd-banner{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:.8rem 1.5rem;color:#fff;
  border-radius:20px;padding:1.2rem 1.5rem;margin:.5rem 0 1rem;box-shadow:0 10px 26px rgba(15,23,42,.18)}
.qd-banner.reject{background:linear-gradient(110deg,#dc2626 0%,#f43f5e 45%,#f97316 100%)}
.qd-banner.approve{background:linear-gradient(110deg,#059669 0%,#10b981 55%,#14b8a6 100%)}
.qd-banner.info{background:linear-gradient(110deg,#4338ca 0%,#6366f1 55%,#8b5cf6 100%)}
.qd-banner .main{flex:1 1 16rem;min-width:0}
.qd-banner .verdict{font-size:clamp(1.3rem,3.6vw,1.9rem);font-weight:800;line-height:1.15}
.qd-banner .sub{margin-top:.4rem;font-size:.92rem;line-height:1.35;opacity:.95;overflow-wrap:anywhere}
.qd-banner .prob{flex:0 0 auto;text-align:right}
.qd-banner .prob b{display:block;font-size:clamp(1.8rem,5vw,2.6rem);line-height:1;white-space:nowrap}
.qd-banner .prob span{font-size:.8rem;opacity:.92}

/* the grid adapts to the width of the content area (sidebar open or closed), not the screen */
.qd-wrap{container-type:inline-size;width:100%}
.qd-grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));grid-auto-rows:1fr;gap:1rem;margin:.4rem 0 1rem}
.qd-grid.four{grid-template-columns:repeat(4,minmax(0,1fr))}
@container (max-width: 860px){ .qd-grid,.qd-grid.four{grid-template-columns:repeat(3,minmax(0,1fr))} }
@container (max-width: 540px){ .qd-grid,.qd-grid.four{grid-template-columns:repeat(2,minmax(0,1fr))} }
@container (max-width: 330px){ .qd-grid,.qd-grid.four{grid-template-columns:minmax(0,1fr)} }

.qc{container-type:inline-size;position:relative;box-sizing:border-box;display:flex;flex-direction:column;gap:.55rem;
  min-height:9.5rem;padding:1rem 1.1rem;border-radius:20px;color:#fff;box-shadow:0 8px 22px rgba(15,23,42,.14)}
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

/* payoff table (Overview) */
.qd-pay table{width:100%;border-collapse:separate;border-spacing:6px;table-layout:fixed}
.qd-pay th{font-size:.82rem;font-weight:600;text-align:center;color:#334155;padding:.2rem}
.qd-pay td{border-radius:14px;padding:.75rem .6rem;text-align:center;font-weight:700;line-height:1.3;overflow-wrap:anywhere}
.qd-pay td small{display:block;font-weight:500;opacity:.85;margin-top:.15rem}
.qd-pay .ok{background:#dcfce7;color:#166534}.qd-pay .meh{background:#fef9c3;color:#854d0e}.qd-pay .bad{background:#fee2e2;color:#991b1b}

/* "what is this ensemble" chips */
.qd-tag{display:inline-block;padding:.12rem .65rem;border-radius:999px;font-size:.78rem;font-weight:700;color:#fff;margin:0 .35rem .35rem 0}
.qd-check{display:flex;align-items:flex-start;gap:.5rem;margin:.25rem 0;line-height:1.35}
</style>
"""


def metric_card(color, label, value_html, pill_text, pill_kind, tip):
    pill = f'<div class="qc-pill {pill_kind}">{escape(pill_text)}</div>' if pill_text else ""
    return (f'<div class="qc qc-{color}"><div class="qc-top"><span class="qc-label">{escape(label)}</span>'
            f'<span class="qc-help" tabindex="0" data-tip="{escape(tip, quote=True)}">?</span></div>'
            f'<div class="qc-value">{value_html}</div>{pill}</div>')


def card_grid(cards, four=False):
    st.html(f'<div class="qd-wrap"><div class="qd-grid{" four" if four else ""}">{"".join(cards)}</div></div>')


def unit_value(unit, value):
    return f'<span class="qc-unit">{escape(unit)}</span>{escape(value)}'


def delta_pill(delta, fmt, good_when, unit=""):
    """Return (text, kind). good_when: 'up' | 'down' | None (neutral)."""
    if abs(delta) < 1e-9:
        return "no change", "neutral"
    arrow = "↑" if delta > 0 else "↓"
    kind = "neutral" if good_when is None else ("good" if (delta > 0) == (good_when == "up") else "bad")
    return f"{arrow} {fmt(delta)}{unit}", kind


def banner(kind, title, sub, big=None, big_sub=""):
    right = f'<div class="prob"><b>{escape(big)}</b><span>{escape(big_sub)}</span></div>' if big else ""
    st.html(f'<div class="qd-banner {kind}"><div class="main"><div class="verdict">{title}</div>'
            f'<div class="sub">{sub}</div></div>{right}</div>')


def checks(items):
    st.html("".join(f'<div class="qd-check"><span>{"✅" if ok else "❌"}</span><span>{text}</span></div>' for ok, text in items))


# ───────────────────────── charts ─────────────────────────
def hbar(df, col, title, label_fmt, domain=None, champion=None, ref=None, ref_color="#dc2626", height=None, x_title=None, x_format=None):
    """Horizontal bars, one per model, coloured by ensemble family, ordered by money saved. Optional dashed reference line."""
    order = df.sort_values("Money_saved", ascending=False)["Model"].tolist()
    d = df.copy()
    d["label"] = d[col].map(label_fmt)
    base = alt.Chart(d).encode(y=alt.Y("Model:N", sort=order, title=None, axis=alt.Axis(labelLimit=170, labelFontSize=12)))
    bars = base.mark_bar(cornerRadiusEnd=5, stroke="#16a34a").encode(
        x=alt.X(f"{col}:Q", title=x_title, scale=alt.Scale(domain=domain) if domain else alt.Scale(),
                axis=alt.Axis(format=x_format) if x_format else alt.Axis()),
        color=alt.Color("Family:N", scale=alt.Scale(domain=list(FAM_COL), range=list(FAM_COL.values())), legend=None),
        strokeWidth=alt.condition(alt.datum.Model == champion, alt.value(3), alt.value(0)) if champion else alt.value(0),
        tooltip=["Model", "Family", alt.Tooltip("label:N", title=title)])
    text = base.mark_text(align="left", dx=5, fontSize=11, color="#334155").encode(x=alt.X(f"{col}:Q"), text="label:N")
    layers = bars + text
    if ref is not None:
        layers = layers + alt.Chart(pd.DataFrame({"x": [ref]})).mark_rule(color=ref_color, strokeDash=[6, 4], strokeWidth=2).encode(x="x:Q")
    return layers.properties(title=alt.TitleParams(title, anchor="start", fontSize=13), height=height or max(120, 29 * len(d) + 26))


def family_legend():
    st.html("".join(f'<span class="qd-tag" style="background:{c}">{f}</span>' for f, c in FAM_COL.items()))


def prep(lb):
    d = lb.copy()
    d["Money_M"] = d["Money_saved"] / 1e6
    return d


# ───────────────────────── static facts about how each model was built (from the notebook) ─────────────────────────
SINGLES = {
    "Decision Tree": ("🌳", "Shallow tree, max_depth = 4", "Asks a handful of yes/no questions about the applicant — easy to read, but too simple to catch many defaulters."),
    "Logistic Regression": ("📈", "Scaled inputs, linear score", "Adds up weighted evidence into one risk score. Transparent, but it can only draw a straight boundary."),
    "KNN": ("👥", "k = 15 nearest neighbours, scaled inputs", "Looks at the 15 most similar past borrowers and copies their majority outcome."),
}
ENSEMBLES = [
    dict(title="Voting", icon="🗳️", color=FAM_COL["Voting"], models=["Voting"],
         idea="Ask several **different** experts and average their votes.",
         build="**Soft voting** over the three single models (Decision Tree + Logistic Regression + KNN): their predicted default probabilities are averaged.",
         fights="Mixes different kinds of mistakes — but if the members are weak, the committee is weak too."),
    dict(title="Bagging", icon="🎒", color=FAM_COL["Bagging"], models=["Bagging (Trees)", "Random Forest"],
         idea="Train many copies of the **same** model on different random slices of the data, then average them.",
         build="**BaggingClassifier**: 200 fully grown trees on bootstrap samples. **Random Forest**: 300 trees, each split sees only a random subset of features.",
         fights="Bagging fights **variance** — an unstable, fully grown tree becomes steady once averaged."),
    dict(title="Boosting", icon="🚀", color=FAM_COL["Boosting"], models=["AdaBoost", "Gradient Boosting", "XGBoost"],
         idea="Models work **in sequence**; each one fixes the previous one's mistakes.",
         build="**AdaBoost**: 200 stumps, learning rate 0.5. **Gradient Boosting**: 200 trees, depth 3, rate 0.05. **XGBoost**: 300 trees, depth 4, rate 0.05, 80% row/column sampling.",
         fights="Boosting fights **bias** — many small, weak trees add up to a strong model."),
    dict(title="Stacking", icon="🧩", color=FAM_COL["Stacking"], models=["Stacking"],
         idea="A **manager** model learns how much to trust each base model.",
         build="Base models: **Random Forest + Gradient Boosting + XGBoost**. Meta-model: logistic regression trained on their out-of-fold predictions (cv = 5, so it never sees a base model's predictions on its own training rows). "
               "Learned trust weights: Random Forest 2.38 · Gradient Boosting 2.53 · XGBoost 1.13.",
         fights="Can beat each base model — but costs more to train (15.9 s vs 1.1 s for Gradient Boosting)."),
]


# ───────────────────────── sidebar ─────────────────────────
st.sidebar.header("⚙️ Model for the live demo")
choice = st.sidebar.radio("Which model should decide?", list(MODEL_FILES), index=0)
st.sidebar.caption(
    "Used in the **🧪 Try it live** tab.\n\n"
    "**Cost-matched 1/6 rule:** reject when P(default) ≥ 1/6, because losing a loan costs 5× the "
    "interest lost by turning away a good customer (0.20 / 1.20 = 1/6).\n\n"
    "**Default 0.5 rule:** reject only when P(default) ≥ 50 % — the setting scored on the leaderboard.\n\n"
    "The **Verdict** tab always compares both.")
fname = MODEL_FILES[choice]
model, COLUMNS = load(fname)
threshold = float(getattr(model, "threshold", 0.5))

# ───────────────────────── shared data ─────────────────────────
D = data_overview()
lb = load_csv("leaderboard.csv")
lb_cost = load_csv("leaderboard_cost_aware_threshold.csv")
cv = load_csv("cv_robustness_table.csv")
S_champ, S_rec = portfolio_stats(CHAMPION_FILE), portfolio_stats(COSTAWARE_FILE)

if lb is not None:
    lbm = prep(lb)
    champ_name = lb.sort_values("Money_saved", ascending=False).iloc[0]["Model"]
    singles_lb = lb[lb.Family == "Single"].sort_values("Money_saved", ascending=False)
    ens_lb = lb[lb.Family != "Single"].sort_values("Money_saved", ascending=False)
    best_single, best_ens = singles_lb.iloc[0], ens_lb.iloc[0]
    acc_spread = (lb.Accuracy.max() - lb.Accuracy.min()) * 100
    money_ratio = lb.Money_saved.max() / lb.Money_saved.min()

st.html(CSS + f"""
<div class="qd-hero"><h1>💰 QuickLoan — Risk Dashboard</h1>
<p>The Great Model Bake-Off: from weak single models to four ensemble committees — judged by <b>money saved</b>, not just accuracy.</p>
<span class="qd-chip">6,000 loans · 10 models · 5 steps</span>
<span class="qd-chip">Recommended: Gradient Boosting · {aed(S_rec["saved"])} saved</span></div>""")

tabs = st.tabs(["🏠 Overview", "1 · Data & Split", "2 · Weak Singles", "3 · Ensembles", "4 · Leaderboard", "5 · Verdict", "🧪 Try it live"])
t_over, t1, t2, t3, t4, t5, t_try = tabs

# ═════════════════════════════════════ OVERVIEW ═════════════════════════════════════
with t_over:
    st.header("The business problem")
    c_txt, c_pay = st.columns([1, 1.15], gap="large")
    with c_txt:
        st.markdown(
            "**QuickLoan** is a UAE fintech that gives small personal loans. For every applicant it must decide: **approve or reject?** "
            "Your job on the risk team: predict who will **default**, and recommend **which model to deploy** — backed by money, not just accuracy.\n\n"
            "**Not all mistakes cost the same:**\n"
            "- Approving someone who then defaults → QuickLoan loses the **whole loan** 💸\n"
            "- Rejecting someone who would have repaid → QuickLoan only loses the **~20% interest** 😕\n\n"
            "So *catching defaulters* matters far more than overall accuracy.")
    with c_pay:
        st.html("""<div class="qd-pay"><table>
<tr><th></th><th>Model <b>approves</b><br>(predict 0)</th><th>Model <b>rejects</b><br>(predict 1)</th></tr>
<tr><th>Actually<br><b>repays</b></th><td class="ok">+0.20 × loan<small>interest earned</small></td><td class="meh">0<small>lost that interest</small></td></tr>
<tr><th>Actually<br><b>defaults</b></th><td class="bad">−1.00 × loan<small>principal lost 💸</small></td><td class="ok">0<small>disaster avoided ✅</small></td></tr>
</table></div>""")
    st.subheader("The money metric — “money saved”")
    st.markdown("How much better a model does than **approving everyone** (which saves AED 0 by definition). Every model in this dashboard is scored with this exact function on the same test set:")
    st.code("""def money_saved(y_true, y_pred, loan_amounts, interest=0.20):
    caught_defaulter = (y_pred == 1) & (y_true == 1)   # correctly rejected a defaulter -> loan SAVED
    wrongly_rejected = (y_pred == 1) & (y_true == 0)   # rejected a good borrower   -> interest LOST
    return loan[caught_defaulter].sum() - interest * loan[wrongly_rejected].sum()""", language="python")

    st.header("The five steps at a glance")
    if lb is not None:
        card_grid([
            metric_card("blue", "Step 1 · Data & split", unit_value("loans", f'{D["rows"]:,}'), f'{D["default_rate"]*100:.2f}% defaulters', "neutral",
                        f'Loaded, one-hot encoded and split {D["n_train"]:,} train / {D["n_test"]:,} test with no leakage. Open the “1 · Data & Split” tab.'),
            metric_card("purple", "Step 2 · Weak singles", unit_value("AED", short(best_single.Money_saved)), f"best: {best_single.Model}", "neutral",
                        f'Three single models. The best one saves {aed(best_single.Money_saved)} but catches only {best_single.Recall*100:.0f}% of defaulters.'),
            metric_card("orange", "Step 3 · Four ensembles", unit_value("AED", short(best_ens.Money_saved)), f"best: {best_ens.Model}", "good",
                        f'Voting, Bagging, Boosting and Stacking. The best committee saves {aed(best_ens.Money_saved)}.'),
            metric_card("green", "Step 4 · Leaderboard", f"{money_ratio:.1f}×", f"accuracy spread only {acc_spread:.1f} pts", "neutral",
                        f'All 10 models on one test set. Money saved varies {money_ratio:.1f}× while accuracy varies just {acc_spread:.2f} percentage points.'),
            metric_card("pink", "Step 5 · Verdict", unit_value("AED", short(S_rec["saved"])), f'{(S_rec["saved"]/S_champ["saved"]-1)*100:+.0f}% with the 1/6 rule', "good",
                        f'Deploy Gradient Boosting with the cost-matched cut-off: {aed(S_rec["saved"])} saved vs approving everyone. Exact: AED {S_rec["saved"]:,.0f}.'),
        ])
    st.info("👆 Use the tabs above to walk through each step. **🧪 Try it live** lets you score a single applicant and see the model's decision.")

# ═════════════════════════════════════ STEP 1 ═════════════════════════════════════
with t1:
    st.header("Step 1 · Load & split")
    st.markdown("Load `loans.csv`, check it is clean, encode the one categorical column, and split into train and test — keeping the test **loan amounts** aside for the money calculation.")
    card_grid([
        metric_card("blue", "📄 Loans (rows)", f'{D["rows"]:,}', f'{D["n_features"]} input columns', "neutral", f'{D["rows"]:,} loans, {D["n_features"]} input columns plus the target column `default`.'),
        metric_card("green", "🧼 Missing values", f'{D["missing"]}', f'{D["dupes"]} duplicate rows', "good", "The data is clean and ready to use: no missing values and no duplicate rows."),
        metric_card("pink", "⚠️ Default rate", f'{D["default_rate"]*100:.2f}%', "minority class", "neutral",
                    f'Only about 1 loan in {round(1/D["default_rate"])} defaults. That imbalance is why accuracy turns out to be a misleading scoreboard.'),
        metric_card("purple", "🏋️ Training set", f'{D["n_train"]:,}', f'{D["dr_train"]*100:.2f}% defaulters', "neutral", "75% of the loans, used to fit every model."),
        metric_card("orange", "🧪 Test set", f'{D["n_test"]:,}', f'{D["test_def"]} defaulters · {D["dr_test"]*100:.2f}%', "neutral",
                    f'25% of the loans, never seen during training. Total loan value {aed(D["test_loans"])}, of which {aed(D["test_def_principal"])} belongs to defaulters.'),
    ])

    st.subheader("1.1 · The data")
    st.dataframe(D["df_head"], hide_index=True, width="stretch")
    with st.expander("Summary statistics of the input columns"):
        st.dataframe(D["describe"], width="stretch")

    st.subheader("1.2 · Class balance — who defaults?")
    g1, g2 = st.columns(2, gap="large")
    with g1:
        bal = D["balance"]
        ch = alt.Chart(bal).mark_bar(cornerRadiusEnd=6).encode(
            y=alt.Y("Outcome:N", sort=None, title=None), x=alt.X("Loans:Q", title="Number of loans"),
            color=alt.Color("Outcome:N", scale=alt.Scale(domain=list(bal.Outcome), range=["#10b981", "#f43f5e"]), legend=None),
            tooltip=["Outcome", "Loans", alt.Tooltip("Share:Q", format=".2%")])
        txt = alt.Chart(bal).mark_text(align="left", dx=5, fontSize=12).encode(
            y=alt.Y("Outcome:N", sort=None), x="Loans:Q", text=alt.Text("Share:Q", format=".2%"))
        st.altair_chart((ch + txt).properties(title=alt.TitleParams("Target balance (default = 1)", anchor="start"), height=150), width="stretch")
    with g2:
        bp = D["by_past"]
        ch2 = alt.Chart(bp).mark_bar(cornerRadiusEnd=6, color="#f59e0b").encode(
            x=alt.X("Past defaults:N", sort=None, title="Number of past defaults", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Default rate:Q", axis=alt.Axis(format="%"), title="Default rate"),
            tooltip=["Past defaults", alt.Tooltip("Default rate:Q", format=".1%"), "Loans"])
        txt2 = alt.Chart(bp).mark_text(dy=-8, fontSize=12).encode(x=alt.X("Past defaults:N", sort=None), y="Default rate:Q", text=alt.Text("Default rate:Q", format=".0%"))
        st.altair_chart((ch2 + txt2).properties(title=alt.TitleParams("Default rate by number of past defaults", anchor="start"), height=150), width="stretch")

    st.subheader("1.3 · Encoding")
    e1, e2 = st.columns([1.25, 1], gap="large")
    with e1:
        st.markdown("`home_ownership` is the only categorical column, so it is **one-hot encoded** into three 0/1 columns. "
                    f"The model therefore trains on **{len(D['feature_cols'])} numeric features**:")
        st.html("".join(f'<span class="qd-tag" style="background:{"#f59e0b" if c.startswith("home_ownership_") else "#6366f1"}">{escape(c)}</span>' for c in D["feature_cols"]))
    with e2:
        st.dataframe(D["encode_example"], hide_index=True, width="stretch")
        st.caption("Before → after: the first five rows.")

    st.subheader("1.4 · Train / test split — and the no-leakage checks")
    sp1, sp2 = st.columns([1, 1.2], gap="large")
    with sp1:
        st.markdown(f"**Stratified 75 / 25 split** (`test_size=0.25`, `random_state=1`, `stratify=y`): "
                    f"the default rate is **{D['dr_train']*100:.2f}%** in train and **{D['dr_test']*100:.2f}%** in test, so both sets are representative. "
                    "The test **loan amounts** are carried through the split and kept aside for the money calculation.")
    with sp2:
        checks([
            (D["leak_target"], "The target `default` is **not** among the features."),
            (D["leak_disjoint"], "No row appears in both the training and the test set."),
            (D["leak_aligned"], "Test loan amounts are aligned row-for-row with the test features."),
            (True, "Feature scaling (needed by Logistic Regression and KNN) lives **inside** each model's pipeline, so it is fitted on training rows only."),
        ])

# ═════════════════════════════════════ STEP 2 ═════════════════════════════════════
with t2:
    st.header("Step 2 · Feel the pain — weak single models")
    st.markdown("Three different kinds of single learner, each scored on the **same test set** with accuracy, **recall on defaulters** (the share of defaulters caught) and **money saved**.")
    if lb is None:
        st.warning("leaderboard.csv is missing from this folder.")
    else:
        card_grid([
            metric_card("blue", "Approve everyone", f'{D["approve_acc"]*100:.1f}%', "accuracy · saves AED 0", "neutral",
                        f'The naive strategy: approve every applicant. It is already {D["approve_acc"]*100:.2f}% accurate (because only {D["dr_test"]*100:.1f}% default), yet it saves nothing and catches no defaulter.'),
            metric_card("pink", "Reject everyone", f'{D["reject_acc"]*100:.1f}%', f'accuracy · {aed(D["reject_money"], signed=True)}', "bad",
                        f'Rejecting everyone catches all defaulters but throws away every good customer. Net effect: {aed(D["reject_money"])}. Exact: AED {D["reject_money"]:,.0f}.'),
        ] + [
            metric_card(c, f"{SINGLES[r.Model][0]} {r.Model}", unit_value("AED", short(r.Money_saved)),
                        f"recall {r.Recall*100:.1f}% · caught {int(r.Defaulters_caught)}/{D['test_def']}", "neutral",
                        f"{SINGLES[r.Model][1]}. Accuracy {r.Accuracy*100:.2f}%. Exact money saved: AED {r.Money_saved:,.0f}.")
            for c, r in zip(["purple", "orange", "green"], [singles_lb[singles_lb.Model == m].iloc[0] for m in SINGLES])
        ])
        st.subheader("2.1 · The three single models")
        cols = st.columns(3, gap="medium")
        for col, (name, (icon, how, what)) in zip(cols, SINGLES.items()):
            r = singles_lb[singles_lb.Model == name].iloc[0]
            with col, st.container(border=True):
                st.markdown(f"**{icon} {name}**")
                st.caption(how)
                st.write(what)
                st.markdown(f"Accuracy **{r.Accuracy*100:.2f}%** · Recall **{r.Recall*100:.1f}%**  \n"
                            f"Caught **{int(r.Defaulters_caught)}** of {D['test_def']} · missed **{int(r.Defaulters_missed)}**  \n"
                            f"Good customers rejected **{int(r.Good_rejected)}**  \n"
                            f"Money saved **{aed(r.Money_saved, keep_zeros=True)}**")
        st.subheader("2.2 · Results")
        sd = prep(singles_lb)
        c_a, c_b, c_c = st.columns(3, gap="medium")
        with c_a:
            st.altair_chart(hbar(sd, "Accuracy", "Accuracy", lambda v: f"{v:.1%}", domain=[0, 1], ref=D["approve_acc"], height=130, x_format="%"), width="stretch")
        with c_b:
            st.altair_chart(hbar(sd, "Recall", "Recall on defaulters", lambda v: f"{v:.1%}", domain=[0, 0.5], height=130, x_format="%"), width="stretch")
        with c_c:
            st.altair_chart(hbar(sd, "Money_M", "Money saved (AED M)", lambda v: f"{v:.2f}M", domain=[0, sd.Money_M.max() * 1.3], height=130), width="stretch")
        st.caption(f"Red dashed line = accuracy of “approve everyone” ({D['approve_acc']*100:.2f}%).")
        worst = singles_lb.iloc[-1]
        st.info(
            f"**What to notice.** On accuracy all three look respectable ({singles_lb.Accuracy.min()*100:.1f}–{singles_lb.Accuracy.max()*100:.1f}%) — but simply approving "
            f"everyone already scores {D['approve_acc']*100:.2f}%. The real weakness is **recall**: the best single model, the {best_single.Model}, catches only "
            f"{int(best_single.Defaulters_caught)} of the {D['test_def']} defaulters ({best_single.Recall*100:.1f}%), and {worst.Model} only {int(worst.Defaulters_caught)} ({worst.Recall*100:.1f}%). "
            f"Every missed defaulter is a whole loan gone, so money saved ranges only from **{aed(worst.Money_saved)} to {aed(best_single.Money_saved)}**. "
            "This is the pain the ensembles are meant to relieve.")

# ═════════════════════════════════════ STEP 3 ═════════════════════════════════════
with t3:
    st.header("Step 3 · Build the four ensembles")
    st.markdown("*One expert can be wrong — a committee wins.* Each ensemble below is a different way to run the committee. "
                "Hyper-parameters are fixed, sensible values — **nothing was tuned on the test set**.")
    if lb is None:
        st.warning("leaderboard.csv is missing from this folder.")
    else:
        for ens in ENSEMBLES:
            with st.container(border=True):
                head, body = st.columns([1, 1.35], gap="large")
                with head:
                    st.markdown(f"### {ens['icon']} {ens['title']}")
                    st.markdown(ens["idea"])
                    st.markdown(f"**How it was built.** {ens['build']}")
                    st.caption(ens["fights"])
                with body:
                    rows = []
                    for m in ens["models"]:
                        r = lb[lb.Model == m].iloc[0]
                        rows.append({"Model": m, "Accuracy": r.Accuracy, "Recall": r.Recall,
                                     "Caught": f"{int(r.Defaulters_caught)}/{D['test_def']}",
                                     "Money saved": aed(r.Money_saved, keep_zeros=True),
                                     f"vs best single ({best_single.Model})": aed(r.Money_saved - best_single.Money_saved, signed=True, keep_zeros=True)})
                    st.dataframe(pd.DataFrame(rows).style.format({"Accuracy": "{:.2%}", "Recall": "{:.1%}"}), hide_index=True, width="stretch")
                    sub = lbm[lbm.Model.isin(ens["models"])]
                    st.altair_chart(hbar(sub, "Money_M", "Money saved (AED M)", lambda v: f"{v:.2f}M", domain=[0, lbm.Money_M.max() * 1.3],
                                         ref=best_single.Money_saved / 1e6, ref_color="#64748b", height=max(70, 34 * len(sub) + 26)), width="stretch")
        st.subheader("All four committees vs the single models")
        family_legend()
        st.altair_chart(hbar(lbm, "Money_M", "Money saved (AED M) — grey dashed line = best single model", lambda v: f"{v:.2f}M",
                             domain=[0, lbm.Money_M.max() * 1.25], champion=champ_name, ref=best_single.Money_saved / 1e6, ref_color="#64748b"), width="stretch")
        below = [m for m in ens_lb.Model if lb[lb.Model == m].iloc[0].Money_saved < best_single.Money_saved]
        stack_r, gb_r = lb[lb.Model == "Stacking"].iloc[0], lb[lb.Model == "Gradient Boosting"].iloc[0]
        st.info(
            f"**What to notice.**\n"
            f"- **The best committee wins:** {best_ens.Model} saves {aed(best_ens.Money_saved)} — {aed(best_ens.Money_saved - best_single.Money_saved)} more than the best single model.\n"
            f"- **Not every ensemble wins:** {' and '.join(below)} fall *below* the single {best_single.Model}. Voting averages in two weak, low-recall members; "
            "AdaBoost's default decision stumps stay too simple to flag the rare defaulter at a 0.5 cut-off.\n"
            "- **Bagging vs boosting:** bagging stabilises an unstable tree and beats the single tree; boosting's best members do better still.\n"
            f"- **Stacking added nothing here:** at {aed(stack_r.Money_saved)} it sits below its own best base model (Gradient Boosting, {aed(gb_r.Money_saved)}) and trains ~14× slower.")

# ═════════════════════════════════════ STEP 4 ═════════════════════════════════════
with t4:
    st.header("Step 4 · The Leaderboard 🏆")
    st.markdown("**Every** model, one table, the **same test set**: Accuracy · Recall (defaulters caught) · Money saved.")
    if lb is None:
        st.warning("leaderboard.csv is missing from this folder.")
    else:
        banner("info", f"🏆 Champion on money saved: {champ_name}",
               f"Catches {int(best_ens.Defaulters_caught)} of {D['test_def']} defaulters while rejecting {int(best_ens.Good_rejected)} good customers.",
               aed(best_ens.Money_saved), "saved vs approving everyone")
        st.subheader("4.1 · The table")
        show = lb[["Model", "Family", "Accuracy", "Recall", "Money_saved", "Defaulters_caught", "Good_rejected", "Rank (accuracy)", "Rank (money)"]].copy()
        show["Money_saved"] = show["Money_saved"].map(lambda v: aed(v, keep_zeros=True))
        show = show.rename(columns={"Money_saved": "Money saved", "Defaulters_caught": "Defaulters caught", "Good_rejected": "Good rejected"})
        st.dataframe(show.style.format({"Accuracy": "{:.2%}", "Recall": "{:.1%}"})
                     .apply(lambda r: ["background-color:#dcfce7"] * len(r) if r.name == 0 else [""] * len(r), axis=1),
                     hide_index=True, width="stretch")
        st.caption("Scored with the default 0.5 decision rule. Green row = champion. Ranks show how differently accuracy and money order the models.")

        st.subheader("4.2 · Three scoreboards, same models")
        family_legend()
        a1, a2, a3 = st.columns(3, gap="medium")
        with a1:
            st.altair_chart(hbar(lbm, "Accuracy", "Accuracy", lambda v: f"{v:.1%}", domain=[0, 1], champion=champ_name, ref=D["approve_acc"], x_format="%"), width="stretch")
        with a2:
            st.altair_chart(hbar(lbm, "Recall", "Recall on defaulters", lambda v: f"{v:.1%}", domain=[0, 0.5], champion=champ_name, x_format="%"), width="stretch")
        with a3:
            st.altair_chart(hbar(lbm, "Money_M", "Money saved (AED M)", lambda v: f"{v:.2f}M", domain=[0, lbm.Money_M.max() * 1.3], champion=champ_name), width="stretch")
        st.caption(f"Green outline = champion. Red dashed line = accuracy of “approve everyone” ({D['approve_acc']*100:.2f}%), which saves AED 0.")

        st.subheader("4.3 · How much does accuracy vary — versus money?")
        dt = lb[lb.Model == best_single.Model].iloc[0]
        card_grid([
            metric_card("blue", "📏 Accuracy spread", f"{acc_spread:.2f} pts", f"{lb.Accuracy.min()*100:.1f}% → {lb.Accuracy.max()*100:.1f}%", "neutral",
                        "Best minus worst accuracy across the ten models: barely any difference."),
            metric_card("green", "💰 Money spread", f"{money_ratio:.1f}×", f"{aed(lb.Money_saved.min())} → {aed(lb.Money_saved.max())}", "good",
                        f"Best money saved divided by worst. Exact: AED {lb.Money_saved.max():,.0f} vs AED {lb.Money_saved.min():,.0f}."),
            metric_card("pink", "🙈 Approve-everyone", f'{D["approve_acc"]*100:.1f}%', "accuracy · saves AED 0", "bad",
                        "A model that does nothing useful already matches the accuracy of the real models — accuracy alone cannot tell them apart."),
            metric_card("orange", f"🌳 {best_single.Model}", f'{int(dt["Rank (accuracy)"])} → {int(dt["Rank (money)"])}', "accuracy rank → money rank", "bad",
                        f"The best single model ranks {int(dt['Rank (accuracy)'])} on accuracy but {int(dt['Rank (money)'])} on money saved."),
            metric_card("purple", "🏅 Ensemble edge", unit_value("AED", short(best_ens.Money_saved - best_single.Money_saved)), f"{best_ens.Model} vs {best_single.Model}", "good",
                        "Extra money saved by the best ensemble over the best single model."),
        ])
        sc_pts = lbm.assign(AccuracyPct=lbm.Accuracy)
        pts = alt.Chart(sc_pts).mark_circle(size=190, opacity=0.92).encode(
            x=alt.X("AccuracyPct:Q", scale=alt.Scale(domain=[0.87, 0.90]), axis=alt.Axis(format=".1%"), title="Accuracy (axis zoomed to 87–90%)"),
            y=alt.Y("Money_M:Q", title="Money saved (AED M)", scale=alt.Scale(domain=[1.5, 5.2])),
            color=alt.Color("Family:N", scale=alt.Scale(domain=list(FAM_COL), range=list(FAM_COL.values())), legend=None),
            tooltip=["Model", "Family", alt.Tooltip("Accuracy:Q", format=".2%"), alt.Tooltip("Money_M:Q", format=".2f", title="Money (AED M)")])
        lab = alt.Chart(sc_pts[sc_pts.Model.isin([champ_name, best_single.Model, "KNN"])]).mark_text(align="left", dx=11, fontSize=12, color="#334155").encode(
            x="AccuracyPct:Q", y="Money_M:Q", text="Model:N")
        rule = alt.Chart(pd.DataFrame({"x": [D["approve_acc"]]})).mark_rule(color="#dc2626", strokeDash=[6, 4], strokeWidth=2).encode(x="x:Q")
        st.altair_chart((pts + lab + rule).properties(height=320, title=alt.TitleParams(
            "Models sit in a thin accuracy band, yet money saved spreads out vertically (red line = approve everyone)", anchor="start", fontSize=13)), width="stretch")
        st.info(
            f"**Reading the leaderboard.** Accuracy spans only **{acc_spread:.2f} points**, while money saved varies **{money_ratio:.1f}×** and recall about "
            f"**{lb.Recall.max()/lb.Recall.min():.1f}×**. Three models tie at 89.33% accuracy yet save {aed(lb[lb.Model=='XGBoost'].iloc[0].Money_saved)}, "
            f"{aed(lb[lb.Model=='Stacking'].iloc[0].Money_saved)} and {aed(lb[lb.Model=='Random Forest'].iloc[0].Money_saved)}. "
            "Accuracy and money move in the same general direction here, but accuracy squeezes a big business gap into a rounding error. "
            "**Recall on defaulters is the metric that tracks money.**\n\n"
            f"**Which model I would deploy:** {champ_name} — it tops the leaderboard on money saved ({aed(lb.Money_saved.max())}). Step 5 adds the cost-matched cut-off and the full recommendation.")

        st.subheader("4.4 · Is the champion real, or just lucky?")
        st.markdown("One test split has only 188 defaulters, so I re-scored every model with **5-fold cross-validation repeated 3 times (15 folds)**, retraining each model from scratch.")
        if cv is not None:
            gbn = "Gradient Boosting"
            col_05, col_16 = f"Folds {gbn} beats it (@0.5)", f"Folds {gbn} beats it (@1/6)"
            cvs = cv.copy()
            cvs["CV money @0.5 (mean)"] = cvs["CV money @0.5 (mean)"].map(lambda v: aed(v, keep_zeros=True))
            cvs["CV money @0.5 (std)"] = cvs["CV money @0.5 (std)"].map(lambda v: aed(v, keep_zeros=True))
            cvs["CV money @1/6 (mean)"] = cvs["CV money @1/6 (mean)"].map(lambda v: aed(v, keep_zeros=True))
            for cname in (col_05, col_16):
                cvs[cname] = cvs[cname].map(lambda v: "—" if pd.isna(v) else f"{int(v)} / 15")
            st.dataframe(cvs.drop(columns=["Position"]).rename(columns={col_05: "GB wins (0.5 rule)", col_16: "GB wins (1/6 rule)"}), hide_index=True, width="stretch")
            all15 = [m for m, v in zip(cv.Model, cv[col_05]) if v == 15]
            st.info(
                f"- **Solid:** Gradient Boosting is #1 in cross-validation too and beats {', '.join(all15)} in **all 15 folds** (0.5 rule).\n"
                f"- **Not solid:** the top few are close — GB beats Bagging in only {int(cv[cv.Model=='Bagging (Trees)'][col_05].iloc[0])}/15 folds, XGBoost {int(cv[cv.Model=='XGBoost'][col_05].iloc[0])}/15 and Stacking "
                f"{int(cv[cv.Model=='Stacking'][col_05].iloc[0])}/15; the per-fold standard deviation (≈ {aed(cv['CV money @0.5 (std)'].mean())}) is as large as those gaps, "
                "so a single 188-defaulter test set cannot reliably separate neighbours.")
        if (HERE / "cv_money_chart.png").exists():
            st.image(str(HERE / "cv_money_chart.png"), width="stretch")
        if lb_cost is not None:
            with st.expander("A free improvement: align the cut-off with the payoff table (cost-matched 1/6 rule)"):
                st.markdown("By default a classifier rejects when P(default) ≥ 0.5, which assumes both mistakes cost the same. From the payoff table, rejecting an applicant with default probability *p* is worth it when "
                            "**p · L > (1 − p) · 0.20 · L ⟺ p > 0.20 / 1.20 = 1/6**. This cut-off comes from the business costs — it is **not tuned on the test set**. Same models, same test set, new cut-off:")
                lc = lb_cost.copy()
                lc["Money @1/6 (test)"] = lc["Money_saved"].map(lambda v: aed(v, keep_zeros=True))
                lc["Money @0.5 (test)"] = lc["Money_at_0_5"].map(lambda v: aed(v, keep_zeros=True))
                lc["CV money @1/6"] = lc["CV_money_1_6"].map(lambda v: aed(v, keep_zeros=True))
                st.dataframe(lc[["Model", "Family", "Accuracy", "Recall", "Money @1/6 (test)", "Money @0.5 (test)", "CV money @1/6"]].style.format({"Accuracy": "{:.2%}", "Recall": "{:.1%}"}),
                             hide_index=True, width="stretch")
                st.caption("AdaBoost's probabilities are poorly spread, so at 1/6 it rejects everybody (accuracy 12.5%, the same as “reject everyone”). "
                           "Logistic Regression tops this single test split, but cross-validation puts it clearly behind Gradient Boosting — that split result was largely luck.")

# ═════════════════════════════════════ STEP 5 ═════════════════════════════════════
with t5:
    st.header("Step 5 · The Verdict")
    st.markdown("*A recommendation to QuickLoan's leadership.*")
    gain, gain_pct = S_rec["saved"] - S_champ["saved"], (S_rec["saved"] / S_champ["saved"] - 1) * 100
    banner("approve", "✅ Deploy Gradient Boosting — with the cost-matched 1/6 rule",
           f'Reject when the predicted default probability is ≥ 1/6 (= interest ÷ (1 + interest)). On the {S_rec["n"]:,}-loan test set it saves '
           f'<b>{aed(S_rec["saved"])}</b> versus approving everyone ({gain_pct:+.0f}% vs the default 0.5 rule).',
           aed(S_rec["saved"]), "saved vs approving everyone")
    card_grid([
        metric_card("green", "💵 Money saved", unit_value("AED", short(S_rec["saved"])), f'{aed(gain, signed=True)} vs 0.5 rule', "good",
                    f'Net AED saved vs approving everyone. Exact: AED {S_rec["saved"]:,.0f}. Loans saved {aed(S_rec["caught_loans"])} minus interest lost {aed(S_rec["lost_interest"])}.'),
        metric_card("purple", "🎯 Defaulters caught", f'{S_rec["caught"]}/{S_rec["total_def"]}', f'recall {S_rec["recall"]*100:.1f}% (was {S_champ["recall"]*100:.1f}%)', "good",
                    "Share of defaulters rejected under the recommended rule."),
        metric_card("orange", "😕 Good customers rejected", f'{S_rec["wrong"]}', f'{S_rec["wrong"]-S_champ["wrong"]:+d} vs 0.5 rule', "bad",
                    f'Repaying customers turned away, out of {S_rec["n_good"]:,} ({S_rec["wrong"]/S_rec["n_good"]*100:.1f}%).'),
        metric_card("blue", "📍 Money per applicant", unit_value("AED", short(S_rec["saved"] / S_rec["n"])), f'across {S_rec["n"]:,} test loans', "neutral",
                    f'Average money saved for each applicant scored. Exact: AED {S_rec["saved"]/S_rec["n"]:,.0f}.'),
    ], four=True)

    st.subheader("5.1 · Money saved vs just approving everyone")
    cmp = pd.DataFrame([
        {"Strategy": "Approve everyone", "Accuracy": D["approve_acc"], "Recall": 0.0, "Defaulters caught": f'0 / {D["test_def"]}', "Good rejected": 0, "Money saved": aed(0)},
        {"Strategy": "Gradient Boosting · default 0.5 rule (bake-off champion)", "Accuracy": S_champ["acc"], "Recall": S_champ["recall"],
         "Defaulters caught": f'{S_champ["caught"]} / {S_champ["total_def"]}', "Good rejected": S_champ["wrong"], "Money saved": aed(S_champ["saved"], keep_zeros=True)},
        {"Strategy": "Gradient Boosting · cost-matched 1/6 rule (recommended)", "Accuracy": S_rec["acc"], "Recall": S_rec["recall"],
         "Defaulters caught": f'{S_rec["caught"]} / {S_rec["total_def"]}', "Good rejected": S_rec["wrong"], "Money saved": aed(S_rec["saved"], keep_zeros=True)},
    ])
    st.dataframe(cmp.style.format({"Accuracy": "{:.2%}", "Recall": "{:.1%}"})
                 .apply(lambda r: ["background-color:#dcfce7"] * len(r) if r.name == 2 else [""] * len(r), axis=1), hide_index=True, width="stretch")
    st.caption(f'Recommended rule: loans saved {aed(S_rec["caught_loans"])} − interest lost {aed(S_rec["lost_interest"])} = net {aed(S_rec["saved"])}. '
               f'It still lets {S_rec["missed"]} defaulters through ({aed(S_rec["missed_principal"])} of principal, {S_rec["missed_principal"]/D["test_def_principal"]*100:.0f}% of all defaulted principal).')

    st.subheader("5.2 · Why this model")
    why1, why2 = st.columns(2, gap="large")
    with why1:
        gb_cv = cv[cv.Model == "Gradient Boosting"].iloc[0] if cv is not None else None
        st.markdown(
            f"- **It topped the leaderboard** ({aed(S_champ['saved'])}) and has the best average in 15-fold cross-validation under both decision rules"
            + (f" ({aed(gb_cv['CV money @0.5 (mean)'])} and {aed(gb_cv['CV money @1/6 (mean)'])} per fold)." if gb_cv is not None else ".") + "\n"
            + (f"- It saves **{aed(S_champ['saved'] - best_single.Money_saved)} more** than the best single model ({best_single.Model}) — and about {aed(gain)} more again once the cut-off matches the costs.\n" if lb is not None else "")
            + ("- **XGBoost** is within noise of it (Gradient Boosting wins only "
               f"{int(cv[cv.Model=='XGBoost'][f'Folds Gradient Boosting beats it (@1/6)'].iloc[0])}–{int(cv[cv.Model=='XGBoost'][f'Folds Gradient Boosting beats it (@0.5)'].iloc[0])} of 15 folds against it) and a perfectly good alternative." if cv is not None else "- **XGBoost** is a perfectly good alternative."))
    with why2:
        st.markdown(
            "- **Stacking** is no better, yet slower and more complex.\n"
            + (f"- **Logistic Regression** tops the single test split at the 1/6 rule, but that did not survive cross-validation: Gradient Boosting beats it in "
               f"{int(cv[cv.Model=='Logistic Regression']['Folds Gradient Boosting beats it (@1/6)'].iloc[0])} of 15 folds.\n" if cv is not None else "")
            +
            "- The **1/6 rule is derived from the payoff table**, not tuned on the test set.")

    st.subheader("5.3 · Trade-offs we accept")
    tr1, tr2, tr3 = st.columns(3, gap="medium")
    with tr1, st.container(border=True):
        st.markdown("**🎯 Accuracy**")
        st.write(f'Accuracy falls about {(S_champ["acc"]-S_rec["acc"])*100:.0f} points ({S_champ["acc"]*100:.1f}% → {S_rec["acc"]*100:.1f}%) and ends up below “approve everyone” '
                 f'({D["approve_acc"]*100:.1f}%) — because accuracy ignores that a missed defaulter costs five times as much as a lost customer.')
    with tr2, st.container(border=True):
        st.markdown("**⚖️ Recall vs customer friction**")
        st.write(f'We catch {S_rec["caught"]-S_champ["caught"]} more defaulters than the default rule but turn away {S_rec["wrong"]-S_champ["wrong"]} more good customers. '
                 "At a 20% interest assumption that trade is clearly profitable.")
    with tr3, st.container(border=True):
        st.markdown("**🔍 Interpretability**")
        st.write("Boosted trees are harder to explain than a logistic regression or a single tree. Mitigation: every decision carries a default probability, and two inputs drive most of the model (chart below).")

    imp_model, imp_cols = load(CHAMPION_FILE)
    if hasattr(imp_model, "feature_importances_"):
        imp = pd.DataFrame({"Feature": imp_cols, "Importance": imp_model.feature_importances_}).sort_values("Importance", ascending=False)
        top2 = imp.head(2)
        imp_ch = alt.Chart(imp.head(8)).mark_bar(cornerRadiusEnd=5, color="#f59e0b").encode(
            y=alt.Y("Feature:N", sort=imp.head(8).Feature.tolist(), title=None), x=alt.X("Importance:Q", axis=alt.Axis(format="%"), title="Share of model importance"),
            tooltip=["Feature", alt.Tooltip("Importance:Q", format=".1%")])
        imp_txt = alt.Chart(imp.head(8)).mark_text(align="left", dx=5, fontSize=11, color="#334155").encode(
            y=alt.Y("Feature:N", sort=imp.head(8).Feature.tolist()), x="Importance:Q", text=alt.Text("Importance:Q", format=".1%"))
        st.altair_chart((imp_ch + imp_txt).properties(height=240, title=alt.TitleParams(
            f"What drives Gradient Boosting's risk score: {top2.iloc[0].Feature} and {top2.iloc[1].Feature} together ≈ {top2.Importance.sum()*100:.0f}%", anchor="start", fontSize=13)), width="stretch")
    if cv is not None:
        dt_cv = cv[cv.Model == "Decision Tree"].iloc[0]["CV money @1/6 (mean)"]
        gb_cv16 = cv[cv.Model == "Gradient Boosting"].iloc[0]["CV money @1/6 (mean)"]
        st.caption(f"If explainability is mandatory, a Decision Tree with the same 1/6 rule is the transparent fallback (cross-validated {aed(dt_cv)}, about {(1-dt_cv/gb_cv16)*100:.0f}% below Gradient Boosting).")

    st.subheader("5.4 · Caveats")
    st.warning(
        "- One dataset of 6,000 loans and a single 188-defaulter test set (per-fold noise ≈ AED 0.5 M), so close rankings among the top models are within noise.\n"
        "- No hyper-parameter tuning; a fixed 20% interest assumption — recompute the cut-off (interest ÷ (1 + interest)) if it changes.\n"
        f"- Even the recommended model lets {S_rec['missed']} defaulters through ({aed(S_rec['missed_principal'])} of principal). Monitor it after deployment.")

# ═════════════════════════════════════ TRY IT LIVE (stretch: decision + money cards) ═════════════════════════════════════
with t_try:
    st.header("🧪 Try it live — decision card & money card")
    st.caption(f"Active model: **{choice}** (change it in the sidebar).")
    st.subheader("Loan decision")
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
    banner(kind, verdict, f"Loan {aed(loan_amount)} · Income {aed(annual_income)} · Debt-to-income {debt_to_income} · "
                          f"Rule: reject if P(default) ≥ {threshold*100:.1f}%", f"{prob*100:.0f}%", "default probability")

    st.subheader("Money saved (portfolio view)")
    s = portfolio_stats(fname)
    base = S_champ                                         # comparison point: the bake-off champion (default 0.5 rule)
    is_base = fname == CHAMPION_FILE
    n, td = s["n"], s["total_def"]

    def pill(delta, fmt, good_when, unit=""):
        return ("baseline (0.5 rule)", "neutral") if is_base else delta_pill(delta, fmt, good_when, unit)

    card_grid([
        metric_card("green", "💵 Money saved vs approve-all", unit_value("AED", short(s["saved"])),
                    *pill(s["saved"] - base["saved"], lambda d: aed(d, signed=True), "up"),
                    f'Net AED saved vs approving everyone: loans of defaulters rejected minus 20% interest lost on good customers rejected. Exact: AED {s["saved"]:,.0f}.'),
        metric_card("purple", "🎯 Defaulters caught", f'{s["caught"]}/{td}',
                    *pill(100 * (s["caught"] / td - base["caught"] / td), lambda d: f"{d:+.1f}", "up", " pts"),
                    f'Defaulters rejected, out of {td} in the test split (recall {100*s["caught"]/td:.1f}%).'),
        metric_card("orange", "😕 Good borrowers rejected", f'{s["wrong"]:,}',
                    *pill(s["wrong"] - base["wrong"], lambda d: f"{d:+,.0f}", "down"),
                    f'Repaying customers turned away, out of {s["n_good"]:,}. Each costs ~20% of the loan in lost interest.'),
        metric_card("blue", "✅ Approval rate", f'{100*s["approval_rate"]:.1f}%',
                    *pill(100 * (s["approval_rate"] - base["approval_rate"]), lambda d: f"{d:+.1f}", None, " pts"),
                    f'Share of the {n:,} test applicants approved ({s["approved"]:,} loans).'),
        metric_card("pink", "⚠️ Default rate of approved", f'{100*s["default_rate_approved"]:.2f}%',
                    *pill(100 * (s["default_rate_approved"] - base["default_rate_approved"]), lambda d: f"{d:+.2f}", "down", " pts"),
                    f'Of the {s["approved"]:,} approved loans, {s["missed"]} still default.'),
    ])
    st.caption(f'Loans saved {aed(s["caught_loans"])} − interest lost {aed(s["lost_interest"])} = net {aed(s["saved"])}. '
               f'Evaluated on a held-out test split of loans.csv (25 %, {n:,} loans). '
               + ("" if is_base else "Pills compare with the bake-off champion (default 0.5 rule). ")
               + "Hover the ? for exact figures.")
