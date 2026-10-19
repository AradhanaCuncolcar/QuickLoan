# app.py — QuickLoan Risk & Compliance Console
# Run locally:  streamlit run app.py
#
# Needs (same folder): loans.csv, loan_model.joblib, loan_model_costaware.joblib,
#                      cv_robustness_table.csv, leaderboard_cost_aware_threshold.csv
# Needs packages:      streamlit, pandas, numpy, scikit-learn, plotly, shap, joblib
#
# Model files are saved by the notebook like this:
#   joblib.dump({'model': clf, 'columns': list(X_train.columns)}, 'loan_model.joblib')
#
# Page map (sidebar navigation)
#   1. Applicant Simulator & XAI  -> prediction, SHAP waterfall, adverse-action basis, human-in-the-loop override
#   2. Bias & Fairness Audit      -> approval rates by group, 80 % (four-fifths) disparate-impact test
#   3. Portfolio Cost-Awareness   -> threshold slider linked to the leaderboard CSV, risk/reward charts
#   4. Live What-If Studio        -> instant re-scoring, gauge, sensitivity, risk map, path to approval
#   5. Live Monitor               -> simulated application stream with auto-refresh, drift alerts
#   6. Batch Scoring              -> upload a CSV, score everyone, download results
#   7. Portfolio Explorer         -> lasso-selectable scatter, global importance, segment heat-maps
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split

# SHAP is imported defensively: if it is missing the rest of the app still works,
# and the XAI panel tells the user exactly what to install.
try:
    import shap

    SHAP_OK, SHAP_ERR = True, ""
except Exception as _e:  # ImportError, or a binary-compat error from numpy/numba
    shap, SHAP_OK, SHAP_ERR = None, False, str(_e)

# ──────────────────────────────────────────────────────────────────────────────
# 0 · CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent  # works no matter which folder Streamlit is launched from

MODEL_FILES = {
    "Gradient Boosting — recommended (cost-matched 1/6 rule)": "loan_model_costaware.joblib",
    "Gradient Boosting — bake-off champion (default 0.5 rule)": "loan_model.joblib",
}
DEFAULT_RULE_LABEL = list(MODEL_FILES)[1]  # used as the comparison baseline for KPI deltas

# Brand palette (matches .streamlit/config.toml)
NAVY, EMERALD, SLATE, RED, AMBER = "#0F172A", "#10B981", "#64748B", "#EF4444", "#F59E0B"
LIGHT = "#E2E8F0"
INDIGO, SKY, VIOLET, ROSE, TEAL = "#6366F1", "#0EA5E9", "#8B5CF6", "#F43F5E", "#14B8A6"
ORANGE, PINK, BLUE = "#F97316", "#EC4899", "#3B82F6"
PALETTE = [VIOLET, EMERALD, ORANGE, SKY, PINK, AMBER, INDIGO, TEAL]  # categorical colours for charts
# Hero / accent gradient per page: (colour 1, colour 2, colour 3)
PAGE_THEMES = {  # per-page accent gradient: (colour 1, colour 2, colour 3)
    "sim": ("#7C3AED", "#C026D3", "#EC4899"),      # violet → magenta → pink
    "whatif": ("#0891B2", "#06B6D4", "#8B5CF6"),   # cyan → aqua → violet
    "monitor": ("#DC2626", "#F43F5E", "#F97316"),  # red → rose → orange
    "batch": ("#4F46E5", "#6366F1", "#0EA5E9"),    # indigo → blue → sky
    "explore": ("#059669", "#14B8A6", "#3B82F6"),  # emerald → teal → blue
    "fair": ("#EA580C", "#F59E0B", "#EF4444"),     # orange → amber → red
    "cost": ("#2563EB", "#0EA5E9", "#10B981")}     # blue → sky → emerald

# Must reproduce the split the notebook used, so "held-out" numbers match the leaderboard CSVs.
SPLIT_KWARGS = dict(test_size=0.25, random_state=1)
FOUR_FIFTHS = 0.80  # ECOA / EEOC "80 % rule" for disparate impact
MIN_GROUP_N = 30  # groups smaller than this are shown but excluded from the DIR test (too noisy)
AGE_BINS = [0, 29, 39, 49, 61, 200]
AGE_LABELS = ["Under 30", "30–39", "40–49", "50–61", "62 and over"]  # 62+ has special ECOA treatment
# Columns that, if someone adds them to loans.csv, are picked up automatically by the fairness audit.
PROTECTED_CANDIDATES = ["gender", "sex", "race", "ethnicity", "marital_status", "religion",
                        "national_origin", "nationality", "disability"]

FEATURES = ["age", "annual_income", "loan_amount", "loan_term_months", "employment_years",
            "credit_history_months", "num_existing_loans", "past_defaults", "debt_to_income",
            "home_ownership"]
FEATURE_LABELS = {
    "age": "Age", "annual_income": "Annual income", "loan_amount": "Loan amount",
    "loan_term_months": "Loan term", "employment_years": "Employment length",
    "credit_history_months": "Credit history length", "num_existing_loans": "Existing loans",
    "past_defaults": "Past defaults", "debt_to_income": "Debt-to-income ratio",
    "home_ownership": "Home ownership",
}
# Plain-language principal reasons for an adverse action notice (one per model feature).
# Reg B (12 CFR 1002.9) requires reasons that are specific and truly reflect what drove the decision.
REASON_TEXT = {
    "past_defaults": "Prior loan defaults on record",
    "debt_to_income": "Requested loan amount is high relative to annual income (debt-to-income ratio)",
    "loan_amount": "Size of the requested loan amount",
    "annual_income": "Level of annual income relative to the request",
    "credit_history_months": "Limited length of credit history",
    "num_existing_loans": "Number of existing loans or credit obligations",
    "employment_years": "Limited length of employment",
    "loan_term_months": "Requested repayment term",
    "home_ownership": "Housing status",
    "age": "Applicant age (compliance review required before this can be stated as a reason)",
}

CREDITOR_NAME = "QuickLoan"
# Replace with the regulator that actually supervises you; ECOA/Reg B is US law, and QuickLoan's data is in AED.
REGULATOR_LINE = ("The federal agency that administers compliance with this law concerning this creditor "
                  "is the agency identified in the creditor's compliance policy.")
ECOA_NOTICE = ("The Federal Equal Credit Opportunity Act prohibits creditors from discriminating against credit "
               "applicants on the basis of race, color, religion, national origin, sex, marital status, age "
               "(provided the applicant has the capacity to enter into a binding contract); because all or part "
               "of the applicant's income derives from any public assistance program; or because the applicant "
               "has in good faith exercised any right under the Consumer Credit Protection Act. ")

PAGE_SIM, PAGE_WHATIF, PAGE_MONITOR, PAGE_BATCH = "🧮 Applicant Simulator & XAI", "🎛️ Live What-If Studio", "🔴 Live Monitor", "📦 Batch Scoring"
PAGE_EXPLORE, PAGE_FAIR, PAGE_COST = "🔬 Portfolio Explorer", "⚖️ Bias & Fairness Audit", "📈 Portfolio Cost-Awareness"
NAV_PAGES = [PAGE_SIM, PAGE_WHATIF, PAGE_MONITOR, PAGE_BATCH, PAGE_EXPLORE, PAGE_FAIR, PAGE_COST]
PAGE_KEYS = dict(zip(NAV_PAGES, ["sim", "whatif", "monitor", "batch", "explore", "fair", "cost"]))
SCOPE_TEST = "Held-out test split (1,500 loans — honest estimate)"
SCOPE_FULL = "Full loans.csv (6,000 loans — models saw 75 % in training)"


# ──────────────────────────────────────────────────────────────────────────────
# 1 · DATA LOADING (cached, with graceful failure)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def read_csv_safe(fname: str, required: tuple = ()) -> tuple:
    """Read a CSV that sits next to app.py. Returns (DataFrame | None, error message | None)
    so the UI can show a friendly message instead of a stack trace."""
    path = HERE / fname
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        return None, f"`{fname}` was not found next to app.py."
    except pd.errors.EmptyDataError:
        return None, f"`{fname}` is empty."
    except Exception as e:  # malformed CSV, permissions, encoding…
        return None, f"`{fname}` could not be read: {e}"
    missing = [c for c in required if c not in df.columns]
    if missing:
        return None, f"`{fname}` is missing expected column(s): {', '.join(missing)}."
    return df, None


@st.cache_resource(show_spinner=False)
def load_model_bundle(fname: str) -> dict:
    """Load {'model','columns'} saved by the notebook and unwrap the underlying Gradient Boosting
    estimator (the cost-aware file wraps it in FixedThresholdClassifier(FrozenEstimator(...)))."""
    bundle = joblib.load(HERE / fname)
    if not isinstance(bundle, dict) or "model" not in bundle or "columns" not in bundle:
        raise ValueError(f"`{fname}` must contain a dict with 'model' and 'columns'.")
    model = bundle["model"]
    return {"model": model, "columns": list(bundle["columns"]),
            "base": unwrap_estimator(model), "threshold": float(getattr(model, "threshold", 0.5))}


def unwrap_estimator(model):
    """Walk through sklearn wrappers until we reach the fitted tree ensemble (the thing SHAP needs)."""
    est = model
    for _ in range(6):
        if type(est).__name__ == "FrozenEstimator":  # it proxies attributes, so unwrap it explicitly
            est = est.estimator
            continue
        if hasattr(est, "estimators_"):  # fitted ensemble reached
            return est
        nxt = getattr(est, "estimator_", None)
        if nxt is None:
            nxt = getattr(est, "estimator", None)
        if nxt is None:
            break
        est = nxt
    return est


def encode(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """One-hot encode home_ownership and align to the exact training columns (missing dummies -> 0)."""
    return pd.get_dummies(df, columns=["home_ownership"]).reindex(columns=columns, fill_value=0).astype(float)


@st.cache_data(show_spinner=False)
def scored_portfolio(model_file: str) -> pd.DataFrame:
    """Score every loan in loans.csv with the chosen model.
    Adds: p_default (model probability), reject (1 = reject, using the model's own cut-off),
    is_test (True if the row belongs to the held-out 25 % split used in the notebook)."""
    df, err = read_csv_safe("loans.csv", required=tuple(FEATURES) + ("default",))
    if err:
        raise FileNotFoundError(err)
    bundle = load_model_bundle(model_file)
    X = encode(df.drop(columns="default"), bundle["columns"])
    out = df.copy()
    out["p_default"] = bundle["base"].predict_proba(X)[:, 1]
    out["reject"] = np.asarray(bundle["model"].predict(X)).astype(int)
    _, test_idx = train_test_split(df.index, stratify=df["default"], **SPLIT_KWARGS)
    out["is_test"] = out.index.isin(test_idx)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 2 · BUSINESS MATHS (money saved, KPIs, threshold sweep)
# ──────────────────────────────────────────────────────────────────────────────
def money_saved(y_true, y_pred, loan, interest: float) -> float:
    """Same payoff as the assignment: principal saved on caught defaulters minus interest lost on
    wrongly rejected good borrowers. Approving everyone scores 0."""
    y_true, y_pred, loan = np.asarray(y_true), np.asarray(y_pred), np.asarray(loan, dtype=float)
    caught = (y_pred == 1) & (y_true == 1)
    wrong = (y_pred == 1) & (y_true == 0)
    return float(loan[caught].sum() - interest * loan[wrong].sum())


def portfolio_metrics(frame: pd.DataFrame, interest: float, reject_col: str = "reject") -> dict:
    """All headline KPIs for a (filtered) scored frame."""
    y, r, loan = frame["default"].to_numpy(), frame[reject_col].to_numpy(), frame["loan_amount"].to_numpy(float)
    defaulters, goods = int((y == 1).sum()), int((y == 0).sum())
    caught = int(((r == 1) & (y == 1)).sum())
    wrong = int(((r == 1) & (y == 0)).sum())
    approved = int((r == 0).sum())
    missed = int(((r == 0) & (y == 1)).sum())
    return {
        "n": len(frame), "money": money_saved(y, r, loan, interest),
        "defaulters": defaulters, "caught": caught, "goods": goods, "wrong": wrong,
        "recall": caught / defaulters if defaulters else float("nan"),
        "approval_rate": approved / len(frame) if len(frame) else float("nan"),
        "default_rate_approved": missed / approved if approved else float("nan"),
        "exposed_principal": float(loan[(r == 0) & (y == 1)].sum()),  # AED lent to defaulters we approved
    }


def threshold_sweep(p, y, loan, interest: float, thresholds) -> pd.DataFrame:
    """Re-score the portfolio at every candidate cut-off: reject when P(default) >= threshold."""
    p, y, loan = np.asarray(p, float), np.asarray(y), np.asarray(loan, float)
    n_def, n_good = max(int((y == 1).sum()), 1), max(int((y == 0).sum()), 1)
    rows = []
    for t in thresholds:
        rej = p >= t
        caught, wrong = rej & (y == 1), rej & (y == 0)
        appr = ~rej
        n_appr = int(appr.sum())
        rows.append({
            "threshold": float(t),
            "money": float(loan[caught].sum() - interest * loan[wrong].sum()),
            "caught": int(caught.sum()), "wrong": int(wrong.sum()),
            "tpr": caught.sum() / n_def,  # share of defaulters caught  (y-axis of a ROC curve)
            "fpr": wrong.sum() / n_good,  # share of good borrowers rejected (x-axis of a ROC curve)
            "approval_rate": n_appr / len(p) if len(p) else np.nan,
            "default_rate_approved": ((appr & (y == 1)).sum() / n_appr) if n_appr else np.nan,
            "exposed_principal": float(loan[appr & (y == 1)].sum()),
        })
    return pd.DataFrame(rows)


def sigmoid(z: float) -> float:
    return float(1.0 / (1.0 + np.exp(-z)))


# ──────────────────────────────────────────────────────────────────────────────
# 3 · FILTERS
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Filters:
    age: tuple
    loan: tuple
    homes: tuple
    scope: str


def apply_filters(df: pd.DataFrame, f: Filters) -> pd.DataFrame:
    mask = df["age"].between(*f.age) & df["loan_amount"].between(*f.loan) & df["home_ownership"].isin(f.homes)
    if f.scope == SCOPE_TEST:
        mask &= df["is_test"]
    return df[mask]


# ──────────────────────────────────────────────────────────────────────────────
# 4 · EXPLAINABILITY (SHAP) + ADVERSE ACTION NOTICE
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_explainer(model_file: str):
    """TreeExplainer on the underlying Gradient Boosting model. Contributions are in log-odds of default;
    positive = pushes toward REJECT, negative = pushes toward APPROVE."""
    return shap.TreeExplainer(load_model_bundle(model_file)["base"])


def explain_applicant(model_file: str, X_row: pd.DataFrame, applicant: dict) -> dict:
    """Return per-feature SHAP contributions (home-ownership dummies merged back into one feature)."""
    bundle = load_model_bundle(model_file)
    explainer = get_explainer(model_file)
    sv = explainer.shap_values(X_row)
    sv = np.asarray(sv[-1] if isinstance(sv, list) else sv)  # old SHAP returns a list per class
    if sv.ndim == 3:  # (rows, features, classes) in newer versions -> positive class
        sv = sv[:, :, -1]
    sv = sv[0]
    base = float(np.ravel(explainer.expected_value)[-1])

    contrib = {}
    for col, val in zip(bundle["columns"], sv):
        key = "home_ownership" if col.startswith("home_ownership_") else col
        contrib[key] = contrib.get(key, 0.0) + float(val)

    raw_margin = float(bundle["base"].decision_function(X_row)[0])  # ground truth log-odds
    rows = [{"feature": f, "label": FEATURE_LABELS[f], "value": fmt_feature(f, applicant[f]),
             "shap": contrib.get(f, 0.0)} for f in FEATURES]
    table = pd.DataFrame(rows).sort_values("shap", key=np.abs, ascending=False).reset_index(drop=True)
    return {"table": table, "base": base, "final": base + float(sv.sum()), "margin": raw_margin,
            "additive_ok": abs(base + float(sv.sum()) - raw_margin) < 1e-3}


def fmt_feature(name: str, v) -> str:
    """Human-friendly feature value for chart labels and notices."""
    if name in ("annual_income", "loan_amount"):
        return f"AED {float(v):,.0f}"
    if name in ("loan_term_months", "credit_history_months"):
        return f"{int(v)} months"
    if name == "employment_years":
        return f"{float(v):.1f} yrs"
    if name == "debt_to_income":
        return f"{float(v):.2f}"
    return str(v)


def principal_reasons(table: pd.DataFrame, k: int = 4, min_effect: float = 0.05) -> list:
    """Top-k features that pushed toward denial (Reg B guidance: up to four principal reasons)."""
    top = table[table["shap"] > min_effect].sort_values("shap", ascending=False).head(k)
    return [{"feature": r.feature, "reason": REASON_TEXT[r.feature], "value": r.value, "shap": r.shap}
            for r in top.itertuples()]


def build_adverse_notice(cur: dict, reasons: list, justification: str = "") -> str:
    """Plain-text notice skeleton: decision, principal reasons, ECOA statement. Not legal advice —
    compliance counsel must review before anything is sent to an applicant."""
    lines = [
        "NOTICE OF ADVERSE ACTION", "=" * 24,
        f"Creditor: {CREDITOR_NAME}",
        f"Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
        f"Application reference: {cur['id']}",
        "Action taken: Your application for credit was declined.", "",
        "Principal reason(s) for this decision:",
    ]
    if reasons:
        lines += [f"  {i}. {r['reason']} (your application: {r['value']})" for i, r in enumerate(reasons, 1)]
    if justification:
        lines.append(f"  • Underwriter assessment: {justification.strip()}")
    if not reasons and not justification:
        lines.append("  (No model-attributed reasons available — underwriter must supply specific reasons.)")
    lines += ["", ECOA_NOTICE + REGULATOR_LINE, "",
              "You have the right to request a statement of specific reasons within 60 days of this notice."]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# 5 · FAIRNESS MATHS
# ──────────────────────────────────────────────────────────────────────────────
def group_stats(frame: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Per-group approval rate, actual repayment rate, and disparate-impact ratio (DIR).
    DIR = group approval rate / highest eligible group's approval rate (the four-fifths rule)."""
    g = frame.assign(_approved=(frame["reject"] == 0).astype(float), _repaid=1.0 - frame["default"])
    s = g.groupby(group_col, observed=True).agg(
        n=("_approved", "size"), approval_rate=("_approved", "mean"), repay_rate=("_repaid", "mean"))
    eligible = s["n"] >= MIN_GROUP_N
    ref = s.loc[eligible, "approval_rate"].max() if eligible.any() else np.nan
    s["dir"] = np.where(eligible & (ref > 0), s["approval_rate"] / ref, np.nan)
    s["eligible"] = eligible
    s["flag"] = s["eligible"] & (s["dir"] < FOUR_FIFTHS)
    s["reference_rate"] = ref
    return s.reset_index().rename(columns={group_col: "group"})


def fairness_attributes(frame: pd.DataFrame) -> list:
    """(label, column, is_protected) for each attribute we can audit with the data at hand."""
    attrs = [("Age band", "age_band", True)]
    attrs += [(c.replace("_", " ").title(), c, True) for c in PROTECTED_CANDIDATES if c in frame.columns]
    attrs.append(("Home ownership (control — not a protected class)", "home_ownership", False))
    return attrs


# ──────────────────────────────────────────────────────────────────────────────
# 6 · PLOTLY CHARTS (all share one style + native transition animation)
# ──────────────────────────────────────────────────────────────────────────────
def style_fig(fig: go.Figure, height: int = 420, title: str | None = None) -> go.Figure:
    """Common look. `transition` makes Plotly animate bars/lines/markers smoothly whenever the
    figure re-renders with new data (e.g. when the threshold slider moves)."""
    fig.update_layout(
        template="plotly_white", height=height, title=dict(text=title, x=0.0, font=dict(size=16, color=NAVY)) if title else None,
        font=dict(color=NAVY, size=13), margin=dict(l=10, r=20, t=85 if title else 20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, x=0), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.75)", colorway=PALETTE, transition=dict(duration=600, easing="cubic-in-out"),
        hoverlabel=dict(bgcolor="white", font_color=NAVY))
    fig.update_xaxes(gridcolor=LIGHT, zeroline=False)
    fig.update_yaxes(gridcolor=LIGHT, zeroline=False)
    return fig


def fig_shap_waterfall(exp: dict, cutoff_prob: float) -> go.Figure:
    """Local explanation: baseline (average applicant) -> each feature's push -> this applicant's score.
    Red bars push toward REJECT, green bars toward APPROVE. Units are log-odds of default."""
    t = exp["table"]
    labels = ["Baseline (average applicant)"] + [f"{r.label}: {r.value}" for r in t.itertuples()] + ["This applicant"]
    values = [exp["base"]] + t["shap"].tolist() + [exp["final"]]
    texts = [f"{exp['base']:+.2f}"] + [f"{v:+.2f}" for v in t["shap"]] + [f"{exp['final']:+.2f}"]
    hover = ([f"Average applicant: default probability {sigmoid(exp['base']):.1%}"] +
             [f"{r.label} = {r.value}<br>{'Raises' if r.shap > 0 else 'Lowers'} default risk ({r.shap:+.3f} log-odds)"
              for r in t.itertuples()] +
             [f"This applicant: default probability {sigmoid(exp['final']):.1%}"])
    fig = go.Figure(go.Waterfall(
        orientation="h", y=labels, x=values, text=texts, textposition="outside",
        measure=["absolute"] + ["relative"] * len(t) + ["total"],
        hovertext=hover, hoverinfo="text",
        increasing=dict(marker=dict(color=ROSE)), decreasing=dict(marker=dict(color=TEAL)),
        totals=dict(marker=dict(color=INDIGO)), connector=dict(line=dict(color=LIGHT, width=1))))
    cut = float(np.log(cutoff_prob / (1 - cutoff_prob)))  # decision cut-off expressed in log-odds
    fig.add_vline(x=cut, line=dict(color=AMBER, dash="dash", width=2),
                  annotation_text=f"Reject cut-off (P = {cutoff_prob:.1%})", annotation_position="top left")
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Log-odds of default  (right of the amber line ⇒ reject)")
    return style_fig(fig, height=470, title="Why this decision? Feature-level contributions (SHAP)")


def fig_group_bars(stats: pd.DataFrame, title: str) -> go.Figure:
    """Grouped horizontal bars: model approval rate vs the actual repayment rate in each group."""
    fig = go.Figure()
    fig.add_bar(y=stats["group"].astype(str), x=stats["approval_rate"], orientation="h", name="Model approval rate",
                marker_color=VIOLET, text=[f"{v:.1%}" for v in stats["approval_rate"]], textposition="outside",
                customdata=stats[["n"]], hovertemplate="%{y}<br>Approval rate %{x:.1%}<br>n = %{customdata[0]}<extra></extra>")
    fig.add_bar(y=stats["group"].astype(str), x=stats["repay_rate"], orientation="h", name="Actual repayment rate",
                marker_color=ORANGE, text=[f"{v:.1%}" for v in stats["repay_rate"]], textposition="outside",
                customdata=stats[["n"]], hovertemplate="%{y}<br>Actually repaid %{x:.1%}<br>n = %{customdata[0]}<extra></extra>")
    ref = stats["reference_rate"].iloc[0] if len(stats) else np.nan
    if pd.notna(ref):
        fig.add_vline(x=FOUR_FIFTHS * ref, line=dict(color=RED, dash="dash"),
                      annotation_text="80 % of top group", annotation_position="top right")
    fig.update_layout(barmode="group")
    fig.update_xaxes(tickformat=".0%", range=[0, 1.12], tickvals=[0, .2, .4, .6, .8, 1], title="Rate")
    fig.update_yaxes(autorange="reversed")
    return style_fig(fig, height=max(300, 90 * len(stats) + 120), title=title)


def fig_dir_bars(stats: pd.DataFrame) -> go.Figure:
    """Disparate-impact ratio per group with the 0.80 line; failing groups are red."""
    s = stats.copy()
    colors = [RED if f else (PALETTE[i % len(PALETTE)] if e else LIGHT) for i, (f, e) in enumerate(zip(s["flag"], s["eligible"]))]
    fig = go.Figure(go.Bar(
        y=s["group"].astype(str), x=s["dir"].fillna(0), orientation="h", marker_color=colors,
        text=[f"{v:.2f}" if pd.notna(v) else f"n<{MIN_GROUP_N}" for v in s["dir"]], textposition="outside",
        hovertemplate="%{y}<br>DIR %{x:.3f}<extra></extra>"))
    fig.add_vline(x=FOUR_FIFTHS, line=dict(color=RED, dash="dash"), annotation_text="0.80 threshold", annotation_font_color=RED, annotation_position="top left")
    fig.update_xaxes(range=[0, 1.15], title="Disparate impact ratio (group rate ÷ highest group rate)")
    fig.update_yaxes(autorange="reversed")
    return style_fig(fig, height=max(260, 70 * len(s) + 110), title="Disparate impact ratio")


def fig_money_curve(sweep: pd.DataFrame, sel: dict, breakeven: float) -> go.Figure:
    """Money saved vs cut-off, with the selected threshold, the cost-matched break-even and the 0.5 rule."""
    fig = go.Figure()
    fig.add_scatter(x=sweep["threshold"], y=sweep["money"], mode="lines", name="Money saved",
                    line=dict(color=BLUE, width=3), fill="tozeroy", fillcolor="rgba(59,130,246,0.18)",
                    hovertemplate="Cut-off %{x:.2f}<br>AED %{y:,.0f}<extra></extra>")
    fig.add_scatter(x=[sel["threshold"]], y=[sel["money"]], mode="markers", name="Selected cut-off",
                    marker=dict(size=16, color=ROSE, line=dict(color="white", width=2)))
    best = sweep.loc[sweep["money"].idxmax()]
    fig.add_scatter(x=[best["threshold"]], y=[best["money"]], mode="markers", name="Best cut-off on this data",
                    marker=dict(size=13, symbol="star", color=AMBER, line=dict(color="white", width=1)))
    fig.add_vline(x=breakeven, line=dict(color=INDIGO, dash="dot"), annotation_text="cost-matched", annotation_position="top left")
    fig.add_vline(x=0.5, line=dict(color=SKY, dash="dot"), annotation_text="default 0.5 rule", annotation_position="top right")
    fig.add_hline(y=0, line=dict(color=LIGHT))
    fig.update_xaxes(title="Reject when P(default) ≥ cut-off", tickformat=".0%")
    fig.update_yaxes(title="Money saved vs approve-all (AED)", tickformat=",.0f")
    return style_fig(fig, height=430, title="Reward: money saved as the cut-off moves")


def fig_roc(sweep: pd.DataFrame, sel: dict, overlay: pd.DataFrame | None) -> go.Figure:
    """ROC-style trade-off: defaulters caught (reward) vs good borrowers rejected (cost of friction)."""
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guessing", line=dict(color=LIGHT, dash="dash"), hoverinfo="skip")
    fig.add_scatter(x=sweep["fpr"], y=sweep["tpr"], mode="lines", name="Gradient Boosting (all cut-offs)",
                    line=dict(color=PINK, width=3), fill="tozeroy", fillcolor="rgba(236,72,153,0.14)",
                    customdata=sweep[["threshold", "money"]],
                    hovertemplate="Cut-off %{customdata[0]:.2f}<br>Good rejected %{x:.1%}<br>Defaulters caught %{y:.1%}"
                                  "<br>AED %{customdata[1]:,.0f}<extra></extra>")
    if overlay is not None and len(overlay):
        fig.add_scatter(x=overlay["fpr"], y=overlay["tpr"], mode="markers", name="Leaderboard models @ 1/6 rule (CSV)",
                        marker=dict(size=np.clip(overlay["Money_saved"] / overlay["Money_saved"].max() * 26, 8, 26),
                                    color=ORANGE, opacity=0.7, line=dict(color="white", width=1)),
                        text=overlay["Model"], customdata=overlay["Money_saved"],
                        hovertemplate="%{text}<br>Good rejected %{x:.1%}<br>Defaulters caught %{y:.1%}"
                                      "<br>AED %{customdata:,.0f}<extra></extra>")
    fig.add_scatter(x=[sel["fpr"]], y=[sel["tpr"]], mode="markers", name="Selected cut-off",
                    marker=dict(size=17, color=ROSE, line=dict(color="white", width=2)))
    fig.update_xaxes(title="Good borrowers rejected (false-positive rate)", tickformat=".0%", range=[0, 1])
    fig.update_yaxes(title="Defaulters caught (recall)", tickformat=".0%", range=[0, 1.02])
    return style_fig(fig, height=430, title="Risk vs reward: ROC view")


def fig_risk_reward(sweep: pd.DataFrame, sel: dict) -> go.Figure:
    """Residual risk (default rate among approved loans) vs reward (money saved)."""
    fig = go.Figure()
    fig.add_scatter(x=sweep["default_rate_approved"], y=sweep["money"], mode="lines+markers", name="Cut-off path",
                    line=dict(color=TEAL, width=3), marker=dict(size=6, color=ORANGE), customdata=sweep[["threshold"]],
                    hovertemplate="Cut-off %{customdata[0]:.2f}<br>Default rate of approved %{x:.2%}<br>AED %{y:,.0f}<extra></extra>")
    fig.add_scatter(x=[sel["default_rate_approved"]], y=[sel["money"]], mode="markers", name="Selected cut-off",
                    marker=dict(size=17, color=ROSE, line=dict(color="white", width=2)))
    fig.update_xaxes(title="Default rate among approved loans (residual risk)", tickformat=".1%")
    fig.update_yaxes(title="Money saved vs approve-all (AED)", tickformat=",.0f")
    return style_fig(fig, height=430, title="Risk vs reward: savings vs residual default risk")


def fig_cv_robustness(cv: pd.DataFrame) -> go.Figure:
    """Cross-validated money saved per model under both decision rules (error bars = fold std at 0.5)."""
    cv = cv.sort_values("CV money @1/6 (mean)")
    fig = go.Figure()
    fig.add_bar(y=cv["Model"], x=cv["CV money @0.5 (mean)"], orientation="h", name="0.5 rule", marker_color=SKY,
                error_x=dict(type="data", array=cv["CV money @0.5 (std)"], color=INDIGO))
    fig.add_bar(y=cv["Model"], x=cv["CV money @1/6 (mean)"], orientation="h", name="Cost-matched 1/6 rule", marker_color=PINK)
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Cross-validated money saved (AED)", tickformat=",.0f")
    return style_fig(fig, height=520, title="Robustness: 15-fold cross-validation")


# ──────────────────────────────────────────────────────────────────────────────
# 7 · UI HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def inject_css(page: str = PAGE_SIM) -> None:
    """Colourful layer on top of config.toml: pastel page background, per-page gradient hero, vivid KPI cards,
    multicolour sidebar/tabs/buttons. Each page gets its own accent gradient (PAGE_THEMES)."""
    c1, c2, c3 = PAGE_THEMES[PAGE_KEYS.get(page, "sim")]
    st.markdown(
        f"""<style>
        .stApp {{background:linear-gradient(160deg,#FDF2F8 0%,#EEF2FF 30%,#ECFEFF 62%,#FFFBEB 100%);}}
        [data-testid="stHeader"] {{background:transparent;}}
        .block-container {{padding-top:1.5rem;}}
        h1, h2, h3 {{letter-spacing:-0.01em;}}
        h2 {{color:{c1};border-bottom:4px solid {c2};padding-bottom:.2rem;display:inline-block;}}
        h3 {{color:{c2};}}
        .hero {{background:linear-gradient(120deg,{c1} 0%,{c2} 55%,{c3} 100%);color:white;padding:1.6rem 2rem;
               border-radius:20px;margin-bottom:1rem;box-shadow:0 10px 28px rgba(0,0,0,.18);}}
        .hero h1 {{color:white;margin:0;font-size:2.1rem;}}
        .hero p {{color:rgba(255,255,255,.92);margin:.3rem 0 0 0;font-size:1.02rem;}}
        
        /* Metric Label Text Truncation Fixes */
        [data-testid="stMetricLabel"] {{ white-space: normal !important; word-wrap: break-word !important; overflow: visible !important; }}
        [data-testid="stMetricLabel"] * {{ white-space: normal !important; overflow: visible !important; }}
        
        /* KPI cards: solid vivid gradients, one per column, white text, delta in a white pill */
        [data-testid="stMetric"] {{border-radius:18px;padding:14px 18px;box-shadow:0 6px 16px rgba(0,0,0,.15);
               background:linear-gradient(135deg,#10B981,#14B8A6);}}
        [data-testid="stHorizontalBlock"] > div:nth-child(2) [data-testid="stMetric"] {{background:linear-gradient(135deg,#6366F1,#8B5CF6);}}
        [data-testid="stHorizontalBlock"] > div:nth-child(3) [data-testid="stMetric"] {{background:linear-gradient(135deg,#F97316,#F59E0B);}}
        [data-testid="stHorizontalBlock"] > div:nth-child(4) [data-testid="stMetric"] {{background:linear-gradient(135deg,#0EA5E9,#3B82F6);}}
        [data-testid="stHorizontalBlock"] > div:nth-child(5) [data-testid="stMetric"] {{background:linear-gradient(135deg,#EC4899,#F43F5E);}}
        [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] *, [data-testid="stMetricValue"], [data-testid="stMetricValue"] * {{color:white !important;}}
        [data-testid="stMetricDelta"] {{background:rgba(255,255,255,.94);border-radius:999px;padding:1px 10px;width:fit-content;}}
        /* sidebar */
        [data-testid="stSidebar"] > div:first-child {{background:linear-gradient(180deg,#1E1B4B 0%,#4C1D95 55%,#9D174D 100%);}}
        /* buttons */
        .stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {{
               background:linear-gradient(90deg,{c1},{c3});border:none;color:white;font-weight:700;box-shadow:0 6px 14px rgba(0,0,0,.2);}}
        .stButton > button:not([kind="primary"]) {{border:2px solid {c1};color:{c1};background:white;font-weight:600;}}
        /* tabs: every tab its own colour */
        .stTabs [data-baseweb="tab-list"] {{gap:8px;}}
        .stTabs [data-baseweb="tab"] {{border-radius:12px 12px 0 0;padding:8px 20px;}}
        .stTabs [data-baseweb="tab"]:nth-child(1) {{background:#EDE9FE;}}
        .stTabs [data-baseweb="tab"]:nth-child(2) {{background:#FFEDD5;}}
        .stTabs [data-baseweb="tab"]:nth-child(3) {{background:#CFFAFE;}}
        .stTabs [data-baseweb="tab"]:nth-child(4) {{background:#FCE7F3;}}
        .stTabs [data-baseweb="tab"]:nth-child(1)[aria-selected="true"] {{background:linear-gradient(90deg,#7C3AED,#C026D3);}}
        .stTabs [data-baseweb="tab"]:nth-child(2)[aria-selected="true"] {{background:linear-gradient(90deg,#EA580C,#F59E0B);}}
        .stTabs [data-baseweb="tab"]:nth-child(3)[aria-selected="true"] {{background:linear-gradient(90deg,#0891B2,#10B981);}}
        .stTabs [data-baseweb="tab"]:nth-child(4)[aria-selected="true"] {{background:linear-gradient(90deg,#DB2777,#F43F5E);}}
        .stTabs [aria-selected="true"] p {{color:white;font-weight:700;}}
        /* expanders and form */
        [data-testid="stExpander"] {{background:rgba(255,255,255,.85);border:1px solid #E9D5FF;border-left:6px solid {c2};border-radius:14px;}}
        [data-testid="stForm"] {{background:linear-gradient(135deg,#F5F3FF 0%,#FDF2F8 50%,#FFF7ED 100%);border:2px solid {c2};border-radius:18px;}}
        </style>""", unsafe_allow_html=True)


def ensure_state() -> None:
    """Initialise session_state keys used for the audit trail and the current decision."""
    st.session_state.setdefault("audit_log", [])  # list[dict] — simulated audit trail
    st.session_state.setdefault("current", None)  # the application currently on screen
    st.session_state.setdefault("override_mode", False)
    st.session_state.setdefault("stream", {"idx": [], "times": [], "ticks": 0, "seed": 42})  # live-monitor state


def aed(v: float) -> str:
    return f"AED {v:,.0f}"


def signed_aed(v: float) -> str:
    # ASCII '-' on purpose: st.metric only colours a delta as negative when it starts with '-'
    return f"{'+' if v >= 0 else '-'}AED {abs(v):,.0f}"


def pct(v: float, digits: int = 1) -> str:
    """Percent string that survives NaN (e.g. when filters leave no defaulters or no approvals)."""
    return "n/a" if not np.isfinite(v) else f"{v * 100:.{digits}f}%"


def pts(a: float, b: float, digits: int = 1) -> str | None:
    """Signed percentage-point difference, or None (no delta shown) if either side is NaN."""
    return None if not (np.isfinite(a) and np.isfinite(b)) else f"{(a - b) * 100:+.{digits}f} pts"


def render_kpis(m: dict, base: dict, base_label: str) -> None:
    """Top-level KPI row. Deltas compare the active decision rule with the other rule on the same loans."""
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("💵 Money saved vs approve-all", aed(m["money"]), signed_aed(m["money"] - base["money"]),
              help=f"Delta vs {base_label}")
    c2.metric("🎯 Defaulters caught", f"{m['caught']}/{m['defaulters']}", pts(m["recall"], base["recall"]) ,
              help=f"Recall delta vs {base_label}")
    c3.metric("😕 Good borrowers rejected", f"{m['wrong']:,}", f"{m['wrong'] - base['wrong']:+,}",
              delta_color="inverse", help=f"Delta vs {base_label}. Fewer is better for customer experience.")
    c4.metric("✅ Approval rate", pct(m["approval_rate"]), pts(m["approval_rate"], base["approval_rate"]),
              delta_color="off", help=f"Delta vs {base_label}")
    c5.metric("⚠️ Default rate of approved", pct(m["default_rate_approved"], 2),
              pts(m["default_rate_approved"], base["default_rate_approved"], 2), delta_color="inverse",
              help=f"Residual portfolio risk. Delta vs {base_label}.")


def sidebar(df: pd.DataFrame | None) -> dict:
    """Navigation + global controls. Returns everything the pages need."""
    with st.sidebar:
        st.markdown("## 💰 QuickLoan")
        st.caption("Risk & Compliance Console")
        page = st.radio("Navigate", NAV_PAGES, label_visibility="collapsed")
        st.divider()

        st.markdown("#### ⚙️ Decision model")
        model_label = st.radio("Which model decides?", list(MODEL_FILES), index=0, label_visibility="collapsed")
        st.caption("**Cost-matched 1/6 rule:** reject when P(default) ≥ i/(1+i). Losing a loan costs 5× the interest lost "
                   "by turning away a good customer.\n\n**Default 0.5 rule:** reject only at P(default) ≥ 50 %.")
        interest = st.number_input("Interest margin assumption", 0.05, 0.60, 0.20, 0.01, format="%.2f",
                                   help="Used in money-saved maths and the break-even cut-off i/(1+i).")
        underwriter = st.text_input("Underwriter ID", value="underwriter_01", help="Written to the audit trail.")
        st.divider()

        st.markdown("#### 🔎 Global filters")
        scope = st.radio("Evaluation data", [SCOPE_TEST, SCOPE_FULL], index=0)
        if df is not None:
            age = st.slider("Age", int(df["age"].min()), int(df["age"].max()), (int(df["age"].min()), int(df["age"].max())))
            lo, hi = int(df["loan_amount"].min()), int(df["loan_amount"].max())
            loan = st.slider("Loan amount (AED)", lo, hi, (lo, hi), step=1000)
            homes = st.multiselect("Home ownership", sorted(df["home_ownership"].unique()),
                                   default=sorted(df["home_ownership"].unique()))
        else:
            age, loan, homes = (21, 64), (0, 10 ** 9), ["rent", "own", "mortgage"]
            st.caption("Filters appear once loans.csv is available.")
    return {"page": page, "model_label": model_label, "model_file": MODEL_FILES[model_label], "interest": interest,
            "underwriter": underwriter.strip(), "filters": Filters(age, loan, tuple(homes), scope)}


# ──────────────────────────────────────────────────────────────────────────────
# 8 · PAGE 1 — SIMULATOR + XAI + HUMAN-IN-THE-LOOP
# ──────────────────────────────────────────────────────────────────────────────
def applicant_form() -> dict | None:
    """Input form. Nothing is scored until the user presses the submit button."""
    with st.form("applicant_form"):
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
            st.caption("Debt-to-income is computed automatically as loan amount ÷ annual income.")
        submitted = st.form_submit_button("Score application", type="primary")
    if not submitted:
        return None
    return dict(age=age, annual_income=annual_income, loan_amount=loan_amount, loan_term_months=loan_term_months,
                employment_years=employment_years, credit_history_months=credit_history_months,
                num_existing_loans=num_existing_loans, past_defaults=past_defaults,
                debt_to_income=round(loan_amount / (annual_income + 1), 2), home_ownership=home_ownership)


def score_application(ctx: dict, applicant: dict) -> dict:
    """Run model + SHAP inside an st.status block so the user sees progress."""
    bundle = load_model_bundle(ctx["model_file"])
    with st.status("Scoring application…", expanded=True) as status:
        st.write("Encoding applicant features…")
        X = encode(pd.DataFrame([applicant]), bundle["columns"])
        st.write("Running the credit model…")
        pred = int(np.asarray(bundle["model"].predict(X))[0])
        prob = float(bundle["base"].predict_proba(X)[0, 1])
        exp = None
        if SHAP_OK:
            st.write("Computing SHAP feature contributions…")
            try:
                exp = explain_applicant(ctx["model_file"], X, applicant)
            except Exception as e:  # keep the decision even if the explanation fails
                st.warning(f"Explanation could not be generated: {e}")
        status.update(label="Decision ready", state="complete", expanded=False)
    reasons = principal_reasons(exp["table"]) if (exp and pred == 1) else []
    return {"id": f"APP-{uuid.uuid4().hex[:8].upper()}", "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "applicant": applicant, "model": ctx["model_label"], "model_file": ctx["model_file"], "prob": prob,
            "cutoff": bundle["threshold"], "model_decision": "Denied" if pred == 1 else "Approved",
            "exp": exp, "reasons": reasons, "status": "pending", "final_decision": None, "justification": ""}


def commit_decision(cur: dict, underwriter: str, action: str, final: str, justification: str = "") -> None:
    """Write one immutable-style row to the simulated audit trail and lock the decision."""
    cur.update(status="accepted" if action == "Accepted model decision" else "overridden",
               final_decision=final, justification=justification.strip())
    st.session_state.audit_log.append({
        "logged_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "application_id": cur["id"], "underwriter": underwriter, "model": cur["model"],
        "p_default": round(cur["prob"], 4), "cutoff": round(cur["cutoff"], 4),
        "model_decision": cur["model_decision"], "human_action": action, "final_decision": final,
        "justification": justification.strip(),
        "top_reasons": "; ".join(r["reason"] for r in cur["reasons"]),
        "inputs": json.dumps(cur["applicant"]),
    })
    st.session_state.override_mode = False


def render_decision(cur: dict) -> None:
    """Prediction banner + probability metrics."""
    p, cut = cur["prob"], cur["cutoff"]
    if cur["model_decision"] == "Denied":
        st.error(f"❌ **Model decision: DENIED** — predicted default risk {p:.1%} is at or above the {cut:.1%} cut-off.")
    else:
        st.success(f"✅ **Model decision: APPROVED** — predicted default risk {p:.1%} is below the {cut:.1%} cut-off.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Default probability", f"{p:.1%}")
    c2.metric("Decision cut-off", f"{cut:.1%}")
    c3.metric("Margin to cut-off", f"{(p - cut) * 100:+.1f} pts", delta_color="inverse",
              help="Positive = above the cut-off (reject side).")


def render_xai(cur: dict) -> None:
    """SHAP waterfall + explanation tables."""
    st.subheader("Why this decision?")
    exp = cur["exp"]
    if not SHAP_OK:
        st.error("SHAP is not installed, so the explanation is unavailable. Run `pip install shap` and restart the app."
                 + (f"\n\nImport error: `{SHAP_ERR}`" if SHAP_ERR else ""))
        return
    if exp is None:
        return
    st.plotly_chart(fig_shap_waterfall(exp, cur["cutoff"]), width="stretch")
    st.caption(f"Baseline default probability for an average applicant is {sigmoid(exp['base']):.1%}; this applicant scores "
               f"{sigmoid(exp['final']):.1%}. Red bars raise default risk (toward denial), green bars lower it.")
    if not exp["additive_ok"]:
        st.caption("⚠️ SHAP contributions differ slightly from the model's raw score; treat magnitudes as approximate.")
    a = cur["applicant"]
    age_row = exp["table"].loc[exp["table"]["feature"] == "age", "shap"].iloc[0]
    if cur["model_decision"] == "Denied" and a["age"] >= 62 and age_row > 0.02:
        st.warning("⚠️ **ECOA age check:** this applicant is 62 or older and age pushed the score toward denial. Reg B only "
                   "permits age in an empirically derived, statistically sound system that does not assign a negative "
                   "value to applicants aged 62+. Escalate to compliance before issuing this decision.")
    with st.expander("Full contribution table"):
        show = exp["table"].rename(columns={"label": "Feature", "value": "Applicant value", "shap": "Effect (log-odds)"})
        st.dataframe(show[["Feature", "Applicant value", "Effect (log-odds)"]]
                     .style.format({"Effect (log-odds)": "{:+.3f}"}), hide_index=True, width="stretch")


def render_hitl(cur: dict, ctx: dict) -> None:
    """Human-in-the-loop controls + adverse action notice + audit trail."""
    st.subheader("Human-in-the-loop review")
    opposite = "Approved" if cur["model_decision"] == "Denied" else "Denied"

    if cur["status"] == "pending":
        st.caption(f"Application **{cur['id']}** is awaiting underwriter review.")
        c1, c2, _ = st.columns([1, 1, 2])
