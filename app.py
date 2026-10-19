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
                    line=dict(color=BLUE, width=3), fill="tozeroy", fillcolor
