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
PAGE_THEMES = {0: ("#7C3AED", "#C026D3", "#EC4899"),   # simulator: violet → magenta → pink
               1: ("#EA580C", "#F59E0B", "#EF4444"),   # fairness: orange → amber → red
               2: ("#2563EB", "#0EA5E9", "#10B981")}   # cost: blue → sky → emerald

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

NAV_PAGES = ["🧮 Applicant Simulator & XAI", "⚖️ Bias & Fairness Audit", "📈 Portfolio Cost-Awareness"]
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
        font=dict(color=NAVY, size=13), margin=dict(l=10, r=20, t=60 if title else 20, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), paper_bgcolor="rgba(0,0,0,0)",
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
                  annotation_text=f"Reject cut-off (P = {cutoff_prob:.1%})", annotation_position="top")
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
                      annotation_text="80 % of top group", annotation_position="bottom right")
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
    fig.add_vline(x=FOUR_FIFTHS, line=dict(color=RED, dash="dash"), annotation_text="0.80 threshold", annotation_font_color=RED)
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
def inject_css(page_idx: int = 0) -> None:
    """Colourful layer on top of config.toml: pastel page background, per-page gradient hero, vivid KPI cards,
    multicolour sidebar/tabs/buttons. Each page gets its own accent gradient (PAGE_THEMES)."""
    c1, c2, c3 = PAGE_THEMES.get(page_idx, PAGE_THEMES[0])
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
        accept = c1.button("✅ Accept Model Decision", key=f"acc_{cur['id']}", type="primary", width="stretch")
        override = c2.button("✍️ Override Decision", key=f"ovr_{cur['id']}", width="stretch")
        if (accept or override) and not ctx["underwriter"]:
            st.error("Enter an Underwriter ID in the sidebar before recording a decision.")
        elif accept:
            commit_decision(cur, ctx["underwriter"], "Accepted model decision", cur["model_decision"])
            st.rerun()
        elif override:
            st.session_state.override_mode = True

        if st.session_state.override_mode:
            # The justification box only exists while the underwriter is overriding.
            st.warning(f"You are about to change the outcome from **{cur['model_decision']}** to **{opposite}**. "
                       "A written justification is mandatory and will be logged.")
            just = st.text_area("Justification (required, min. 25 characters)", key=f"just_{cur['id']}",
                                placeholder="E.g. verified 3 years of stable salary slips not captured by the model…")
            b1, b2, _ = st.columns([1, 1, 2])
            if b1.button(f"Confirm override → {opposite}", key=f"cfm_{cur['id']}", type="primary", width="stretch"):
                if len(just.strip()) < 25:
                    st.error("Please give a justification of at least 25 characters.")
                elif not ctx["underwriter"]:
                    st.error("Enter an Underwriter ID in the sidebar.")
                else:
                    commit_decision(cur, ctx["underwriter"], "Overrode model decision", opposite, just)
                    st.rerun()
            if b2.button("Cancel", key=f"can_{cur['id']}", width="stretch"):
                st.session_state.override_mode = False
                st.rerun()
    else:
        verb = "accepted the model decision" if cur["status"] == "accepted" else "overrode the model"
        box = st.success if cur["final_decision"] == "Approved" else st.error
        box(f"**Final decision: {cur['final_decision'].upper()}** — underwriter `{ctx['underwriter'] or 'n/a'}` {verb}. "
            f"Logged to the audit trail.")
        if cur["justification"]:
            st.info(f"**Override justification:** {cur['justification']}")

    render_adverse_notice(cur)

    with st.expander(f"🗂️ Audit trail ({len(st.session_state.audit_log)} entries this session)"):
        if st.session_state.audit_log:
            log = pd.DataFrame(st.session_state.audit_log)
            st.dataframe(log.drop(columns=["inputs"]), hide_index=True, width="stretch")
            st.download_button("Download audit trail (CSV)", log.to_csv(index=False).encode("utf-8"),
                               file_name="audit_trail.csv", mime="text/csv")
        else:
            st.caption("No decisions recorded yet.")
        st.caption("Stored in `st.session_state` only: it disappears when the browser session ends. "
                   "A production system must write to an append-only, access-controlled database.")


def render_adverse_notice(cur: dict) -> None:
    """Adverse-action basis: shown only when the *effective* decision is a denial."""
    effective = cur["final_decision"] or cur["model_decision"]
    if effective != "Denied":
        return
    draft = cur["status"] == "pending"
    human_denial = cur["status"] == "overridden" and cur["model_decision"] == "Approved"
    st.subheader("Adverse action notice" + (" — draft" if draft else ""))
    if human_denial:
        st.warning("The model approved this applicant, so its SHAP drivers cannot justify a denial. The reason below is "
                   "the underwriter's justification and must itself be specific and lawful.")
    reasons = [] if human_denial else cur["reasons"]
    if reasons:
        st.markdown("**Principal reasons for denial** (ranked by influence):")
        for i, r in enumerate(reasons, 1):
            st.markdown(f"{i}. {r['reason']} — *your application: {r['value']}*")
    elif not human_denial:
        st.info("No single feature pushed the score toward denial by a material amount; an underwriter must supply specific reasons.")
    notice = build_adverse_notice(cur, reasons, cur["justification"] if human_denial else "")
    with st.expander("Preview notice text"):
        st.code(notice, language=None)
    st.download_button("Download notice basis (.txt)", notice.encode("utf-8"), file_name=f"adverse_action_{cur['id']}.txt",
                       mime="text/plain", disabled=draft, help="Available once the underwriter records a final decision." if draft else None)
    st.caption("Starting point only — compliance counsel must review wording, delivery timing and any FCRA disclosures "
               "(if a consumer report was used) before sending.")


def page_simulator(ctx: dict) -> None:
    st.header("Applicant Simulator & Explainable AI")
    st.caption(f"Active model: **{ctx['model_label']}**")
    applicant = applicant_form()
    if applicant is not None:
        try:
            st.session_state.current = score_application(ctx, applicant)
            st.session_state.override_mode = False
        except Exception as e:
            st.error(f"Scoring failed: {e}")
            return
    cur = st.session_state.current
    if cur is None:
        st.info("Fill in the form and press **Score application** to see the decision and its explanation.")
        return
    if cur["model_file"] != ctx["model_file"] and cur["status"] == "pending":
        st.warning("You switched models after scoring. Re-submit the form to score with the newly selected model.")
    render_decision(cur)
    render_xai(cur)
    render_hitl(cur, ctx)


# ──────────────────────────────────────────────────────────────────────────────
# 9 · PAGE 2 — BIAS & FAIRNESS AUDIT
# ──────────────────────────────────────────────────────────────────────────────
def page_fairness(ctx: dict, df_all: pd.DataFrame, df_all_other: pd.DataFrame) -> None:
    st.header("Bias & Fairness Audit (ECOA)")
    st.caption("Does the model approve some groups materially less often than others? "
               "Rule of thumb: a group's approval rate below 80 % of the highest group's rate is flagged.")
    audit_scope = st.radio("Audit population", [SCOPE_FULL, SCOPE_TEST], horizontal=True,
                           help="The full file gives bigger groups; the held-out split avoids rows the model trained on.")
    f = replace(ctx["filters"], scope=audit_scope)  # sidebar filters still apply (age/loan/home)

    with st.spinner("Auditing approval rates…"):
        frame = apply_filters(df_all, f).copy()
        other = apply_filters(df_all_other, f).copy()
        if frame.empty:
            st.warning("No loans match the current filters.")
            return
        for d in (frame, other):
            d["age_band"] = pd.cut(d["age"], bins=AGE_BINS, labels=AGE_LABELS)
        attrs = fairness_attributes(frame)
        results = {col: group_stats(frame, col) for _, col, _ in attrs}

    # ── Visual alert ────────────────────────────────────────────────────────
    flagged = [(label, row) for label, col, protected in attrs if protected
               for row in results[col].itertuples() if row.flag]
    if flagged:
        for label, row in flagged:
            st.warning(f"⚠️ **Potential disparate impact — {label}:** group **{row.group}** has a disparate impact ratio of "
                       f"**{row.dir:.2f}** (below {FOUR_FIFTHS:.2f}). Approval rate {row.approval_rate:.1%} vs "
                       f"{row.reference_rate:.1%} for the most-approved group. Investigate before relying on this model.")
    else:
        st.success("✅ No protected class falls below the 80 % threshold in this population.")
    n_extra = [c for c in PROTECTED_CANDIDATES if c in frame.columns]
    if not n_extra:
        st.info("**Coverage note:** loans.csv contains age as its only protected attribute (no sex, race, marital status, etc.), "
                "so those classes cannot be tested directly. If you add columns such as `gender` or `ethnicity`, they are "
                "audited automatically. Use a documented proxy method where collecting them is restricted.")

    # ── Per-attribute charts ────────────────────────────────────────────────
    tabs = st.tabs([label.split(" (")[0] for label, _, _ in attrs])
    for tab, (label, col, protected) in zip(tabs, attrs):
        with tab:
            s = results[col]
            if not protected:
                st.caption("Control attribute: not protected under ECOA, shown for comparison and to spot possible proxy effects.")
            c1, c2 = st.columns([3, 2])
            c1.plotly_chart(fig_group_bars(s, f"Approval vs actual repayment by {label.split(' (')[0].lower()}"), width="stretch")
            c2.plotly_chart(fig_dir_bars(s), width="stretch")
            small = s[~s["eligible"]]
            if len(small):
                st.caption(f"Groups with fewer than {MIN_GROUP_N} loans are excluded from the ratio test: "
                           + ", ".join(map(str, small["group"])))

    # ── Diagnostics ─────────────────────────────────────────────────────────
    with st.expander("📋 Group statistics table"):
        for label, col, _ in attrs:
            st.markdown(f"**{label}**")
            t = results[col][["group", "n", "approval_rate", "repay_rate", "dir", "flag"]].rename(columns={
                "group": "Group", "n": "Loans", "approval_rate": "Approval rate", "repay_rate": "Actual repayment rate",
                "dir": "DIR", "flag": "Below 0.80"})
            st.dataframe(t.style.format({"Approval rate": "{:.1%}", "Actual repayment rate": "{:.1%}", "DIR": "{:.3f}"}),
                         hide_index=True, width="stretch")
    with st.expander("🔁 Compare both decision rules (age band DIR)"):
        both = []
        for name, d in ((ctx["model_label"], frame), ("Other rule", other)):
            s = group_stats(d, "age_band")
            both.append(s[["group", "approval_rate", "dir"]].rename(
                columns={"approval_rate": f"Approval rate — {name[:28]}", "dir": f"DIR — {name[:28]}"}).set_index("group"))
        st.dataframe(pd.concat(both, axis=1).reset_index().rename(columns={"group": "Age band"}), hide_index=True, width="stretch")
    with st.expander("📘 How to read this (and what it does not prove)"):
        st.markdown(
            "- **Disparate impact ratio** = a group's approval rate ÷ the highest group's rate. Below 0.80 is a screening flag, "
            "not a legal finding; follow up with statistical tests and a business-necessity / less-discriminatory-alternative review.\n"
            "- **Age** is permitted as a model input only under Reg B's conditions; applicants aged 62+ must not be treated less favourably than younger ones.\n"
            "- Gaps can reflect genuine differences in repayment (see the navy bars) — but proxies such as income or home ownership can also encode protected traits.\n"
            "- ECOA / Reg B is a US framework while this dataset is in AED; map the audit to the rules of your actual regulator.")


# ──────────────────────────────────────────────────────────────────────────────
# 10 · PAGE 3 — PORTFOLIO COST-AWARENESS
# ──────────────────────────────────────────────────────────────────────────────
def page_cost(ctx: dict, fdf: pd.DataFrame, df_all: pd.DataFrame) -> None:
    st.header("Portfolio-Level Cost-Awareness")
    st.caption("Slide the cut-off to see how money saved, defaulters caught and residual risk trade off on the filtered loans.")
    interest = ctx["interest"]
    breakeven = interest / (1 + interest)  # reject iff p·L ≥ (1-p)·i·L  ⇒  p ≥ i/(1+i)

    lb, lb_err = read_csv_safe("leaderboard_cost_aware_threshold.csv", ("Model", "Money_saved", "Defaulters_caught", "Good_rejected"))
    cv, cv_err = read_csv_safe("cv_robustness_table.csv", ("Model", "CV money @0.5 (mean)", "CV money @0.5 (std)", "CV money @1/6 (mean)"))

    # ── Controls ────────────────────────────────────────────────────────────
    cc1, cc2 = st.columns([1, 3])
    pin = cc1.toggle("Pin to cost-matched cut-off", value=True,
                     help=f"Uses the break-even P(default) = i/(1+i) = {breakeven:.3f} for the interest margin in the sidebar.")
    thr_slider = cc2.slider("Reject when P(default) ≥", 0.02, 0.90, 0.50, 0.01, disabled=pin, format="%.2f",
                            help="Unpin the toggle to move the cut-off yourself.")
    thr = breakeven if pin else thr_slider
    view = st.radio("Chart view", ["ROC trade-off (caught vs rejected)", "Risk vs reward (savings vs residual default rate)"], horizontal=True)

    if fdf.empty:
        st.warning("No loans match the current filters.")
        return

    # ── Compute sweep + selected operating point ────────────────────────────
    with st.spinner("Re-scoring the portfolio at every cut-off…"):
        grid = np.round(np.arange(0.01, 0.91, 0.01), 2)
        sweep = threshold_sweep(fdf["p_default"], fdf["default"], fdf["loan_amount"], interest, grid)
        sel = threshold_sweep(fdf["p_default"], fdf["default"], fdf["loan_amount"], interest, [thr]).iloc[0].to_dict()
        ref = threshold_sweep(fdf["p_default"], fdf["default"], fdf["loan_amount"], interest, [0.5]).iloc[0].to_dict()

    # ── KPI row at the selected cut-off (delta = vs the default 0.5 rule) ───
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Cut-off in use", f"{thr:.3f}", f"{(thr - 0.5):+.3f} vs 0.5 rule", delta_color="off")
    k2.metric("💵 Money saved", aed(sel["money"]), signed_aed(sel["money"] - ref["money"]))
    k3.metric("🎯 Defaulters caught", f"{int(sel['caught'])}", f"{int(sel['caught'] - ref['caught']):+d}")
    k4.metric("😕 Good rejected", f"{int(sel['wrong']):,}", f"{int(sel['wrong'] - ref['wrong']):+,}", delta_color="inverse")
    k5.metric("🏦 Principal exposed to missed defaults", aed(sel["exposed_principal"]),
              signed_aed(sel["exposed_principal"] - ref["exposed_principal"]), delta_color="inverse")

    # ── Overlay: leaderboard CSV models at the 1/6 rule ─────────────────────
    unfiltered_test = ctx["filters"].scope == SCOPE_TEST and len(fdf) == int(df_all["is_test"].sum())
    overlay = None
    if lb is not None and unfiltered_test and abs(interest - 0.20) < 1e-9:
        overlay = lb.assign(tpr=lb["Defaulters_caught"] / max(int((fdf["default"] == 1).sum()), 1),
                            fpr=lb["Good_rejected"] / max(int((fdf["default"] == 0).sum()), 1))

    # ── Charts ──────────────────────────────────────────────────────────────
    left, right = st.columns(2)
    left.plotly_chart(fig_money_curve(sweep, sel, breakeven), width="stretch")
    if view.startswith("ROC"):
        right.plotly_chart(fig_roc(sweep, sel, overlay), width="stretch")
        if overlay is None:
            st.caption("The leaderboard-model overlay appears when you evaluate on the unfiltered held-out split at a 20 % interest margin.")
    else:
        right.plotly_chart(fig_risk_reward(sweep, sel), width="stretch")
    best = sweep.loc[sweep["money"].idxmax()]
    st.caption(f"Best cut-off on this data: **{best['threshold']:.2f}** (AED {best['money']:,.0f}). It is picked in hindsight on a "
               "single sample, so treat it as an upper bound. The cost-matched cut-off is the principled choice.")

    # ── Diagnostics hidden in expanders ─────────────────────────────────────
    with st.expander("🔗 Reconcile with leaderboard_cost_aware_threshold.csv"):
        if lb_err:
            st.error(lb_err)
        else:
            gb = lb[lb["Model"] == "Gradient Boosting"]
            if gb.empty or not unfiltered_test or abs(interest - 0.20) > 1e-9:
                st.info("Reconciliation is only meaningful on the unfiltered held-out split with a 20 % interest margin.")
            else:
                at16 = threshold_sweep(fdf["p_default"], fdf["default"], fdf["loan_amount"], interest, [1 / 6]).iloc[0]
                rec = pd.DataFrame({
                    "Rule": ["Cost-matched 1/6", "Default 0.5"],
                    "Live money saved (AED)": [at16["money"], ref["money"]],
                    "CSV money saved (AED)": [float(gb["Money_saved"].iloc[0]), float(gb["Money_at_0_5"].iloc[0])
                                              if "Money_at_0_5" in gb.columns else np.nan]})
                rec["Difference"] = rec["Live money saved (AED)"] - rec["CSV money saved (AED)"]
                st.dataframe(rec.style.format({c: "{:,.0f}" for c in rec.columns[1:]}), hide_index=True, width="stretch")
    with st.expander("🏁 Leaderboard at the cost-matched 1/6 rule (CSV)"):
        if lb_err:
            st.error(lb_err)
        else:
            st.dataframe(lb, hide_index=True, width="stretch")
    with st.expander("🧪 Cross-validation robustness (CSV)"):
        if cv_err:
            st.error(cv_err)
        else:
            st.plotly_chart(fig_cv_robustness(cv), width="stretch")
            st.caption("Error bars show fold-to-fold standard deviation at the 0.5 rule (the table has no std for the 1/6 rule).")
            st.dataframe(cv, hide_index=True, width="stretch")


# ──────────────────────────────────────────────────────────────────────────────
# 11 · MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(page_title="QuickLoan Risk & Compliance Console", page_icon="💰", layout="wide")
    ensure_state()

    raw, data_err = read_csv_safe("loans.csv", tuple(FEATURES) + ("default",))
    ctx = sidebar(raw)
    inject_css(NAV_PAGES.index(ctx["page"]))

    # Load the active model (hard requirement for every page).
    try:
        with st.spinner("Loading models…"):
            load_model_bundle(ctx["model_file"])
    except FileNotFoundError:
        st.error(f"Model file `{ctx['model_file']}` was not found next to app.py. Re-run the notebook to regenerate it.")
        st.stop()
    except Exception as e:
        st.error(f"Could not load `{ctx['model_file']}`: {e}. A scikit-learn version mismatch is the usual cause "
                 "(see requirements.txt).")
        st.stop()

    st.markdown('<div class="hero"><h1>💰 QuickLoan — Risk & Compliance Console</h1>'
                '<p>Explainable decisions · Fair-lending audit · Cost-aware portfolio</p></div>', unsafe_allow_html=True)

    # Score the portfolio with the active rule and the other rule (for KPI deltas + fairness comparison).
    scored = scored_other = None
    if data_err:
        st.warning(f"{data_err} Portfolio KPIs, the fairness audit and the cost tab are disabled; the simulator still works.")
    else:
        other_file = next(f for f in MODEL_FILES.values() if f != ctx["model_file"])
        try:
            with st.spinner("Scoring the portfolio…"):
                scored, scored_other = scored_portfolio(ctx["model_file"]), scored_portfolio(other_file)
        except Exception as e:
            st.warning(f"Portfolio scoring failed: {e}")

    fdf = None
    if scored is not None:
        fdf = apply_filters(scored, ctx["filters"])
        if fdf.empty:
            st.warning("No loans match the current filters — widen them in the sidebar.")
        else:
            base_label = [k for k in MODEL_FILES if MODEL_FILES[k] != ctx["model_file"]][0]
            fdf_other = scored_other.loc[fdf.index]
            render_kpis(portfolio_metrics(fdf, ctx["interest"]), portfolio_metrics(fdf_other, ctx["interest"]),
                        base_label.split(" — ")[1])
            st.caption(f"KPIs: {len(fdf):,} loans · {ctx['filters'].scope.split(' (')[0]} · deltas vs the other decision rule.")
    st.divider()

    page = ctx["page"]
    if page == NAV_PAGES[0]:
        page_simulator(ctx)
    elif scored is None:
        st.info("This page needs `loans.csv` and a working model to score the portfolio.")
    elif page == NAV_PAGES[1]:
        page_fairness(ctx, scored, scored_other)
    else:
        page_cost(ctx, fdf, scored)


if __name__ == "__main__":  # Streamlit executes the script as __main__
    main()
