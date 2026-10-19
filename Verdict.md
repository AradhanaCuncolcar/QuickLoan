# QuickLoan — Verdict to Leadership

**Recommendation: deploy Gradient Boosting, run with a cost-matched decision rule — reject when the predicted default probability is ≥ 1/6 (= interest ÷ (1 + interest)).**

| Test set: 1,500 loans, AED 75.2 M requested, 188 defaulters (AED 12.27 M principal) | Accuracy | Defaulters caught | Good customers rejected | **Money saved vs approving everyone** |
|---|---|---|---|---|
| Approve everyone | 87.5 % | 0 / 188 | 0 | AED 0 |
| Gradient Boosting, default 0.5 rule (bake-off champion) | 89.6 % | 63 / 188 (33.5 %) | 31 | **AED 4,768,240** |
| **Gradient Boosting, cost-matched 1/6 rule (recommended)** | 84.7 % | 101 / 188 (53.7 %) | 143 (10.9 %) | **AED 5,815,160** (≈ AED 3,900 per applicant) |

**Why this model — in money terms.** It topped the leaderboard and also had the best average in 15-fold cross-validation under both decision rules. It saves AED 0.99 M more than the best single model (Decision Tree) and about AED 1.0 M more once the cut-off is matched to the costs. XGBoost is statistically indistinguishable and is a perfectly good alternative; Stacking is no better yet far slower and more complex; Logistic Regression's strong-looking single-split score at the 1/6 rule did not survive cross-validation.

**Trade-offs we accept.**
- **Accuracy:** we give up about 5 points (89.6 % → 84.7 %) and end up below "approve everyone" on accuracy — because accuracy ignores that a missed defaulter costs five times as much as a lost customer.
- **Recall vs customer friction:** we catch 38 more defaulters than the default rule but turn away 112 more good customers; at a 20 % interest assumption that trade is clearly profitable (the net saving would only disappear at an interest rate far above any realistic level).
- **Interpretability:** boosted trees are harder to explain than a logistic regression or a single tree. Mitigation: every decision carries a default probability, and `debt_to_income` and `past_defaults` drive about 68 % of the model's importance. If explainability is mandatory, a Decision Tree with the same 1/6 rule is the transparent fallback (cross-validated AED 4.63 M, ~10 % below Gradient Boosting).

**Caveats.** One synthetic-looking dataset of 6,000 loans, a single 188-defaulter test set (per-fold noise ≈ AED 0.5 M), no hyper-parameter tuning, and a fixed 20 % interest assumption. Even the recommended model still lets 87 defaulters (AED 4.49 M of principal) through; the cut-off must be recomputed if the interest or loss assumptions change, and the model should be monitored after deployment.

*Full evidence: QuickLoan_Notebook.ipynb (leaderboard, cross-validation, cost-matched cut-off).*
