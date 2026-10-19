# QuickLoan — Submission

| File | What it is | Brief item |
|---|---|---|
| `QuickLoan_Notebook.ipynb` (+ `.html`) | Executed notebook, outputs visible: load/split → weak singles → Voting, Bagging, Boosting, Stacking → leaderboard → CV check → cost-matched cut-off → verdict → stretch | 1 |
| `leaderboard.csv`, `leaderboard_chart.png` | Leaderboard: accuracy, recall, money saved for every model (same test set) | 2 |
| `Verdict.md` | Recommendation to leadership (also in the notebook) | 3 |
| `loan_model.joblib`, `app.py`, `loans.csv` | Stretch: dashboard with decision + money cards | 4 |
| `loan_model_costaware.joblib` | Same champion with the recommended 1/6 cut-off | extra |
| `leaderboard_cost_aware_threshold.csv`, `cv_robustness_table.csv`, `cv_money_chart.png` | Supporting evidence | extra |

**Run the dashboard:** `pip install pandas numpy scikit-learn xgboost matplotlib seaborn streamlit joblib`, then `streamlit run app.py` in this folder.
To show the recommended production setting, rename `loan_model_costaware.joblib` to `loan_model.joblib`.

**Re-run the notebook:** keep it next to `loans.csv`; it takes about 8 minutes (the 15-fold cross-validation dominates) and regenerates every file above.
The `.joblib` files were saved with scikit-learn 1.8; with an older version, re-run the notebook to regenerate them (the cost-aware file needs scikit-learn >= 1.6).
