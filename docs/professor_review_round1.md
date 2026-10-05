> **Historical note (Oct 4, 2026):** this is the original review record and it names models that were later taken out of the project after a line-by-line check against the official Data Science-I syllabus (25MCA3001, DDU GNUMS portal). See the addendum at the end of professor_review_round3.md for what changed.

# Professor Review — Round 1: ColdChain AI (Data Science-I Term Work)

Reviewer: Professor Sharma (strict mode). Project read in full: README.md,
src/make_data.py, src/eda.py, src/features.py, src/train.py,
results/metrics.csv, plus the proposal (Data_Science_1_hpw.pdf).
Code was NOT modified. Numbers below were verified by re-running the
dataset (not the training).

Verdict up front: the pipeline runs end-to-end and stays inside the
syllabus, which is good. But the evaluation methodology has one serious
flaw, the models barely beat naive baselines, and several small sloppiness
issues would cost marks in a viva. Fix everything below before Round 2.

---

## ERRORS TO FIX (numbered, most serious first)

### 1. You shuffled a time series — your test scores are optimistic (train.py)
`train.py` uses `train_test_split(..., random_state=42)` (shuffle=True, the
default) for BOTH tasks. This data is hourly sensor readings with strong
autocorrelation (temp at hour t depends on temp at t−1; your own simulator
uses an AR(0.6) process). Shuffling interleaves test hours between training
hours, so the model is evaluated on points whose temporal neighbours it has
already seen. In deployment you only ever have the PAST — never the future.
**Fix:** use a chronological split (e.g. first ~80% of days for training,
last ~20% for testing) or `sklearn.model_selection.TimeSeriesSplit`. Keep
the shuffled split only if you can defend why — you currently cannot.

### 2. No naive baselines — so you cannot prove the models learned anything (train.py, results/metrics.csv)
You report RMSE ≈ 1.02 and accuracy ≈ 0.89 with no baseline next to them.
Verified on your data:
- Regression: naive persistence baseline ("predict temp_next = current
  temp") gives RMSE = **1.17**. Your best model (Ridge, RMSE 1.017) is only
  ~13% better than guessing "no change".
- Classification: majority-class baseline ("always predict normal") gives
  accuracy = **0.86**. Your KNN classifier's 0.889 is just 3 points above
  doing nothing.
**Fix:** add both baselines as rows in metrics.csv and discuss the gap. A
professor's first question will be "better than what?" — right now you have
no answer.

### 3. Perfectly duplicated feature: `power_bad` and `compressor_on` (make_data.py, train.py)
In `make_data.py`, `compressor = 1 - power_bad`. Verified on your data:
corr(power_bad, compressor_on) = **−1.0 exactly**. You feed both into every
model. For LinearRegression this makes the design matrix singular and the
coefficients uninterpretable (sklearn silently absorbs it via SVD, but any
"feature importance" story you tell from linear coefficients is garbage).
**Fix:** drop one of them (keep `power_bad`) and re-run.

### 4. No documented model-selection rule — accuracy and F1 crown different winners (results/metrics.csv)
Classification results: RandomForest has the best **accuracy** (0.8872) but
SVC has the best **F1** (0.5309). Your files never state which metric picks
the winner or which model you would deploy. On a 14%-minority problem,
accuracy is the wrong criterion (see Error 2).
**Fix:** state the selection metric up front (should be F1 or
recall-oriented, given a missed excursion spoils medicines) and name ONE
champion model per task with justification.

### 5. Inconsistent class-imbalance handling makes the comparison unfair (train.py)
You set `class_weight="balanced"` for LogisticRegression, DecisionTree,
RandomForest and SVC — but KNN and NaiveBayes get nothing (they don't
support it, which is exactly the point: say so). Right now the leaderboard
compares balanced models against unbalanced ones.
**Fix:** either note this explicitly as a limitation, or put all models on
equal footing (e.g. tune the decision threshold, or use resampling) and
re-run.

### 6. Missing evaluation artefacts (results/)
`results/` has EDA plots only. There is no confusion matrix, no ROC/PR
curve, no predicted-vs-actual scatter, no residual plot, no feature
importances. For an imbalanced alerting task, a confusion matrix and a
precision–recall discussion are near-mandatory.
**Fix:** add at least: confusion matrix for the champion classifier,
predicted-vs-actual + residuals for the champion regressor.

### 7. Bad encodings for cyclic/ordinal features (make_data.py, train.py)
`hour` is fed as a raw integer 0–23, so hour 23 and hour 0 look maximally
different to KNN/SVM/linear models although they are adjacent. `day` is
`(t // 24) % 7` — an arbitrary Monday alignment with false ordinality
(Sunday=6 > Monday=0 means nothing).
**Fix:** encode hour cyclically (sin/cos) or one-hot; either drop `day` or
one-hot it. Justify the choice in the report.

### 8. Hyperparameter grids hit their edges — optimum may lie outside (train.py, metrics.csv)
KNN (both tasks) picks `n_neighbors=15`, the LARGEST value in the grid.
RandomForestReg picks `max_depth=5`, the SHALLOWEST non-None value.
You have no evidence the true optimum is inside your grid.
**Fix:** widen the grids (e.g. k up to 31; max_depth up to 15–20) and
re-run, or explicitly note it as a limitation.

### 9. Your "excursion" class is one-sided — the model has never seen a freeze (make_data.py, data)
Verified: all 403 excursion rows are temp > 8°C; **zero** are temp < 2°C
(overall temp range 3.43–13.10°C). The classifier is really a
"too-hot detector", not an excursion detector.
**Fix:** either make the simulator produce sub-2°C events too, or rename
and scope the task honestly ("high-temperature excursion").

### 10. 90 days vs 120 days — your own files disagree (README.md, make_data.py)
README says "hourly readings, 90 days"; `make_data.py` header comment says
"90 days"; the code sets `HOURS = 120 * 24` (120 days, 2877 rows after
shifts). Pick one number and make every file agree.

### 11. README promises docs that do not exist (README.md, docs/)
README: "docs/ - experiment notes and viva Q&A". The `docs/` directory is
empty. Either write those files or fix the README.

### 12. Name typo (README.md, proposal)
"Zeeshan Anasari (Roll No. 70)" — the student's name is Zeeshan **Ansari**.
The typo is also in the proposal PDF. Fix everywhere before submission.

### 13. Theatre-of-tuning (train.py)
`GridSearchCV(LinearRegression(), {})` — a grid search over an empty grid
tunes nothing; call it a plain 5-fold CV or just fit. Tuning
`GaussianNB(var_smoothing=[1e-9, 1e-8, 1e-7])` moves the score by ~0.000.
Don't pad the "tuning" story; a viva examiner will ask what each grid
actually changed.

---

## SILLY QUESTIONS (answer every one before the viva)

1. What does `random_state=42` do? What changes in your results if you set
   it to 0, or delete it everywhere?
2. You scaled features for KNN, SVM, Ridge and Lasso but NOT for decision
   trees and random forests. Why? What breaks if you scale trees anyway?
3. `corr(power_bad, compressor_on)` is exactly −1.0. Why do both columns
   exist, and what does that do to your LinearRegression coefficients?
4. Your KNN classifier accuracy is 0.889. The majority baseline is 0.860.
   In one sentence: did KNN actually learn anything useful?
5. You ran `GridSearchCV` on `LinearRegression()` with an empty param grid
   `{}`. What, exactly, was "searched"?
6. Best KNN uses `n_neighbors=15` — the largest value in your grid. How do
   you know 25 isn't better?
7. You used `stratify=y_clf` in the classification split but not in the
   regression split. What does `stratify` do, and why the difference?
8. Why 5-fold CV inside GridSearchCV? Why not tune on a single
   train/validation split — what does the averaging buy you?
9. What is the difference between GridSearchCV and RandomizedSearchCV, and
   why did you use the random one ONLY for the two RandomForest models?
10. `temp_roll3 = df["temp"].rolling(3).mean()` — which hours go into the
    average for row t? Does it peek at the future? (Show the indices.)
11. You called `precision_score(yte, pred)` with no `average` argument.
    What does it compute on binary labels, and which class is "positive"?
12. Your `day` column is `(t // 24) % 7` starting from t=0. Which weekday is
    t=0, and how would you prove it?

---

## IMPORTANT QUESTIONS (deep — prepare 4–5 line answers)

1. You shuffled time-series data with `train_test_split`. Explain precisely
   WHY that is wrong for sensor data (autocorrelation, future leakage,
   deployment reality) and what validation scheme replaces it.
2. RandomForest has the best accuracy (0.887) but SVC the best F1 (0.531)
   on the excursion task. You are deploying ONE model to guard medicines
   worth lakhs. Which do you ship, and why is accuracy the wrong answer
   here? What does a false negative cost vs a false positive?
3. The excursion class is 14% of hours. You balanced 4 of 6 classifiers.
   Is the leaderboard fair? Name TWO other ways to handle imbalance
   (resampling, threshold moving, etc.) and their trade-offs.
4. Best regression RMSE is 1.017; naive persistence gives 1.17. Your model
   is only slightly better than "tomorrow = today". What does that tell
   you about the problem difficulty — and about the features you built?
5. Why do lag-1/lag-2 temperature features help at all? Connect them to the
   physics: what property of a cold room does your simulator's
   `0.60 * temp[i-1]` term represent, and which feature captures it?
6. Bias–variance: DecisionTreeReg with max_depth=3 underfits (R² 0.54);
   RandomForest with 200 trees at depth 5 does better (R² 0.58). Explain the
   bias-variance trade-off across your grids, and why the forest's best
   depth is shallow here.
7. All 403 excursions in your data are over-heating events; there is not a
   single sub-2°C freeze. What can your classifier NOT do, and how would
   this fail in a real cold room?
8. `hour` as a raw 0–23 integer: explain geometrically why this encoding
   misleads KNN and SVM, and give the correct encoding.
9. Suppose this goes live and the facility replaces its compressor with a
   stronger one. What happens to your trained model over the next month?
   (Concept drift — how would you detect it, and what is your retraining
   plan?)
10. With REAL sensor data instead of your simulator: name three things that
    break (missing readings, sensor spikes/drift, unlogged door events…)
    and, for each, the preprocessing step from your syllabus you would add.

---

## MISSING FROM SYLLABUS CHECK

Allowed (per task brief): LinearRegression, Ridge, Lasso,
LogisticRegression, KNN, NaiveBayes, DecisionTree, RandomForest, SVM,
with GridSearchCV / RandomizedSearchCV tuning.

Audit of `train.py` imports and usage:
- Regression: LinearRegression ✓, Ridge ✓, Lasso ✓, KNeighborsRegressor ✓,
  DecisionTreeRegressor ✓, RandomForestRegressor ✓, SVR ✓
- Classification: LogisticRegression ✓, KNeighborsClassifier ✓,
  GaussianNB ✓, DecisionTreeClassifier ✓, RandomForestClassifier ✓,
  SVC ✓
- Preprocessing/search: StandardScaler ✓, Pipeline ✓, train_test_split ✓,
  GridSearchCV ✓, RandomizedSearchCV ✓
- Metrics: RMSE (via mean_squared_error), MAE, R², accuracy, precision,
  recall, F1 ✓

**Result: CLEAN.** Every model, preprocessor, search method and metric used
is inside the Data Science-I syllabus named above. Nothing exotic
(no XGBoost, no neural networks, no PCA, no external AutoML). No syllabus
violation to flag.

---

*Round 1 complete. Fix Errors 1–9 at minimum, then request Round 2. Come to
the viva able to answer every silly question in under 30 seconds each.*
