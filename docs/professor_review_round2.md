> **Historical note (Oct 4, 2026):** this is the original review record and it names models that were later taken out of the project after a line-by-line check against the official Data Science-I syllabus (25MCA3001, DDU GNUMS portal). See the addendum at the end of professor_review_round3.md for what changed.

# Professor Review — Round 2: ColdChain AI (Data Science-I Term Work)

Reviewer: Professor Sharma (strict mode). Re-read in full: src/make_data.py,
src/eda.py, src/features.py, src/train.py, src/plots.py,
results/metrics.csv, README.md, docs/experiments.md. Verified by re-running
the dataset (not the training) and by inspecting all result artefacts.
Code was NOT modified.

Verdict up front: all 13 round-1 fixes are genuinely implemented — this is
not a cosmetic pass, the student did the work. BUT two new errors found
this round are serious enough to block a pass: the regression baseline is
computed with the wrong lag (it flatters every model), and oversampling
before TimeSeriesSplit leaks duplicate rows across CV folds. Fix those,
then we are done.

---

## FIX VERIFICATION (13 claimed fixes)

### 1. Chronological split + TimeSeriesSplit — FIXED
`train.py` has no `train_test_split` anywhere. `chrono_split()` takes the
first 80% of rows for train and the last 20% for test, and every
GridSearchCV/RandomizedSearchCV uses `TimeSeriesSplit(n_splits=5)`.
CV is run on the training block only, so no model ever trains on the
future. Correct.

### 2. Naive baselines added and beaten — FIXED (with a caveat, see New Error A)
`metrics.csv` now contains `Baseline-persistence` (regression) and
`Baseline-majority` (classification). Best regression RMSE 1.0341 beats
baseline 1.5079; best classification accuracy 0.8889 / F1 0.5152 beats
majority 0.8715 / F1 0.0. Caveat: the persistence baseline itself is
computed incorrectly (New Error A) — the comparison is valid in form,
wrong in substance.

### 3. compressor_on dropped — FIXED
`features.py` drops `compressor_on` (with a comment explaining why), and
the engineered feature set contains no `compressor_on` column (verified).
`power_bad` is kept. The raw CSV still has both with corr = −1.0, but raw
data is not fed to any model. Correct.

### 4. Explicit model selection rule — FIXED
`train.py` header states the rule before any training: regression winner
= lowest test RMSE; classification winner = highest test F1, with the
reason given (accuracy misleads at ~14% minority). `docs/experiments.md`
names the champions: Lasso (RMSE 1.0341) and DecisionTreeClf (F1 0.5152).
One documented champion per task. Correct.

### 5. Fair class-imbalance handling — FIXED
`LogisticRegression`, `DecisionTree`, `RandomForest`, `SVC` use
`class_weight="balanced"`; KNN and NaiveBayes (which have no
`class_weight`) are trained on a randomly oversampled copy via the
`oversample()` helper, and the code comment says exactly why. The
leaderboard is now a fair comparison. Caveat: oversampling is applied
BEFORE TimeSeriesSplit, which leaks duplicates across CV folds
(New Error B).

### 6. New evaluation artefacts — FIXED
`results/` contains `confusion_matrix.png` (640×480, valid),
`residuals.png` (640×480, valid) and `feat_importance.png` (700×500,
valid). All three open as real plots. `plots.py` rebuilds the champions
from `metrics.csv` deterministically. Correct.

### 7. Cyclic hour encoding, day dropped — FIXED
`features.py` creates `hour_sin`/`hour_cos` and drops both `hour` and
`day`. Verified: engineered columns contain `hour_sin`, `hour_cos` and
neither `hour` nor `day`. Correct.

### 8. Grids widened, winners off the edges — FIXED (residual edges noted)
KNN grids now go to 35, RandomForest `max_depth` to 15/None. Winners are
interior: Lasso `alpha=0.01` (grid 0.001–1.0), RandomForestClf
`max_depth=15` (grid 5/10/15/None), KNNRegressor `k=25` (grid 3–35).
Residual: several non-winners still sit on grid edges (SVC `C=10` at max,
LogisticRegression `C=0.1` at min, KNNClassifier `k=3` at min,
DecisionTreeReg `max_depth=3` at min, RandomForestReg `max_depth=5` at
min) — their tunes are incomplete, though it does not change the
champions. Note, don't ignore.

### 9. Both excursion types present — FIXED
Verified on the data: 405 excursion rows = 373 heat (>8°C) + 32 freeze
(<2°C). Overall temp range is −1.06 to 13.59°C. The thermostat-fault
mechanism in `make_data.py` works. The classifier is now a genuine
excursion detector.

### 10. 120 days consistent everywhere — FIXED
README says 120 days, `make_data.py` header and `HOURS = 120 * 24` say
120 days, and the raw CSV has 2879 rows (= 120×24 − 1 after the target
shift). All files agree.

### 11. docs/ no longer empty — FIXED
`docs/experiments.md` exists (148 lines): one section per model with
what it is, hyperparameters tried, best params, results, advantages,
disadvantages and a verdict. Minor: README also promises "viva Q&A" and
no such file exists — either add it or fix the README line.

### 12. Name spelling — FIXED
README reads "Zeeshan Ansari (Roll No. 70)". Correct.

### 13. No fake tuning — FIXED
`LinearRegression` and `GaussianNB` are fit directly (`"direct"` branch)
with comments explaining that a grid search would be theatre. Honest.

---

## NEW ERRORS (found this round)

### N1. SERIOUS — the regression baseline uses the wrong lag (train.py, docs/experiments.md)
`train.py` computes the persistence baseline as `base_pred = Xte["temp_lag1"]`.
`temp_lag1` at row t is the temperature at hour t−1, while the target
`temp_next` is the temperature at hour t+1. That is a **two-hour gap**, not
persistence. True persistence ("next hour = current hour") uses `Xte["temp"]`
and gives RMSE ≈ 1.17 (my round-1 verified value); your code reports 1.5079.
`docs/experiments.md` even misdescribes it: "predict that next hour
temperature will be the same as the current hour (temp_lag1)" — `temp_lag1`
is NOT the current hour. Effect: every model looks ~45% better than the
baseline instead of ~12% better. The whole "we beat the baseline" story is
built on a weakened baseline.
**Fix:** use `Xte["temp"]` for the baseline, re-run, and update
metrics.csv + experiments.md with the honest number.

### N2. SERIOUS — oversampling before TimeSeriesSplit leaks across folds (train.py)
`Xtr_os, ytr_os = oversample(Xtr, ...)` duplicates minority rows, and THEN
`GridSearchCV(..., cv=tscv)` splits that oversampled set into folds. The
same physical hour (original + its copies) lands in both the training and
validation folds, so the CV scores that selected KNN's k=3 and NaiveBayes'
smoothing are optimistic, and the KNN/NB test scores are selected on
leaky validation. The comparison is fair in intent, biased in execution.
**Fix:** oversample inside each CV fold (e.g. resample within a manual
TimeSeriesSplit loop), or at minimum document this as a limitation and
show that the test-set ranking is unaffected.

### N3. Latent bug in plots.py regressor branch
The regressor plot builder only handles Ridge/Lasso/LinearRegression.
If any other model ever wins, the `else` branch silently fits a plain
`LinearRegression` and saves it under the winner's name — a mislabeled
residual plot. It happens to be correct today (Lasso won), but the
`SystemExit` guard used in the classifier branch should be used here too.

### N4. README is now stale in three places
(a) "Every model is tuned with GridSearchCV / RandomizedSearchCV using
5-fold cross-validation" — false: LinearRegression and NaiveBayes are
fit directly, and CV is TimeSeriesSplit. (b) No champion named, no
selection rule — a reader must dig into train.py. (c) `pip install`
lists seaborn, which no file imports.

### N5. Undocumented tie-break (metrics.csv, experiments.md)
DecisionTreeClf and RandomForestClf tie at F1 = 0.5152 (accuracy 0.8889
both). `plots.py` picks the first via `idxmax` (the tree) and
`experiments.md` declares the tree the winner, but no tie-break rule is
stated anywhere. "Simpler model wins ties" is a fine rule — write it down.

---

## VERDICT: NEEDS ROUND 3

All 13 round-1 fixes are real — credit where due. But N1 and N2 are
methodological errors, not cosmetics: a wrong baseline and leaky CV are
exactly the things a viva examiner fails students on. Fix N1 and N2
(re-run, update metrics.csv + experiments.md), address N3–N5, then request
Round 3. Round 3 will be short if the numbers are honest.

---

## VIVA QUESTIONS — the 6 most dangerous (prepare these first)

1. RandomForest has the best accuracy and SVC is close, but you ship ONE
   model to guard medicines worth lakhs. Which do you deploy and why is
   accuracy the wrong criterion? What does a false negative cost versus a
   false positive here?
2. Your best classifier accuracy is 0.8889; always predicting "normal"
   gives 0.8715. In one sentence: did your classifier actually learn
   anything useful? Which single number proves it?
3. Your best RMSE is 1.0341 and naive persistence is ~1.17. You are only
   ~12% better than "next hour = this hour". What does that tell you about
   the problem difficulty — and about the features you built?
4. You balanced 4 classifiers with `class_weight` and 2 with oversampling.
   Name TWO other ways to handle class imbalance, and give one trade-off
   of each versus what you did.
5. Your KNN classifier picked k=3 — the smallest value in your grid — and
   your SVC picked C=10, the largest. How do you know the true optimum
   isn't outside the grid?
6. Suppose this goes live and the facility replaces its compressor with a
   stronger one. What happens to your trained model's accuracy over the
   next month? How would you detect it, and what is your retraining plan?

---

*Round 2 complete. Fix N1–N5. Come to Round 3 with honest numbers — I check
the baseline myself.*
