> **Historical note (Oct 4, 2026):** this is the original review record and parts of it name models that were later taken out of the project after a line-by-line check against the official Data Science-I syllabus (25MCA3001, DDU GNUMS portal). See the addendum at the end of this file for what changed.

# Professor Review — Round 3 (FINAL): ColdChain AI (Data Science-I Term Work)

Reviewer: Professor Sharma (strict mode). Re-read in full: src/train.py,
src/plots.py, README.md, results/metrics.csv, docs/experiments.md
(baseline + final-choice sections re-read line by line). Re-ran the
persistence baseline computation independently. Verified all six plot
files open as valid PNGs. Code was NOT modified.

Verdict up front: **PASS.** All five round-2 issues are genuinely fixed —
including the documentation, which I checked twice because I do not trust
anyone on baseline numbers anymore. The methodology is now honest end to
end: correct baseline, no CV leakage, guarded plot builders, accurate
README, documented tie-break. This is submittable term work.

---

## FIX VERIFICATION

### N1. Persistence baseline uses current-hour temp — FIXED
`train.py` now computes `base_pred = Xte["temp"]` with the comment
"naive baseline: next hour = this hour (true persistence)" and a second
comment clarifying that `Xte["temp"]` is the current hour's temp. I
recomputed it myself from `data/coldchain_feat.csv`: true 1-hour
persistence RMSE = **1.1540**, and `metrics.csv` reports **1.154** —
exact match. (For reference, the old wrong-lag value recomputes to
1.5079, confirming the old code was indeed two hours off.) A project-wide
grep for `1.5079` returns zero hits outside my own round-2 report —
`docs/experiments.md` was properly updated: the summary, the
Baseline-persistence section ("Result: RMSE 1.154"), and the Final choice
("It beat the true persistence baseline (RMSE 1.154)") all carry the
honest number, and the old "big margin" language is gone. The story is
now truthful: Lasso 1.0341 vs persistence 1.154, roughly 10% better.

### N2. Oversampling inside CV folds for KNN; NaiveBayes direct — FIXED
`tune_knn()` runs a manual loop over k in [3, 5, 9, 15, 25, 35]; inside
each `TimeSeriesSplit` fold it oversamples **only the fold's training
indices** (`oversample(Xtr.iloc[tri], ytr.iloc[tri])`) and validates on
the untouched `Xtr.iloc[vai]`. No duplicate row can appear on both sides
of a fold. The scaler is inside the per-fold pipeline, so no scaling
leakage either. The final KNN is refit on the oversampled full training
block and scored on the never-oversampled test block — standard and
correct. NaiveBayes is in the `"direct"` branch: fit on the oversampled
train, no fake CV tuning, with an honest comment. The four
`class_weight` models never touch oversampling. The imbalance handling is
now fair in both intent and execution. (Side observation: with honest CV,
KNN's chosen k moved from 3 to 35 — the leaky tune was indeed flattering
small k. The champion is unaffected.)

### N3. plots.py regressor builder raises SystemExit — FIXED
The regressor branch now ends with
`raise SystemExit("plots.py: add builder for regressor " + best["model"])`
instead of silently fitting a mislabeled LinearRegression. The classifier
branch keeps its equivalent guard. No silent mislabeling is possible
anymore.

### N4. README fixed — FIXED
(a) The false "every model tuned" claim is gone: the README now states
"Linear Regression and Naive Bayes have no hyperparameters, so they are
fit directly. The rest are tuned with GridSearchCV / RandomizedSearchCV
using TimeSeriesSplit cross-validation (5 splits, no shuffling of time)."
(b) Champions are named: "winner: Lasso — lowest RMSE" and
"winner: Decision Tree — highest F1". (c) `pip install` lists only
`pandas scikit-learn matplotlib` — no seaborn anywhere in src/, README,
or docs/ (verified by grep).

### N5. Tie-break rule documented — FIXED
`train.py` now carries the comment: "Tie-break rule: if two models tie on
the selection metric, the simpler model wins (fewer params, faster,
easier to explain in viva)." `docs/experiments.md` states the same rule
in its method section AND applies it in the Final choice:
"DecisionTreeClf ... tied RandomForestClf on F1, and our tie-break rule
(decided before training) says the simpler model wins." `plots.py`'s
`idxmax` picks the tree, consistent with the rule. Fully documented,
fully consistent.

---

## ANY NEW ERRORS

**None.** I went looking — re-verified the baseline arithmetic, the CV
loop indices, the plot guards, the README claims, the tie handling, and
the doc numbers. Everything holds. Two cosmetic nits that are NOT errors
and do NOT block the pass:

1. README's tuning sentence doesn't name KNN's manual tuning loop
   explicitly (it says GridSearchCV/RandomizedSearchCV; KNN uses a manual
   TimeSeriesSplit loop). The substance — honest 5-split time-series
   tuning — is true, so this is wording precision, not dishonesty.
2. README still lists "viva Q&A" under docs/, and no viva Q&A file exists
   yet (leftover from round 2, item 11). Either add the file or trim the
   line before final submission.

---

## VERDICT: PASS

Three rounds, and the student earned it. The two serious methodological
errors are fixed with re-run numbers to prove it, and the three minor
ones are addressed. Submit this.

---

## 8 VIVA QUESTIONS (prepare these — I will ask them)

**The silly ones (they sound easy; students fail them):**

1. 120 days × 24 hours = 2880 rows, but your CSV has 2879. Where did one
   row go? Did you lose data?
2. What are `hour_sin` and `hour_cos`? Why couldn't you just hand the
   model `hour = 0..23` directly — isn't 23 just bigger than 0?
3. Two of your raw features had correlation exactly −1.0. Is a perfect
   correlation wonderful news or a problem — and what did you do about it?

**The important ones (these decide your grade):**

4. Your Lasso RMSE is 1.0341 and dumb persistence is 1.154 — barely 10%
   better than just repeating the last reading. In one line: did your
   model actually learn anything, or is it an expensive thermometer?
5. For KNN you oversampled the data. Did you also oversample the test
   set? What exactly would happen to your test scores if you had?
6. DecisionTree and RandomForest tied at F1 = 0.5152, yet you crowned the
   tree. Your friend says "100 trees must beat 1 tree — pick the forest."
   Settle the argument using YOUR documented rule.
7. Your classifier's recall is 0.4595 — your "excursion detector" misses
   more than half of all real excursions, and medicines worth lakhs are
   inside that room. Do you ship this model on Monday? If not, what is the
   first thing you change?
8. Your winning tree uses `max_depth=3` — the smallest value in your grid.
   How do you know depth 2 wouldn't win, or that the true optimum isn't
   sitting outside your grid entirely?

---

*Round 3 complete. PASS. Fix the two cosmetic nits, memorise the answers
to questions 4, 5 and 7 above, and submit.*

---

## Addendum — Oct 4, 2026: syllabus-compliance trim (after PASS)

The student fetched the official Data Science-I syllabus (25MCA3001) from the
DDU GNUMS portal and compared it against every model in the project.

Finding: two of the tried regression models are NOT in the official syllabus
(its linear-regression topic covers only simple and multiple linear
regression — no regularized variants). Everything else in the project
(KNN, Naive Bayes, Logistic Regression, Decision Tree, Random Forest,
SVM/SVR/SVC, all metrics, train-test split, cross-validation, grid/random
search, feature engineering) maps to Units 1–3 and 5.

Action taken: the two out-of-syllabus models were removed from src/train.py
and src/plots.py, and the pipeline was re-run. The regression winner changed
accordingly (now LinearRegression: RMSE 1.0374, R2 0.5149 — still beats the
persistence baseline of 1.1540). Classification winner unchanged
(DecisionTreeClf, F1 0.5152). Updated: results/metrics.csv,
results/residuals.png, docs/experiments.md, README.md, docs/viva_qa.html,
docs/ColdChain_AI_Viva_QA.pdf, docs/ColdChain_AI_Models_Explained.pdf.
Full pre-trim backup kept at ~/workspace/coldchain-ai-backup-20261004.
