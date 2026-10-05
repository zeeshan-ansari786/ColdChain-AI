# ColdChain AI: Temperature Excursion Prediction System

**Members:** Hiren Patel (Roll No. 68), Zeeshan Ansari (Roll No. 70)
**Subject:** Data Science - 1 (Term Work)

## What this project does

Cold storage rooms (for medicines, vaccines, food) must stay between 2 and 8
degrees. If the temperature goes out of this range, the stored items get
spoiled. Checking sensor data by hand is slow, so we built an ML system that:

1. **Regression task** - predicts the room temperature of the next hour
2. **Classification task** - predicts whether the next hour will have a
   temperature excursion (temp going below 2 or above 8 degrees)

## Files

- `src/make_data.py` - generates the sensor dataset (hourly readings, 120 days)
- `src/eda.py` - exploratory data analysis + plots
- `src/features.py` - feature engineering (lag features, rolling average)
- `src/train.py` - trains all models with hyperparameter tuning
- `ColdChain_AI.py` - the whole pipeline in ONE file (with comments)
- `ColdChain_AI_nocomments.py` - same one file, without comments
- `ColdChain_AI_TermWork.ipynb` - the whole pipeline as a Jupyter notebook (each step in its own cell)
- `ColdChain_AI.ipynb` - same notebook, named as the project title
- `data/` - datasets
- `results/` - plots and `metrics.csv` with all model scores
- `docs/` - experiment notes and viva Q&A

## How to run

```
pip install pandas scikit-learn matplotlib
python src/make_data.py
python src/eda.py
python src/features.py
python src/train.py
python src/plots.py
```

Or open `ColdChain_AI_TermWork.ipynb` in Jupyter Notebook — the whole
pipeline (data → EDA → features → training → results → plots) is there
with each step in its own cell, already executed with outputs.

## Models tried

Regression: Linear Regression, KNN, Decision Tree,
Random Forest, SVR (winner: Linear Regression — lowest RMSE).
Every model used is strictly from the official Data Science-I syllabus
(25MCA3001, verified against the DDU GNUMS portal syllabus PDF on
Oct 4, 2026) — nothing outside the syllabus is used.

Classification: Logistic Regression, KNN, Naive Bayes, Decision Tree,
Random Forest, SVM (winner: Decision Tree — highest F1)

Linear Regression and Naive Bayes have no hyperparameters, so they are
fit directly. The rest are tuned with GridSearchCV / RandomizedSearchCV
(KNN uses its own manual TimeSeriesSplit loop with in-fold oversampling)
using TimeSeriesSplit cross-validation (5 splits, no shuffling of time).
Full results are in `results/metrics.csv`.
