# Experiment Notes — ColdChain AI

We made fake sensor data for a cold storage room: 120 days of hourly readings (2880 rows). About 14% of the hours are temperature excursions, and there are two kinds — heat (temp goes above 8C) and freeze (temp goes below 2C). Task 1 is regression (predict next hour temperature). Task 2 is classification (predict if next hour will be an excursion or not).

Since this is time data, we did NOT shuffle it. First 80% of hours are train, last 20% are test. For cross-validation we used TimeSeriesSplit with 5 splits, so the model never trains on the future and tests on the past. For classification we tuned on F1, because only ~14% of hours are excursions and accuracy alone would mislead. KNN and NaiveBayes have no class_weight option, so for them we oversampled the minority class — but always INSIDE each CV fold, never before the split, so the validation scores stay honest.

Model selection rule (fixed before training): regression winner = lowest RMSE on test set. Classification winner = highest F1 on test set. Tie-break: if two models tie on the metric, the simpler model wins (fewer params, faster, easier to explain in viva).

Summary of results: **LinearRegression won regression** (RMSE 1.0374, R2 0.5149). **DecisionTreeClf won classification** (accuracy 0.8889, F1 0.5152). Both winners beat their naive baselines — persistence baseline RMSE was 1.154, and the majority-class baseline got F1 0.0.

Note (Oct 4, 2026): every model below is taken strictly from the official Data Science-I syllabus (25MCA3001), verified line-by-line against the DDU GNUMS portal syllabus PDF. Nothing outside the syllabus is used.

## Baseline-persistence — Task: regression
- What it is: A dumb rule — predict that next hour temperature will be the same as this hour's temperature.
- Hyperparameters tried: none — this is a baseline, not a trained model.
- Best params found: -
- Result: RMSE 1.154, MAE 0.8104, R2 0.3997.
- Baselines beaten: this IS the baseline. Every real model must beat RMSE 1.154, and they all did.
- Advantages: Needs no training. Very easy to understand.
- Disadvantages: It cannot predict any change. It just repeats the current value.
- Verdict: baseline only — it exists so we can check that our real models actually learned something.

## LinearRegression — Task: regression
- What it is: It draws a straight line through the data. The simplest model in our syllabus.
- Hyperparameters tried: none — fit directly. LinearRegression has no hyperparameters, so running a grid search on it would be a fake search.
- Best params found: {} (no tuning)
- Result: RMSE 1.0374, MAE 0.7046, R2 0.5149. Training time: 0.0 s.
- Advantages: Very easy to explain in viva. Trains instantly.
- Disadvantages: Can only learn straight-line relations. If two features say the same thing, the weights become unstable.
- Verdict: CHOSEN — lowest RMSE of every syllabus model (1.0374), trains instantly, and it is the first topic in Unit 2 of the official syllabus.

## KNNRegressor — Task: regression
- What it is: To predict, it finds the k most similar hours from training data and averages their temperatures.
- Hyperparameters tried: n_neighbors = [3, 5, 9, 15, 25, 35] with GridSearchCV.
- Best params found: n_neighbors = 25
- Result: RMSE 1.0941, MAE 0.7322, R2 0.4604. Training time: 0.5 s.
- Advantages: No assumptions about the data shape. Simple idea, easy to explain.
- Disadvantages: Slow at prediction time on big data (it compares with every training row). RMSE 1.0941 is clearly worse than LinearRegression.
- Verdict: NOT CHOSEN — worse score than the linear models, and it does not scale well.

## DecisionTreeReg — Task: regression
- What it is: It splits the data with yes/no questions (like "is door_open_min > 10?") until it can predict the temperature.
- Hyperparameters tried: max_depth = [3, 5, 8, 12, None], min_samples_split = [2, 5, 10] with GridSearchCV.
- Best params found: max_depth = 3, min_samples_split = 2
- Result: RMSE 1.107, MAE 0.7681, R2 0.4476. Training time: 0.8 s.
- Advantages: Very easy to draw and explain. Needs no feature scaling.
- Disadvantages: Worst score of all trained models (RMSE 1.107). Single trees overfit easily.
- Verdict: NOT CHOSEN — the worst regressor we tried.

## RandomForestReg — Task: regression
- What it is: Many decision trees trained on random parts of the data, then their predictions are averaged.
- Hyperparameters tried: n_estimators = [50, 100, 200], max_depth = [5, 10, 15, None], min_samples_split = [2, 5] with RandomizedSearchCV (10 tries).
- Best params found: n_estimators = 200, min_samples_split = 5, max_depth = 5
- Result: RMSE 1.0612, MAE 0.7228, R2 0.4923. Training time: 29.0 s.
- Advantages: Handles non-linear relations. More stable than a single tree.
- Disadvantages: Slowest model we trained (29 seconds for 200 trees). Still worse than LinearRegression. Hard to explain tree-by-tree in viva.
- Verdict: NOT CHOSEN — slower and worse than LinearRegression.

## SVR — Task: regression
- What it is: Support Vector Regression — it tries to fit a tube around the data and ignores small errors inside the tube.
- Hyperparameters tried: C = [0.1, 1, 10], gamma = ["scale", 0.01, 0.1] with GridSearchCV.
- Best params found: C = 10, gamma = 0.01
- Result: RMSE 1.0573, MAE 0.642, R2 0.496. Training time: 3.8 s.
- Advantages: Good at non-linear patterns. Interesting — it got the lowest MAE of all models (0.642).
- Disadvantages: RMSE (our selection metric) is worse than LinearRegression's. Sensitive to C and gamma, and needs scaled features.
- Verdict: NOT CHOSEN — lowest MAE is nice, but our rule picks on RMSE, and LinearRegression wins there.

## Baseline-majority — Task: classification
- What it is: A dumb rule — always predict "no excursion" (the majority class).
- Hyperparameters tried: none — this is a baseline, not a trained model.
- Best params found: -
- Result: accuracy 0.8715, precision 0.0, recall 0.0, F1 0.0.
- Baselines beaten: this IS the baseline. It shows why accuracy misleads — 0.8715 accuracy looks good, but F1 is 0.0 because it never catches a single real excursion.
- Advantages: Needs no training.
- Disadvantages: It catches zero excursions, so it is useless for our actual goal.
- Verdict: baseline only — it proves we must judge models by F1, not accuracy.

## LogisticRegression — Task: classification
- What it is: It draws a straight boundary between "excursion" and "no excursion" and gives a probability.
- Hyperparameters tried: C = [0.1, 1.0, 10.0] with GridSearchCV, scoring on F1. Trained with class_weight="balanced".
- Best params found: C = 0.1
- Result: accuracy 0.8264, precision 0.3839, recall 0.5811, F1 0.4624. Training time: 0.3 s.
- Advantages: Fast and easy to explain. Gives probabilities, not just yes/no.
- Disadvantages: Can only learn a straight boundary. F1 0.4624 is below our winner.
- Verdict: NOT CHOSEN — decent, but DecisionTreeClf beat it on F1.

## KNNClassifier — Task: classification
- What it is: To classify, it looks at the k most similar hours and takes a majority vote.
- Hyperparameters tried: k = [3, 5, 9, 15, 25, 35] with a manual CV loop — inside each fold the minority class was oversampled, so the validation scores stay honest.
- Best params found: n_neighbors = 35
- Result: accuracy 0.7882, precision 0.3261, recall 0.6081, F1 0.4245. Training time: 1.2 s.
- Advantages: Highest recall of all models (0.6081) — it catches the most excursions. Simple idea.
- Disadvantages: Lowest accuracy (0.7882) and lowest precision — it raises too many false alarms. F1 0.4245 is well below the winner.
- Verdict: NOT CHOSEN — catches a lot but cries wolf too often.

## NaiveBayes — Task: classification
- What it is: It uses Bayes theorem and assumes every feature is independent of the others.
- Hyperparameters tried: none — nothing to tune, so we fit it directly on the oversampled training data (a grid search here would be a fake search).
- Best params found: {} (no tuning)
- Result: accuracy 0.7986, precision 0.325, recall 0.527, F1 0.4021. Training time: 0.0 s.
- Advantages: Trains instantly. Works okay on small data.
- Disadvantages: The independence assumption is wrong here (temp and humidity move together). Lowest F1 of all trained models.
- Verdict: NOT CHOSEN — weakest classifier we tried.

## DecisionTreeClf — Task: classification
- What it is: It splits the data with yes/no questions (like "is temp > 7.5?") until it can say excursion or not.
- Hyperparameters tried: max_depth = [3, 5, 8, 12, None], min_samples_split = [2, 5, 10] with GridSearchCV, scoring on F1. Trained with class_weight="balanced".
- Best params found: max_depth = 3, min_samples_split = 2
- Result: accuracy 0.8889, precision 0.5862, recall 0.4595, F1 0.5152. Training time: 0.9 s.
- Advantages: Highest F1 of all models (0.5152, tied with RandomForest). Very easy to draw and explain in viva. Fast.
- Disadvantages: Single trees can overfit — but here the best tree was shallow (depth 3), so it stayed simple. Recall 0.4595 means it still misses about half the excursions.
- Verdict: CHOSEN — tied RandomForestClf on F1, and our tie-break rule says the simpler model wins.

## RandomForestClf — Task: classification
- What it is: Many decision trees trained on random parts of the data, then they vote.
- Hyperparameters tried: n_estimators = [50, 100, 200], max_depth = [5, 10, 15, None], min_samples_split = [2, 5] with RandomizedSearchCV (10 tries), scoring on F1. Trained with class_weight="balanced".
- Best params found: n_estimators = 100, min_samples_split = 2, max_depth = 15
- Result: accuracy 0.8889, precision 0.5862, recall 0.4595, F1 0.5152. Training time: 12.0 s.
- Advantages: Same best F1 as the decision tree (0.5152). Usually more stable than a single tree.
- Disadvantages: 100 trees take 12 seconds to train vs 0.9 for one tree. Much harder to explain in viva.
- Verdict: NOT CHOSEN — tied the decision tree on F1, but the tie-break rule picks the simpler model.

## SVC — Task: classification
- What it is: Support Vector Classifier — it finds the boundary with the biggest gap between the two classes.
- Hyperparameters tried: C = [0.1, 1, 10], gamma = ["scale", 0.01, 0.1] with GridSearchCV, scoring on F1. Trained with class_weight="balanced".
- Best params found: C = 10, gamma = 0.01
- Result: accuracy 0.8611, precision 0.4651, recall 0.5405, F1 0.5. Training time: 2.6 s.
- Advantages: Close to the winner (F1 0.5). Good balance of precision and recall.
- Disadvantages: Still below 0.5152. Sensitive to C and gamma, needs scaled features, harder to explain than a tree.
- Verdict: NOT CHOSEN — second best, but the rule picks the highest F1.

## Final choice

- **Regression task (predict next hour temperature): LinearRegression** — RMSE 1.0374, R2 0.5149. It beat the true persistence baseline (RMSE 1.154) and every other syllabus model, trains instantly, and is the first topic of Unit 2 in the official syllabus.
- **Classification task (predict excursion): DecisionTreeClf** — accuracy 0.8889, F1 0.5152. It tied RandomForestClf on F1, and our tie-break rule (decided before training) says the simpler model wins: one shallow tree (depth 3) vs 100 trees.

Honest limits we admit: the classification winner still misses about half the real excursions (recall 0.4595), and the majority baseline's 0.8715 accuracy shows why we judged on F1 instead of accuracy. The regression R2 of 0.5149 means the model explains about half the temperature variation — good enough for a student project, not production-ready.
