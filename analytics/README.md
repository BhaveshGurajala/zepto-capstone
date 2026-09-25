# Module 2 — Analytics Pipeline (`/analytics`)

Profile → clean → visualise → model → evaluate → tune → save, all on the Titanic dataset.

## How to run

From the repo root, after installing `requirements.txt`:

```bash
cd analytics
jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb       # or open and "Run All"
jupyter nbconvert --to notebook --execute --inplace 02_modeling.ipynb  # ~1 minute (GridSearchCV)
python predict.py                                                       # reload the saved pipeline
```

Run the notebooks in order. Both are committed **with their outputs**, so everything below can be checked without running anything.

## Files

| File | Purpose |
|---|---|
| `01_eda.ipynb` | Tasks 1–6: load once, profile, clean, univariate / bivariate / multivariate analysis, z-score check |
| `02_modeling.ipynb` | Tasks 7–15: split, pipeline, 3 classifiers, evaluation, imbalance, GridSearchCV + OOB, regression, comparison, save/reload |
| `titanic.csv` | The one offline copy of the raw dataset, saved by `01_eda.ipynb` right after `sns.load_dataset('titanic')` |
| `cleaning.py` | The shared cleaning rules (threshold rule, row drops, deck category, EDA age imputation) |
| `models/titanic_best_pipeline.joblib` | The full fitted pipeline: preprocessing + tuned Random Forest |
| `predict.py` | Reloads the pipeline with `joblib.load` and predicts on raw rows |
| `charts/` | PNG copies of every chart (supporting files only; all interpretation is in text) |

## One load, one story

`sns.load_dataset('titanic')` appears **once** in the whole module, in the Task 1 load cell of `01_eda.ipynb`. The DataFrame is saved straight away with `df.to_csv("titanic.csv", index=False)`. If `titanic.csv` already exists, that cell reads it instead, so the notebook also runs offline. `02_modeling.ipynb` only ever calls `pd.read_csv("titanic.csv")`. Both notebooks import the same cleaning rules from `cleaning.py`.

Why the saved CSV is the raw data and not the cleaned data: the modeling notebook must impute `age` **after** the train/test split, using training rows only. If the EDA's imputed ages were fed into the model, test-set information would already be baked into the training data. So only the rules that don't learn anything from the data (dropping 2 rows, the "Unknown" deck category) run before the split.

---

# Part A — Profiling, cleaning and the data story

## Task 1 — Profile

891 rows × 15 columns. Target balance: **549 died (61.62%) / 342 survived (38.38%)**. Missing values:

| Column | Missing % |
|---|---|
| deck | 77.22 |
| age | 19.87 |
| embarked | 0.22 |
| embark_town | 0.22 |

`df.info()`, `df.describe()` and `df.shape` are printed in the notebook.

## Task 2 — Missing values

**Missing-value decisions.** I measured each column first, then applied the threshold rule (under 5% → drop rows, 5–30% → impute, above that → too unreliable to impute).

| Column | Missing (measured) | Rule | Decision |
|---|---|---|---|
| `embarked` | **0.22%** (2 rows) | < 5% → drop rows | Drop the 2 rows. |
| `embark_town` | **0.22%** (the same 2 rows) | < 5% → drop rows | Dropped along with `embarked`. It's the same information spelled out as a town name. |
| `age` | **19.87%** (177 rows) | 5–30% → impute | Impute with the **median age of passengers with the same sex and class**. Age changes a lot between groups (1st-class men median 40, 3rd-class women 21.5), so a group median is closer to the truth than one overall median. I used the median, not the mean, because it isn't pulled up by the few elderly passengers. |
| `deck` | **77.22%** (688 rows) | > 30% → too high to impute | **Keep the column and mark missing values as their own category, `"Unknown"`.** |

**Why deck gets an "Unknown" category instead of being dropped:** with 77% missing, any imputed deck would mostly be made up. But whether the deck is missing is itself useful information. The cell below shows 175 of the 203 known decks belong to 1st-class passengers, and survival is **67% when the deck is known vs 30% when it is missing**. Dropping the column would throw that signal away. Filling it in would invent data. A separate `"Unknown"` level keeps the signal without inventing anything.

Only 2 of 891 rows are removed, so **889 rows** remain.

## Task 3 — Univariate analysis

**Outliers (IQR rule).** On the cleaned data, `age` has **32 outliers**: everyone above the upper fence of 57.75 years. There's no lower fence problem, since that fence is below 0. `fare` has **114 outliers**: everyone above £65.66, the expensive 1st-class tickets and the £512 maximum. On the raw, un-imputed ages the age count is only 11. Imputing 177 ages with group medians squeezes the middle of the distribution (the IQR shrinks from 17.9 to 14.5), which pulls the upper fence in. So the 32 is partly an effect of imputation. These are real passengers, not data errors, so they are kept.

**Skewness of `fare`.** **mean (32.10) > median (14.45) > mode (8.05)**. When mean > median > mode, a long tail of large values is pulling the mean to the right. So **`fare` is strongly right-skewed** (skewness = 4.80). Most passengers paid a cheap 3rd-class fare, and a small number of very expensive tickets stretch the tail. The histogram shows it too: a tall spike near £0–£15 and a thin tail out to £512.

## Task 4 — Bivariate analysis

Survival rates, computed with boolean masks (`&`, `|`):

| Group | Survival % |
|---|---|
| (a) female | **74.04** |
| (a) male | **18.89** |
| (b) 1st class | **62.62** |
| (b) 2nd class | **47.28** |
| (b) 3rd class | **24.24** |

(c) sex **and** class together:

| | 1st | 2nd | 3rd |
|---|---|---|---|
| female | **96.74** | **92.11** | **50.00** |
| male | **36.89** | **15.74** | **13.54** |

With `|`: women **or** children under 16 survived at 71.59%, vs 16.39% for adult men.

Correlation matrix: exactly `survived, pclass, age, sibsp, parch, fare`. `adult_male` and `alone` are excluded.

**The two strongest correlations**, ranking all 15 off-diagonal pairs by absolute value:

1. **`pclass` – `fare`: −0.55.** The strongest relationship in the matrix. A higher class number (3rd class) goes with a lower fare. That's expected, since class is mostly what you paid for. It also means the two carry overlapping information for a model.
2. **`sibsp` – `parch`: +0.41.** Passengers with siblings/spouses aboard also tend to have parents/children aboard, because families travelled together. Both columns mostly measure "family size".

A close third is `pclass` – `age` (−0.41): 1st-class passengers were older on average. Part of this comes from imputing age by class, which pushes each class's missing ages to that class's median. For the target, `survived` correlates most with `pclass` (−0.34) and `fare` (+0.26). Sex isn't in this numeric matrix, but the bivariate rates above show it matters more than either (74% vs 19% survival).

## Task 5 — The data story (5 charts)

**Chart 1 — Survival rate by class and sex** (`charts/03_story_class_sex.png`)

**Interpretation.** Sex is the biggest single factor: in every class, women survived far more often than men. Class sets the level within each sex, from 97% of 1st-class women down to only 50% of 3rd-class women, and from 37% of 1st-class men to 14% of 3rd-class men. "Women and children first" was clearly applied, but a woman in 3rd class still had only a coin-flip chance.

**Chart 2 — Age of survivors vs non-survivors, by sex** (`charts/04_story_age_sex.png`)

**Interpretation.** For men, deaths far outnumber survivals at almost every age. The exception is young boys: boys aged 12 and under survived at 57%, versus 9–19% in every other male age band. For women, survivors outnumber deaths at almost every age, so age mainly matters for males, where being a child was nearly as protective as being female.

**Chart 3 — Fare vs age by class, coloured by outcome** (`charts/05_story_fare_age_class.png`)

**Interpretation.** 1st class is mostly blue (survived), and within it survivors paid clearly more (median fare £77 vs £45 for those who died). 3rd class is a dense orange (died) cluster at £7–£8, where fare barely separates the outcomes (median £8.52 vs £8.05). So money helped mostly by buying a better class: a higher fare improved the odds in 1st and 2nd class, but in 3rd class almost everyone paid the same low price.

**Chart 4 — Survival rate by family size and sex** (`charts/06_story_family_size.png`)

**Interpretation.** A small family (2–4 people) was the best position: 81% of women and 32% of men survived, double the rate for men travelling alone (16%). Large families of 5+ did very badly (27% of women, 3% of men); they were mostly in 3rd class and would have struggled to stay together and reach the boats. This up-then-down pattern is why `sibsp` and `parch` show almost no linear correlation with `survived` in the heatmap.

**Chart 5 — Survival rate by port and class** (`charts/07_story_port_class.png`)

**Interpretation.** Cherbourg passengers survived more often than Southampton passengers in every class (69% vs 58% in 1st class), while Southampton's 3rd class was the worst cell: 19% of 353 passengers, the largest group aboard. The port didn't cause survival; Cherbourg simply carried a larger share of wealthy 1st-class passengers, and Southampton most of the 3rd-class men. The Queenstown 1st and 2nd class cells hold only 2–3 people, so those percentages shouldn't be trusted.

**The story in one paragraph.**

Survival on the Titanic was decided mostly by **who you were** (sex and age) and **where you were** (class). Women survived at 74% vs 19% for men, and children, especially boys, were the one group of males with good odds. On top of that, class set a ceiling: 1st-class passengers were closer to the boats and had better access. Nearly all 1st- and 2nd-class women survived, while 3rd-class passengers, especially Southampton's large 3rd-class group and large families, died in the greatest numbers. Fare and port look important on their own, but mostly because they stand in for class. This suggests `sex`, `pclass` and `age` should be the strongest features in the model notebook.

## Task 6 — Exploratory z-score check

| | mean before | std before | mean after | std after |
|---|---|---|---|---|
| age | 29.0654 | 13.2702 | 0.0000 | 1.0000 |
| fare | 32.0967 | 49.6975 | 0.0000 | 1.0000 |

**Result.** After `z = (x − mean) / std`, both columns have **mean 0.0000 and standard deviation 1.0000** (table above), so the transform worked. The shape of each distribution doesn't change. `fare` is exactly as right-skewed after scaling, with a long tail out to about z = 9.7. Standardizing only moves and rescales the values. It doesn't remove skew or outliers.

This was just a sanity check during EDA. The model notebook does **not** reuse these z-scores. It fits its own `StandardScaler` on the training split only, so no test-set information leaks into training.

---

# Part B — Predictive modeling

## Task 7 — Stratified split

**Why stratify?** Task 1 showed the target is imbalanced: **61.6% died vs 38.4% survived** (61.75% / 38.25% on the 889 rows left after dropping the 2 rows with no port; about 1.6 : 1). A plain random 80/20 split could, by chance, put noticeably more or fewer survivors in the 178-row test set. That would make every metric, especially precision and recall for the minority "survived" class, depend on the luck of the split. With `stratify=y` both splits keep the same ratio (38.26% survived in train, 38.20% in test, as the table shows), so the test set is a fair miniature of the full data. The split happens **before** any imputation, encoding or scaling.

## Feature choice

**Which columns go into the model and why.** I use `pclass, sex, age, sibsp, parch, fare, embarked`. The other columns are left out on purpose:

- `alive` is just `survived` written as "yes"/"no". Using it would leak the answer.
- `class`, `embark_town`, `who`, `adult_male` and `alone` are re-codings of columns already in the list (`pclass`, `embarked`, `sex`+`age`, `sibsp`+`parch`), so they add no new information.
- `deck` is 77% "Unknown" and overlaps heavily with `pclass`.

Only row-level rules (`apply_row_rules`) are applied before the split. They drop 2 rows with no port and don't compute any statistics from the data. **`age` is deliberately left with its 177 missing values.** It is imputed inside the pipeline after the split, so the imputation median comes from training rows only.

## Task 8 — Preprocessing

**Preprocessing choices.** Everything sits in a `ColumnTransformer`, wrapped in a `Pipeline` together with the model. Calling `pipeline.fit(X_train, …)` fits the imputers, encoder and scaler on the training rows only. `predict(X_test)` then only calls `transform` on the test rows. The code enforces the fit-on-train rule, so I can't forget it.

| Columns | Missing values | Encoding / scaling |
|---|---|---|
| `pclass, age, sibsp, parch, fare` | `SimpleImputer(median)`. Only `age` actually has gaps. | `StandardScaler` |
| `sex, embarked` | `SimpleImputer(most_frequent)`, a safety net for new data | `OneHotEncoder(handle_unknown="ignore")` |

This is simpler than notebook 01's sex + class group median for `age`. A single train-only median (28.0) is easy to reproduce inside a pipeline, and the trees can still pick up age differences between groups. One-hot encoding is used for `sex` and `embarked` because they have no natural order, and label numbers like 0/1/2 would suggest one. `handle_unknown="ignore"` means an unseen category in new data won't crash the model.

## Task 9 — Three classifiers + the decision tree

Logistic Regression, Decision Tree and Random Forest are all trained on the identical `X_train` / `y_train`. The tree is drawn with `plot_tree(feature_names=..., class_names=["died", "survived"])` (`charts/10_decision_tree.png`).

**Reading the tree.** The root split is `sex_female <= 0.5`, so sex is the most informative feature (the "True" branch on the left is male). On the male side the next split is `age <= 3.5 years`: very young boys mostly survived (11 of 13 in the training data). Among older males, class and fare separate a large group of 343 with only 11% survival. On the female side the split is `pclass <= 2.5`. 1st and 2nd class women survived almost without exception (127 of 134). 3rd-class women are split on fare, and those who paid more than about £23 (large families on a shared ticket) mostly died. The tree has learned the same story the EDA told. I limited it to `max_depth=4` and `min_samples_leaf=5` so it stays readable and doesn't memorize the training set. Numeric thresholds in the plot are in z-score units because the tree sits after the scaler; the table converts them back to years and pounds.

## Task 10 — Evaluation (test set)

| Model | Accuracy | Precision | Recall | F1 | ROC AUC | Confusion matrix [TN FP / FN TP] |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.809 | 0.783 | 0.691 | 0.734 | **0.861** | [97 13 / 21 47] |
| Decision Tree | 0.798 | 0.776 | 0.662 | 0.714 | 0.851 | [97 13 / 23 45] |
| Random Forest | 0.803 | 0.762 | **0.706** | 0.733 | 0.824 | [95 15 / 20 48] |

The ROC curves are in `charts/12_roc_curves.png` and the confusion-matrix plots in `charts/11_confusion_matrices.png`.

**How the three models compare** (test set, 178 passengers, 68 of whom survived):

- **Logistic Regression** has the best AUC (**0.861**) and the highest accuracy of the three untuned models (0.809). Its errors are balanced: 13 false alarms, 21 missed survivors.
- **Decision Tree** is slightly behind on everything (F1 0.714, AUC 0.851). It is the easiest to explain, but a single shallow tree loses some detail.
- **Random Forest (untuned, 300 trees)** has the best recall (**0.706**, 48 of 68 survivors found) but the lowest AUC (0.824). With no depth limit it scores 0.985 on the training data vs 0.803 on test. It is overfitting, and that makes its probability estimates less smooth, which is what AUC measures. Task 12 tunes this.

All three sit close together at 0.80–0.81 accuracy. They mostly miss the same kind of passenger: a man who survived or a 3rd-class woman who died, which are the cases that go against the main pattern.

## Task 11 — Imbalance handling (Logistic Regression)

| Variant | Precision | Recall | F1 | Accuracy | AUC |
|---|---|---|---|---|---|
| (a) baseline | **0.783** | 0.691 | 0.734 | **0.809** | 0.861 |
| (b) class_weight='balanced' | 0.718 | **0.750** | 0.734 | 0.792 | 0.861 |
| (c) SMOTE (train fold only) | 0.735 | 0.735 | **0.735** | 0.798 | **0.867** |

**Conclusion.** The training data is 1.61 : 1 (died : survived). All three variants use Logistic Regression on the same split.

- **(a) Baseline** has the highest precision (0.783) but misses the most survivors (recall 0.691). The model leans toward the majority "died" class.
- **(b) `class_weight='balanced'`** makes each survivor count about 1.6× more in the loss. Recall rises to **0.750** (4 more survivors found), precision falls to 0.718, and F1 stays at 0.734.
- **(c) SMOTE**, applied only inside `fit` on the training fold (439 : 439 after resampling, test set untouched), lands in between: precision = recall = **0.735**. It gives the **best F1 (0.735) and best AUC (0.867)**.

**SMOTE worked best, but only by a small margin.** It gave the best balance between catching survivors and not raising false alarms, and slightly improved ranking quality (AUC). The imbalance here is mild (38% minority), so all three F1 scores are within 0.001 of each other. The real effect of any strategy is to **trade precision for recall**. `class_weight='balanced'` gets most of SMOTE's recall gain with no synthetic rows, so it's the simpler choice if missing a survivor is the costly mistake. SMOTE sits inside an imblearn `Pipeline`, so it only runs during `fit()` and never on test data or during cross-validation scoring.

## Task 12 — GridSearchCV + OOB

**Tuning result.** `GridSearchCV` tried all 36 combinations of `n_estimators` × `max_depth` × `max_features` with 5-fold stratified CV on the training split, scoring by F1. The model was built as `RandomForestClassifier(oob_score=True, …)`.

- **Best parameters: `n_estimators=400`, `max_depth=8`, `max_features=0.5`.**
- **Best CV F1: 0.769.**
- **OOB score: 0.826.** This is accuracy on the training rows each tree didn't see in its bootstrap sample, a free built-in validation estimate. It's close to the test accuracy of 0.820, so the model generalizes as expected.

`max_depth=8` beat `None` (unlimited) in every combination. Capping depth is what fixed the default forest's overfitting. At the best depth (8), using half the features at each split (`0.5`) beat `sqrt` / `log2`, since with only 10 input columns `sqrt` gives each split just 3 to choose from. On the test set the tuned forest improves accuracy (0.803 → **0.820**), precision (0.762 → **0.833**), F1 (0.733 → **0.738**) and AUC (0.824 → 0.839), at the cost of some recall (0.706 → 0.662).

## Task 13 — Regression: predicting `fare`

Features: `pclass, sex, age, sibsp, parch, embarked, survived`, using the same train/test rows as the classifiers and the same kind of pipeline (median impute + scale, one-hot with `drop="first"`).

**Results.** On the same 178 test rows the linear model gets **MAE = £19.75, RMSE = £41.27, R² = 0.347, Adjusted R² = 0.316**. Adjusted R² uses n = 178 and p = 8 predictors after one-hot encoding. About a third of the variation in fare is explained, mostly from `pclass` (−£26 per standard deviation) and port, with family size (`sibsp`, `parch`) adding a smaller share. RMSE is about twice MAE, which means a few very large errors dominate, not many medium ones.

**Heteroscedasticity: yes, clearly.** The residual plot is not a random, even band around zero. It fans out. For low predicted fares the residuals are tight (mean |error| **£9.71**). For the top third of predictions the mean |error| is **£35.40** with a standard deviation of **£59.20**, including one ticket under-predicted by over £400. The size of the error grows with the prediction (corr(|residual|, predicted) = 0.32). The residual histogram is also right-skewed rather than bell-shaped, and the model even predicts slightly negative fares for some 3rd-class passengers. All of this follows from `fare` being heavily right-skewed (notebook 01). A linear model on raw fare breaks the constant-variance assumption, so its confidence intervals would be unreliable. Modelling `log(fare)` instead would be the natural next step.

## Task 14 — Model comparison

| Model | Accuracy | Precision | Recall | F1 | ROC AUC | ‖ | MAE (£) | RMSE (£) | R² | Adj. R² |
|---|---|---|---|---|---|---|---|---|---|---|
| | *classification (0–1)* | | | | | ‖ | *regression* | | | |
| Logistic Regression | 0.809 | 0.783 | 0.691 | 0.734 | **0.861** | ‖ | — | — | — | — |
| Decision Tree | 0.798 | 0.776 | 0.662 | 0.714 | 0.851 | ‖ | — | — | — | — |
| Random Forest (untuned, 300 trees) | 0.803 | 0.762 | **0.706** | 0.733 | 0.824 | ‖ | — | — | — | — |
| **Random Forest (tuned)** | **0.820** | **0.833** | 0.662 | **0.738** | 0.839 | ‖ | — | — | — | — |
| Linear Regression (fare) | — | — | — | — | — | ‖ | 19.753 | 41.270 | 0.347 | 0.316 |

The table keeps the two model types apart on purpose. Classification metrics are proportions between 0 and 1, where higher is better. Regression metrics are in **pounds** (MAE, RMSE, where lower is better) or are variance-explained ratios (R², Adjusted R²). Numbers in one group can't be compared with numbers in the other.

### Recommendation

**I would deploy the tuned Random Forest**, because it has the highest test accuracy (**0.820**), precision (**0.833**) and F1 (**0.738**) of the four classifiers. Its OOB score (0.826) and CV F1 (0.769) agree with the test result, so the gain isn't a lucky split. Logistic Regression is a strong runner-up, with the best AUC (**0.861**) and an F1 only 0.004 lower, so it would be the better pick if the product needed well-ranked probabilities or an easily explained model. The tuned forest's weak spot is recall (0.662). If missing a survivor were the expensive mistake, I would lower its decision threshold or add `class_weight='balanced'`, which raised Logistic Regression's recall by about 6 points in Task 11.

## Task 15 — Saved pipeline

`joblib.dump(best_pipeline, "models/titanic_best_pipeline.joblib")` saves the whole `Pipeline`: the `ColumnTransformer` (imputers, one-hot encoder, scaler) plus the tuned `RandomForestClassifier`. It is not the bare estimator. The last cells of `02_modeling.ipynb` and `predict.py` reload it with `joblib.load`. They confirm it gives **identical predictions** on the test set (accuracy 0.820 after reload) and that it predicts on **raw** new rows, including a passenger with a missing age:

```
 pclass    sex  age  sibsp  parch   fare embarked  P(survive) prediction
      1 female 29.0      0      0 110.00        C       0.999   survived
      3   male 24.0      0      0   7.25        S       0.078       died
      3 female  NaN      1      3  25.47        Q       0.580   survived
```

The file was saved with scikit-learn 1.8. Loading it needs the same scikit-learn version (pinned in `requirements.txt`). If you use a different version, re-run `02_modeling.ipynb` to regenerate it.
