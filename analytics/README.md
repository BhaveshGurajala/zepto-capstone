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
| `age` | **19.87%** (177 rows) | 5–30% → impute | Impute. I filled each missing age with the median age of passengers with the same sex and class, because different groups have very different ages (the median is 40 for 1st-class men but 21.5 for 3rd-class women), so one number for everyone would not give appropriate values. I used the median instead of the mean because a few very old passengers would pull the mean up. |
| `deck` | **77.22%** (688 rows) | > 30% → too high to impute | **Keep the column and mark missing values as their own category, `"Unknown"`.** |

**Why deck gets an "Unknown" category instead of being dropped:** The `deck` column is missing for 77.22% of passengers, which is above the 30% limit, so filling it in would not be reliable. But the missing values are not random: of the 203 passengers whose deck is known, 175 are in 1st class. Survival is also very different: 67% when the deck is known and 30% when it is missing. So instead of deleting the column, I kept it and marked the missing values as "Unknown", because whether the deck is missing is itself useful information.

Only 2 of 891 rows are removed, so **889 rows** remain.

## Task 3 — Univariate analysis

**Outliers (IQR rule).** `age` has 32 outliers, all passengers older than about 58 (the upper fence is 57.75). On the raw data, before the missing ages were filled in, it was only 11: filling 177 missing ages with medians packs the middle of the data closer together, so more old passengers land outside the range. `fare` has 114 outliers, meaning every fare above £65.66, mostly expensive 1st-class tickets, with the highest at £512. I kept all of them because they are real passengers, not data errors.

**Skewness of `fare`.** `fare` is right-skewed. The mean is 32.10, the median is 14.45 and the mode is 8.05. Since mean > median > mode, a few very large values are pulling the mean up, and the skewness value of 4.80 shows the skew is strong. This happens because most passengers paid cheap 3rd-class fares of around £8, while a few expensive tickets up to £512 stretch the tail to the right. The histogram shows the same thing: a tall bar near £0–15 and then a long thin tail.

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

**The two strongest correlations.** Of all the pairs, the two strongest are:

1. **`pclass` – `fare`: −0.55.** This is the strongest pair. A higher class number (3rd class) goes with a lower fare, which makes sense because you pay more for 1st class. For a model, it also means these two columns carry overlapping information.
2. **`sibsp` – `parch`: +0.41.** People with siblings or a spouse aboard also tend to have parents or children aboard, because families travel together. So both columns really measure family size.

## Task 5 — The data story (5 charts)

**Chart 1 — Survival rate by class and sex** (`charts/03_story_class_sex.png`)

**Interpretation.** Sex matters most: in every class, women survived far more often than men (97% vs 37% in 1st class). Class matters too, because survival drops from 1st to 3rd class for both women and men. "Women and children first" was followed, but even a woman in 3rd class had only a 50/50 chance.

**Chart 2 — Age of survivors vs non-survivors, by sex** (`charts/04_story_age_sex.png`)

**Interpretation.** At almost every age, more men died than survived. The exception is young boys: boys aged 12 and under survived at 57%, while every other male age group survived at only 9–19%. For women it is the opposite, and more survived than died at almost every age. So age mainly matters for males, where being a child was nearly as protective as being female.

**Chart 3 — Fare vs age by class, coloured by outcome** (`charts/05_story_fare_age_class.png`)

**Interpretation.** 1st class is mostly blue (survived) dots, and survivors there also paid more than those who died (median fare £77 vs £45). 3rd class is a dense cluster of orange (died) dots at cheap fares of about £7–8, and fare barely separates who lived and who died (median £8.52 vs £8.05). So money helped mainly by buying a better class. Within 1st and 2nd class a higher fare meant better odds, but in 3rd class almost everyone paid the same low price.

**Chart 4 — Survival rate by family size and sex** (`charts/06_story_family_size.png`)

**Interpretation.** A small family (2–4 people) was the best position: men in small families survived at double the rate of men travelling alone (32% vs 16%). Large families of 5+ did very badly for both sexes (27% of women, 3% of men), because they were mostly in 3rd class and would have found it hard to stay together and reach the lifeboats. So survival goes up, then down, as family size grows. That's why `sibsp` and `parch` showed almost no correlation with `survived` in the heatmap: correlation only catches straight-line patterns.

**Chart 5 — Survival rate by port and class** (`charts/07_story_port_class.png`)

**Interpretation.** Cherbourg passengers survived more often than Southampton passengers in every class (69% vs 58% in 1st class). Southampton's 3rd class was the worst group: only 19% of 353 passengers survived, and it was also the biggest group on the ship. The port itself didn't cause survival; Cherbourg simply had more wealthy 1st-class passengers, while Southampton had most of the 3rd-class men. Queenstown's 1st and 2nd class boxes contain only 2–3 people each, so those percentages aren't reliable.

**The story in one paragraph.** Survival was decided mostly by who you were (sex and age) and where you were (class). Women survived at 74% vs 19% for men, and children, especially boys, were the only group of males with good odds. Class then set the limit on top of that: almost all 1st- and 2nd-class women survived, while 3rd-class passengers died in the greatest numbers, especially Southampton's large 3rd-class group and large families. Fare and port look important, but mostly because they stand in for class, so `sex`, `pclass` and `age` should be the strongest features for the model in notebook 02.

## Task 6 — Exploratory z-score check

| | mean before | std before | mean after | std after |
|---|---|---|---|---|
| age | 29.0654 | 13.2702 | 0.0000 | 1.0000 |
| fare | 32.0967 | 49.6975 | 0.0000 | 1.0000 |

**Result.** We standardized `age` and `fare` with z = (x − mean) / std. After that, both columns have mean 0 and standard deviation 1, which shows the formula worked. Before, age had mean 29.07 and std 13.27, and fare had mean 32.10 and std 49.70. The shape doesn't change: fare is still just as right-skewed after scaling, because standardizing only shifts and rescales the numbers and doesn't remove skew or outliers.

This was only a check during EDA. The model in notebook 02 doesn't use these z-scores. It fits its own scaler on the training data only, so no test information leaks in.

---

# Part B — Predictive modeling

## Task 7 — Stratified split

**Why stratify?** The target is imbalanced: 61.6% died vs 38.4% survived, about 1.6 : 1 (61.75% vs 38.25% on the 889 rows we model). A normal random 80/20 split could, by bad luck, put too many or too few survivors in the 178-row test set, and then every score, especially precision and recall for "survived", would depend on luck. Using `stratify=y` keeps the same ratio in both parts: 38.26% survived in train and 38.20% in test. The split happens before any filling-in, encoding or scaling.

## Feature choice

**Which columns go into the model and why.** The model uses seven columns: `pclass`, `sex`, `age`, `sibsp`, `parch`, `fare` and `embarked`. `alive` is left out on purpose because it is just the `survived` column written again as "yes"/"no", so using it would give the model the answer. `class`, `embark_town`, `who`, `adult_male` and `alone` are copies or combinations of columns already in the list, so they add nothing new. `deck` is 77% "Unknown" and overlaps heavily with `pclass`, so it is left out too. Before the split, only the 2 rows with no port are dropped. `age` keeps its 177 missing values on purpose, so that the pipeline can fill them using the training data only.

## Task 8 — Preprocessing

**Preprocessing choices.** All the preprocessing steps (filling missing values, encoding and scaling) sit inside one scikit-learn `Pipeline` together with the model. When we call `fit` on the training data, every step learns only from training rows, so it never learns anything from the test set and there is no leakage.

| Columns | Missing values | Encoding / scaling |
|---|---|---|
| `pclass, age, sibsp, parch, fare` | `SimpleImputer(median)`. Only `age` actually has gaps. | `StandardScaler` |
| `sex, embarked` | `SimpleImputer(most_frequent)`, a safety net for new data | `OneHotEncoder(handle_unknown="ignore")` |

The number columns are filled with the median from the training data and then scaled with `StandardScaler`. The text columns `sex` and `embarked` are filled with the most common value and then one-hot encoded, because they have no natural order and numbers like 0, 1, 2 would wrongly suggest one. Age uses one simple training median here instead of notebook 01's sex + class median, because it is easier to do inside a pipeline, and the tree models can still pick up age differences between groups.

## Task 9 — Three classifiers + the decision tree

Logistic Regression, Decision Tree and Random Forest are all trained on the identical `X_train` / `y_train`. The tree is drawn with `plot_tree(feature_names=..., class_names=["died", "survived"])` (`charts/10_decision_tree.png`).

**Reading the tree.** The tree's first split is on sex (`sex_female <= 0.5`), so sex is the most useful feature; the left "True" branch is the males. On the male side, the next split is age ≤ 3.5 years: very young boys mostly survived (11 of 13 in the training data), and among older males, class and fare pick out a big group of 343 with only 11% survival. On the female side, the split is on class: 1st- and 2nd-class women almost all survived (127 of 134), while 3rd-class women are split on fare, and those who paid more than about £23 (large families on one shared ticket) mostly died. So the tree learned the same story as the EDA. I limited it to `max_depth=4` and `min_samples_leaf=5` so it stays readable and doesn't just memorize the training data. The numbers in the plot are in z-score units because of the scaler, so the table in the notebook converts them back to years and pounds.

## Task 10 — Evaluation (test set)

| Model | Accuracy | Precision | Recall | F1 | ROC AUC | Confusion matrix [TN FP / FN TP] |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.809 | 0.783 | 0.691 | 0.734 | **0.861** | [97 13 / 21 47] |
| Decision Tree | 0.798 | 0.776 | 0.662 | 0.714 | 0.851 | [97 13 / 23 45] |
| Random Forest | 0.803 | 0.762 | **0.706** | 0.733 | 0.824 | [95 15 / 20 48] |

The ROC curves are in `charts/12_roc_curves.png` and the confusion-matrix plots in `charts/11_confusion_matrices.png`.

**How the three models compare** (test set, 178 passengers, 68 of whom survived). Logistic Regression has the best AUC (0.861) and the best accuracy of the three (0.809), and its mistakes are balanced: 13 false alarms and 21 missed survivors. The Decision Tree is slightly behind on everything; it is the easiest to explain, but one small tree loses some detail. The Random Forest has the best recall (it found 48 of 68 survivors) but the lowest AUC (0.824), and it scores 0.985 on the training data vs 0.803 on test, so it is overfitting (memorizing); Task 12 fixes this by tuning. All three are close (0.80–0.81 accuracy), and they mostly get the same people wrong: men who survived or 3rd-class women who died, i.e. people who go against the main pattern.

## Task 11 — Imbalance handling (Logistic Regression)

| Variant | Precision | Recall | F1 | Accuracy | AUC |
|---|---|---|---|---|---|
| (a) baseline | **0.783** | 0.691 | 0.734 | **0.809** | 0.861 |
| (b) class_weight='balanced' | 0.718 | **0.750** | 0.734 | 0.792 | 0.861 |
| (c) SMOTE (train fold only) | 0.735 | 0.735 | **0.735** | 0.798 | **0.867** |

**Conclusion.** The baseline has the highest precision (0.783) but misses the most survivors, because it leans towards the bigger "died" class. With `class_weight='balanced'`, each survivor counts about 1.6× more, so recall goes up to 0.750 and precision goes down to 0.718. SMOTE creates synthetic survivor rows in the training data only (439 : 439 after resampling) and leaves the test set untouched; it gives the best F1 (0.735) and the best AUC (0.867), with precision equal to recall. Comparing all three, SMOTE worked best, but only by a tiny margin, because the imbalance is mild (38% survivors). The real effect of each method is a trade of precision for recall, and `class_weight='balanced'` gets most of the same recall gain without creating fake rows, so it is the simpler choice.

## Task 12 — GridSearchCV + OOB

**Tuning result.** `GridSearchCV` tried 36 combinations of `n_estimators` × `max_depth` × `max_features`, using 5-fold cross-validation on the training data, and picked the combination with the best F1. The forest was built with `oob_score=True`. The best settings are `n_estimators=400`, `max_depth=8` and `max_features=0.5`, with a best CV F1 of 0.769. The OOB score of 0.826 means each tree is tested on the training rows it didn't see, which gives a free built-in check, and it is very close to the test accuracy of 0.820, so the model generalizes well. `max_depth=8` beat unlimited depth every time, so capping the depth fixed the overfitting. At depth 8, using half the features at each split (0.5) beat `sqrt` / `log2`, because with only 10 columns `sqrt` gives each split just 3 to choose from. The test results improved: accuracy from 0.803 to 0.820, precision from 0.762 to 0.833, F1 from 0.733 to 0.738 and AUC from 0.824 to 0.839, while recall dropped a bit (0.706 to 0.662).

Note: 200 and 400 trees score within 0.001 of each other, so on some machines the search picks 200 trees instead. The conclusions don't change.

## Task 13 — Regression: predicting `fare`

Features: `pclass, sex, age, sibsp, parch, embarked, survived`, using the same train/test rows as the classifiers and the same kind of pipeline (median impute + scale, one-hot with `drop="first"`).

**Results.** On the same 178 test passengers, the model gets MAE = £19.75, RMSE = £41.27, R² = 0.347 and Adjusted R² = 0.316. It explains about a third of the variation in fare, and most of that comes from `pclass` and port, with family size adding a smaller share. RMSE is about twice the MAE, which means a few very big errors dominate.

**Heteroscedasticity: yes, clearly.** The residual plot isn't an even band around zero; it fans out. For low predicted fares the errors are small (average £9.71), but for the top third of predictions they are large (average £35.40), and one ticket was under-predicted by more than £400. The model even predicts slightly negative fares for some 3rd-class passengers. This happens because fare is heavily right-skewed, and a straight-line model on raw fare can't handle that, so predicting `log(fare)` would be the natural next step.

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

I would deploy the tuned Random Forest, because it has the highest accuracy (0.820), precision (0.833) and F1 (0.738) of the four classifiers. Its OOB score (0.826) and CV F1 (0.769) agree with the test result, so it wasn't just a lucky split. Logistic Regression is a strong runner-up with the best AUC (0.861) and an F1 only 0.004 lower, so it would be the better pick if we needed well-ranked probabilities or a model that's easy to explain. The tuned forest's weak spot is recall (0.662), so if missing a survivor were costly, I would lower its decision threshold or add `class_weight='balanced'`, which raised Logistic Regression's recall by about 6 points in Task 11.

## Task 15 — Saved pipeline

`joblib.dump(best_pipeline, "models/titanic_best_pipeline.joblib")` saves the whole `Pipeline`: the `ColumnTransformer` (imputers, one-hot encoder, scaler) plus the tuned `RandomForestClassifier`. It is not the bare estimator. The last cells of `02_modeling.ipynb` and `predict.py` reload it with `joblib.load`. They confirm it gives **identical predictions** on the test set (accuracy 0.820 after reload) and that it predicts on **raw** new rows, including a passenger with a missing age:

```
 pclass    sex  age  sibsp  parch   fare embarked  P(survive) prediction
      1 female 29.0      0      0 110.00        C       0.999   survived
      3   male 24.0      0      0   7.25        S       0.078       died
      3 female  NaN      1      3  25.47        Q       0.580   survived
```

The file was saved with scikit-learn 1.8. Loading it needs the same scikit-learn version (pinned in `requirements.txt`). If you use a different version, re-run `02_modeling.ipynb` to regenerate it.
