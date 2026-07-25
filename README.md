# Data Science Portfolio

Welcome to my Data Science Portfolio.
This repository documents my journey into Data Science through practical projects covering data analysis, machine learning and statistical modelling. The focus is on reproducible workflows, transparent data preparation and interpretable results.

## Technologies

Python · Pandas · NumPy · Scikit-learn · Imbalanced-learn · XGBoost · Matplotlib · Seaborn

---

## Projects

### 🔬 Diabetes Prediction

> Binary classification of diabetes risk using the Pima Indians Diabetes Dataset.

| | |
|---|---|
| **Status** | ✅ v1.1 complete |
| **Type** | Binary Classification |
| **Stack** | Python, pandas, scikit-learn, imbalanced-learn, XGBoost |
| **Dataset** | [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) (Kaggle, 768 rows, 8 features) |

#### What was done

**Data cleaning & EDA**
Explored feature distributions with pair plots (before and after imputation). Identified physiologically implausible zero values in Glucose, BloodPressure, SkinThickness, Insulin and BMI as missing data. Applied median imputation on the training set and transferred the same medians to the test set. Added binary missing-indicator flags for each imputed column to preserve information about missingness.

**Feature engineering**
Added clinically motivated interaction terms (e.g. `Glucose x BMI`, `Age x BMI`, `Pregnancies x Insulin x BMI`) selected by permutation importance. Added binary threshold features based on medical reference values (e.g. fasting glucose ≥ 126 mg/dL for diabetes diagnosis, BMI ≥ 30 for obesity), used by XGBoost only — they introduced redundancy and hurt other models.

**Class imbalance**
The dataset has a ~65/35 class split. Four strategies were systematically compared for each model: RandomUnderSampler, RandomOverSampler, SMOTE, and class_weight='balanced'. The best strategy was selected per model based on the CV-to-test-F1 gap rather than CV score alone, to avoid selecting strategies that look good on cross-validation but overfit.

**Hyperparameter tuning**
All models were tuned via GridSearchCV (cv=5, scoring=F1). Resampling strategy, regularisation strength, and feature set were treated as separate optimisation axes. For KNN, a combinatorial feature search over all subsets up to size 5 (from 15 candidate features, ~5000 combinations) identified the optimal 5-feature set. For XGBoost, the interaction between learning rate and n_estimators was explicitly analysed.

**Individual models trained and evaluated**

| Model | Resampling | Features | Test F1 | ROC-AUC |
|---|---|---|---|---|
| **XGBoost** | scale_pos_weight | 26 (poly + threshold) | **0.703** | **0.827** |
| Random Forest | class_weight='balanced' | 13 (poly, low-importance dropped) | 0.692 | 0.819 |
| KNN | RandomUnderSampler | 5 (combinatorial search) | 0.691 | 0.809 |
| SVM (RBF) | RandomOverSampler | 14 (poly, importance-filtered) | 0.672 | 0.797 |
| Logistic Regression | SMOTE, L2, C=10 | 14 (poly, importance-filtered) | 0.661 | 0.818 |
| MLP | SMOTEENN | 19 (poly) | 0.657 | 0.833 |

**Ensemble: Soft Voting (KNN + RF + XGBoost + MLP)**
Three voting strategies were compared: hard voting (all 5 models, majority ≥ 3), soft voting (unweighted average of predicted probabilities), and weighted soft voting (weights proportional to CV-F1 scores). Soft voting with KNN, RF, XGBoost and MLP — excluding Logistic Regression and SVM, which reduced ensemble performance — achieved the best result. The decision threshold was tuned by scanning the full precision/recall/F1 curve; the default threshold of 0.5 proved optimal.

| Ensemble | F1 | Precision | Recall | ROC-AUC |
|---|---|---|---|---|
| **Soft Voting (KNN + RF + XGB + MLP)** | **0.710** | 0.590 | **0.891** | **0.833** |
| Hard Voting (all 5 models) | — | — | — | — |
| Weighted Soft Voting (KNN + RF + XGB + MLP) | — | — | — | — |

The ensemble improves Recall substantially over any individual model (0.891 vs. 0.873 for KNN), which is the clinically relevant direction: in a diabetes screening context, minimising false negatives (missed positive cases) matters more than minimising false alarms.

**Key findings**
- The soft voting ensemble (Test F1 = 0.710) outperforms all individual models, with notably higher recall.
- All individual models converge in a narrow Test F1 range (0.66–0.70), suggesting the dataset size (~768 rows) is the main limiting factor rather than model choice.
- Polynomial interaction features helped KNN and XGBoost but introduced multicollinearity in Logistic Regression, requiring stronger regularisation (L2, C=10).
- SMOTE consistently improved CV scores but widened the CV–test gap, indicating a tendency to overfit on small datasets. Model-native class weighting (RF, XGBoost) proved more reliable.
- MLP achieved the highest ROC-AUC (0.833) despite the lowest F1, reflecting good probability calibration with a conservative decision boundary at 0.5.

#### Next steps (v1.2)

- **MLflow** — experiment tracking for all GridSearchCV runs
- **SHAP** — model interpretability and feature contribution analysis
- **Calibration** — evaluate and improve probability calibration for ensemble soft voting

---

More projects will be added over time as the portfolio grows.
