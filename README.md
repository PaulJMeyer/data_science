# Data Science Portfolio

Welcome to my Data Science Portfolio.
This repository documents my journey into Data Science through practical projects covering data analysis, machine learning and statistical modelling. The focus is on reproducible workflows, transparent data preparation and interpretable results.

## Technologies
Python
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn

## Projects
🔬 Diabetes Prediction (In Progress)

Goal: Predict whether a patient is likely to have diabetes based on medical measurements.

Dataset: [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) (Kaggle)

| | |
|---|---|
| **Status** | 🚧 In Progress |
| **Type** | Binary Classification |
| **Stack** | Python, pandas, scikit-learn, imbalanced-learn |

### Completed steps:
- Data cleaning and exploratory data analysis (EDA)
- Missing value imputation (median-based, with missing indicator flags)
- Feature engineering (polynomial and interaction features based on permutation importance)
- Class imbalance handling (evaluated RandomUnderSampler, RandomOverSampler, SMOTE, class_weight)
- Hyperparameter tuning via GridSearchCV for all models
- Feature selection via combinatorial search (KNN) and importance-based filtering (RF, LogReg)

### Models trained and evaluated (Test F1):

| Model | Strategy | Test F1 |
|---|---|---|
| K-Nearest Neighbors | 5 selected features, RandomUnderSampler | 0.6906 |
| Random Forest | 12 features, class_weight='balanced' | 0.6923 |
| Logistic Regression | 14 features, SMOTE, L2, C=10 | 0.6614 |

### Next steps:

**Threshold tuning**
All models currently use the default decision threshold of 0.5. In a medical context, missing a positive case (false negative) is more costly than a false alarm. Shifting the threshold using `predict_proba` allows explicit control over the precision/recall trade-off and will be evaluated for all three models using ROC AUC, PR AUC and Matthews Correlation Coefficient

**Additional models**
To broaden the comparison and explore a wider range of approaches:
- Support Vector Machine (SVM) with RBF kernel
- Multi-Layer Perceptron (MLP, sklearn)
- XGBoost (gradient boosting)

**Voting Classifier**
After evaluating all models (KNN, RF, LogReg, SVM, MLP, XGBoost), the five best-performing ones will be combined into a Voting Classifier. Using an odd number of models (5) avoids tie-breaking issues in Hard Voting for binary classification. Both Hard Voting (majority decision) and Soft Voting (averaged predicted probabilities) will be tested and compared.


More projects will be added over time as the portfolio grows.
