# Titanic Analytics

This is my module 2 project using titanic dataset.

I loaded the data using seaborn and saved it as `titanic.csv` so it can work even without internet.

## Files

* analytics_pipeline.ipynb
* titanic.csv
* best_titanic_pipeline.joblib
* README.md

## How to run

I used VS code for this project.

.\data_pipeline\.venv\Scripts\Activate.ps1


Install libs if needed:

pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib jupyter
Then open:
analytics/analytics_pipeline.ipynb
and run the notebook.

## Data
Dataset has 891 rows and 15 columns.

Missing values:

* age - 19.87%
* embarked - 0.22%
* embark_town - 0.22%
* deck - 77.22%
I removed rows for embarked and embark_town since missing data is very small.
For age I used median because around 20% data was missing.
Deck had too much missing data so I removed that column.
## Outliers
Using IQR:
* Age outliers - 65
* Fare outliers - 114
Fare values:
* Mean - 32.10
* Median - 14.45
* Mode - 8.05
Fare is right skewed because mean > median > mode.
## Survival
Female survival was around 74% and male was around 19%.
Class survival:
* 1st class - 62.62%
* 2nd class - 47.28%
* 3rd class - 24.24%
So women and higher class passengers had better survival.

## Correlation
Strongest correlations were:

* pclass and fare - around -0.55
* sibsp and parch - around 0.41

## Charts

I made charts for:

* survival by sex
* survival by class
* sex and class
* fare and survival

From charts, women had higher survival and first class also had better survival.

Passengers who paid higher fare also had better survival in general.

## Standardization

I scaled age and fare using StandardScaler.
After scaling mean was almost 0 and std was almost 1.
## Train Test Split
Survival classes were not fully balanced.

* 61.75% not survived
* 38.25% survived

So I used stratify while doing train test split.

## Preprocessing

Numeric columns:

* pclass
* age
* sibsp
* parch
* fare

I used median imputation and StandardScaler.

Categorical:

* sex
* embarked

I used most frequent value and OneHotEncoder.

Preprocessing was fitted only on train data.

## Models

I trained 3 models:

* Logistic Regression
* Decision Tree
* Random Forest

I compared accuracy, precision, recall, F1 and AUC.

I also made confusion matrix, ROC curve and decision tree plot.

## Imbalance

I compared:

* normal logistic regression
* class_weight balanced
* SMOTE

SMOTE was used only on train data.

I compared precision, recall and F1 for all 3.

## Random Forest Tuning

I used GridSearchCV for:

* n_estimators
* max_depth
* max_features

Also checked OOB score using `oob_score=True`.

## Regression

I used Linear Regression to predict fare.

Metrics used:

* MAE
* RMSE
* R2
* Adjusted R2

I also checked residual plot for heteroscedasticity.

## Final Model

I compared all classifier results and selected the model with best overall performance.

The final complete pipeline is saved as:

best_titanic_pipeline.joblib

I also loaded it again using joblib and checked that predictions are same.
## Model Comparison

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8090 | 0.7833 | 0.6912 | 0.7344 | 0.8610 |
| Decision Tree | 0.7640 | 0.7600 | 0.5588 | 0.6441 | 0.8374 |
| Random Forest | 0.8202 | 0.7813 | 0.7353 | 0.7576 | 0.8179 |
| Tuned Random Forest | 0.8258 | 0.8491 | 0.6618 | 0.7438 | 0.8372 |

## Final Recommendation

Logistic Regression had the highest AUC of 0.8610.
Random Forest had the highest F1 among the original three models at 0.7576.
The tuned Random Forest improved accuracy to 0.8258 but its AUC was still lower than Logistic Regression.
I selected Logistic Regression as the final model because it gave the best overall AUC.
