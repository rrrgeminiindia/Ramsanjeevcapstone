# MODULE 2 - TITANIC ANALYTICS PIPELINE

import warnings
warnings.filterwarnings("ignore")

import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from imblearn.over_sampling import SMOTE
import os
os.makedirs("charts", exist_ok=True)

# PART A - LOAD AND PROFILE DATA

df = sns.load_dataset("titanic")
df.to_csv("titanic.csv", index=False)
print("\nSHAPE")
print(df.shape)
print("\nINFO")
df.info()
print("\nDESCRIBE")
print(df.describe())
print("\nMISSING VALUE PERCENTAGE")
missing_percentage = (df.isnull().sum() / len(df)) * 100
print(missing_percentage[missing_percentage > 0])


# MISSING VALUES

cleaned_df = df.copy()

cleaned_df = cleaned_df.dropna(subset=["embarked", "embark_town"])
age_median = cleaned_df["age"].median()
cleaned_df["age"] = cleaned_df["age"].fillna(age_median)

cleaned_df = cleaned_df.drop(columns=["deck"])

print("\nMISSING VALUES AFTER CLEANING")
print(cleaned_df.isnull().sum())


# UNIVARIATE ANALYSIS


plt.figure(figsize=(8, 5))
sns.histplot(cleaned_df["age"], kde=True)
plt.title("Age Distribution")
plt.savefig("charts/age_histogram.png", bbox_inches="tight")
plt.show()


plt.figure(figsize=(8, 4))
sns.boxplot(x=cleaned_df["age"])
plt.title("Age Box Plot")
plt.savefig("charts/age_boxplot.png", bbox_inches="tight")
plt.show()


plt.figure(figsize=(8, 5))
sns.histplot(cleaned_df["fare"], kde=True)
plt.title("Fare Distribution")
plt.savefig("charts/fare_histogram.png", bbox_inches="tight")
plt.show()

plt.figure(figsize=(8, 4))
sns.boxplot(x=cleaned_df["fare"])
plt.title("Fare Box Plot")
plt.savefig("charts/fare_boxplot.png", bbox_inches="tight")
plt.show()

# IQR OUTLIERS

def count_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = series[(series < lower) | (series > upper)]
    return len(outliers)


age_outliers = count_outliers(cleaned_df["age"])
fare_outliers = count_outliers(cleaned_df["fare"])

print("\nOUTLIERS")
print("Age Outliers:", age_outliers)
print("Fare Outliers:", fare_outliers)

# FARE MEAN, MEDIAN AND MODE

fare_mean = cleaned_df["fare"].mean()
fare_median = cleaned_df["fare"].median()
fare_mode = cleaned_df["fare"].mode()[0]

print("\nFARE STATISTICS")
print("Mean:", fare_mean)
print("Median:", fare_median)
print("Mode:", fare_mode)

if fare_mean > fare_median > fare_mode:
    print("Fare is right-skewed because Mean > Median > Mode.")
elif fare_mean < fare_median < fare_mode:
    print("Fare is left-skewed because Mean < Median < Mode.")
else:
    print("Fare is approximately symmetric.")


#  SURVIVAL BY SEX

female = cleaned_df[cleaned_df["sex"] == "female"]
male = cleaned_df[cleaned_df["sex"] == "male"]

female_survival = female["survived"].mean() * 100
male_survival = male["survived"].mean() * 100

print("\nSURVIVAL BY SEX")
print("Female:", round(female_survival, 2), "%")
print("Male:", round(male_survival, 2), "%")


# SURVIVAL BY PASSENGER CLASS

print("\nSURVIVAL BY CLASS")

for pclass in [1, 2, 3]:
    temp = cleaned_df[cleaned_df["pclass"] == pclass]
    rate = temp["survived"].mean() * 100
    print("Class", pclass, ":", round(rate, 2), "%")

# SURVIVAL BY SEX AND CLASS - BOOLEAN MASKING WITH &

print("\nSURVIVAL BY SEX AND CLASS")

for sex in ["female", "male"]:
    for pclass in [1, 2, 3]:
        temp = cleaned_df[(cleaned_df["sex"] == sex) & (cleaned_df["pclass"] == pclass)]
        rate = temp["survived"].mean() * 100
        print(sex, "Class", pclass, ":", round(rate, 2), "%")


# Boolean masking using |
female_or_first_class = cleaned_df[(cleaned_df["sex"] == "female") | (cleaned_df["pclass"] == 1)]

print("\nFemale OR First Class rows:", female_or_first_class.shape[0])


# CORRELATION MATRIX

corr_columns = ["survived", "pclass", "age", "sibsp", "parch", "fare"]

corr_matrix = cleaned_df[corr_columns].corr()

print("\nCORRELATION MATRIX")
print(corr_matrix)

plt.figure(figsize=(9, 7))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Titanic Correlation Matrix")
plt.savefig("charts/correlation_heatmap.png", bbox_inches="tight")

plt.show()

corr_pairs = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).stack().reset_index()

corr_pairs.columns = ["Feature 1", "Feature 2", "Correlation"]
corr_pairs["Absolute Correlation"] = corr_pairs["Correlation"].abs()

strongest_two = corr_pairs.sort_values("Absolute Correlation", ascending=False).head(2)

print("\nTWO STRONGEST CORRELATIONS")
print(strongest_two)

for _, row in strongest_two.iterrows():
    print(
        row["Feature 1"],
        "and",
        row["Feature 2"],
        "have correlation",
        round(row["Correlation"], 3)
    )


# MULTIVARIATE DATA STORY - CHART 1

plt.figure(figsize=(7, 5))
sns.barplot(data=cleaned_df, x="sex", y="survived")
plt.title("Survival Rate by Sex")

plt.savefig("charts/survival_by_sex.png", bbox_inches="tight")

plt.show()

print("Interpretation:")
print("Female passengers had a much higher survival rate than male passengers.")
print("This shows that sex was strongly associated with survival on the Titanic.")


# CHART 

plt.figure(figsize=(7, 5))
sns.barplot(data=cleaned_df, x="pclass", y="survived")
plt.title("Survival Rate by Passenger Class")
plt.savefig("charts/survival_by_class.png", bbox_inches="tight")

plt.show()

print("Interpretation:")
print("First-class passengers had the highest survival rate and third-class passengers had the lowest.")
print("This shows that passenger class was strongly related to survival.")


# CHART 3

plt.figure(figsize=(8, 5))
sns.barplot(data=cleaned_df, x="pclass", y="survived", hue="sex")
plt.title("Survival by Sex and Passenger Class")
plt.savefig("charts/survival_by_sex_class.png", bbox_inches="tight")

plt.show()

print("Interpretation:")
print("Women had higher survival rates than men in every passenger class.")
print("First-class and second-class women had especially high survival rates.")
print("Both sex and passenger class influenced survival.")


# CHART 4

plt.figure(figsize=(8, 5))
sns.boxplot(data=cleaned_df, x="survived", y="fare")
plt.title("Fare by Survival")
plt.savefig("charts/fare_survival.png", bbox_inches="tight")    
plt.show()

print("Interpretation:")
print("Passengers who survived generally paid higher fares.")
print("Higher fares were related to higher passenger classes, which also had better survival rates.")

# EDA STANDARDIZATION CHECK

print("\nBEFORE SCALING")
print(cleaned_df[["age", "fare"]].agg(["mean", "std"]))

eda_scaler = StandardScaler()

scaled_values = eda_scaler.fit_transform(cleaned_df[["age", "fare"]])
scaled_df = pd.DataFrame(scaled_values, columns=["age", "fare"])

print("\nAFTER SCALING")
print("Mean")
print(scaled_df.mean())

print("\nStandard Deviation")
print(scaled_df.std(ddof=0))

print("After standardization, age and fare have approximately mean 0 and standard deviation 1.")


# PART B - CLASSIFICATION

features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]

model_df = cleaned_df[features + ["survived"]].copy()

original_missing_age = df.loc[model_df.index, "age"].isnull()
model_df.loc[original_missing_age, "age"] = np.nan

X = model_df[features]
y = model_df["survived"]

# CLASS BALANCE

class_balance = y.value_counts(normalize=True) * 100

print("\nCLASS BALANCE")
print("Not Survived:", round(class_balance[0], 2), "%")
print("Survived:", round(class_balance[1], 2), "%")

print("Stratification is used because the two classes are not equally represented.")


# PREPROCESSING

num_cols = ["pclass", "age", "sibsp", "parch", "fare"]
cat_cols = ["sex", "embarked"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, num_cols),
    ("cat", categorical_pipeline, cat_cols)
])

# TRAIN TEST SPLIT - BEFORE FITTING PREPROCESSOR

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTRAIN / TEST")
print(f"Train: {len(X_train):,} samples | Survival Rate: {y_train.mean()*100:.1f}%")
print(f"Test: {len(X_test):,} samples | Survival Rate: {y_test.mean()*100:.1f}%")

# THREE CLASSIFIERS

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
}

results = []
trained_pipes = {}


for name, model in models.items():

    pipe = Pipeline([
        ("prep", clone(preprocessor)),
        ("model", model)
    ])

    t0 = time.time()

    pipe.fit(X_train, y_train)

    elapsed = time.time() - t0

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    cv_auc = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc")

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "CV AUC Mean": cv_auc.mean(),
        "CV AUC Std": cv_auc.std(),
        "Time": elapsed
    })

    trained_pipes[name] = pipe


results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print(results_df)

# CONFUSION MATRICES

for name, pipe in trained_pipes.items():

    y_pred = pipe.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not Survived", "Survived"]
    )

    disp.plot()
    plt.title(name)
    plt.savefig(f"charts/{name.replace(' ', '_').lower()}_confusion_matrix.png", bbox_inches="tight")
    plt.show()

# ROC CURVES

plt.figure(figsize=(8, 6))

for name, pipe in trained_pipes.items():

    y_prob = pipe.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    plt.plot(fpr, tpr, label=f"{name} AUC={auc:.3f}")


plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.savefig("charts/roc_curve.png", bbox_inches="tight")

plt.show()

# DECISION TREE VISUALIZATION

tree_pipe = trained_pipes["Decision Tree"]

feature_names = tree_pipe.named_steps["prep"].get_feature_names_out()

plt.figure(figsize=(30, 16))

plot_tree(
    tree_pipe.named_steps["model"],
    feature_names=feature_names,
    class_names=["Not Survived", "Survived"],
    filled=True
)
plt.savefig("charts/decision_tree.png", bbox_inches="tight")

plt.show()


imbalance_preprocessor = clone(preprocessor)

X_train_processed = imbalance_preprocessor.fit_transform(X_train)
X_test_processed = imbalance_preprocessor.transform(X_test)


# Baseline
baseline_model = LogisticRegression(max_iter=1000, random_state=42)

baseline_model.fit(X_train_processed, y_train)

baseline_pred = baseline_model.predict(X_test_processed)


# Class weight balanced
balanced_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

balanced_model.fit(X_train_processed, y_train)

balanced_pred = balanced_model.predict(X_test_processed)


# SMOTE - training data ONLY
smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train_processed,
    y_train
)

print("\nBEFORE SMOTE")
print(y_train.value_counts())

print("\nAFTER SMOTE")
print(pd.Series(y_train_smote).value_counts())


smote_model = LogisticRegression(max_iter=1000, random_state=42)

smote_model.fit(X_train_smote, y_train_smote)

smote_pred = smote_model.predict(X_test_processed)


predictions = {
    "Baseline": baseline_pred,
    "Class Weight Balanced": balanced_pred,
    "SMOTE": smote_pred
}

imbalance_results = []

for method, prediction in predictions.items():

    imbalance_results.append({
        "Method": method,
        "Precision": precision_score(y_test, prediction),
        "Recall": recall_score(y_test, prediction),
        "F1": f1_score(y_test, prediction)
    })


imbalance_df = pd.DataFrame(imbalance_results)

print("\nIMBALANCE COMPARISON")
print(imbalance_df)

best_imbalance = imbalance_df.sort_values("F1", ascending=False).iloc[0]

print(
    "\nBest imbalance strategy:",
    best_imbalance["Method"],
    "| F1:",
    round(best_imbalance["F1"], 4)
)

# RANDOM FOREST GRID SEARCH

rf_pipeline = Pipeline([
    ("prep", clone(preprocessor)),
    ("model", RandomForestClassifier(oob_score=True, random_state=42, n_jobs=-1))
])

param_grid = {
    "model__n_estimators": [50, 100, 200],
    "model__max_depth": [5, 10, None],
    "model__max_features": ["sqrt", "log2"]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    rf_pipeline,
    param_grid,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=1
)

print("\nRUNNING GRID SEARCH")

grid_search.fit(X_train, y_train)

print("\nBEST PARAMETERS")
print(grid_search.best_params_)

print("\nBEST CV ROC-AUC")
print(grid_search.best_score_)


best_rf = grid_search.best_estimator_

oob_score = best_rf.named_steps["model"].oob_score_

print("\nOOB SCORE")
print(oob_score)


rf_pred = best_rf.predict(X_test)
rf_prob = best_rf.predict_proba(X_test)[:, 1]

tuned_rf_results = {
    "Model": "Tuned Random Forest",
    "Accuracy": accuracy_score(y_test, rf_pred),
    "Precision": precision_score(y_test, rf_pred),
    "Recall": recall_score(y_test, rf_pred),
    "F1": f1_score(y_test, rf_pred),
    "AUC": roc_auc_score(y_test, rf_prob)
}

print("\nTUNED RANDOM FOREST RESULTS")
print(tuned_rf_results)

# REGRESSION - PREDICT FARE

reg_features = ["survived", "pclass", "sex", "age", "sibsp", "parch", "embarked"]

reg_df = cleaned_df[reg_features + ["fare"]].copy()

original_reg_age_missing = df.loc[reg_df.index, "age"].isnull()
reg_df.loc[original_reg_age_missing, "age"] = np.nan

X_reg = reg_df[reg_features]
y_reg = reg_df["fare"]


X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg,
    y_reg,
    test_size=0.2,
    random_state=42
)


reg_num_cols = ["survived", "pclass", "age", "sibsp", "parch"]
reg_cat_cols = ["sex", "embarked"]


reg_preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), reg_num_cols),

    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]), reg_cat_cols)
])


linear_model = Pipeline([
    ("prep", reg_preprocessor),
    ("model", LinearRegression())
])


linear_model.fit(X_reg_train, y_reg_train)

fare_pred = linear_model.predict(X_reg_test)


mae = mean_absolute_error(y_reg_test, fare_pred)

rmse = np.sqrt(
    mean_squared_error(y_reg_test, fare_pred)
)

r2 = r2_score(y_reg_test, fare_pred)

n = len(y_reg_test)

p = linear_model.named_steps["prep"].transform(X_reg_test).shape[1]

adjusted_r2 = 1 - ((1 - r2) * (n - 1)) / (n - p - 1)


print("\nREGRESSION RESULTS")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)
print("Adjusted R2:", adjusted_r2)

# RESIDUAL PLOT

residuals = y_reg_test - fare_pred

plt.figure(figsize=(8, 5))

plt.scatter(fare_pred, residuals)

plt.axhline(y=0, linestyle="--")

plt.xlabel("Predicted Fare")
plt.ylabel("Residual")
plt.title("Residual Plot")
plt.savefig("charts/residual_plot.png", bbox_inches="tight")
plt.show()


residual_check = pd.DataFrame({
    "Predicted": fare_pred,
    "Residual": residuals
})

residual_check["Fare Group"] = pd.qcut(
    residual_check["Predicted"],
    q=4,
    duplicates="drop"
)

residual_spread = residual_check.groupby(
    "Fare Group",
    observed=True
)["Residual"].std()

spread_ratio = residual_spread.max() / residual_spread.min()

print("\nRESIDUAL SPREAD")
print(residual_spread)

if spread_ratio > 1.5:
    print("The residuals show evidence of heteroscedasticity because their spread is not constant.")
else:
    print("The residuals do not show strong evidence of heteroscedasticity.")

# FINAL MODEL COMPARISON

classification_results = results_df[
    ["Model", "Accuracy", "Precision", "Recall", "F1", "AUC"]
].copy()


classification_results.loc[len(classification_results)] = [
    "Tuned Random Forest",
    tuned_rf_results["Accuracy"],
    tuned_rf_results["Precision"],
    tuned_rf_results["Recall"],
    tuned_rf_results["F1"],
    tuned_rf_results["AUC"]
]


regression_results = pd.DataFrame({
    "Model": ["Linear Regression"],
    "MAE": [mae],
    "RMSE": [rmse],
    "R2": [r2],
    "Adjusted R2": [adjusted_r2]
})


print("\nCLASSIFICATION METRICS")
print(classification_results)

print("\nREGRESSION METRICS")
print(regression_results)

print("\nClassification and regression metrics are shown separately because they measure different types of model performance.")


best_classifier = classification_results.sort_values(
    "AUC",
    ascending=False
).iloc[0]



print(
    f"I would deploy {best_classifier['Model']} because it achieved "
    f"Accuracy={best_classifier['Accuracy']:.4f}, "
    f"Precision={best_classifier['Precision']:.4f}, "
    f"Recall={best_classifier['Recall']:.4f}, "
    f"F1={best_classifier['F1']:.4f}, "
    f"and AUC={best_classifier['AUC']:.4f}."
)


# SAVE BEST COMPLETE PIPELINE

best_model_name = best_classifier["Model"]

if best_model_name == "Tuned Random Forest":
    best_pipeline = best_rf
else:
    best_pipeline = trained_pipes[best_model_name]


print("\nSAVING MODEL:", best_model_name)

joblib.dump(best_pipeline, "best_titanic_pipeline.joblib")

# RELOAD MODEL AND TEST

loaded_model = joblib.load("best_titanic_pipeline.joblib")

loaded_predictions = loaded_model.predict(X_test)

original_predictions = best_pipeline.predict(X_test)

print("\nPredictions identical:",np.array_equal(loaded_predictions,original_predictions))