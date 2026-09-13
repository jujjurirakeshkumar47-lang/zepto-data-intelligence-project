import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve,
    mean_absolute_error, mean_squared_error, r2_score
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

df = pd.read_csv("analytics/titanic.csv")

print("Shape:", df.shape)
print("\nInformation:")
df.info()

print("\nDescription:")
print(df.describe())

print("\nMissing Values:")
missing = df.isnull().sum()
missing_percent = (missing / len(df)) * 100
print(pd.DataFrame({
    "Count": missing,
    "Percentage": missing_percent
})[missing > 0])

df = df.dropna(subset=["embarked"])

df["age"] = df["age"].fillna(df["age"].median())

df["embark_town"] = df["embark_town"].fillna(
    df["embark_town"].mode()[0]
)

df = df.drop(columns=["deck"])

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

plt.figure()
plt.hist(df["age"])
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

plt.figure()
plt.boxplot(df["age"])
plt.title("Age Boxplot")
plt.ylabel("Age")
plt.show()

plt.figure()
plt.hist(df["fare"])
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Count")
plt.show()

plt.figure()
plt.boxplot(df["fare"])
plt.title("Fare Boxplot")
plt.ylabel("Fare")
plt.show()

q1 = df["age"].quantile(0.25)
q3 = df["age"].quantile(0.75)
iqr = q3 - q1

age_outliers = df[
    (df["age"] < q1 - 1.5 * iqr) |
    (df["age"] > q3 + 1.5 * iqr)
]

q1 = df["fare"].quantile(0.25)
q3 = df["fare"].quantile(0.75)
iqr = q3 - q1

fare_outliers = df[
    (df["fare"] < q1 - 1.5 * iqr) |
    (df["fare"] > q3 + 1.5 * iqr)
]

print("\nAge Outliers:", len(age_outliers))
print("Fare Outliers:", len(fare_outliers))

print("\nFare Mean:", df["fare"].mean())
print("Fare Median:", df["fare"].median())
print("Fare Mode:", df["fare"].mode()[0])

if df["fare"].mean() > df["fare"].median() > df["fare"].mode()[0]:
    print("Fare is right-skewed.")
elif df["fare"].mean() < df["fare"].median() < df["fare"].mode()[0]:
    print("Fare is left-skewed.")
else:
    print("Fare is not strongly skewed.")

print("\nSurvival by Sex")

female = df.loc[df["sex"] == "female", "survived"].mean()
male = df.loc[df["sex"] == "male", "survived"].mean()

print("Female:", female)
print("Male:", male)

print("\nSurvival by Passenger Class")

for pclass in [1, 2, 3]:
    rate = df.loc[df["pclass"] == pclass, "survived"].mean()
    print("Class", pclass, ":", rate)

print("\nSurvival by Sex and Class")

for sex in ["female", "male"]:
    for pclass in [1, 2, 3]:
        rate = df.loc[
            (df["sex"] == sex) & (df["pclass"] == pclass),
            "survived"
        ].mean()
        print(sex, "Class", pclass, ":", rate)

mask = (
    ((df["sex"] == "female") & (df["pclass"] == 1)) |
    ((df["sex"] == "male") & (df["pclass"] == 1))
)

print("\nFirst Class Combined Survival Rate:")
print(df.loc[mask, "survived"].mean())

columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

correlation = df[columns].corr()

print("\nCorrelation Matrix:")
print(correlation)

plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True)
plt.title("Correlation Heatmap")
plt.show()

corr = correlation.abs()
np.fill_diagonal(corr.values, 0)

pairs = corr.unstack().sort_values(ascending=False)

print("\nTop Two Correlations:")

shown = 0
used = set()

for (a, b), value in pairs.items():
    pair = tuple(sorted([a, b]))
    if pair not in used:
        print(a, "-", b, ":", correlation.loc[a, b])
        used.add(pair)
        shown += 1

    if shown == 2:
        break

plt.figure()
sns.barplot(data=df, x="pclass", y="survived", hue="sex")
plt.title("Survival Rate by Class and Sex")
plt.show()

plt.figure()
sns.boxplot(data=df, x="pclass", y="fare", hue="survived")
plt.title("Fare by Class and Survival")
plt.show()

plt.figure()
sns.scatterplot(data=df, x="age", y="fare", hue="survived")
plt.title("Age, Fare and Survival")
plt.show()

plt.figure()
sns.barplot(data=df, x="sibsp", y="survived", hue="sex")
plt.title("Survival by Siblings/Spouses and Sex")
plt.show()

scaled_data = df[["age", "fare"]].copy()

print("\nBefore Standardization:")
print(scaled_data.agg(["mean", "std"]))

scaler = StandardScaler()

scaled_data[["age", "fare"]] = scaler.fit_transform(
    scaled_data[["age", "fare"]]
)

print("\nAfter Standardization:")
print(scaled_data.agg(["mean", "std"]))

features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked"
]

X = df[features]
y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

categorical_features = [
    "sex",
    "embarked"
]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

logistic = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

tree = Pipeline([
    ("preprocessor", preprocessor),
    ("model", DecisionTreeClassifier(random_state=42))
])

forest = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

logistic.fit(X_train, y_train)
tree.fit(X_train, y_train)
forest.fit(X_train, y_train)

models = {
    "Logistic Regression": logistic,
    "Decision Tree": tree,
    "Random Forest": forest
}

results = []

for name, model in models.items():

    prediction = model.predict(X_test)
    probability = model.predict_proba(X_test)[:, 1]

    results.append([
        name,
        accuracy_score(y_test, prediction),
        precision_score(y_test, prediction, zero_division=0),
        recall_score(y_test, prediction, zero_division=0),
        f1_score(y_test, prediction, zero_division=0),
        roc_auc_score(y_test, probability)
    ])

    print("\n", name)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, prediction))

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC AUC"
    ]
)

print("\nClassification Results:")
print(results_df)

plt.figure(figsize=(18, 8))

tree_features = tree.named_steps[
    "preprocessor"
].get_feature_names_out()

plot_tree(
    tree.named_steps["model"],
    feature_names=tree_features,
    class_names=["Not Survived", "Survived"],
    filled=True,
    max_depth=4
)

plt.title("Decision Tree")
plt.show()

plt.figure(figsize=(8, 6))

for name, model in models.items():

    probability = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(
        y_test,
        probability
    )

    auc = roc_auc_score(
        y_test,
        probability
    )

    plt.plot(
        fpr,
        tpr,
        label=name + " AUC = " + str(round(auc, 3))
    )

plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.show()

print("\nClass Balance:")
print(y.value_counts())
print(y.value_counts(normalize=True))

baseline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

balanced = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

smote_model = ImbPipeline([
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", LogisticRegression(max_iter=1000))
])

baseline.fit(X_train, y_train)
balanced.fit(X_train, y_train)
smote_model.fit(X_train, y_train)

imbalance_results = []

for name, model in {
    "Baseline": baseline,
    "Balanced": balanced,
    "SMOTE": smote_model
}.items():

    prediction = model.predict(X_test)

    imbalance_results.append([
        name,
        precision_score(y_test, prediction, zero_division=0),
        recall_score(y_test, prediction, zero_division=0),
        f1_score(y_test, prediction, zero_division=0)
    ])

imbalance_df = pd.DataFrame(
    imbalance_results,
    columns=["Method", "Precision", "Recall", "F1"]
)

print("\nImbalance Comparison:")
print(imbalance_df)

rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        oob_score=True,
        random_state=42
    ))
])

parameters = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 5, 10],
    "model__max_features": ["sqrt", "log2"]
}

grid = GridSearchCV(
    rf_pipeline,
    parameters,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("\nBest Random Forest Parameters:")
print(grid.best_params_)

best_rf = grid.best_estimator_

print("Random Forest OOB Score:")
print(best_rf.named_steps["model"].oob_score_)

rf_prediction = best_rf.predict(X_test)
rf_probability = best_rf.predict_proba(X_test)[:, 1]

print("\nTuned Random Forest Test Results:")
print("Accuracy:", accuracy_score(y_test, rf_prediction))
print("Precision:", precision_score(y_test, rf_prediction))
print("Recall:", recall_score(y_test, rf_prediction))
print("F1:", f1_score(y_test, rf_prediction))
print("ROC AUC:", roc_auc_score(y_test, rf_probability))

regression_features = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "survived",
    "embarked"
]

X_reg = df[regression_features]
y_reg = df["fare"]

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg,
    y_reg,
    test_size=0.20,
    random_state=42
)

reg_numeric = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "survived"
]

reg_categorical = [
    "sex",
    "embarked"
]

reg_preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), reg_numeric),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ]), reg_categorical)
])

regression_model = Pipeline([
    ("preprocessor", reg_preprocessor),
    ("model", LinearRegression())
])

regression_model.fit(
    X_reg_train,
    y_reg_train
)

fare_prediction = regression_model.predict(
    X_reg_test
)

mae = mean_absolute_error(
    y_reg_test,
    fare_prediction
)

rmse = np.sqrt(
    mean_squared_error(
        y_reg_test,
        fare_prediction
    )
)

r2 = r2_score(
    y_reg_test,
    fare_prediction
)

processed_regression = regression_model.named_steps[
    "preprocessor"
].transform(X_reg_test)

p = processed_regression.shape[1]
n = len(y_reg_test)

adjusted_r2 = 1 - (
    (1 - r2) * (n - 1) / (n - p - 1)
)

print("\nRegression Results:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)
print("Adjusted R2:", adjusted_r2)

residuals = y_reg_test - fare_prediction

plt.figure()
plt.scatter(fare_prediction, residuals)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted Fare")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

residual_correlation = np.corrcoef(
    fare_prediction,
    abs(residuals)
)[0, 1]

print("\nResidual Correlation:", residual_correlation)

if abs(residual_correlation) > 0.2:
    print("Conclusion: The residual spread suggests possible heteroscedasticity.")
else:
    print("Conclusion: The residual spread does not show strong evidence of heteroscedasticity.")

tuned_results = {
    "Model": "Tuned Random Forest",
    "Accuracy": accuracy_score(y_test, rf_prediction),
    "Precision": precision_score(y_test, rf_prediction, zero_division=0),
    "Recall": recall_score(y_test, rf_prediction, zero_division=0),
    "F1": f1_score(y_test, rf_prediction, zero_division=0),
    "ROC AUC": roc_auc_score(y_test, rf_probability)
}

final_results = pd.concat(
    [
        results_df,
        pd.DataFrame([tuned_results])
    ],
    ignore_index=True
)

print("\nFinal Classification Comparison:")
print(final_results)

best_model_name = final_results.loc[
    final_results["F1"].idxmax(),
    "Model"
]

print("\nRecommended Classifier:")
print(best_model_name)

best_models = {
    "Logistic Regression": logistic,
    "Decision Tree": tree,
    "Random Forest": forest,
    "Tuned Random Forest": best_rf
}

best_model = best_models[best_model_name]

joblib.dump(
    best_model,
    "analytics/titanic_best_pipeline.joblib"
)

print("\nPipeline saved successfully.")

loaded_model = joblib.load(
    "analytics/titanic_best_pipeline.joblib"
)

sample = X_test.iloc[[0]]

print("\nPrediction using Reloaded Pipeline:")
print(loaded_model.predict(sample))

if hasattr(loaded_model, "predict_proba"):
    print("Prediction Probability:")
    print(loaded_model.predict_proba(sample))

results_df.to_csv(
    "analytics/classification_metrics.csv",
    index=False
)

imbalance_df.to_csv(
    "analytics/imbalance_comparison.csv",
    index=False
)

final_results.to_csv(
    "analytics/final_model_comparison.csv",
    index=False
)

pd.DataFrame({
    "Metric": ["MAE", "RMSE", "R2", "Adjusted R2"],
    "Value": [mae, rmse, r2, adjusted_r2]
}).to_csv(
    "analytics/regression_metrics.csv",
    index=False
)

print("\nAll Tasks 1-15 completed.")