import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, mean_absolute_error, mean_squared_error, r2_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

os.makedirs("analytics/charts", exist_ok=True)

# TASK 1
df = sns.load_dataset("titanic")
df.to_csv("analytics/titanic.csv", index=False)
print(df.info())
print(df.describe())
print("Shape:", df.shape)

missing = df.isnull().sum()
missing_pct = missing / len(df) * 100
print(pd.DataFrame({"Missing": missing[missing > 0], "Percentage": missing_pct[missing > 0]}))

# TASK 2
for col in missing[missing > 0].index:
    p = missing_pct[col]
    print(col, round(p, 2), "%", "Drop rows" if p < 5 else "Impute" if p <= 30 else "Drop column")

df = df.dropna(subset=["embarked"])
df["age"] = df["age"].fillna(df["age"].median())
df["embark_town"] = df["embark_town"].fillna(df["embark_town"].mode()[0])
df = df.drop(columns=["deck"])

# TASK 3
for col in ["age", "fare"]:
    plt.figure()
    plt.hist(df[col])
    plt.title(col + " Distribution")
    plt.savefig(f"analytics/charts/{col}_histogram.png")
    plt.show()

    plt.figure()
    plt.boxplot(df[col])
    plt.title(col + " Box Plot")
    plt.savefig(f"analytics/charts/{col}_boxplot.png")
    plt.show()

def outliers(s):
    q1, q3 = s.quantile([.25, .75])
    iqr = q3 - q1
    return ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()

print("Age outliers:", outliers(df["age"]))
print("Fare outliers:", outliers(df["fare"]))

mean = df["fare"].mean()
median = df["fare"].median()
mode = df["fare"].mode()[0]
print("Fare Mean:", mean, "Median:", median, "Mode:", mode)

if mean > median > mode:
    print("Fare is right-skewed")
elif mean < median < mode:
    print("Fare is left-skewed")
else:
    print("No clear skew")

# TASK 4
for sex in ["female", "male"]:
    print(sex, df.loc[df["sex"] == sex, "survived"].mean())

for pclass in [1, 2, 3]:
    print("Class", pclass, df.loc[df["pclass"] == pclass, "survived"].mean())

for sex in ["female", "male"]:
    for pclass in [1, 2, 3]:
        mask = (df["sex"] == sex) & (df["pclass"] == pclass)
        print(sex, pclass, df.loc[mask, "survived"].mean())

mask = ((df["sex"] == "female") & (df["pclass"] == 1)) | ((df["sex"] == "male") & (df["pclass"] == 1))
print("First class combined:", df.loc[mask, "survived"].mean())

cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
corr = df[cols].corr()
print(corr)

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True)
plt.title("Correlation Heatmap")
plt.savefig("analytics/charts/correlation_heatmap.png")
plt.show()

pairs = corr.where(~np.tril(np.ones(corr.shape)).astype(bool)).stack().sort_values(key=abs, ascending=False)
print("Top 2 correlations:")
print(pairs.head(2))

# TASK 5
plt.figure()
sns.barplot(data=df, x="pclass", y="survived", hue="sex")
plt.title("Survival by Class and Sex")
plt.savefig("analytics/charts/survival_class_sex.png")
plt.show()

plt.figure()
sns.boxplot(data=df, x="pclass", y="fare", hue="survived")
plt.title("Fare by Class and Survival")
plt.savefig("analytics/charts/fare_class_survival.png")
plt.show()

plt.figure()
sns.scatterplot(data=df, x="age", y="fare", hue="survived")
plt.title("Age, Fare and Survival")
plt.savefig("analytics/charts/age_fare_survival.png")
plt.show()

plt.figure()
sns.barplot(data=df, x="sibsp", y="survived", hue="sex")
plt.title("Survival by Siblings/Spouses and Sex")
plt.savefig("analytics/charts/sibsp_sex_survival.png")
plt.show()

print("Chart 1: Females and first-class passengers generally had higher survival rates.")
print("Chart 2: First-class passengers generally paid higher fares.")
print("Chart 3: Higher fares are generally associated with better survival.")
print("Chart 4: Survival varies with family size and sex.")

# TASK 6
scaled = df[["age", "fare"]].copy()
print("Before:", scaled.mean().to_dict(), scaled.std().to_dict())
scaled = pd.DataFrame(StandardScaler().fit_transform(scaled), columns=["age", "fare"])
print("After:", scaled.mean().to_dict(), scaled.std().to_dict())

# TASK 7
X = df.drop(columns=["survived"])
y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.2, random_state=42, stratify=y
)

print("Class balance:", y.value_counts(normalize=True))

# TASK 8
num = ["pclass", "age", "sibsp", "parch", "fare"]
cat = ["sex", "embarked"]

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), num),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]), cat)
])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("Before preprocessing:", X_train.shape)
print("After preprocessing:", X_train_processed.shape)

# TASK 9
logistic = LogisticRegression(max_iter=1000, random_state=42)
tree = DecisionTreeClassifier(random_state=42)
forest = RandomForestClassifier(n_estimators=100, random_state=42)

for model in [logistic, tree, forest]:
    model.fit(X_train_processed, y_train)

predictions = {
    "Logistic Regression": logistic.predict(X_test_processed),
    "Decision Tree": tree.predict(X_test_processed),
    "Random Forest": forest.predict(X_test_processed)
}

probabilities = {
    "Logistic Regression": logistic.predict_proba(X_test_processed)[:, 1],
    "Decision Tree": tree.predict_proba(X_test_processed)[:, 1],
    "Random Forest": forest.predict_proba(X_test_processed)[:, 1]
}

plt.figure(figsize=(18, 10))
plot_tree(
    tree,
    feature_names=preprocessor.get_feature_names_out(),
    class_names=["Not Survived", "Survived"],
    filled=True,
    max_depth=4
)
plt.savefig("analytics/charts/decision_tree.png")
plt.show()

# TASK 10
results = []

for name in predictions:
    pred = predictions[name]
    prob = probabilities[name]

    cm = confusion_matrix(y_test, pred)
    acc = accuracy_score(y_test, pred)
    pre = precision_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    auc = roc_auc_score(y_test, prob)

    print(name, "\nConfusion Matrix:\n", cm)

    results.append([name, acc, pre, rec, f1, auc])

results_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "Precision", "Recall", "F1", "ROC AUC"]
)

print(results_df)

plt.figure(figsize=(8, 6))
for name in probabilities:
    fpr, tpr, _ = roc_curve(y_test, probabilities[name])
    plt.plot(fpr, tpr, label=f"{name} AUC={roc_auc_score(y_test, probabilities[name]):.3f}")

plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("analytics/charts/roc_curves.png")
plt.show()

# TASK 11
baseline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000, random_state=42))
])

balanced = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
])

smote = ImbPipeline([
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", LogisticRegression(max_iter=1000, random_state=42))
])

imbalance = []

for name, model in {
    "Baseline": baseline,
    "Balanced": balanced,
    "SMOTE": smote
}.items():

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    imbalance.append([
        name,
        precision_score(y_test, pred, zero_division=0),
        recall_score(y_test, pred, zero_division=0),
        f1_score(y_test, pred, zero_division=0)
    ])

imbalance_df = pd.DataFrame(
    imbalance,
    columns=["Method", "Precision", "Recall", "F1"]
)

print(imbalance_df)

# TASK 12
rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        oob_score=True,
        random_state=42,
        n_jobs=1
    ))
])

grid = GridSearchCV(
    rf_pipeline,
    {
        "model__n_estimators": [100],
        "model__max_depth": [5, 10],
        "model__max_features": ["sqrt"]
    },
    cv=3,
    scoring="accuracy",
    n_jobs=1
)

grid.fit(X_train, y_train)
best_rf = grid.best_estimator_

print("Best parameters:", grid.best_params_)
print("CV score:", grid.best_score_)
print("OOB score:", best_rf.named_steps["model"].oob_score_)

tuned_pred = best_rf.predict(X_test)
tuned_prob = best_rf.predict_proba(X_test)[:, 1]

print(
    "Tuned RF:",
    accuracy_score(y_test, tuned_pred),
    precision_score(y_test, tuned_pred, zero_division=0),
    recall_score(y_test, tuned_pred, zero_division=0),
    f1_score(y_test, tuned_pred, zero_division=0),
    roc_auc_score(y_test, tuned_prob)
)

# TASK 13
reg_features = ["pclass", "sex", "age", "sibsp", "parch", "survived", "embarked"]

Xr = df[reg_features]
yr = df["fare"]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    Xr, yr, test_size=.2, random_state=42
)

reg_num = ["pclass", "age", "sibsp", "parch", "survived"]
reg_cat = ["sex", "embarked"]

reg_preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), reg_num),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]), reg_cat)
])

regression = Pipeline([
    ("preprocessor", reg_preprocessor),
    ("model", LinearRegression())
])

regression.fit(Xr_train, yr_train)
fare_pred = regression.predict(Xr_test)

mae = mean_absolute_error(yr_test, fare_pred)
rmse = np.sqrt(mean_squared_error(yr_test, fare_pred))
r2 = r2_score(yr_test, fare_pred)

p = regression.named_steps["preprocessor"].transform(Xr_test).shape[1]
n = len(yr_test)

adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)
print("Adjusted R2:", adj_r2)

residuals = yr_test - fare_pred

plt.figure()
plt.scatter(fare_pred, residuals)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted Fare")
plt.ylabel("Residuals")
plt.title("Fare Residual Plot")
plt.savefig("analytics/charts/fare_residual_plot.png")
plt.show()

res_corr = np.corrcoef(fare_pred, abs(residuals))[0, 1]
print("Residual correlation:", res_corr)

if abs(res_corr) > .20:
    print("Possible heteroscedasticity")
else:
    print("No strong evidence of heteroscedasticity")

# TASK 14
tuned_row = pd.DataFrame([[
    "Tuned Random Forest",
    accuracy_score(y_test, tuned_pred),
    precision_score(y_test, tuned_pred, zero_division=0),
    recall_score(y_test, tuned_pred, zero_division=0),
    f1_score(y_test, tuned_pred, zero_division=0),
    roc_auc_score(y_test, tuned_prob)
]], columns=results_df.columns)

final_results = pd.concat([results_df, tuned_row], ignore_index=True)

print("\nCLASSIFICATION")
print(final_results)

print("\nREGRESSION")
print(pd.DataFrame({
    "Metric": ["MAE", "RMSE", "R2", "Adjusted R2"],
    "Value": [mae, rmse, r2, adj_r2]
}))

best_name = final_results.loc[final_results["F1"].idxmax(), "Model"]
best_row = final_results.loc[final_results["F1"].idxmax()]

print(
    f"Recommended classifier: {best_name}. "
    f"It has F1={best_row['F1']:.3f}, "
    f"Precision={best_row['Precision']:.3f}, "
    f"Recall={best_row['Recall']:.3f}, "
    f"ROC AUC={best_row['ROC AUC']:.3f}. "
    "It provides the strongest overall classification performance."
)

# TASK 15
if best_name == "Tuned Random Forest":
    final_pipeline = best_rf
else:
    final_pipeline = {
        "Logistic Regression": baseline,
        "Decision Tree": Pipeline([
            ("preprocessor", preprocessor),
            ("model", DecisionTreeClassifier(random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(n_estimators=100, random_state=42))
        ])
    }[best_name]

final_pipeline.fit(X_train, y_train)

joblib.dump(
    final_pipeline,
    "analytics/titanic_best_pipeline.joblib"
)

loaded = joblib.load(
    "analytics/titanic_best_pipeline.joblib"
)

sample = X_test.iloc[[0]]

print("Reloaded pipeline prediction:", loaded.predict(sample))

results_df.to_csv(
    "analytics/classification_metrics.csv",
    index=False
)

imbalance_df.to_csv(
    "analytics/imbalance_comparison.csv",
    index=False
)

final_results.to_csv(
    "analytics/final_classification_results.csv",
    index=False
)

pd.DataFrame({
    "Metric": ["MAE", "RMSE", "R2", "Adjusted R2"],
    "Value": [mae, rmse, r2, adj_r2]
}).to_csv(
    "analytics/regression_metrics.csv",
    index=False
)

print("ALL TASKS 1-15 COMPLETED")