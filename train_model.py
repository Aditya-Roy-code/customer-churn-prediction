import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("data/customer_churn.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# Clean column names
df.columns = df.columns.str.strip()


# ==========================================
# 2. CHECK REQUIRED COLUMNS
# ==========================================

required_columns = [
    "Names",
    "Age",
    "Total_Purchase",
    "Account_Manager",
    "Years",
    "Num_Sites",
    "Onboard_date",
    "Location",
    "Company",
    "Churn"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in CSV: {missing_columns}"
    )


# ==========================================
# 3. REMOVE NAME
# ==========================================

# Customer name is not useful for prediction
df = df.drop(columns=["Names"])


# ==========================================
# 4. CONVERT DATE
# ==========================================

df["Onboard_date"] = pd.to_datetime(
    df["Onboard_date"],
    errors="coerce"
)

# Create useful date features
df["Onboard_year"] = df["Onboard_date"].dt.year
df["Onboard_month"] = df["Onboard_date"].dt.month

# Remove original date column
df = df.drop(columns=["Onboard_date"])


# ==========================================
# 5. CLEAN CHURN COLUMN
# ==========================================

print("\nOriginal Churn values:")
print(df["Churn"].value_counts(dropna=False))

if df["Churn"].dtype == "object":

    df["Churn"] = (
        df["Churn"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "yes": 1,
            "no": 0,
            "true": 1,
            "false": 0,
            "1": 1,
            "0": 0
        })
    )

# Remove rows where Churn could not be converted
df = df.dropna(subset=["Churn"])

df["Churn"] = df["Churn"].astype(int)


# ==========================================
# 6. FEATURES AND TARGET
# ==========================================

X = df.drop(columns=["Churn"])
y = df["Churn"]


# ==========================================
# 7. IDENTIFY DATA TYPES
# ==========================================

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X.select_dtypes(
    exclude=["object"]
).columns.tolist()

print("\nCategorical features:")
print(categorical_features)

print("\nNumerical features:")
print(numerical_features)


# ==========================================
# 8. PREPROCESSING
# ==========================================

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    )
])

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(handle_unknown="ignore")
    )
])


preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_pipeline,
        numerical_features
    ),
    (
        "categorical",
        categorical_pipeline,
        categorical_features
    )
])


# ==========================================
# 9. RANDOM FOREST MODEL
# ==========================================

model = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        )
    )
])


# ==========================================
# 10. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining model...")


# ==========================================
# 11. TRAIN MODEL
# ==========================================

model.fit(
    X_train,
    y_train
)


# ==========================================
# 12. PREDICTION
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 13. MODEL EVALUATION
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n================================")
print("       MODEL PERFORMANCE")
print("================================")

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ==========================================
# 14. SAVE MODEL
# ==========================================

joblib.dump(
    model,
    "models/churn_model.pkl"
)

print("\n================================")
print("MODEL SAVED SUCCESSFULLY!")
print("================================")

print(
    "File: models/churn_model.pkl"
)