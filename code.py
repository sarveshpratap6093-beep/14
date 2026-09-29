# ============================================================
# RISK FACTOR PREDICTION AND ALARM SYSTEM
# ============================================================
#
# Machine Learning Model:
# XGBoost Multiclass Classification
#
# Risk Factor:
# 0 = LOW
# 1 = MODERATE
# 2 = HIGH
# 3 = VERY HIGH
#
# Alarm:
# Risk Factor 0       -> Alarm 0
# Risk Factor >= 1    -> Alarm 1
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier


# ============================================================
# 2. LOAD DATASET
# ============================================================

# Put the dataset in the same folder as this Python file.
# Change the filename if necessary.

DATASET_FILE = "pneumonia_dataset.csv"

df = pd.read_csv(DATASET_FILE)

print("Dataset loaded successfully")
print("Number of rows:", len(df))
print("Number of columns:", len(df.columns))

print("\nDataset columns:")
print(df.columns.tolist())


# ============================================================
# 3. DEFINE INPUT FEATURES
# ============================================================

features = [
    "age_years",
    "spo2",
    "heart_rate_bpm",
    "respiratory_rate_bpm",
    "systolic_bp_mmhg",
    "diastolic_bp_mmhg"
]

target = "risk_factor"


# ============================================================
# 4. CREATE X AND Y
# ============================================================

X = df[features].copy()

y = df[target].copy()


print("\nInput features:")
print(features)

print("\nTarget:")
print(target)


# ============================================================
# 5. CHECK RISK FACTOR DISTRIBUTION
# ============================================================

print("\nRisk Factor Distribution:")

print(
    y.value_counts()
    .sort_index()
)


# ============================================================
# 6. PATIENT-LEVEL TRAIN / TEST SPLIT
# ============================================================

# Patient ID is used only for grouping.
# It is NOT used as a prediction feature.

groups = df["patient_id"]


splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)


train_idx, test_idx = next(
    splitter.split(
        X,
        y,
        groups=groups
    )
)


X_train = X.iloc[train_idx].copy()
X_test = X.iloc[test_idx].copy()

y_train = y.iloc[train_idx].copy()
y_test = y.iloc[test_idx].copy()


print("\nTrain/Test Split")
print("----------------")

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))


# ============================================================
# 7. HANDLE MISSING VALUES
# ============================================================

# Median imputation is fitted ONLY on training data.

imputer = SimpleImputer(
    strategy="median"
)


X_train = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=features,
    index=X_train.index
)


X_test = pd.DataFrame(
    imputer.transform(X_test),
    columns=features,
    index=X_test.index
)


print("\nMissing values handled using median imputation.")


# ============================================================
# 8. CREATE XGBOOST MODEL
# ============================================================

xgb_model = XGBClassifier(

    objective="multi:softprob",

    num_class=4,

    n_estimators=300,

    max_depth=6,

    learning_rate=0.05,

    subsample=0.8,

    colsample_bytree=0.8,

    eval_metric="mlogloss",

    random_state=42,

    tree_method="hist",

    n_jobs=-1
)


# ============================================================
# 9. TRAIN MODEL
# ============================================================

print("\nTraining XGBoost model...")

xgb_model.fit(
    X_train,
    y_train
)

print("Training completed successfully.")


# ============================================================
# 10. PREDICT RISK FACTOR
# ============================================================

y_pred = xgb_model.predict(
    X_test
)


# ============================================================
# 11. MODEL ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n========================================")
print("MODEL PERFORMANCE")
print("========================================")

print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)


# ============================================================
# 12. CLASSIFICATION REPORT
# ============================================================

risk_names = [
    "LOW",
    "MODERATE",
    "HIGH",
    "VERY HIGH"
]


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2, 3],
        target_names=risk_names,
        zero_division=0
    )
)


# ============================================================
# 13. CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1, 2, 3]
)

print(cm)


# ============================================================
# 14. FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({

    "Feature": features,

    "Importance":
        xgb_model.feature_importances_

})


feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)


print("\nFeature Importance:")

print(feature_importance)


# ============================================================
# 15. SAVE TRAINED MODEL
# ============================================================

import joblib


joblib.dump(
    xgb_model,
    "pneumonia_xgboost_model.pkl"
)


joblib.dump(
    imputer,
    "pneumonia_imputer.pkl"
)


joblib.dump(
    features,
    "pneumonia_features.pkl"
)


print("\nModel files saved successfully.")


# ============================================================
# 16. PATIENT PREDICTION FUNCTION
# ============================================================

def predict_patient(
    age,
    spo2,
    heart_rate,
    respiratory_rate,
    systolic_bp,
    diastolic_bp
):

    # Create patient dataframe

    patient = pd.DataFrame({

        "age_years": [age],

        "spo2": [spo2],

        "heart_rate_bpm": [heart_rate],

        "respiratory_rate_bpm": [
            respiratory_rate
        ],

        "systolic_bp_mmhg": [
            systolic_bp
        ],

        "diastolic_bp_mmhg": [
            diastolic_bp
        ]
    })


    # Apply the same imputer
    # used during model training

    patient = pd.DataFrame(

        imputer.transform(patient),

        columns=features
    )


    # Predict risk factor

    prediction = xgb_model.predict(
        patient
    )[0]


    # Get probabilities

    probabilities = xgb_model.predict_proba(
        patient
    )[0]


    # Convert risk number to name

    predicted_risk = risk_names[
        prediction
    ]


    # ========================================================
    # ALARM LOGIC
    # ========================================================

    if prediction == 0:

        alarm = 0

    else:

        alarm = 1


    return (
        prediction,
        predicted_risk,
        probabilities,
        alarm
    )


# ============================================================
# 17. TAKE PATIENT INPUT
# ============================================================

print("\n========================================")
print("PATIENT RISK PREDICTION")
print("========================================")


age = float(
    input("Enter Age: ")
)

spo2 = float(
    input("Enter SpO2: ")
)

heart_rate = float(
    input("Enter Heart Rate: ")
)

respiratory_rate = float(
    input("Enter Respiratory Rate: ")
)

systolic_bp = float(
    input("Enter Systolic BP: ")
)

diastolic_bp = float(
    input("Enter Diastolic BP: ")
)


# ============================================================
# 18. RUN PREDICTION
# ============================================================

(
    prediction,
    predicted_risk,
    probabilities,
    alarm
) = predict_patient(

    age,

    spo2,

    heart_rate,

    respiratory_rate,

    systolic_bp,

    diastolic_bp
)


# ============================================================
# 19. DISPLAY FINAL RESULT
# ============================================================

print("\n========================================")
print("FINAL RESULT")
print("========================================")


print(
    "Predicted Risk:",
    predicted_risk
)


print(
    "Risk Factor:",
    prediction
)


print("\nRisk Probabilities:")


for i in range(4):

    print(
        f"{risk_names[i]}: "
        f"{probabilities[i] * 100:.2f}%"
    )


print("\n----------------------------------------")


print(
    "Alarm:",
    alarm
)


if alarm == 1:

    print(
        "ALARM STATUS: RAISED"
    )

else:

    print(
        "ALARM STATUS: NOT RAISED"
    )


print("========================================")
