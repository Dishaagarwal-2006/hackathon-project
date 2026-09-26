import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)


df = pd.read_csv("Dataset/transactions.csv")

features = [
    "amount",
    "transaction_hour",
    "transaction_frequency",
    "account_age_days",
    "location_change",
    "time_since_last_transaction",
    "device_change"
]

X = df[features]
y = df["is_fraud"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


model = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


model.fit(X_train, y_train)


predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]


print("\n========== MODEL EVALUATION ==========")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions
    )
)

print("\nROC-AUC:")
print(
    round(
        roc_auc_score(
            y_test,
            probabilities
        ),
        4
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        predictions
    )
)


joblib.dump(
    {
        "model": model,
        "features": features
    },
    "fraud_model.pkl"
)

print("\n======================================")
print("Model saved as fraud_model.pkl")
print("======================================")