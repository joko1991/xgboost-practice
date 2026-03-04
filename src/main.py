from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import xgboost as xgb


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load breast cancer dataset and return features + target."""
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")  # 0=malignant, 1=benign
    return X, y


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
    """Train a simple XGBoost classifier."""
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    return model


def main() -> None:
    X, y = load_data()

    # Stratify keeps class proportions similar in train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = train_model(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}\n")

    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, y_pred), "\n")

    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=["malignant", "benign"]))

    # Feature importance (top 10)
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nTop 10 features by importance:")
    print(importances.head(10).to_string())


if __name__ == "__main__":
    main()