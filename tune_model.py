import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score


# =========================
# LOAD SAVED HOG FEATURES
# =========================

print("Loading saved HOG features...")

data = np.load("hog_features.npz")

X = data["X"]
y = data["y"]

print("Feature shape:", X.shape)
print("Total samples:", len(X))


# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# =========================
# RANDOM FOREST - TUNED
# =========================

print("\nTraining Tuned Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(
    y_test,
    rf_pred
) * 100

print(
    "Tuned Random Forest Accuracy:",
    round(rf_accuracy, 2),
    "%"
)


# =========================
# EXTRA TREES
# =========================

print("\nTraining Extra Trees...")

extra_model = ExtraTreesClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

extra_model.fit(X_train, y_train)

extra_pred = extra_model.predict(X_test)

extra_accuracy = accuracy_score(
    y_test,
    extra_pred
) * 100

print(
    "Extra Trees Accuracy:",
    round(extra_accuracy, 2),
    "%"
)


# =========================
# SAVE BEST MODEL
# =========================

if extra_accuracy >= rf_accuracy:

    best_model = extra_model
    best_accuracy = extra_accuracy
    model_name = "Extra Trees"

    joblib.dump(
        best_model,
        "best_hog_crack_model.pkl"
    )

else:

    best_model = rf_model
    best_accuracy = rf_accuracy
    model_name = "Random Forest"

    joblib.dump(
        best_model,
        "best_hog_crack_model.pkl"
    )


print("\n==============================")
print("BEST MODEL:", model_name)
print("BEST ACCURACY:", round(best_accuracy, 2), "%")
print("MODEL SAVED: best_hog_crack_model.pkl")
print("==============================")