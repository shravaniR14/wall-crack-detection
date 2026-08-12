import os
import cv2
import numpy as np
import joblib

from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# =========================
# SETTINGS
# =========================

POSITIVE_PATH = "Positive"
NEGATIVE_PATH = "Negative"

IMG_SIZE = (128, 128)


# =========================
# HOG FEATURE EXTRACTION
# =========================

def extract_hog_features(folder_path, label):

    features = []
    labels = []

    files = os.listdir(folder_path)

    print(f"\nProcessing {folder_path}: {len(files)} images")

    for i, filename in enumerate(files):

        filepath = os.path.join(folder_path, filename)

        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

        if image is None:
            continue

        image = cv2.resize(image, IMG_SIZE)

        hog_features = hog(
            image,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2)
        )

        features.append(hog_features)
        labels.append(label)

        # Progress every 500 images
        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1}/{len(files)}")


    return features, labels


# =========================
# LOAD + HOG FEATURES
# =========================

print("Starting HOG feature extraction...")


positive_features, positive_labels = extract_hog_features(
    POSITIVE_PATH,
    1
)

negative_features, negative_labels = extract_hog_features(
    NEGATIVE_PATH,
    0
)


# Combine datasets

X = np.array(
    positive_features + negative_features
)

y = np.array(
    positive_labels + negative_labels
)


print("\nHOG extraction completed!")

print("Total samples:", len(X))
print("Feature shape:", X.shape)

print("Positive samples:", np.sum(y == 1))
print("Negative samples:", np.sum(y == 0))


# =========================
# SAVE HOG FEATURES
# =========================

np.savez(
    "hog_features.npz",
    X=X,
    y=y
)

print("\nHOG features saved as hog_features.npz")


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

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# =========================
# LOGISTIC REGRESSION
# =========================

print("\nTraining Logistic Regression...")

lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

lr_accuracy = accuracy_score(
    y_test,
    y_pred_lr
) * 100

print(
    "HOG + Logistic Regression Accuracy:",
    round(lr_accuracy, 2),
    "%"
)


# =========================
# RANDOM FOREST
# =========================

print("\nTraining Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

rf_accuracy = accuracy_score(
    y_test,
    y_pred_rf
) * 100

print(
    "HOG + Random Forest Accuracy:",
    round(rf_accuracy, 2),
    "%"
)


# =========================
# SAVE BEST MODEL
# =========================

if rf_accuracy >= lr_accuracy:

    joblib.dump(
        rf_model,
        "rf_hog_crack_model.pkl"
    )

    print("\nBest model: Random Forest")
    print("Saved as: rf_hog_crack_model.pkl")

else:

    joblib.dump(
        lr_model,
        "lr_hog_crack_model.pkl"
    )

    print("\nBest model: Logistic Regression")
    print("Saved as: lr_hog_crack_model.pkl")


print("\n================================")
print("TRAINING COMPLETED SUCCESSFULLY")
print("================================")