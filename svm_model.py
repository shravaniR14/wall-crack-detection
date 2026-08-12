import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

print("Loading saved HOG features...")

# Load HOG features
data = np.load("hog_features.npz")

X = data["X"]
y = data["y"]

print("Feature shape:", X.shape)
print("Total samples:", len(X))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# RBF SVM
print("\nTraining RBF SVM...")

svm_model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale",
    probability=True,
    random_state=42
)

svm_model.fit(X_train, y_train)

# Prediction
print("Making predictions...")

y_pred = svm_model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred) * 100

print("\n==============================")
print("RBF SVM Accuracy:", round(accuracy, 2), "%")
print("==============================")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(svm_model, "svm_rbf_hog_crack_model.pkl")

print("\nModel saved as:")
print("svm_rbf_hog_crack_model.pkl")