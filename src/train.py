import joblib
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocess import load_and_preprocess

# Load and preprocess data
X, y, vectorizer, mlb = load_and_preprocess("../data/dataset.csv")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define classifier
classifier = OneVsRestClassifier(LinearSVC())
classifier.fit(X_train, y_train)

# Evaluation
y_pred = classifier.predict(X_test)
print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=mlb.classes_))

# Save model and encoders
joblib.dump(classifier, "../models/model.pkl")
joblib.dump(vectorizer, "../models/vectorizer.pkl")
joblib.dump(mlb, "../models/mlb.pkl")
