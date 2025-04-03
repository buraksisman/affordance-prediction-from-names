import joblib

def predict_affordances(object_name: str):
    # Load model
    model = joblib.load("../models/model.pkl")
    vectorizer = joblib.load("../models/vectorizer.pkl")
    mlb = joblib.load("../models/mlb.pkl")

    # Transform input
    X = vectorizer.transform([object_name])
    y_pred = model.predict(X)

    # Decode multi-label output
    predicted_labels = mlb.inverse_transform(y_pred)

    return predicted_labels[0] if predicted_labels else []

if __name__ == "__main__":
    print("🔍 Affordance Predictor")
    print("------------------------")
    while True:
        obj = input("Enter an object name (or 'exit'): ").strip()
        if obj.lower() == "exit":
            break
        predicted = predict_affordances(obj)
        print(f"Predicted affordances: {', '.join(predicted) if predicted else 'None'}\n")
