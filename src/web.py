import streamlit as st
import joblib

# Load model and encoders
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")
mlb = joblib.load("models/mlb.pkl")

# App config
st.set_page_config(page_title="Affordance Predictor", page_icon="🧠")
st.title("🔍 Affordance Prediction from Object Names")
st.write("Enter an object name to see its predicted affordances.")

# Input box
object_name = st.text_input("Object name:", placeholder="e.g., knife, bowl, hammer")

# Predict on submit
if st.button("Predict") and object_name.strip():
    X = vectorizer.transform([object_name])
    y_pred = model.predict(X)
    labels = mlb.inverse_transform(y_pred)

    if labels and labels[0]:
        st.success("**Predicted Affordances:**")
        st.write(", ".join(labels[0]))
    else:
        st.warning("No affordances predicted.")
