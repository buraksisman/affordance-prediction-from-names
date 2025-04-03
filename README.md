# Affordance Prediction from Object Names

This project uses machine learning to predict **object affordances**—such as `grasp`, `cut`, `contain`, or `scoop`—based solely on an object's **name** (e.g., "spoon", "hammer", "bowl"). The system takes a simple word or phrase as input and returns the set of likely affordances associated with it.

---

## Purpose of the Project

The ability to understand object affordances is a key component of **robotic reasoning**, **tool use**, and **human-robot interaction**. While traditional methods rely on visual perception or simulation, this project explores whether we can infer affordances using **only the name** of an object.

This project aims to:

- Demonstrate how **language can inform affordance inference**
- Provide a **lightweight affordance prediction module** that can integrate with robotic systems

---

## Features

- Multi-label classification using **Support Vector Machines**
- Text vectorization with **TF-IDF**
- Command-line and **FastAPI** interface
- Interactive **Streamlit web UI**
- Easily extendable for research, robotics, or educational use

---

## Project Structure

```
affordance-prediction-from-names/
├── data/                 # CSV dataset of object names + affordances
├── models/               # Trained model + encoders
├── src/
│   ├── preprocess.py     # Preprocessing utilities
│   ├── train.py          # Model training script
│   ├── predict.py        # CLI-based prediction
│   ├── api.py            # FastAPI web service
│   └── web.py            # Streamlit user interface
├── requirements.txt
└── README.md
```

---

## Dataset

I use a custom-built dataset (`data/dataset.csv`) with over **100 common objects**, each labeled with one or more affordances like:

```csv
object_name,affordances
knife,"grasp,cut"
bowl,"grasp,contain"
hammer,"grasp,pound"
```

Affordances are based on a validated set used in affordance and tool-use research.

---

## Installation

```bash
git clone https://github.com/buraksisman/affordance-prediction-from-names.git
cd affordance-prediction-from-names
pip install -r requirements.txt
```

---

## Training the Model

```bash
# Train the SVM model
python -m src.train
```

---

## Predict from CLI

```bash
python -m src.predict
# Then enter: screwdriver
```

---

## Run the FastAPI Server

```bash
PYTHONPATH=. uvicorn src.api:app --reload
# Access docs at: http://127.0.0.1:8000/docs
```

---

## Run the Streamlit Web App

```bash
streamlit run src/web.py
```

---

## Example

**Input**:
```json
{ "object_name": "ladle" }
```

**Output**:
```json
{ "affordances": ["grasp", "scoop"] }
```

---


## 📄 License

This project is released under the [MIT License](LICENSE).

---

## 🤖 Created by

**Burak Sisman** – TU Delft, Department of Cognitive Robotics  
Contact: [b.sisman@tudelft.nl]