import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer

def load_and_preprocess(data_path):
    # Load dataset
    df = pd.read_csv(data_path)

    # Convert affordances to lists
    df['affordances'] = df['affordances'].apply(lambda x: x.split(','))

    # Vectorize object names
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['object_name'])

    # Binarize multi-label affordance outputs
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df['affordances'])

    return X, y, vectorizer, mlb
