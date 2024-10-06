from preprocessing.text_processing import *
from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize_with_tfidf(data):
    if 'sentence' in data.columns:
        vectorizer = TfidfVectorizer()

        tfidf_matrix = vectorizer.fit_transform(data['sentence'])

        print("TF-IDF Vectorization complete.")
        return tfidf_matrix, vectorizer
    else:
        print("Column 'sentence' not found in the DataFrame.")