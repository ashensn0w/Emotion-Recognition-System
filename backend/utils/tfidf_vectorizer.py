from preprocessing.text_processing import *
from sklearn.feature_extraction.text import TfidfVectorizer

def vectorize_with_tfidf(data):
    if 'sentence' in data.columns:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.85, min_df=2)

        tfidf_matrix = vectorizer.fit_transform(data['sentence'])

        print("TF-IDF Vectorization complete.")
        return tfidf_matrix, vectorizer
    else:
        print("Column 'sentence' not found in the DataFrame.")