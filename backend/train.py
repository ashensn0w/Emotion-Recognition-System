from preprocessing.text_processing import *
from preprocessing.narrative_features import *
import pandas as pd

file_path = './backend/data/sample_dataset.csv'
data = load_dataset(file_path)

# Extract sentences
sentences = data['sentence'].tolist()

if data is not None:
    # Print the first 10 sentences and their feature vectors
    print("First 10 Sentences and their Feature Vectors:")
    for i, sentence in enumerate(sentences[:10]):
        features = extract_narrative_features(sentence)
        feature_vector = [features[feature] for feature in sorted(features.keys())]
        print(f"Sentence {i+1}: {sentence}")
        print(f"Feature Vector: {feature_vector}")
        print()

    # Generate and print the feature matrix for the entire dataset
    feature_matrix = create_feature_matrix(sentences)
    print("Feature Matrix:")
    print(feature_matrix)

    # Print the initial data
    print_table(data, title="Original Data")

    # Convert to lowercase
    convert_to_lowercase(data)
    print_table(data, title="Data After Lowercase Conversion")

    # Remove punctuation
    remove_punctuation(data)
    print_table(data, title="Data After Punctuation Removal")

    # Remove numbers
    remove_numbers(data)
    print_table(data, title="Data After Numbers Removal")

    # Tokenize sentences (Now done AFTER all string-based steps)
    tokenize_sentences(data)
    print_table(data, title="Data After Tokenization")

    # Remove stopwords
    remove_stopwords(data)
    print_table(data, title="Data After Stopwords Removal")

    # Lemmatize tokens
    lemmatize_tokens(data)
    print_table(data, title="Data After Lemmatization")

    # Join tokens back into sentences
    join_tokens(data)
    print_table(data, title="Data After Joining Tokens")

    # Vectorize the processed text using TF-IDF
    tfidf_matrix, vectorizer = vectorize_with_tfidf(data)

    # Get feature names (i.e., the words)
    feature_names = vectorizer.get_feature_names_out()

    # Create a DataFrame from the TF-IDF matrix
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

    # Add the 'emotion' column to the TF-IDF DataFrame
    tfidf_df['emotion'] = data['emotion'].values

    # Display the resulting DataFrame
    print(tfidf_df.head())