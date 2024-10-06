from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from preprocessing.narrative_features_eng import *
from preprocessing.narrative_features_fil import *
from rich.console import Console
from rich.table import Table
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.save_load import *
import json
import nltk
import numpy as np
import pandas as pd
import pickle
import spacy
import stopwordsiso as stopwords
import string

nltk.download('punkt')
nltk.download('stopwords')

nlp = spacy.load("en_core_web_md")

filipino_stopwords = stopwords.stopwords('tl')
english_stopwords = set(stopwords.stopwords('english'))
# <-------------------------------------------------------------------------------------------------------------->
def load_dataset(file_path):
    try:
        data = pd.read_csv(file_path)
        print("Dataset loaded successfully!")
        return data
    except FileNotFoundError:
        print("File not found. Please check the file path.")
    except pd.errors.EmptyDataError:
        print("File is empty. Please check the file content.")
    except pd.errors.ParserError:
        print("Error parsing file. Please check the file format.")
    except Exception as e:
        print(f"An error occurred: {e}")

file_path = './backend/data/sample.csv'
data = load_dataset(file_path)

sentences = data['sentence'].tolist()
# <-------------------------------------------------------------------------------------------------------------->
def format_list_as_string(token_list):
    return str(token_list).replace("'", '"')
# <-------------------------------------------------------------------------------------------------------------->
def print_table(data, title="Table", num_samples=20):    
    table = Table(title=title)
    
    # Add column names to the table
    for col in data.columns:
        table.add_column(col)

    # Add rows to the table
    for _, row in data.head(num_samples).iterrows():
        formatted_row = [format_list_as_string(row[col]) if isinstance(row[col], list) else row[col] for col in data.columns]
        table.add_row(*map(str, formatted_row))
    
    # Display the table
    console = Console()
    console.print(table)
# <-------------------------------------------------------------------------------------------------------------->
# Check if the dataset is loaded successfully
if data is not None:
    print_table(data, title="Original Data")
    # <-------------------------------------------------------------------------------------------------------------->
    def convert_to_lowercase(data):
        if 'sentence' in data.columns:
            data['sentence'] = data['sentence'].str.lower()
            print("Sentence has been converted to lowercase.")
        else:
            print("Column 'sentence' not found in the DataFrame.")

    convert_to_lowercase(data)
    print_table(data, title="Data After Lowercase Conversion")
    # <-------------------------------------------------------------------------------------------------------------->
    def remove_punctuation(data):
        if 'sentence' in data.columns:
            data['sentence'] = data['sentence'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
            print("Punctuation has been removed.")
        else:
            print("Column 'sentence' not found in the DataFrame.")

    remove_punctuation(data)
    print_table(data, title="Data After Punctuation Removal")
    # <-------------------------------------------------------------------------------------------------------------->
    def remove_numbers(data):
        if 'sentence' in data.columns:
            data['sentence'] = data['sentence'].str.replace(r'\d+', '', regex=True)
            print("Numbers have been removed.")
        else:
            print("Column 'sentence' not found in the DataFrame.")

    remove_numbers(data)
    print_table(data, title="Data After Numbers Removal")
    # <-------------------------------------------------------------------------------------------------------------->
    def tokenize_sentences(data):
        if 'sentence' in data.columns:
            data['sentence'] = data['sentence'].apply(lambda x: word_tokenize(x))
            print("Sentences have been tokenized.")
        else:
            print("Column 'sentence' not found in the DataFrame.")

    tokenize_sentences(data)
    print_table(data, title="Data After Tokenization")
    # <-------------------------------------------------------------------------------------------------------------->
    def remove_stopwords(data):
        if 'sentence' in data.columns:
            all_stopwords = english_stopwords.union(set(filipino_stopwords))
            
            data['sentence'] = data['sentence'].apply(lambda tokens: [word for word in tokens if word.lower() not in all_stopwords])
            print("Stopwords have been removed.")
        else:
            print("Column 'sentence' not found in the DataFrame.")

    remove_stopwords(data)
    print_table(data, title="Data After Stopwords Removal")
    # <-------------------------------------------------------------------------------------------------------------->
    def lemmatize_filo(data):
        with open('./backend/data/filipino_lemmatizer.json', 'r', encoding='utf-8') as json_file:
            lemma_dict = json.load(json_file)

        token_to_lemma = {}
        for lemma, tokenval in lemma_dict['lemma_dict'].items():
            for token in tokenval:
                token_to_lemma[token] = lemma

        for index, row in data.iterrows():
            tokens = row['sentence']
            
            if all(len(token) == 1 for token in tokens):
                tokens = ''.join(tokens).split()
            
            updated_tokens = [token_to_lemma.get(token, token) for token in tokens]
            data.at[index, 'sentence'] = updated_tokens

    lemmatize_filo(data)
    print_table(data, title="Data After Lemmatization in Filipino")
    # <-------------------------------------------------------------------------------------------------------------->
    def lemmatize_eng(data):
        if 'sentence' in data.columns:
            data['sentence'] = data['sentence'].apply(lambda tokens: [nlp(token)[0].lemma_ for token in tokens])
            print("Tokens have been lemmatized.")
        else:
            print("Column 'sentence' not found in the DataFrame.")

    lemmatize_eng(data)
    print_table(data, title="Data After Lemmatization in English")
    # <-------------------------------------------------------------------------------------------------------------->
    def join_tokens(data):
        if 'sentence' in data.columns:
            data['sentence'] = data['sentence'].apply(lambda tokens: ' '.join(tokens))
            print("Tokens have been joined back into sentences.")
        else:
            print("Column 'sentence' not found in the DataFrame.")

    join_tokens(data)
    print_table(data, title="Data After Joining Tokens")
    # <-------------------------------------------------------------------------------------------------------------->
    # Load the saved TF-IDF vectorizer
    tfidf_vectorizer = load_model_with_name('tfidf_vectorizer_model.pkl')

    # Check if the TF-IDF vectorizer was loaded successfully
    if tfidf_vectorizer:
        # Transform the sentences using the loaded TF-IDF vectorizer
        tfidf_matrix = tfidf_vectorizer.transform(data['sentence'])

        # Convert the TF-IDF matrix to a DataFrame for better visualization (optional)
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

        print("TF-IDF transformation successful!")
        print_table(tfidf_df, title="TF-IDF Features")
    else:
        print("TF-IDF vectorizer could not be loaded.")

    # # Combine Filipino and English feature vectors using element-wise maximum
    # def combine_features(fil_features_df, eng_features_df):
    #     # Ensure both DataFrames have the same structure
    #     assert fil_features_df.shape == eng_features_df.shape, "Feature dataframes must have the same shape"
        
    #     # Element-wise maximum between Filipino and English features
    #     combined_features = np.maximum(fil_features_df.values, eng_features_df.values)
        
    #     # Convert back to DataFrame with the same column names
    #     combined_features_df = pd.DataFrame(combined_features, columns=fil_features_df.columns)
        
    #     return combined_features_df

    # def process_data(df):
    #     # Extract Filipino and English features
    #     fil_features_df = extract_fil_features_from_dataframe(df)
    #     eng_features_df = extract_eng_features_from_dataframe(df)

    #     # Combine the features
    #     combined_features_df = combine_features(fil_features_df, eng_features_df)

    #     return combined_features_df

    # # Read narrative features data from CSV
    # narrative_file_path = './backend/data/testing_data_sample.csv'
    # narrative_features_df = pd.read_csv(narrative_file_path)

    # # Apply the feature extraction and combination process
    # combined_features_df = process_data(narrative_features_df)

    # # Now combine tfidf_df, combined_features_df, and 'emotion'
    # # Drop the 'emotion' column from tfidf_df before concatenation
    # # tfidf_no_emotion = tfidf_df.drop(columns=['emotion'])

    # # Concatenate TF-IDF features and combined narrative features
    # final_combined_df = pd.concat([tfidf_df, combined_features_df], axis=1)

    # # Add the 'emotion' column back as the last column
    # final_combined_df['emotion'] = tfidf_df['emotion'].values

    # # Print the final combined DataFrame
    # print("Final combined DataFrame:")
    # print(final_combined_df.head())

    # # Save the final combined features DataFrame to CSV
    # final_combined_df.to_csv('./backend/data/feature vectors/complete_vectorized_data.csv', index=False)
    # # 
    # # Load the pre-trained model
    # model_path = './backend/models/best_emotion_recognition_glm_model.pkl'
    # try:
    #     with open(model_path, 'rb') as model_file:
    #         emo_recog_model = pickle.load(model_file)
    #     print("Model loaded successfully.")
    # except FileNotFoundError:
    #     print(f"Model file '{model_path}' not found. Please check the path and try again.")
    #     emo_recog_model = None  # Make sure this doesn't cause further issues

    # # Prepare the data for prediction (drop the 'emotion' column)
    # X = final_combined_df.drop(columns=['emotion'])

    # # Check if the model was loaded successfully
    # if emo_recog_model is not None:
    #     # Assuming 'X' is your feature data (without the 'emotion' column)
    #     predictions = emo_recog_model.predict(X)
    #     print("Predictions made successfully.")
    #     final_combined_df['predicted_emotion'] = predictions
    # else:
    #     print("Model not loaded. Unable to make predictions.")

    # # Display the final DataFrame with predictions
    # print("Final DataFrame with predictions:")
    # print(final_combined_df.head())

    # # Save the predictions to a new CSV file
    # final_combined_df.to_csv('./backend/data/feature vectors/final_predictions.csv', index=False)