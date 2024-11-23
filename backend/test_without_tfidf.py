from nltk.corpus import stopwords
from nltk.data import find
from nltk.tokenize import word_tokenize
from preprocessing.narrative_features_eng import *
from preprocessing.narrative_features_fil import *
from rich.console import Console
from rich.table import Table
from utils.save_load import *
import json
import nltk
import numpy as np
import pandas as pd
import spacy
import stopwordsiso as stopwords
import string

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

file_path = './backend/data/training_data.csv'
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

    def combine_features(fil_features_df, eng_features_df):
        # Ensure both DataFrames have the same structure
        assert fil_features_df.shape == eng_features_df.shape, "Feature dataframes must have the same shape"
        
        # Element-wise maximum between Filipino and English features
        combined_features = np.maximum(fil_features_df.values, eng_features_df.values)
        
        # Convert back to DataFrame with the same column names
        combined_features_df = pd.DataFrame(combined_features, columns=fil_features_df.columns)
        
        return combined_features_df

    def process_data(df):
        # Extract Filipino and English features
        fil_features_df = extract_fil_features_from_dataframe(df)
        eng_features_df = extract_eng_features_from_dataframe(df)

        # Combine the features
        combined_features_df = combine_features(fil_features_df, eng_features_df)

        combined_features_df['emotion'] = df['emotion']

        return combined_features_df

    # Read narrative features data from CSV
    narrative_file_path = './backend/data/training_data.csv'
    narrative_features_df = pd.read_csv(narrative_file_path)

    # Apply the feature extraction and combination process
    combined_features_df = process_data(narrative_features_df)

    combined_features_df.to_csv('./backend/data/feature vectors/trained_complete_vectorized_data_without_tfidf.csv', index=False)
    # <-------------------------------------------------------------------------------------------------------------->
    # Load the original dataset and make sure the 'emotion' column is intact
    file_path = './backend/data/training_data.csv'
    data = load_dataset(file_path)

    # Check if the 'emotion' column is present in the original data
    if 'emotion' in data.columns:
        # Add the original 'emotion' column from the data to final_combined_df
        combined_features_df['emotion'] = data['emotion']
    else:
        print("The 'emotion' column is missing from the dataset.")

    # Load the saved emotion recognition model
    emo_recog_model = load_model_with_name('emotion_recognition_model_without_tfidf.pkl')

    # Prepare the data for prediction (drop the 'emotion' column from feature data)
    X = combined_features_df.drop(columns=['emotion'])

    # Check if the model was loaded successfully
    if emo_recog_model is not None:
        # Assuming 'X' is your feature data (without the 'emotion' column)
        predictions = emo_recog_model.predict(X)
        print("Predictions made successfully.")
        combined_features_df['predicted_emotion'] = predictions
    else:
        print("Model not loaded. Unable to make predictions.")

    # Select the columns you want for output
    output_df = pd.DataFrame({
        'sentence': data['sentence'],  # From the original dataset
        'emotion': combined_features_df['emotion'],  # Actual emotion
        'predicted_emotion': combined_features_df['predicted_emotion']  # Predicted emotion
    })

    # Save the output to a new CSV file
    output_df.to_csv('./backend/data/feature vectors/without_tfidf_final_predictions.csv', index=False)
    output_df.to_csv('./frontend/backend outputs/without_tfidf_output.csv', index=False)

    # Display the first few rows in the console for review
    print_table(output_df, title="Sentences with Actual and Predicted Emotions")