from preprocessing.text_processing import *
from preprocessing.narrative_features_eng import *
from utils.tfidf_vectorizer import *
from utils.save_load import *
from rich.console import Console
from rich.table import Table
import pandas as pd

# Function to load a dataset from a file path
def load_dataset(file_path):
    try:
        data = pd.read_csv(file_path)
        print(f"Dataset loaded successfully from {file_path}!")
        return data
    except FileNotFoundError:
        print(f"File not found at {file_path}. Please check the file path.")
    except pd.errors.EmptyDataError:
        print(f"File at {file_path} is empty. Please check the file content.")
    except pd.errors.ParserError:
        print(f"Error parsing file at {file_path}. Please check the file format.")
    except Exception as e:
        print(f"An error occurred while loading {file_path}: {e}")

# Define file paths for Filipino and English datasets
filipino_file_path = './backend/data/training_data_sample_fil.csv'
english_file_path = './backend/data/training_data_sample_eng.csv'

# Load both Filipino and English datasets
filipino_data = load_dataset(filipino_file_path)
english_data = load_dataset(english_file_path)

# Convert the 'sentence' column from both datasets to a list of sentences
filipino_sentences = filipino_data['sentence'].tolist()
english_sentences = english_data['sentence'].tolist()

# Function to format a list as a string
def format_list_as_string(token_list):
    return str(token_list).replace("'", '"')

# Function to print a sample table from the dataset
def print_table(data, title="Table", num_samples=10):    
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

# Check if both datasets are loaded successfully
if filipino_data is not None and english_data is not None:
    print("Both datasets loaded successfully!\n")
    print_table(filipino_data, title="Original Filipino Data")
    print_table(english_data, title="Original English Data")
# <-------------------------------------------------------------------------------------------------------------->
    # LOWERCASE CONVERSION
    convert_to_lowercase(filipino_data, dataset_name="Filipino Data")
    convert_to_lowercase(english_data, dataset_name="English Data")

    # Print both datasets after lowercase conversion
    print("Filipino Data After Lowercase Conversion:")
    print_table(filipino_data, title="Filipino Data After Lowercase Conversion")
    print("\nEnglish Data After Lowercase Conversion:")
    print_table(english_data, title="English Data After Lowercase Conversion")
# <-------------------------------------------------------------------------------------------------------------->
    # PUNCTUATION REMOVAL
    remove_punctuation(filipino_data, dataset_name="Filipino Data")
    remove_punctuation(english_data, dataset_name="English Data")

    # Print both datasets after punctuation removal
    print("Filipino Data After Punctuation Removal:")
    print_table(filipino_data, title="Filipino Data After Punctuation Removal")
    print("\nEnglish Data After Punctuation Removal:")
    print_table(english_data, title="English Data After Punctuation Removal")
# <-------------------------------------------------------------------------------------------------------------->
    # NUMBERS REMOVAL
    remove_numbers(filipino_data, dataset_name="Filipino Data")
    remove_numbers(english_data, dataset_name="English Data")

    # Print both datasets after number removal
    print("Filipino Data After Numbers Removal:")
    print_table(filipino_data, title="Filipino Data After Numbers Removal")
    print("\nEnglish Data After Numbers Removal:")
    print_table(english_data, title="English Data After Numbers Removal")
# <-------------------------------------------------------------------------------------------------------------->
    # TOKENIZATION
    tokenize_sentences(filipino_data, dataset_name="Filipino Data")
    tokenize_sentences(english_data, dataset_name="English Data")

    # Print both datasets after tokenization
    print("Filipino Data After Tokenization:")
    print_table(filipino_data, title="Filipino Data After Tokenization")
    print("\nEnglish Data After Tokenization:")
    print_table(english_data, title="English Data After Tokenization")
# <-------------------------------------------------------------------------------------------------------------->
    # STOPWORDS REMOVAL
    remove_stopwords(filipino_data, dataset_name="Filipino Data", stopwords_list=filipino_stopwords)
    # Remove stopwords from the English dataset using NLTK's English stopwords list
    remove_stopwords(english_data, dataset_name="English Data", stopwords_list=english_stopwords)

    # Print both datasets after stopword removal
    print("Filipino Data After Stopwords Removal:")
    print_table(filipino_data, title="Filipino Data After Stopwords Removal")
    print("\nEnglish Data After Stopwords Removal:")
    print_table(english_data, title="English Data After Stopwords Removal")
# <-------------------------------------------------------------------------------------------------------------->
    # LEMMATIZATION
    lemmatize_filo(filipino_data, dataset_name="Filipino Data")
    lemmatize_eng(english_data, dataset_name="English Data")

    # Print both datasets after lemmatization
    print("Filipino Data After Lemmatization:")
    print_table(filipino_data, title="Filipino Data After Lemmatization")
    print("\nEnglish Data After Lemmatization:")
    print_table(english_data, title="English Data After Lemmatization")
# <-------------------------------------------------------------------------------------------------------------->
    # JOIN TOKENS
    join_tokens(filipino_data, dataset_name="Filipino Data")
    join_tokens(english_data, dataset_name="English Data")

    # Print both datasets after joining back the tokens
    print("Filipino Data After Joining Back the Tokens:")
    print_table(filipino_data, title="Filipino Data After Joining Back the Tokens")
    print("\nEnglish Data After Joining Back the Tokens:")
    print_table(english_data, title="English Data After Joining Back the Tokens")
# <-------------------------------------------------------------------------------------------------------------->
    #TF-IDF VECTORIZER

    # Vectorize the Filipino dataset
    filipino_tfidf_matrix, filipino_vectorizer = vectorize_with_tfidf(filipino_data)
    filipino_feature_names = filipino_vectorizer.get_feature_names_out()
    filipino_tfidf_df = pd.DataFrame(filipino_tfidf_matrix.toarray(), columns=filipino_feature_names)
    filipino_tfidf_df['emotion'] = filipino_data['emotion'].values

    # Save the Filipino TF-IDF DataFrame to a CSV file
    # filipino_tfidf_df.to_csv('./backend/data/feature vectors/filipino_tfidf_vectorized_data.csv', index=False)
    print(filipino_tfidf_df.head())

    # Save the Filipino vectorizer model
    save_model_with_name(filipino_vectorizer, "filipino_tfidf_vectorizer_model.pkl")

    # Vectorize the English dataset
    english_tfidf_matrix, english_vectorizer = vectorize_with_tfidf(english_data)
    english_feature_names = english_vectorizer.get_feature_names_out()
    english_tfidf_df = pd.DataFrame(english_tfidf_matrix.toarray(), columns=english_feature_names)
    english_tfidf_df['emotion'] = english_data['emotion'].values

    # # Save the English TF-IDF DataFrame to a CSV file
    # english_tfidf_df.to_csv('./backend/data/feature vectors/english_tfidf_vectorized_data.csv', index=False)
    print(english_tfidf_df.head())

    # Save the English vectorizer model
    save_model_with_name(english_vectorizer, "english_tfidf_vectorizer_model.pkl")
# <-------------------------------------------------------------------------------------------------------------->
    # Define file path for Filipino dataset
    filipino_narrative_file_path = './backend/data/training_data_sample_fil.csv'

    # Load the Filipino dataset again
    filipino_data_narrative = load_dataset(filipino_narrative_file_path)

    # Convert the 'sentence' column from Filipino dataset to a list of sentences
    filipino_sentences = filipino_data_narrative['sentence'].tolist()

    # GENERATE FEATURE VECTORS FROM FILIPINO NARRATIVE FEATURES
    fil_features_df = extract_features_from_dataframe(filipino_data_narrative)

    # Print the features DataFrame
    print(fil_features_df)

    # Save the features DataFrame to CSV
    # fil_features_df.to_csv('./backend/data/feature vectors/filipino_narratives_vectorized_data.csv', index=False)
# <-------------------------------------------------------------------------------------------------------------->
    # Define file path for English dataset
    english_narrative_file_path = './backend/data/training_data_sample_eng.csv'

    # Load the English dataset again
    english_data_narrative = load_dataset(english_narrative_file_path)

    # Convert the 'sentence' column from English dataset to a list of sentences
    english_sentences = english_data_narrative['sentence'].tolist()

    # GENERATE FEATURE VECTORS FROM ENGLISH NARRATIVE FEATURES
    eng_features_df = extract_features_from_dataframe(english_data_narrative)

    # Print the features DataFrame
    print(eng_features_df)

    # Save the features DataFrame to CSV
    # eng_features_df.to_csv('./backend/data/feature vectors/english_narratives_vectorized_data.csv', index=False)
# <-------------------------------------------------------------------------------------------------------------->
    # Merge Filipino and English TF-IDF tokens alphabetically
    all_token_columns = sorted(set(filipino_tfidf_df.columns) | set(english_tfidf_df.columns))

    # Reindex Filipino TF-IDF to have all tokens and fill missing ones with 0
    filipino_tfidf_df = filipino_tfidf_df.reindex(columns=all_token_columns, fill_value=0.0)

    # Reindex English TF-IDF to have all tokens and fill missing ones with 0
    english_tfidf_df = english_tfidf_df.reindex(columns=all_token_columns, fill_value=0.0)

    # Align Narrative Features with the merged tokens
    narrative_feature_columns = fil_features_df.columns  # Narrative feature columns (same for both)

    # Reindex the Filipino and English narrative features to match TF-IDF token columns (adding narrative features after tokens)
    fil_combined_df = pd.concat([filipino_tfidf_df, fil_features_df], axis=1)
    eng_combined_df = pd.concat([english_tfidf_df, eng_features_df], axis=1)

    # Combine both Filipino and English datasets row-wise
    combined_df = pd.concat([fil_combined_df, eng_combined_df], ignore_index=True)

    # Ensure the 'emotion' column is at the end of the DataFrame
    emotion_column = combined_df.pop('emotion')
    combined_df['emotion'] = emotion_column

    # Save the final combined DataFrame to CSV
    combined_df.to_csv('./backend/data/combined_final_feature_vectors.csv', index=False)

    # Preview the first few rows of the final combined DataFrame
    print(combined_df.head())