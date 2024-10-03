import pandas as pd
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('stopwords')

nlp = spacy.load("en_core_web_trf")

filipino_stopwords = [
    "a", "ako", "ang", "ano", "at", "ay", "ibang", "ito", "iyon", "ka",
    "kami", "kanila", "kanya", "kayo", "laki", "mga", "na", "ng", "ni",
    "nito", "nang", "sa", "sila", "tayo", "walang", "yung", "si", "bawat",
    "kung", "hindi", "para", "dahil", "doon", "baka", "kapag", "saan",
    "sino", "siya", "tama", "yan", "o", "pala", "pero", "wala", "huwag",
    "muna", "na", "naman", "pag", "sana", "tulad", "upang", "bago", 
    "dati", "iba", "madami", "nakita", "pagkatapos", "pati", "sabi", "sana"
]

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

def format_list_as_string(token_list):
    return str(token_list).replace("'", '"')

def print_table(data, title="Table", num_samples=10):
    from rich.console import Console
    from rich.table import Table
    
    table = Table(title=title)
    
    for col in data.columns:
        table.add_column(col)

    for _, row in data.head(num_samples).iterrows():
        formatted_row = [format_list_as_string(row[col]) if isinstance(row[col], list) else row[col] for col in data.columns]
        table.add_row(*map(str, formatted_row))
    
    console = Console()
    console.print(table)

def convert_to_lowercase(data):
    if 'sentence' in data.columns:
        data['sentence'] = data['sentence'].str.lower()
        
        data.to_csv(index=False)
        
        print("Sentence has been converted to lowercase.")
    else:
        print("Column 'sentence' not found in the DataFrame.")

def remove_punctuation(data):
    if 'sentence' in data.columns:
        data['sentence'] = data['sentence'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
        print("Punctuation has been removed.")
    else:
        print("Column 'sentence' not found in the DataFrame.")

def remove_numbers(data):
    if 'sentence' in data.columns:
        data['sentence'] = data['sentence'].str.replace(r'\d+', '', regex=True)
        print('-' * 120)
        print("Numbers have been removed.")
    else:
        print("Column 'sentence' not found in the DataFrame.")

def tokenize_sentences(data):
    if 'sentence' in data.columns:
        data['sentence'] = data['sentence'].apply(lambda x: word_tokenize(x))
        print('-' * 120)
        print("Sentences have been tokenized.")
    else:
        print("Column 'sentence' not found in the DataFrame.")

def remove_stopwords(data):
    if 'sentence' in data.columns:
        english_stopwords = set(stopwords.words('english'))
        all_stopwords = english_stopwords.union(set(filipino_stopwords))
        
        data['sentence'] = data['sentence'].apply(lambda tokens: [word for word in tokens if word.lower() not in all_stopwords])
        print('-' * 120)
        print("Stopwords have been removed.")
    else:
        print("Column 'sentence' not found in the DataFrame.")

def lemmatize_tokens(data):
    if 'sentence' in data.columns:
        data['sentence'] = data['sentence'].apply(lambda tokens: [nlp(token)[0].lemma_ for token in tokens])
        print('-' * 120)
        print("Tokens have been lemmatized.")
    else:
        print("Column 'sentence' not found in the DataFrame.")

def join_tokens(data):
    if 'sentence' in data.columns:
        data['sentence'] = data['sentence'].apply(lambda tokens: ' '.join(tokens))
        print('-' * 120)
        print("Tokens have been joined back into sentences.\n")
    else:
        print("Column 'sentence' not found in the DataFrame.")

def vectorize_with_tfidf(data):
    if 'sentence' in data.columns:
        vectorizer = TfidfVectorizer()

        tfidf_matrix = vectorizer.fit_transform(data['sentence'])

        print("TF-IDF Vectorization complete.")
        return tfidf_matrix, vectorizer
    else:
        print("Column 'sentence' not found in the DataFrame.")