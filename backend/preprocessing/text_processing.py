import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy
import json
import stopwordsiso as filipino_stopwords

nltk.download('punkt')
nltk.download('stopwords')

nlp = spacy.load("en_core_web_md")

filipino_stopwords = filipino_stopwords.stopwords('tl')
english_stopwords = set(stopwords.words('english'))

# Function to convert the 'sentence' column to lowercase for a given dataset
def convert_to_lowercase(data, dataset_name="dataset"):
    if 'sentence' in data.columns:
        data['sentence'] = data['sentence'].str.lower()
        print(f"Sentence column in {dataset_name} has been converted to lowercase.\n")
    else:
        print(f"Column 'sentence' not found in the {dataset_name} DataFrame.")

# Function to remove punctuation from the 'sentence' column for a given dataset
def remove_punctuation(data, dataset_name="dataset"):
    if 'sentence' in data.columns:
        data['sentence'] = data['sentence'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
        print(f"Punctuation has been removed from {dataset_name}.\n")
    else:
        print(f"Column 'sentence' not found in the {dataset_name} DataFrame.")

# Function to remove numbers from the 'sentence' column for a given dataset
def remove_numbers(data, dataset_name="dataset"):
    if 'sentence' in data.columns:
        data['sentence'] = data['sentence'].str.replace(r'\d+', '', regex=True)
        print(f"Numbers have been removed from {dataset_name}.\n")
    else:
        print(f"Column 'sentence' not found in the {dataset_name} DataFrame.")

# Function to tokenize the 'sentence' column for a given dataset
def tokenize_sentences(data, dataset_name="dataset"):
    if 'sentence' in data.columns:
        data['sentence'] = data['sentence'].apply(lambda x: word_tokenize(x))
        print(f"Sentences in {dataset_name} have been tokenized.\n")
    else:
        print(f"Column 'sentence' not found in the {dataset_name} DataFrame.")

# Function to remove stopwords from the 'sentence' column for a given dataset
def remove_stopwords(data, dataset_name="dataset", stopwords_list=None):
    if 'sentence' in data.columns:
        if stopwords_list is None:
            print(f"Stopwords list is not provided for {dataset_name}.")
            return
        data['sentence'] = data['sentence'].apply(lambda tokens: [word for word in tokens if word.lower() not in stopwords_list])
        print(f"Stopwords have been removed from {dataset_name}.\n")
    else:
        print(f"Column 'sentence' not found in the {dataset_name} DataFrame.")

# Function to lemmatize Filipino sentences
def lemmatize_filo(data, dataset_name="dataset"):
    with open('./backend/data/filipino_lemmatizer.json', 'r', encoding='utf-8') as json_file:
        lemma_dict = json.load(json_file)

    token_to_lemma = {}
    for lemma, tokenval in lemma_dict['lemma_dict'].items():
        for token in tokenval:
            token_to_lemma[token] = lemma

    for index, row in data.iterrows():
        tokens = row['sentence']
        
        # Handle cases where tokens may be a single character
        if all(len(token) == 1 for token in tokens):
            tokens = ''.join(tokens).split()
        
        updated_tokens = [token_to_lemma.get(token, token) for token in tokens]
        data.at[index, 'sentence'] = updated_tokens

    print(f"Sentences in {dataset_name} have been lemmatized.\n")

# Function to lemmatize English sentences
def lemmatize_eng(data, dataset_name="dataset"):
    if 'sentence' in data.columns:
        data['sentence'] = data['sentence'].apply(lambda tokens: [nlp(token)[0].lemma_ for token in tokens])
        print(f"Sentences in {dataset_name} have been lemmatized.\n")
    else:
        print("Column 'sentence' not found in the DataFrame.")

# Function to join back the tokens for a given dataset
def join_tokens(data, dataset_name="dataset"):
    if 'sentence' in data.columns:
        data['sentence'] = data['sentence'].apply(lambda tokens: ' '.join(tokens))
        print(f"Tokens in {dataset_name} have been joined back into sentences.\n")
    else:
        print(f"Column 'sentence' not found in the {dataset_name} DataFrame.")