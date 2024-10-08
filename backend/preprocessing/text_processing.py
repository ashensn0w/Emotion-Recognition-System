import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy
import json
import stopwordsiso as filipino_stopwords
from nltk.data import find

# List of resources to check
resources = ['tokenizers/punkt', 'corpora/stopwords']

# Check if the resources are already downloaded
for resource in resources:
    try:
        find(resource)
        print(f"{resource} is already downloaded.")
    except LookupError:
        print(f"{resource} not found. Downloading...")
        nltk.download(resource)

nlp = spacy.load("en_core_web_md")

filipino_stopwords = filipino_stopwords.stopwords('tl')
english_stopwords = set(stopwords.words('english'))

# Function to convert the 'sentence' column to lowercase
def convert_to_lowercase(data):
    if 'sentence' in data.columns:
        data['sentence'] = data['sentence'].str.lower()
        print("Sentence has been converted to lowercase.")
    else:
        print("Column 'sentence' not found in the DataFrame.")

# Function to remove punctuation from the 'sentence' column
def remove_punctuation(data):
    if 'sentence' in data.columns:
        data['sentence'] = data['sentence'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
        print("Punctuation has been removed.")
    else:
        print("Column 'sentence' not found in the DataFrame.")

# Function to remove numbers from the 'sentence' column
def remove_numbers(data):
    if 'sentence' in data.columns:
        data['sentence'] = data['sentence'].str.replace(r'\d+', '', regex=True)
        print("Numbers have been removed.")
    else:
        print("Column 'sentence' not found in the DataFrame.")

# Function to tokenize the 'sentence' column
def tokenize_sentences(data):
    if 'sentence' in data.columns:
        data['sentence'] = data['sentence'].apply(lambda x: word_tokenize(x))
        print("Sentences have been tokenized.")
    else:
        print("Column 'sentence' not found in the DataFrame.")

# Function to remove stopwords from the 'sentence' column
def remove_stopwords(data):
    if 'sentence' in data.columns:
        all_stopwords = english_stopwords.union(set(filipino_stopwords))
        
        data['sentence'] = data['sentence'].apply(lambda tokens: [word for word in tokens if word.lower() not in all_stopwords])
        print("Stopwords have been removed.")
    else:
        print("Column 'sentence' not found in the DataFrame.")

# Function to lemmatize Filipino sentences
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

# Function to lemmatize English sentences
def lemmatize_eng(data):
    if 'sentence' in data.columns:
        data['sentence'] = data['sentence'].apply(lambda tokens: [nlp(token)[0].lemma_ for token in tokens])
        print("Tokens have been lemmatized.")
    else:
        print("Column 'sentence' not found in the DataFrame.")

# Function to join back the tokens for a given dataset
def join_tokens(data):
    if 'sentence' in data.columns:
        data['sentence'] = data['sentence'].apply(lambda tokens: ' '.join(tokens))
        data['sentence'] = data['sentence'].str.lower()
        print("Tokens have been joined back into sentences.")
    else:
        print("Column 'sentence' not found in the DataFrame.")