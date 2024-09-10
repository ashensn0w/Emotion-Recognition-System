from preprocessing.text_processing import load_dataset, convert_to_lowercase, remove_punctuation, remove_numbers, tokenize_sentences, remove_stopwords, lemmatize_tokens, print_table

file_path = './backend/data/sample_dataset.csv'
data = load_dataset(file_path)

if data is not None:
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

    # Tokenize sentences
    tokenize_sentences(data)
    print_table(data, title="Data After Tokenization")

    # Remove stopwords
    remove_stopwords(data)
    print_table(data, title="Data After Stopwords Removal")

    # Lemmatize tokens
    lemmatize_tokens(data)
    print_table(data, title="Data After Lemmatization")