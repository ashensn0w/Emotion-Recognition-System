from preprocessing.text_processing import *
from preprocessing.narrative_features_fil import *
from preprocessing.narrative_features_eng import *
from utils.tfidf_vectorizer import *
from utils.save_load import *
from utils.resampling import *
from rich.console import Console
from rich.table import Table
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
import numpy as np

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

file_path = './backend/data/sample.csv'
data = load_dataset(file_path)

sentences = data['sentence'].tolist()

# Function to format a list as a string
def format_list_as_string(token_list):
    return str(token_list).replace("'", '"')

# Function to print a sample table from the dataset
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

# Check if the dataset is loaded successfully
if data is not None:
    print_table(data, title="Original Data")
# <-------------------------------------------------------------------------------------------------------------->
    # LOWERCASE CONVERSION
    convert_to_lowercase(data)
    print_table(data, title="Data After Lowercase Conversion")
# <-------------------------------------------------------------------------------------------------------------->
    # PUNCTUATION REMOVAL
    remove_punctuation(data)
    print_table(data, title="Data After Punctuation Removal")
# <-------------------------------------------------------------------------------------------------------------->
    # NUMBERS REMOVAL
    remove_numbers(data)
    print_table(data, title="Data After Numbers Removal")
# <-------------------------------------------------------------------------------------------------------------->
    # TOKENIZATION
    tokenize_sentences(data)
    print_table(data, title="Data After Tokenization")
# <-------------------------------------------------------------------------------------------------------------->
    # STOPWORDS REMOVAL
    remove_stopwords(data)
    print_table(data, title="Data After Stopwords Removal")
# <-------------------------------------------------------------------------------------------------------------->
    # LEMMATIZATION
    lemmatize_filo(data)
    print_table(data, title="Data After Lemmatization in Filipino")

    lemmatize_eng(data)
    print_table(data, title="Data After Lemmatization in English")
# <-------------------------------------------------------------------------------------------------------------->
    # JOIN TOKENS
    join_tokens(data)
    print_table(data, title="Data After Joining Tokens")
# <-------------------------------------------------------------------------------------------------------------->
    # TF-IDF VECTORIZER

    # Vectorize the dataset
    tfidf_matrix, vectorizer = vectorize_with_tfidf(data)
    feature_names = vectorizer.get_feature_names_out()
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
    tfidf_df['emotion'] = data['emotion'].values

    # Save the TF-IDF DataFrame to a CSV file
    tfidf_df.to_csv('./backend/data/feature vectors/tfidf_vectorized_data.csv', index=False)
    print(tfidf_df.head())

    # Save the vectorizer model
    save_model_with_name(vectorizer, "tfidf_vectorizer_model.pkl")
# <-------------------------------------------------------------------------------------------------------------->
    # Combine Filipino and English feature vectors using element-wise maximum
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

        return combined_features_df

    # Read narrative features data from CSV
    narrative_file_path = './backend/data/sample.csv'
    narrative_features_df = pd.read_csv(narrative_file_path)

    # Apply the feature extraction and combination process
    combined_features_df = process_data(narrative_features_df)

    # Concatenate TF-IDF features and combined narrative features
    final_combined_df = pd.concat([tfidf_df, combined_features_df], axis=1)

    # Print the final combined DataFrame
    print("Final combined DataFrame:")
    print(final_combined_df.head())

    # Save the final combined features DataFrame to CSV
    final_combined_df.to_csv('./backend/data/feature vectors/complete_vectorized_data.csv', index=False)
# <-------------------------------------------------------------------------------------------------------------->
    # RANDOM RESAMPLING

    # Separate features (X) and target (y)
    X = final_combined_df.drop(columns=['emotion'])
    y = final_combined_df['emotion']

    print("Class distribution before resampling:")
    print(y.value_counts())
    print("\nData before resampling:")
    print(final_combined_df.head())

    # Call the resampling function from resampling.py
    X_resampled, y_resampled = resample_data(X, y)

    # Create the resampled DataFrame
    resampled_df = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=['emotion'])], axis=1)

    # Save the resampled DataFrame to a CSV file
    resampled_df.to_csv('./backend/data/feature vectors/resampled_combined_df.csv', index=False)

    print("\nData after resampling:")
    print(resampled_df.head())
# # <-------------------------------------------------------------------------------------------------------------->
    #TRAINING THE MODEL WITH GLM (MULTINOMIAL LOGISTIC REGRESSION)

    # SPLIT DATA INTO FEATURES (X) AND TARGET (y)
    X = resampled_df.drop(columns=['emotion'])  # Features are all columns except 'emotion'
    y = resampled_df['emotion']  # Target variable is the 'emotion' column

    # SPLIT THE DATA INTO TRAINING (80%) AND TESTING (20%) SETS
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # PERFORM K-FOLD CROSS-VALIDATION ON THE TRAINING SET
    k = 5  # Number of folds for cross-validation
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)  # Stratified K-Fold to maintain class distribution

    # LOGISTIC REGRESSION WITH GRID SEARCH FOR HYPERPARAMETER TUNING
    param_grid = {
        'C': [0.01, 0.1, 1, 10],  # Regularization parameter for Logistic Regression
        'solver': ['liblinear', 'lbfgs']  # Solvers to be tested
    }

    # CREATE A LOGISTIC REGRESSION MODEL
    glm = LogisticRegression(max_iter=1000)  # Initialize the logistic regression model with a max iteration limit

    # GRID SEARCH WITH CROSS-VALIDATION
    grid_search = GridSearchCV(glm, param_grid, cv=kf, scoring='f1_macro', n_jobs=-1)  # Set up grid search with cross-validation
    grid_search.fit(X_train, y_train)  # Fit grid search to the training data

    # PRINT BEST PARAMETERS AFTER CROSS-VALIDATION
    best_params = grid_search.best_params_  # Retrieve the best hyperparameters found
    print(f"Best hyperparameters found through cross-validation: {best_params}")

    # OUTPUT K-FOLD CROSS-VALIDATION SCORES
    cv_results = grid_search.cv_results_  # Get the results of the cross-validation
    for mean_score, std_dev, params in zip(cv_results['mean_test_score'], cv_results['std_test_score'], cv_results['params']):
        print(f"Mean F1 Score: {mean_score:.4f} ± {std_dev:.4f} for params: {params}")  # Print mean F1 score and std deviation for each parameter combination

    # TRAIN FINAL MODEL USING THE BEST HYPERPARAMETERS ON THE ENTIRE TRAINING SET
    best_glm = grid_search.best_estimator_  # Retrieve the best estimator from grid search
    best_glm.fit(X_train, y_train)  # Train the model using the entire training set

    # PREDICT THE TEST SET
    y_pred = best_glm.predict(X_test)  # Make predictions on the test set

    # GENERATE CLASSIFICATION REPORT (PRECISION, RECALL, F1-SCORE)
    report = classification_report(y_test, y_pred)  # Create a classification report
    print("\nClassification Report on Test Set:")  # Print the header for the report
    print(report)  # Print the classification report

    save_model_with_name(best_glm, "best_emotion_recognition_glm_model.pkl")