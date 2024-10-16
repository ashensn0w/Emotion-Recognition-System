from nltk.corpus import stopwords
from nltk.data import find
from nltk.tokenize import word_tokenize
from preprocessing.narrative_features_eng import *
from preprocessing.narrative_features_fil import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QFrame, QSpacerItem, QSizePolicy, QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QScrollArea
from rich.console import Console
from rich.table import Table
from utils.save_load import *
import json
import nltk
import numpy as np
import os
import pandas as pd
import shutil
import spacy
import stopwordsiso as stopwords
import string
import sys

# List of resources to check
resources = ['tokenizers/punkt', 'corpora/stopwords']

# Check if the resources are already downloaded
for resource in resources:
    try:
        find(resource)
        # print(f"{resource} is already downloaded.")
    except LookupError:
        # print(f"{resource} not found. Downloading...")
        nltk.download(resource)

nlp = spacy.load("en_core_web_md")

filipino_stopwords = stopwords.stopwords('tl')
english_stopwords = set(stopwords.stopwords('english'))

# First Window (Uploading Novella)
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Emotion Recognition in Filipino Novellas")
        self.setFixedSize(1500, 800)

        # Initialize file_path variable
        self.file_path = ""

        # Main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # Layouts
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        # Left side (Red background with text)
        left_widget = QWidget()
        left_widget.setStyleSheet("background-color: #8B1E3F;")
        left_layout.addStretch(1)
        title_label = QLabel("EMOTION\nRECOGNITION\nIN FILIPINO\nNOVELLAS")
        title_label.setAlignment(Qt.AlignLeft)
        title_label.setFont(QFont("Arial", 28, QFont.Bold))
        title_label.setStyleSheet("color: white; margin-left: 4px;")
        left_layout.addWidget(title_label)
        left_layout.addStretch(1)
        left_widget.setLayout(left_layout)
        left_widget.setFixedWidth(400)

        # Right side (Upload section)
        right_layout.addStretch(1)
        upload_label = QLabel("UPLOAD YOUR NOVELLA HERE")
        upload_label.setAlignment(Qt.AlignCenter)
        upload_label.setFont(QFont("Arial", 28, QFont.Bold))
        right_layout.addWidget(upload_label)

        # File upload button
        upload_button = QPushButton()
        upload_button.setFixedSize(500, 300)
        upload_button.setStyleSheet("""
            QPushButton {
                border: 2px dashed black;
                background-color: #f9f9f9;
                margin: 50px;
                border-radius: 30%;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """)
        upload_button.clicked.connect(self.open_file_dialog)
        upload_text = QLabel("Only .TXT Format is Supported\n\nCLICK TO BROWSE")
        upload_text.setFont(QFont("Arial", 13))
        upload_text.setAlignment(Qt.AlignCenter)
        upload_button_layout = QVBoxLayout(upload_button)
        upload_button_layout.addWidget(upload_text)
        upload_button_layout.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(upload_button, alignment=Qt.AlignCenter)

        # Label to display selected file name
        self.file_name_label = QLabel("")
        self.file_name_label.setAlignment(Qt.AlignCenter)
        self.file_name_label.setFont(QFont("Arial", 12))
        right_layout.addWidget(self.file_name_label, alignment=Qt.AlignCenter)

        # Preview Novella Button
        self.preview_button = QPushButton("Preview Novella")
        self.preview_button.setFixedSize(250, 50)
        self.preview_button.setFont(QFont("Arial", 14, QFont.Bold))
        self.preview_button.setStyleSheet("""
            background-color: #1338BE; 
            color: white;
            border-radius: 10px;
        """)
        self.preview_button.clicked.connect(self.show_preview_page)
        right_layout.addWidget(self.preview_button, alignment=Qt.AlignCenter)
        right_layout.addStretch(1)

        # Add widgets to main layout
        main_layout.addWidget(left_widget)
        main_layout.addLayout(right_layout)
        main_widget.setLayout(main_layout)

    def open_file_dialog(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Novella", "", "Text Files (*.txt);;All Files (*)", options=options)
        
        if file_name:
            # Extract the file name from the full path
            base_name = os.path.basename(file_name)
            
            # Define the destination directory
            dest_dir = "./backend/data/novellas"
            
            # Create the directory if it doesn't exist
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            
            # Set the destination path
            dest_file = os.path.join(dest_dir, base_name)
            
            # Copy the file to the destination directory
            shutil.copy(file_name, dest_file)
            
            # Store the file path for later use
            self.file_path = dest_file
            
            # Update the label to show the file was selected and saved
            self.file_name_label.setText(f"Selected file: {base_name}")
    
    def show_preview_page(self):
        if self.file_path:
            self.preview_window = PreviewWindow(self.file_path, self)  # Pass self as parent
            self.preview_window.show()
            self.hide()  # Hide main window
        else:
            print("Please upload a novella first.")

# Second Window (Displaying Novella)
class PreviewWindow(QMainWindow):
    def __init__(self, file_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preview Window")
        self.setFixedSize(1500, 800)

        # Store the parent reference
        self.parent = parent
        self.file_path = file_path 

        # Main layout with two columns
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()

        # Left side (Red background with text)
        left_widget = QWidget()
        left_widget.setStyleSheet("background-color: #8B1E3F;")
        left_layout.addStretch(1)
        title_label = QLabel("EMOTION\nRECOGNITION\nIN FILIPINO\nNOVELLAS")
        title_label.setAlignment(Qt.AlignLeft)
        title_label.setFont(QFont("Arial", 28, QFont.Bold))
        title_label.setStyleSheet("color: white; margin-left: 4px;")
        left_layout.addWidget(title_label)
        left_layout.addStretch(1)
        left_widget.setLayout(left_layout)
        left_widget.setFixedWidth(400)

        # Right column layout (for the preview content)
        right_layout = QVBoxLayout()

        # Create a scroll area for the content
        scroll_area = QScrollArea()
        scroll_area.setFixedHeight(630)
        scroll_area.setStyleSheet("margin-bottom: 20px;")
        scroll_area.setWidgetResizable(True)

        # Create a frame to hold the content in the scroll area
        content_frame = QFrame()
        content_layout = QVBoxLayout(content_frame)

        # Read the text file
        self.title = ""
        self.content = ""
        self.paragraph_count = 0  # Initialize paragraph count
        self.word_count = 0  # Initialize word count
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                if lines:
                    self.title = lines[0].strip()  # First line as title
                    self.content = "".join(lines[1:]).strip()  # Remaining lines as content
                    
                    # Count the number of paragraphs
                    self.paragraph_count = self.content.count('\n\n') + 1  # Add 1 to account for the last paragraph
                    # Count the number of words
                    self.word_count = len(self.content.split())

        except Exception as e:
            print(f"Error reading file: {e}")

         # Create a horizontal layout for the paragraph and word count labels
        count_layout = QHBoxLayout()
        
         # Paragraph count label (lower left)
        self.paragraph_count_label = QLabel(f"Paragraphs: <span style='color: #8B1E3F;'>{self.paragraph_count}</span>")
        self.paragraph_count_label.setFont(QFont("Arial", 12))
        count_layout.addWidget(self.paragraph_count_label, alignment=Qt.AlignLeft)
        
        # Word count label (lower right)
        self.word_count_label = QLabel(f"Words: <span style='color: #8B1E3F;'>{self.word_count}</span>")
        self.word_count_label.setFont(QFont("Arial", 12))
        count_layout.addWidget(self.word_count_label, alignment=Qt.AlignRight)

        # Add the count layout to the right layout
        right_layout.addLayout(count_layout)

        # Display title and content
        self.title_label = QLabel(self.title)
        self.title_label.setFont(QFont("Arial", 24))
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("margin-top: 20px; margin-bottom: 10px;")
        content_layout.addWidget(self.title_label)

        self.content_label = QLabel(self.content)
        self.content_label.setFont(QFont("Arial", 12))
        self.content_label.setWordWrap(True)
        content_layout.addWidget(self.content_label)

        # Set the content frame to the scroll area
        scroll_area.setWidget(content_frame)

        # Add the scroll area to the right layout
        right_layout.addWidget(scroll_area)

        # Back button to return to the main window
        self.back_button = QPushButton("Back")
        self.back_button.setFixedSize(250, 40)
        self.back_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.back_button.setStyleSheet(""" 
            background-color: #1338BE; 
            color: white; 
            border-radius: 10px; 
        """)
        self.back_button.clicked.connect(self.back_to_main)
        right_layout.addWidget(self.back_button, alignment=Qt.AlignCenter)

        # Process Novella button
        self.process_button = QPushButton("Process Novella")
        self.process_button.setFixedSize(250, 40)
        self.process_button.setFont(QFont("Arial", 12, QFont.Bold))
        self.process_button.setStyleSheet(""" 
            background-color: #1338BE; 
            color: white; 
            border-radius: 10px; 
        """)
        self.process_button.clicked.connect(self.process_novella)
        right_layout.addWidget(self.process_button, alignment=Qt.AlignCenter)

        # Add stretch at the bottom for vertical centering
        right_layout.addStretch(1)

        # Add widgets to the main layout
        main_layout.addWidget(left_widget)
        main_layout.addLayout(right_layout)

        # Set the central widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def back_to_main(self):
        self.close()  # Close the preview window
        if self.parent:  # Check if parent is set
            self.parent.show()  # Show the main window again

    def process_novella(self):
        print("Processing novella...")
        # Use self.file_path instead of self.file_path
        with open(self.file_path, 'r', encoding='utf-8') as file:
            text_content = file.read()

        # Split the text into lines based on double newlines and store in sentences
        sentences = [line.strip() for line in text_content.strip().split('\n\n') if line.strip()]

        remaining_sentences = sentences[1:] if len(sentences) > 1 else []  # Get the rest of the sentences for analysis
        # print(remaining_sentences)

        # Print the sentences variable to see the output
        # print(sentences)

        # Create a DataFrame from the list of sentences
        data = pd.DataFrame(remaining_sentences, columns=['sentence'])

        def format_list_as_string(token_list):
            return str(token_list).replace("'", '"')

        # Check if the dataset is loaded successfully
        if data is not None:
            # <-------------------------------------------------------------------------------------------------------------->
            def convert_to_lowercase(data):
                if 'sentence' in data.columns:
                    data['sentence'] = data['sentence'].str.lower()
                    # print("Sentence has been converted to lowercase.")
                else:
                    print("Column 'sentence' not found in the DataFrame.")

            convert_to_lowercase(data)
            # <-------------------------------------------------------------------------------------------------------------->
            def remove_punctuation(data):
                if 'sentence' in data.columns:
                    data['sentence'] = data['sentence'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
                    # print("Punctuation has been removed.")
                else:
                    print("Column 'sentence' not found in the DataFrame.")

            remove_punctuation(data)
            # <-------------------------------------------------------------------------------------------------------------->
            def remove_numbers(data):
                if 'sentence' in data.columns:
                    data['sentence'] = data['sentence'].str.replace(r'\d+', '', regex=True)
                    # print("Numbers have been removed.")
                else:
                    print("Column 'sentence' not found in the DataFrame.")

            remove_numbers(data)
            # <-------------------------------------------------------------------------------------------------------------->
            def tokenize_sentences(data):
                if 'sentence' in data.columns:
                    data['sentence'] = data['sentence'].apply(lambda x: word_tokenize(x))
                    # print("Sentences have been tokenized.")
                else:
                    print("Column 'sentence' not found in the DataFrame.")

            tokenize_sentences(data)
            # <-------------------------------------------------------------------------------------------------------------->
            def remove_stopwords(data):
                if 'sentence' in data.columns:
                    all_stopwords = english_stopwords.union(set(filipino_stopwords))
                    
                    data['sentence'] = data['sentence'].apply(lambda tokens: [word for word in tokens if word.lower() not in all_stopwords])
                    # print("Stopwords have been removed.")
                else:
                    print("Column 'sentence' not found in the DataFrame.")

            remove_stopwords(data)
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
            # <-------------------------------------------------------------------------------------------------------------->
            def lemmatize_eng(data):
                if 'sentence' in data.columns:
                    data['sentence'] = data['sentence'].apply(lambda tokens: [nlp(token)[0].lemma_ for token in tokens])
                    # print("Tokens have been lemmatized.")
                else:
                    print("Column 'sentence' not found in the DataFrame.")

            lemmatize_eng(data)
            # <-------------------------------------------------------------------------------------------------------------->
            def join_tokens(data):
                if 'sentence' in data.columns:
                    data['sentence'] = data['sentence'].apply(lambda tokens: ' '.join(tokens))
                    # data['sentence'] = data['sentence'].str.lower()
                    # print("Tokens have been joined back into sentences.")
                else:
                    print("Column 'sentence' not found in the DataFrame.")

            join_tokens(data)
            # <-------------------------------------------------------------------------------------------------------------->
            # Load the saved TF-IDF vectorizer
            tfidf_vectorizer = load_model_with_name('tfidf_vectorizer_model.pkl')

            # Check if the TF-IDF vectorizer was loaded successfully
            if tfidf_vectorizer:
                # Transform the sentences using the loaded TF-IDF vectorizer
                tfidf_matrix = tfidf_vectorizer.transform(data['sentence'])

                # Convert the TF-IDF matrix to a DataFrame for better visualization (optional)
                tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

                # print("TF-IDF transformation successful!")
            else:
                print("TF-IDF vectorizer could not be loaded.")
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
            
            with open(self.file_path, 'r', encoding='utf-8') as file:
                text_content = file.read()

            # # Split the text into lines based on double newlines and store in sentences
            sentences = [line.strip() for line in text_content.strip().split('\n\n') if line.strip()]

            remaining_sentences = sentences[1:] if len(sentences) > 1 else []  # Get the rest of the sentences for analysis

            # Print the sentences variable to see the output
            # print(sentences)

            # Create a DataFrame from the list of sentences
            data = pd.DataFrame(remaining_sentences, columns=['sentence'])
    
            # Apply the feature extraction and combination process
            combined_features_df = process_data(data)

            # Concatenate TF-IDF features and combined narrative features
            final_combined_df = pd.concat([tfidf_df, combined_features_df], axis=1)

            # Print the final combined DataFrame
            # print("Final combined DataFrame:")
            # print(final_combined_df.head())

            # Remove any columns that were not part of the model's training
            if 'emotion' in final_combined_df.columns:
                final_combined_df.drop(columns=['emotion'], inplace=True)
                
            # Load the saved emotion recognition model
            emo_recog_model = load_model_with_name('best_emotion_recognition_glm_model.pkl')

            # Check if the model was loaded successfully
            if emo_recog_model is not None:
                # Ensure that the final_combined_df matches the expected feature set
                predictions = emo_recog_model.predict(final_combined_df)
                print("Predictions made successfully.")
                data['predicted_emotion'] = predictions  # Add predictions to the DataFrame
            else:
                print("Model not loaded. Unable to make predictions.")
                return

            # Prepare the output DataFrame
            output_df = data[['sentence', 'predicted_emotion']]

            # Save the output to a new CSV file
            output_df.to_csv('./backend/data/feature vectors/new_input_predictions.csv', index=False)

        # Simulating processing and then opening the results window
        self.results_window = ResultsWindow(self.parent, self.parent.file_path)
        self.results_window.show()
        self.close()  # Close the preview window after opening results


# Third Window (Results Page)
class ResultsWindow(QWidget):
    def __init__(self, parent, file_path):  # Add file_path parameter
        super().__init__()
        self.parent = parent
        self.file_path = file_path  # Store the file_path if needed
        self.setWindowTitle("Results")
        self.setFixedSize(1500, 800)

        # Main layout with two columns
        main_layout = QHBoxLayout()

        # Left column layout (for the title)
        left_widget = QWidget()
        left_widget.setStyleSheet("background-color: #8B1E3F;")

        left_layout = QVBoxLayout()

        title_label = QLabel("EMOTION\nRECOGNITION\nIN FILIPINO\nNOVELLAS")
        title_label.setAlignment(Qt.AlignLeft)
        title_label.setFont(QFont("Arial", 28, QFont.Bold))
        title_label.setStyleSheet("color: white; margin-left: 4px;")
        left_layout.addStretch(1)
        left_layout.addWidget(title_label)
        left_layout.addStretch(1)

        left_widget.setLayout(left_layout)
        left_widget.setFixedWidth(400)

        # Right column layout (for the results content)
        right_layout = QVBoxLayout()

        # Add stretch at the top for vertical centering
        right_layout.addStretch(1)

        # Load the novella title from the first line of the file
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                self.novella_title = file.readline().strip()  # Read the first line as the title
        except Exception as e:
            self.novella_title = "Unknown Title"  # Fallback if there is an error reading the file

        # Results title
        results_title = QLabel("RESULTS")
        results_title.setFont(QFont("Verdana", 26, QFont.Bold))
        results_title.setAlignment(Qt.AlignCenter)
        results_title.setStyleSheet("margin-bottom: 20px;")

        # Add the results title to the right layout
        right_layout.addWidget(results_title)

        # Novella title label (display the title at the top)
        novella_title_label = QLabel(f"\"{self.novella_title}\"")
        novella_title_label.setFont(QFont("Verdana", 22))
        novella_title_label.setAlignment(Qt.AlignCenter)
        novella_title_label.setStyleSheet("color: #050088; margin-bottom: 30px;")  # Add some space below the title

        # Add the novella title label to the right layout
        right_layout.addWidget(novella_title_label)

        def display_emotion_counts_and_sentences(csv_file):
            # Read the CSV file
            df = pd.read_csv(csv_file)

            # Group sentences by predicted emotion
            emotion_sentences = {}
            for _, row in df.iterrows():
                emotion = row['predicted_emotion']
                sentence = row['sentence']  # Adjust this if your column name is different

                if emotion not in emotion_sentences:
                    emotion_sentences[emotion] = []
                emotion_sentences[emotion].append(sentence)

            # Create a dictionary to hold the counts
            emotion_counts = {emotion: len(sentences) for emotion, sentences in emotion_sentences.items()}

            return emotion_counts, emotion_sentences  # Return both counts and sentences
    
        # Load emotion counts and sentences
        emotion_counts, emotion_sentences = display_emotion_counts_and_sentences('./backend/data/feature vectors/new_input_predictions.csv')

        # Find the dominant emotion
        dominant_emotion = max(emotion_counts, key=emotion_counts.get)  # Get the emotion with the highest count

        # Create a label for the dominant emotion with HTML formatting
        dominant_emotion_label = QLabel(
            f"<span style='color: black;'>Dominant Emotion:</span> "
            f"<span style='color: #8B1E3F; font-weight: bold'>{dominant_emotion.capitalize()}</span>"
        )
        dominant_emotion_label.setFont(QFont("Verdana", 22))
        dominant_emotion_label.setAlignment(Qt.AlignCenter)
        dominant_emotion_label.setStyleSheet("margin-top: 20px; margin-bottom: 20px;")

        # Add the dominant emotion label to the right layout
        right_layout.addWidget(dominant_emotion_label)

        # Add stretch at the bottom for vertical centering
        right_layout.addStretch(1)

        # List of all possible emotions
        all_emotions = ['fear', 'sadness', 'anger', 'joy']

        # Create UI elements for each emotion dynamically
        for emotion in all_emotions:
            count = emotion_counts.get(emotion, 0)

            hbox = QHBoxLayout()

            emotion_label = QLabel(f"{emotion.capitalize()} : ")
            emotion_label.setFont(QFont("Verdana", 18))

            count_label = QLabel(f"{count}    ")  # Use the count from the emotion_counts dictionary
            count_label.setFont(QFont("Arial", 18))

            # Button to view sentences
            view_button = QPushButton("View Sentences")
            view_button.setFixedSize(200, 40)
            view_button.setFont(QFont("Arial", 10))
            view_button.setStyleSheet(
            """
            background-color: #1338BE; 
            color: white;
            border-radius: 10px;
            """)

            # Connect the button to the appropriate function, passing the sentences
            view_button.clicked.connect(lambda _, e=emotion, s=emotion_sentences.get(emotion, []): self.go_to_sentences_page(e, s))

            # Add widgets to the horizontal layout
            hbox.addWidget(emotion_label)
            hbox.addWidget(count_label)
            hbox.addWidget(view_button)

            # Center the emotion row horizontally
            hbox.setAlignment(Qt.AlignCenter)
            
            # Add the emotion row to the right layout
            right_layout.addLayout(hbox)

            # Add a spacer between emotion rows
            spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
            right_layout.addItem(spacer)

        # Add stretch at the bottom for vertical centering
        right_layout.addStretch(1)

        # Back button to upload a new file
        back_button = QPushButton("Back")
        back_button.setFixedSize(150, 40)
        back_button.setFont(QFont("Arial", 12, QFont.Bold))
        back_button.setStyleSheet(""" 
            background-color: #1338BE;
            color: white;
            border-radius: 10px;
            """)
        back_button.clicked.connect(self.go_back)

        # Add the back button to the right layout
        right_layout.addWidget(back_button, alignment=Qt.AlignCenter)

        # Add a spacer above the back button to move it higher
        spacer_above_back_button = QSpacerItem(20, 60, QSizePolicy.Minimum, QSizePolicy.Expanding)
        right_layout.addItem(spacer_above_back_button)

        # Add widgets to the main layout
        main_layout.addWidget(left_widget)
        main_layout.addLayout(right_layout)
        self.setLayout(main_layout)

    def go_to_sentences_page(self, emotion, sentences):
        self.sentences_window = SentencesWindow(emotion, sentences, self)  # Pass the sentences
        self.sentences_window.show()
        self.close()
        
    def go_back(self):
        self.parent.show()
        self.close()

# Fourth Window (Sentences Page)
class SentencesWindow(QWidget):
    def __init__(self, emotion, sentences, parent):
        super().__init__()
        self.parent = parent
        self.setWindowTitle(f"{emotion} Sentences")
        self.setFixedSize(1500, 800)

        # Main layout with two columns
        main_layout = QHBoxLayout()

        # Left column layout (for the title)
        left_widget = QWidget()
        left_widget.setStyleSheet("background-color: #8B1E3F;")

        left_layout = QVBoxLayout()

        title_label = QLabel("EMOTION\nRECOGNITION\nIN FILIPINO\nNOVELLAS")
        title_label.setAlignment(Qt.AlignLeft)
        title_label.setFont(QFont("Arial", 28, QFont.Bold))
        title_label.setStyleSheet("color: white; margin-left: 4px;")
        left_layout.addStretch(1)
        left_layout.addWidget(title_label)
        left_layout.addStretch(1)

        left_widget.setLayout(left_layout)
        left_widget.setFixedWidth(400)

        # Right column layout (for sentences content)
        right_layout = QVBoxLayout()

        # Add stretch at the top for vertical centering
        right_layout.addStretch(1)

        # Display the selected emotion
        emotion_title = QLabel(f"{emotion.capitalize()} Sentences")
        emotion_title.setFont(QFont("Verdana", 26, QFont.Bold))
        emotion_title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(emotion_title)

        # Scroll area to contain sentences
        sentence_scroll_area = QScrollArea()
        sentence_scroll_area.setFixedHeight(580)
        sentence_scroll_area.setWidgetResizable(True)

        # Content inside the scroll area
        sentence_widget = QWidget()
        sentence_layout = QVBoxLayout(sentence_widget)

        # Add the sentences passed from ResultsWindow
        for index, sentence in enumerate(sentences, start=1):
            sentence_label = QLabel(f"{index}. {sentence}")
            sentence_label.setFont(QFont("Verdana", 13))
            sentence_label.setWordWrap(True)
            sentence_label.setContentsMargins(0, 5, 0, 5) 
            sentence_layout.addWidget(sentence_label)

        sentence_layout.addStretch(1)

        sentence_scroll_area.setWidget(sentence_widget)
        right_layout.addWidget(sentence_scroll_area)

        # Add a "Back" button to return to the previous window
        back_button = QPushButton("Back")
        back_button.setFixedSize(150, 60)
        back_button.setFont(QFont("Arial", 12, QFont.Bold))
        back_button.setStyleSheet(""" 
            background-color: #1338BE;
            color: white;
            border-radius: 10px;
            margin-top: 20px;
            """)
        back_button.clicked.connect(self.go_back)

        right_layout.addWidget(back_button, alignment=Qt.AlignCenter)

        # Add stretch at the bottom for vertical centering
        right_layout.addStretch(1)

        # Add widgets to the main layout
        main_layout.addWidget(left_widget)
        main_layout.addLayout(right_layout)
        self.setLayout(main_layout)

    def go_back(self):
        self.parent.show()
        self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())