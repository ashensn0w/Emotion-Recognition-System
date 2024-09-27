import sys
from PyQt5.QtWidgets import QSpacerItem, QSizePolicy, QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QScrollArea
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont

# Main Window for the application
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Emotion Recognition in Filipino Novellas")
        self.setGeometry(250, 150, 1500, 800)

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

        # Submit button
        submit_button = QPushButton("Submit Novella")
        submit_button.setFixedSize(250, 50)
        submit_button.setFont(QFont("Arial", 14, QFont.Bold))
        submit_button.setStyleSheet("""
            background-color: #1338BE; 
            color: white;
            border-radius: 10px;
        """)
        submit_button.clicked.connect(self.go_to_results_page)  # Connect to function that opens results page
        right_layout.addWidget(submit_button, alignment=Qt.AlignCenter)
        right_layout.addStretch(1)

        # Add widgets to main layout
        main_layout.addWidget(left_widget)
        main_layout.addLayout(right_layout)
        main_widget.setLayout(main_layout)

    def open_file_dialog(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Novella", "", "Text Files (*.txt);;All Files (*)", options=options)
        if file_name:
            self.file_name_label.setText(f"Selected file: {file_name.split('/')[-1]}")

    def go_to_results_page(self):
        self.results_window = ResultsWindow(self)  # Pass self to allow navigation to sentences page
        self.results_window.show()
        self.close()


# Second Window (Results Page)
class ResultsWindow(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.setWindowTitle("Results")
        self.setGeometry(250, 150, 1500, 800)

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

        # Results title
        results_title = QLabel("RESULTS")
        results_title.setFont(QFont("Arial", 28, QFont.Bold))
        results_title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(results_title)

        spacer_between_title_and_emotions = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        right_layout.addItem(spacer_between_title_and_emotions)
        right_layout.addItem(spacer_between_title_and_emotions)

        # Emotion categories and counts
        emotions = [("JOY", 87), ("FEAR", 281), ("SADNESS", 166), ("ANGER", 138)]

        for emotion, count in emotions:
            # Horizontal layout for each emotion
            hbox = QHBoxLayout()

            emotion_label = QLabel(f"{emotion} : ")
            emotion_label.setFont(QFont("Arial", 18))

            count_label = QLabel(f"{count}")
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

            view_button.clicked.connect(lambda _, e=emotion: self.go_to_sentences_page(e))

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

        # Add widgets to the main layout
        main_layout.addWidget(left_widget)
        main_layout.addLayout(right_layout)
        self.setLayout(main_layout)

    def go_to_sentences_page(self, emotion):
        self.sentences_window = SentencesWindow(emotion, self)  # Create an instance of the sentences window with the selected emotion
        self.sentences_window.show()
        self.close()


# Third Window (Sentences Page)
class SentencesWindow(QWidget):
    def __init__(self, emotion, parent):
        super().__init__()
        self.parent = parent
        self.setWindowTitle(f"{emotion} Sentences")
        self.setGeometry(250, 150, 1500, 800)

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
        emotion_title = QLabel(f"{emotion} Sentences")
        emotion_title.setFont(QFont("Arial", 28, QFont.Bold))
        emotion_title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(emotion_title)

        spacer_between_title_and_sentences = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        right_layout.addItem(spacer_between_title_and_sentences)

        # Example sentences related to the selected emotion
        sentences = {
            "JOY": ["sentence 1", "sentence 2", "sentence 3", "sentence 4", "sentence 5", "sentence 6"],
            "FEAR": ["sentence 1", "sentence 2", "sentence 3", "sentence 4", "sentence 5", "sentence 6"],
            "SADNESS": ["sentence 1", "sentence 2", "sentence 3", "sentence 4", "sentence 5", "sentence 6"],
            "ANGER": ["sentence 1", "sentence 2", "sentence 3", "sentence 4", "sentence 5", "sentence 6"]
        }

        sentence_scroll_area = QScrollArea()
        sentence_widget = QWidget()
        sentence_layout = QVBoxLayout(sentence_widget)

        for sentence in sentences.get(emotion, []):
            sentence_label = QLabel(f"• {sentence}")
            sentence_label.setFont(QFont("Arial", 18))
            sentence_layout.addWidget(sentence_label)

        sentence_scroll_area.setWidget(sentence_widget)
        right_layout.addWidget(sentence_scroll_area)

        # Add a "Back" button to return to the previous window
        back_button = QPushButton("Back")
        back_button.setFixedSize(150, 40)
        back_button.setFont(QFont("Arial", 12, QFont.Bold))
        back_button.setStyleSheet("""
            background-color: #1338BE; 
            color: white;
            border-radius: 10px;
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


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
