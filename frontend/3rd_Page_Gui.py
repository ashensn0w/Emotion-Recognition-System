import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QScrollArea
from PyQt5.QtCore import Qt

class EmotionRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Main layout with two columns
        main_layout = QHBoxLayout()

        # Left column layout (for the title)
        left_layout = QVBoxLayout()
        left_label = QLabel('EMOTION\nRECOGNITION\nIN FILIPINO\nNOVELLAS')
        left_label.setAlignment(Qt.AlignCenter)
        left_label.setStyleSheet("background-color: #a33234; color: white; font-size: 24px; padding: 50px;")
        left_layout.addWidget(left_label)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Right column layout (for the content)
        right_layout = QVBoxLayout()

        # Emotion Title and Sentence Count
        emotion_title = QLabel('FEAR\n281 sentences')
        emotion_title.setStyleSheet("font-size: 24px; font-weight: bold; padding-left: 200px;")
        emotion_title.setAlignment(Qt.AlignLeft)
        right_layout.addWidget(emotion_title)

        # Scroll area for sentences
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)

        # Example sentences
        for i in range(281):  # Add 15 example sentences to demonstrate scrolling
            sentence = QLabel('Example Sentence')
            sentence.setStyleSheet("font-size: 18px; color: gray; padding-left: 10px;" if i >= 13 else "font-size: 18px; color: black; padding-left: 10px;")
            content_layout.addWidget(sentence)

        scroll_area.setWidget(content_widget)
        right_layout.addWidget(scroll_area)

        # Add layouts to the main layout
        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 3)

        self.setLayout(main_layout)
        self.setWindowTitle('Emotion Recognition UI')
        self.setGeometry(100, 100, 900, 512)
        self.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = EmotionRecognitionApp()
    sys.exit(app.exec_())
