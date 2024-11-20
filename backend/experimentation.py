# from sklearn.metrics import confusion_matrix
# from rich.console import Console
# from rich.table import Table
# import pandas as pd

# # Load the dataset containing actual and predicted emotions
# output_df = pd.read_csv('./frontend/backend outputs/output.csv')

# # Unique emotions in the dataset
# emotions = output_df['emotion'].unique()

# # Initialize console for displaying tables
# console = Console()

# # Function to create a formatted table
# def create_table(metric_name, emotion, cm):
#     TP = cm[1, 1]  # True Positive
#     FP = cm[0, 1]  # False Positive
#     FN = cm[1, 0]  # False Negative
#     TN = cm[0, 0]  # True Negative

#     table = Table(title=f"{metric_name.upper()} - {emotion.upper()}", expand=True)
#     table.add_column("", justify="center", style="cyan")
#     table.add_column("CLASSIFIED", justify="center", style="green")
#     table.add_column("NOT CLASSIFIED", justify="center", style="green")
    
#     table.add_row("CLASSIFIED", str(TP), str(FN))
#     table.add_row("NOT CLASSIFIED", str(FP), str(TN))
#     return table

# # Function to compute and display tables for all metrics and emotions
# def display_all_metrics(df, emotions):
#     for emotion in emotions:
#         # Binary classification for each emotion
#         y_true = (df['emotion'] == emotion).astype(int)
#         y_pred = (df['predicted_emotion'] == emotion).astype(int)
        
#         # Compute confusion matrix
#         cm = confusion_matrix(y_true, y_pred)
        
#         # Display Precision table
#         precision_table = create_table("Precision", emotion, cm)
#         console.print(precision_table)

#         # Display Recall table
#         recall_table = create_table("Recall", emotion, cm)
#         console.print(recall_table)

#         # Display F1 Score table
#         f1_table = create_table("F1 Score", emotion, cm)
#         console.print(f1_table)

# # Display all metrics for all emotions
# display_all_metrics(output_df, emotions)
######################################################################################

# TO ANSWER SOP#1

from sklearn.metrics import confusion_matrix
from rich.console import Console
from rich.table import Table
import pandas as pd

# Load the dataset with and without TF-IDF outputs
tfidf_df = pd.read_csv('./frontend/backend outputs/with_tfidf_output.csv')
non_tfidf_df = pd.read_csv('./frontend/backend outputs/without_tfidf_output.csv')

# Unique emotions in the dataset
emotions = tfidf_df['emotion'].unique()

# Initialize console for displaying tables
console = Console()

# Function to create a formatted table for each configuration
def create_table(metric_name, emotion, with_tfidf, cm):
    TP = cm[1, 1]  # True Positive
    FP = cm[0, 1]  # False Positive
    FN = cm[1, 0]  # False Negative
    TN = cm[0, 0]  # True Negative
    
    # Title
    tfidf_mode = "WITH TF-IDF" if with_tfidf else "WITHOUT TF-IDF"
    
    # Create table
    table = Table(title=f"{metric_name.upper()} - {tfidf_mode} {emotion.upper()}", expand=True)
    table.add_column("", justify="center", style="cyan")
    table.add_column("CLASSIFIED", justify="center", style="green")
    table.add_column("NOT CLASSIFIED", justify="center", style="green")
    
    table.add_row("CLASSIFIED", str(TP), str(FN))
    table.add_row("NOT CLASSIFIED", str(FP), str(TN))
    
    return table

# Function to compute and display all metrics for TF-IDF and Non-TF-IDF
def display_metrics(tfidf_data, non_tfidf_data, emotions):
    for emotion in emotions:
        # For TF-IDF
        y_true_tfidf = (tfidf_data['emotion'] == emotion).astype(int)
        y_pred_tfidf = (tfidf_data['predicted_emotion'] == emotion).astype(int)
        cm_tfidf = confusion_matrix(y_true_tfidf, y_pred_tfidf)
        
        # Create and display tables for TF-IDF
        precision_tfidf = create_table("Precision", emotion, True, cm_tfidf)
        recall_tfidf = create_table("Recall", emotion, True, cm_tfidf)
        f1_tfidf = create_table("F1 Score", emotion, True, cm_tfidf)
        
        console.print(precision_tfidf)
        console.print(recall_tfidf)
        console.print(f1_tfidf)
        
        # For Non-TF-IDF
        y_true_non_tfidf = (non_tfidf_data['emotion'] == emotion).astype(int)
        y_pred_non_tfidf = (non_tfidf_data['predicted_emotion'] == emotion).astype(int)
        cm_non_tfidf = confusion_matrix(y_true_non_tfidf, y_pred_non_tfidf)
        
        # Create and display tables for Non-TF-IDF
        precision_non_tfidf = create_table("Precision", emotion, False, cm_non_tfidf)
        recall_non_tfidf = create_table("Recall", emotion, False, cm_non_tfidf)
        f1_non_tfidf = create_table("F1 Score", emotion, False, cm_non_tfidf)
        
        console.print(precision_non_tfidf)
        console.print(recall_non_tfidf)
        console.print(f1_non_tfidf)
        
        # Separator between emotion tables
        console.print("<---------------------------------------------------------------------------->\n", style="bold cyan")

# Call the function to display all metrics
display_metrics(tfidf_df, non_tfidf_df, emotions)