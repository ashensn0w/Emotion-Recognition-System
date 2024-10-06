import pickle
import os

def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filename}")

# Get the current directory of this file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define the path to the models folder
models_folder = os.path.join(current_dir, "..", "models")
if not os.path.exists(models_folder):
    os.makedirs(models_folder)

# Function to save the model with a specific name
def save_model_with_name(model, model_name):
    save_model(model, os.path.join(models_folder, model_name))