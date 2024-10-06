import pickle
import os

# Save the model
def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filename}")

# Load the model
def load_model(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filename}")
        return model
    else:
        print(f"File {filename} does not exist.")
        return None

# Get the current directory of this file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define the path to the models folder
models_folder = os.path.join(current_dir, "..", "models")

if not os.path.exists(models_folder):
    os.makedirs(models_folder)

# Function to save the model with a specific name
def save_model_with_name(model, model_name):
    save_model(model, os.path.join(models_folder, model_name))

# Function to load the model with a specific name
def load_model_with_name(model_name):
    return load_model(os.path.join(models_folder, model_name))