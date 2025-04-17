# main.py
# optional script that imports and runs the main components
# of the project pipeline in order (preprocessing → training → evaluation)

# main.py - runs the full model pipeline

# load and preprocess data
import preprocess

# train all models
import train_model

# evaluate all models
import evaluate

# utils is for helper functions used across other files
import utils
