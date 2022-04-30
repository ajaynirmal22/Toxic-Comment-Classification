# TCC_PRML_Project
Course Project "Toxic Comment Classification" | Pattern Recognition and Machine Learning 

# Usage | Follow these steps to use this repo
1. Add the dataset "all_data.csv" in "datasets/" folder
2. Run the command git clone https://github.com/screenygeek/TCC_PRML_Project.git
3. Run the command pip install -r requirements.txt
4. Run the command python run.py 

# Observations
1. Execuing run.py will result in:-
    1.  Creating and exporting TfIdf Vectorizer Object in "saved_model/vectorzier/" 
    2.  Training and exporting 6 Logistic Regression Models (for 6 labels being used) in "saved_model/"

# Flask Web Framework | Get real time results on texual comments
## Usage
1. python api/server.py
    1. This will run the Flask API at localhost:5000
    2. You can use this API for two purposes:-
        1. Predicting toxicity and other label's levels for a single/multiple comments, given as an input by a user 
        2. Predicting toxicity and other label's levels across various Tweets of a Twitter user (input from a user) and classifying them as a toxic publisher or non toxic publisher

# EDA | Exploratory Data Analysis

# Data Preprocessing

# Training

# Testing

# Validation
