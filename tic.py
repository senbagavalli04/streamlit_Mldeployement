import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import streamlit as st


# Load data
data = pd.read_csv("tic-tac-toe.csv")

# Encode features
X = data.drop('class', axis=1).apply(lambda x: x.map({'x': 1, 'o': -1, 'b': 0}))


# Extract target labels
y = data['class']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)

# Check the number of input features
print("Number of input features:", X_train.shape[1])

import pickle

# Save the trained model using pickle
with open('ff.pkl', 'wb') as file:
    pickle.dump(random_forest,file)
