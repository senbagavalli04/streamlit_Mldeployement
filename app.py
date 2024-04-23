import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open('ff.pkl', 'rb') as file:
    random_forest = pickle.load(file)

def make_prediction(move1, move2, move3, move4, move5, move6, move7, move8, move9):
    # Convert moves to numerical representation (e.g., 0 for 'X', 1 for 'O', 2 for 'Empty')
    moves_encoded = []
    for move in [move1, move2, move3, move4, move5, move6, move7, move8, move9]:
        if move == 'X':
            moves_encoded.append(0)
        elif move == 'O':
            moves_encoded.append(1)
        else:  # 'Empty'
            moves_encoded.append(2)
    return moves_encoded

# Title of the Streamlit app
st.title("Tic Tac Toe Move Predictor")

# User input for 9 moves
st.header("Input Moves")
move1 = st.selectbox("Move 1", ['X', 'O', 'Empty'])
move2 = st.selectbox("Move 2", ['X', 'O', 'Empty'])
move3 = st.selectbox("Move 3", ['X', 'O', 'Empty'])
move4 = st.selectbox("Move 4", ['X', 'O', 'Empty'])
move5 = st.selectbox("Move 5", ['X', 'O', 'Empty'])
move6 = st.selectbox("Move 6", ['X', 'O', 'Empty'])
move7 = st.selectbox("Move 7", ['X', 'O', 'Empty'])
move8 = st.selectbox("Move 8", ['X', 'O', 'Empty'])
move9 = st.selectbox("Move 9", ['X', 'O', 'Empty'])

# Make prediction when user clicks the button
if st.button("Predict"):
    if random_forest is not None:
        moves_encoded = make_prediction(move1, move2, move3, move4, move5, move6, move7, move8, move9)
        # Ensure that there are 9 features in moves_encoded
        if len(moves_encoded) == 9:
            prediction = random_forest.predict([moves_encoded])
            if prediction == 0:
                st.success("Predicted winner: X")
            elif prediction == 1:
                st.success("Predicted winner: O")
            else:
                st.warning("Prediction: Draw")
        else:
            st.error("Incorrect number of features in input data. Please select 9 moves.")
    else:
        st.error("Model not loaded. Please upload or select a trained model.")
