import streamlit as st
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.datasets import imdb
import numpy as np

# Load model
from keras.layers import SpatialDropout1D

model = load_model("rnn-lstm-sentiment_model.h5", compile=False,
                   custom_objects={"SpatialDropout1D": SpatialDropout1D})

# Load IMDB word index
word_index = imdb.get_word_index()
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

# Function to predict sentiment
def predict_sentiment(review, maxlen=100):
    tokens = review.lower().split()
    encoded = [1]  # <START>
    for word in tokens:
        encoded.append(word_index.get(word, 2))  # 2 = <UNK>
    padded = pad_sequences([encoded], maxlen=maxlen)
    prob = model.predict(padded, verbose=0)[0][0]
    if prob >= 0.6:
        sentiment = f"😊 Positive with {prob*100:.2f}% confidence"
    else:
        sentiment = f"😞 Negative with {(1 - prob)*100:.2f}% confidence"
    return sentiment

# Streamlit UI
st.title("🎬 IMDB Movie Review Sentiment Analyzer")
st.write("Enter a movie review and get the predicted sentiment!")

user_input = st.text_area("Your Movie Review")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review first.")
    else:
        prediction = predict_sentiment(user_input)
        st.success(prediction)