import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import string
from fuzzywuzzy import process
from random import choice

# Load model pipeline dan vectorizer dari file .sav
with open('D:\\kuliah nopal\\semester 5\\bengkod\\chatbot_model.sav', 'rb') as model_file:
    pipe = pickle.load(model_file)

with open('D:\\kuliah nopal\\semester 5\\bengkod\\vectorizer.sav', 'rb') as vect_file:
    vect = pickle.load(vect_file)

class JSONParser:
    def __init__(self):
        self.text = []
        self.intents = []
        self.responses = {}

    def parse(self, json_path):
        with open(json_path) as data_file:
            self.data = json.load(data_file)

        for intent in self.data['intents']:
            for pattern in intent['patterns']:
                self.text.append(pattern)
                self.intents.append(intent['tag'])
            for resp in intent['responses']:
                if intent['tag'] in self.responses.keys():
                    self.responses[intent['tag']].append(resp)
                else:
                    self.responses[intent['tag']] = [resp]

        self.df = pd.DataFrame({'text_input': self.text,
                                'intents': self.intents})

    def get_dataframe(self):
        return self.df

    def get_response(self, intent):
        return choice(self.responses[intent])

def preprocess(chat):
    chat = chat.lower()
    tandabaca = tuple(string.punctuation)
    chat = ''.join(ch for ch in chat if ch not in tandabaca)
    return chat

def match_pattern(user_input, patterns):
    best_match = process.extractOne(user_input, patterns)
    return best_match

# Load kembali data intents dari file JSON
path = "D:\\kuliah nopal\\semester 5\\bengkod\\intents1112.json"  # Update with the correct path to your intents file
jp = JSONParser()
jp.parse(path)
df = jp.get_dataframe()
df['text_input_prep'] = df.text_input.apply(preprocess)

# Streamlit app
st.title("Chatbot")
st.write("Anda Terhubung dengan chatbot Kami")

# Chat input
user_input = st.text_input("Anda :")

if st.button("Kirim"):
    if user_input:
        chat_prep = preprocess(user_input)
        best_match = match_pattern(chat_prep, df['text_input_prep'].tolist())

        if best_match[1] >= 80:  # Threshold for fuzzy matching
            matched_index = df[df['text_input_prep'] == best_match[0]].index[0]
            pred_label = df['intents'][matched_index]
            max_prob = 1.0
        else:
            res = pipe.predict_proba([chat_prep])
            max_prob = max(res[0])
            max_idx = np.argmax(res[0])
            pred_label = pipe.classes_[max_idx]

        if max_prob < 0.20:
            response = "Bot : Maaf Kak, aku ga ngerti"
        else:
            response = "Bot : " + jp.get_response(pred_label)

        st.write(response)

        # If intent-nya 'bye', clear input
        if pred_label == 'bye':
            st.write("Bot : Sampai jumpa!")
            st.text_input("Anda :", "", key='bye_input')  # Clear input for goodbye
