#Importing required libraries
import streamlit as st  # For web development
import requests
from dotenv import load_dotenv
import os

load_dotenv()

st.title("Sentence Textual Similarity") # Document title
API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/stsb-mpnet-base-v2"
headers = {"Authorization": os.getenv('bearer')}
# st.write(headers)
def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

@st.cache(allow_output_mutation=True) # Caching the model to avoid loading it again and again
def question_answer(text1, text2): # Function to find similarity between two sentences
    payload = {"inputs": {"source_sentence": text1, "sentences": [text2]}}
    sim = query(payload)
    norm_sim = (sim[0]+1)/2 # Normalizing the similarity score
    return norm_sim # Returning the normalized similarity score

st.subheader("Enter the two sentences to compare") # Subheader for the document
with st.form('form'):
    text1 = st.text_area("Text 1")  # Input : Text area for the Sentence 1
    text2 = st.text_area("Text 2")  # Input : Text area for the Sentence 2
    result = st.form_submit_button("Compare")   # Button to compare the two sentences
    if result:
        st.write("Similarity Score: ", question_answer(text1, text2))
    
