#Importing required libraries
import streamlit as st  # For web development
from sentence_transformers import SentenceTransformer   # For Sentence embeddings
from sklearn.metrics.pairwise import cosine_similarity  # For Cosine similarity between Embeddings

st.title("Sentence Textual Similarity") # Document title

@st.cache(allow_output_mutation=True) # Caching the model to avoid loading it again and again
def question_answer(text1, text2): # Function to find similarity between two sentences
    s = SentenceTransformer('stsb-mpnet-base-v2') # Loading the model
    text1 = s.encode(text1) # Encoding the first sentence
    text2 = s.encode(text2) # Encoding the second sentence
    sim = cosine_similarity([text1],[text2]) # Finding the cosine similarity between the two sentences
    norm_sim = (sim+1)/2 # Normalizing the similarity score
    return norm_sim.item() # Returning the normalized similarity score

st.subheader("Enter the two sentences to compare") # Subheader for the document
text1 = st.text_area("Text 1")  # Input : Text area for the Sentence 1
text2 = st.text_area("Text 2")  # Input : Text area for the Sentence 2
result = st.button("Compare")   # Button to compare the two sentences

if text1 and text2 : # Checking if both the sentences are entered
    st.write("Similarity Score: ", question_answer(text1, text2))
