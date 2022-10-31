import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.title("Sentence Textual Similarity")

@st.cache(allow_output_mutation=True)
def question_answer(text1, text2):
    s = SentenceTransformer('stsb-mpnet-base-v2')
    text1 = s.encode(text1)
    text2 = s.encode(text2)
    sim = cosine_similarity([text1],[text2])
    norm_sim = (sim+1)/2
    return norm_sim.item()

st.subheader("Enter the two sentences to compare")
text1 = st.text_area("Text 1")
text2 = st.text_area("Text 2")
result = st.button("Compare")

if text1 or text2 :
    st.write("Similarity Score: ", question_answer(text1, text2))
