import pandas as pd
import os
import joblib
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import time

# File paths for saving/loading model and embeddings based on the number of articles
def get_file_names(n_articles):
    suffix = f"_{n_articles}" if n_articles else "_all"
    embeddings_file = f'embeddings{suffix}.pkl'
    model_file = f'bert_model{suffix}'
    return embeddings_file, model_file

# Function to load data
def load_data(n_articles=None):
    data = pd.read_csv('articles.csv')
    if n_articles is not None:
        data = data.head(n_articles)
    return data['content'].tolist(), data['publication'].tolist(), data['author'].tolist(), data['url'].tolist()

# Function to train the model
def train_model(documents, embeddings_file, model_file):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(documents, convert_to_tensor=True)
    joblib.dump(embeddings, embeddings_file)
    model.save(model_file)
    return embeddings, model

# Function to load or train model
def load_or_train_model(n_articles):
    embeddings_file, model_file = get_file_names(n_articles)
    if os.path.exists(embeddings_file) and os.path.exists(model_file):
        embeddings = joblib.load(embeddings_file)
        model = SentenceTransformer(model_file)
    else:
        documents, _, _, _ = load_data(n_articles)
        embeddings, model = train_model(documents, embeddings_file, model_file)
    return embeddings, model

# Streamlit UI
st.title("Plagiarism Checker")

# Option for number of articles
article_option = st.selectbox("Select number of articles for model training", 
                              ["100 articles", "500 articles", "5000 articles", "All articles"])

# Mapping for number of articles
article_map = {"100 articles": 100, "500 articles": 500, "5000 articles": 5000, "All articles": None}
selected_articles = article_map[article_option]

# Load or train the model based on selection
embeddings, model = load_or_train_model(selected_articles)

# Function to calculate similarity
def calculate_similarity(text, embeddings, model):
    input_embedding = model.encode(text, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(input_embedding, embeddings)[0]
    return cosine_scores

# Streamlit UI for plagiarism check
user_input = st.text_area("Enter Text to Check", "")
similarity_threshold = st.slider("Cosine Similarity Threshold", 0.0, 1.0, 0.8, 0.01)

if st.button("Check for Plagiarism"):
    if user_input:
        start_time = time.time()  # Start the timer

        similarity_scores = calculate_similarity(user_input, embeddings, model)

        elapsed_time = time.time() - start_time  # Calculate elapsed time

        matched_documents = []
        for i, score in enumerate(similarity_scores):
            if score > similarity_threshold:
                matched_documents.append({
                    "Document ID": i,
                    "Similarity": round(score.item(), 2),
                    "Publication": publications[i],
                    "Author": authors[i],
                    "Content": documents[i][:200] + "...",
                    "URL": urls[i]
                })

        if matched_documents:
            st.write(pd.DataFrame(matched_documents))
            st.write(f"Time taken for similarity detection: {elapsed_time:.2f} seconds")
        else:
            st.write("No similar documents found.")
            st.write(f"Time taken for similarity detection: {elapsed_time:.2f} seconds")
    else:
        st.write("Please enter some text to check.")
