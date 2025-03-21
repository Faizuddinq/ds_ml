import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load job roles dataset
@st.cache_data
def load_data():
    return pd.read_csv("job_roles_skills.csv")  # Ensure the CSV file is in the same directory

df = load_data()

# Convert skills into TF-IDF vectors
vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(", "))
tfidf_matrix = vectorizer.fit_transform(df["Skills"])

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to recommend similar roles
def recommend_roles(input_role, df, similarity_matrix, top_n=3):
    if input_role not in df["Job Role"].values:
        return []

    # Find index of the input job role
    idx = df[df["Job Role"] == input_role].index[0]

    # Get similarity scores and sort
    sim_scores = sorted(list(enumerate(similarity_matrix[idx])), key=lambda x: x[1], reverse=True)[1:top_n+1]

    # Get the recommended job roles
    return [(df.iloc[i[0]]["Job Role"], i[1]) for i in sim_scores]

# Streamlit UI
st.title(" Job Role Recommendation Engine")

# User input: Select a job role
selected_role = st.selectbox("Select a job role:", df["Job Role"].tolist())

if st.button("Find Similar Roles"):
    recommendations = recommend_roles(selected_role, df, cosine_sim)

    if recommendations:
        st.subheader(" Top 3 Similar Roles:")
        for role, score in recommendations:
            st.write(f"**{role}** (Similarity: {score:.2f})")
    else:
        st.write("No similar roles found.")
