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
st.set_page_config(page_title="Job Role Recommender", page_icon="🔍", layout="wide")

# Sidebar for instructions
with st.sidebar:
    st.title("⚙️ About This App")
    st.write(
        """
        This AI-powered tool recommends the **top 3 most similar job roles** based on required skills.
        
        - 🔎 **Type a job role** or **select from the list**.
        - 📊 **See job similarity scores** visually.
        - 🚀 **Helps career changers & job seekers** find related roles!
        """
    )
    st.markdown("---")
    st.write("📌 **Created by [Your Name]**")

# Main UI
st.title("🔍 Job Role Recommendation Engine")

# Search bar for job roles
selected_role = st.text_input("Enter a job role:", "").strip()

# Alternative: Use dropdown if no input
if not selected_role:
    selected_role = st.selectbox("Or select from the list:", df["Job Role"].tolist())

# Find similar jobs when button is clicked
if st.button("Find Similar Roles"):
    recommendations = recommend_roles(selected_role, df, cosine_sim)

    if recommendations:
        st.subheader(f"📌 Top 3 Similar Roles to **{selected_role}**:")
        for role, score in recommendations:
            st.write(f"### {role}  \nSimilarity Score: ")
            st.progress(float(score))  # Visual bar representation
    else:
        st.warning("⚠️ No similar roles found. Try a different job title.")
