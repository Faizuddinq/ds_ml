import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# ============================
# Load Job Roles Dataset
# ============================
@st.cache_data
def load_data():
    file_path = "job_roles_skills.csv"  
    if not os.path.exists(file_path):
        st.error("Error: The dataset file 'job_roles_skills.csv' was not found. Please check the file path.")
        return None
    return pd.read_csv(file_path)

df = load_data()

# Exit if dataset is not found
if df is None:
    st.stop()

# ==================================================
# Convert Skills into TF-IDF Vectors for Matching
# ==================================================
vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(", "))
tfidf_matrix = vectorizer.fit_transform(df["Skills"])

# ===============================
# Compute Cosine Similarity
# ===============================
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# =============================================
# Function to Recommend Similar Job Roles
# =============================================
def recommend_roles(input_role, df, similarity_matrix, top_n=3):
    input_role = input_role.strip().title()  # Case-insensitive matching

    if input_role not in df["Job Role"].values:
        return []

    idx = df[df["Job Role"] == input_role].index[0]

    sim_scores = sorted(
        list(enumerate(similarity_matrix[idx])),
        key=lambda x: x[1], reverse=True
    )[1:top_n+1]

    return [(df.iloc[i[0]]["Job Role"], round(i[1], 2)) for i in sim_scores]

# ==============================
# Streamlit UI Configuration
# ==============================
st.set_page_config(page_title="Job Role Recommender", page_icon="🔍", layout="wide")

# ============================
# 🔹 Custom Styling
# ============================
st.markdown(
    """
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #F8F9FA;
        }
        .title {
            text-align: center;
            color: #2C3E50;
        }
        .sidebar .sidebar-content {
            background-color: #2C3E50;
            color: white;
        }
        .stButton > button {
            background-color: #007BFF;
            color: white;
            font-size: 16px;
            border-radius: 5px;
            padding: 10px 20px;
        }
        .stTable {
            background-color: white;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ======================
# 📌 Sidebar Instructions
# ======================
with st.sidebar:
    st.title("🔹 About This App")
    st.write(
        """
        This tool helps you find **similar job roles** based on required skills.
        
        - Type a job role or select from the list.
        - See the top 3 most similar roles.
        - Helps career changers and job seekers explore new opportunities.
        """
    )
    st.markdown("---")
    st.write("Developed by Aiman Suhail")

# =====================
# 🔍 Main UI Component
# =====================
st.markdown("<h1 class='title'>🔍 Job Role Recommendation Engine</h1>", unsafe_allow_html=True)

# Search bar for job roles with autocomplete
selected_role = st.text_input("🔎 Enter a job role:", "").strip()

# Alternative: Dropdown if no input is given
if not selected_role:
    selected_role = st.selectbox("Or select from the list:", df["Job Role"].tolist())

# =================================
# 🚀 Button to Generate Recommendations
# =================================
if st.button("Find Similar Roles"):
    recommendations = recommend_roles(selected_role, df, cosine_sim)

    if recommendations:
        st.subheader(f"📌 Top 3 Similar Roles to **{selected_role}**")
        
        # Convert recommendations to DataFrame
        result_df = pd.DataFrame(recommendations, columns=["Job Role", "Similarity Score"])
        
        # Customizing Progress Bar Colors based on similarity score
        def get_progress_color(score):
            if score >= 0.75:
                return "green"
            elif score >= 0.50:
                return "orange"
            else:
                return "red"

        # Display table with styled progress bars
        for index, row in result_df.iterrows():
            st.write(f"### {row['Job Role']}  \n**Similarity Score:** `{row['Similarity Score']}`")
            st.progress(row["Similarity Score"])

    else:
        st.warning("⚠️ No similar roles found. Try a different job title.")
