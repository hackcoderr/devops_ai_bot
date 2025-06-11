import pandas as pd
from sentence_transformers import SentenceTransformer # type: ignore
import faiss # type: ignore
import numpy as np
import streamlit as st

# Load dataset
df = pd.read_csv("aws_devops_issues_500_dataset.csv")
texts = (df["Issue Subject"] + " - " + df["Issue Solution"]).tolist()

# Load model and generate embeddings
@st.cache_resource
def load_model_and_index():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return model, index, df.to_dict(orient="records")

model, index, metadata = load_model_and_index()

# UI
st.title("🛠️ DevOps AI Assistant")
st.write("Ask any DevOps-related issue and get the closest solution from the knowledge base.")

query = st.text_input("🔍 Describe your issue here:")
top_k = st.slider("How many results to show?", 1, 5, 3)

if query:
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    results = [metadata[i] for i in indices[0]]

    st.markdown("### 📋 Results")
    for res in results:
        st.markdown(f"**{res['Issue Subject']}**")
        st.markdown(f"💡 {res['Issue Solution']}")
        st.markdown(f"🆔 Ticket ID: `{res['Ticket ID']}`")
        st.markdown("---")