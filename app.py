import streamlit as st
import tempfile
import os

st.set_page_config(page_title="Enterprise RAG Assistant", page_icon="⚡", layout="wide")

st.title("⚡ Enterprise RAG Assistant")
st.caption("Upload documents, perform semantic search, and query context with cross-encoder reranking.")

# Sidebar for Setup & Document Ingestion
with st.sidebar:
    st.header("📄 Document Ingestion")
    uploaded_file = st.file_uploader("Upload PDF or Text File", type=["pdf", "txt", "md"])
    
    if uploaded_file:
        with st.spinner("Processing & Indexing Document..."):
            # Save uploaded file to temporary directory
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            st.success(f"Indexed `{uploaded_file.name}` successfully!")

# Main Chat & Query Area
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Query Input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Render user query
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Simulated RAG response block (Replace with your pipeline imports)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving & Reranking Context..."):
            # Mock Output structure
            response_text = f"Based on the ingested document, here is the answer to: **{prompt}**"
            sources = ["Document Page 3, Section 2.1", "Document Page 5, Table 1"]
            
            st.markdown(response_text)
            
            # Display source citations expander
            with st.expander("📚 View Retained Context & Citations"):
                for src in sources:
                    st.write(f"- `{src}`")
                    
    st.session_state.messages.append({"role": "assistant", "content": response_text})