import streamlit as st
import requests
import json
import pandas as pd

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Ultra Doc-Intelligence POC", layout="wide")

st.title("Ultra Doc-Intelligence")
st.markdown("### Logistics Document Intelligence System (TMS Integration)")

with st.sidebar:
    st.header("Settings")
    st.success("Inference Engine: Local")
    st.info("Upload a logistics document (PDF, DOCX, TXT) to start.")

tab1, tab2 = st.tabs(["Q&A & Upload", "Structured Extraction"])

with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Upload Document")
        uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt"])
        
        if uploaded_file:
            if st.button("Process Document"):
                with st.spinner("Processing..."):
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    response = requests.post(f"{API_URL}/upload", files=files)
                    
                    if response.status_code == 200:
                        st.success(f"Successfully processed: {uploaded_file.name}")
                    else:
                        st.error(f"Error: {response.json().get('detail')}")

    with col2:
        st.subheader("Document Query")
        question = st.text_input("Enter your query")
        
        if st.button("Ask"):
            if not question:
                st.warning("Please enter a question.")
            else:
                with st.spinner("Processing request..."):
                    response = requests.post(f"{API_URL}/ask", json={"question": question})
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.markdown(f"**Result:** {data['answer']}")
                        
                        st.divider()
                        st.markdown(f"**Match Strength:** `{data['confidence_score']}`")
                        
                        with st.expander("View Sources"):
                            for i, source in enumerate(data['sources']):
                                st.markdown(f"**Source {i+1}:**\n{source}")
                    else:
                        st.error(f"Error: {response.json().get('detail')}")

with tab2:
    st.subheader("Extract Structured Logistics Fields")
    if st.button("Run Extraction"):
        with st.spinner("Extracting data..."):
            response = requests.post(f"{API_URL}/extract")
            
            if response.status_code == 200:
                extracted_data = response.json()
                
                # Display as a table
                df = pd.DataFrame(list(extracted_data.items()), columns=["Field", "Value"])
                st.table(df)
                
                with st.expander("View Raw JSON"):
                    st.json(extracted_data)
            else:
                st.error(f"Error: {response.json().get('detail')}")

st.sidebar.markdown("---")
st.sidebar.caption("Internal Document Intelligence Prototype")
