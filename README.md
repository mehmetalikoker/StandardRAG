# What's StandardRAG Project
 It is a Standard RAG project developed using Langchain,streamlit.
 
## How It Works
for rag.py
- Pulling data from web source or documents
- Data to be used is divided into small pieces for each use.
- The divided chunks values are saved to Vector Database (Langchain Chroma Database).
- Using the role structure with prompt (Langchain ChatPromptTemplate)

for ragwithui.py
The screen has been enhanced for the rag.py project using StreamLit.
- for open : terminal -> streamlit run ragwithui.py

## Requirements
- OPENAI_API_KEY
- LANGCHAIN_API_KEY
- LANGCHAIN_PROJECT Info
