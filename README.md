
# E-commerce Report Recommendation System

## Project Purpose: Develop a vector-based retrieval system for e-commerce product categorization using FAISS, HuggingFace Embeddings, and LangChain. The system helps match product names to relevant categories efficiently.

## Project Description
1. Data Preprocessing:
  + Read product category data from an Excel file.
  + Normalize text data for consistency.
    
2. Embedding Generation:
  + Use HuggingFace's SentenceTransformer models to generate embeddings for product names.
  + Supported models:
    intfloat/multilingual-e5-large (default)
    BAAI/bge-m3
    Alibaba-NLP/gte-Qwen2-7B-instruct
    sentence-transformers/all-MiniLM-L6-v2

3. Vector Database Construction: Store embeddings in a FAISS vector database for fast similarity searches.
   
4. Query Processing & Similarity Search:
  + Normalize product names from a dataset.
  + Perform similarity search using FAISS to find the top 5 closest matches.

5 Output Storage:
  + Store results in JSONL format (dict_cate_group.jsonl).
  + Convert JSONL results to CSV format for easy analysis.

## Project Operation Steps
1. Load and preprocess product category data.
2. Generate embeddings using a HuggingFace model.
3. Construct a FAISS vector database from the embeddings.
4. Read product queries and retrieve the top 5 most relevant categories.
5. Save the retrieved results in JSONL and CSV formats.

