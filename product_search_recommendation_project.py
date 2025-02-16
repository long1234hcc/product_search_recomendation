# !pip install -U langchain-community
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from langchain.chains import RetrievalQA
from tqdm import tqdm
import json
tqdm.pandas()
import csv
import json




def normalized(text):
    return str(text).lower().strip()
# Đọc dữ liệu từ file CSV
file_path = r'/hdd/sv10/svc/docker-svc/jupyter/data/Team_DC/long/Retrival/lst_cate_unique_handle.xlsx'
df = pd.read_excel(file_path)

df['name'] = df['name'].apply(normalized)
print(df.shape)

# HuggingFaceEmbeddings SentenceTransformer
# custom_embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-m3')
custom_embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
# custom_embeddings = HuggingFaceEmbeddings(model_name='Alibaba-NLP/gte-Qwen2-7B-instruct')
# custom_embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')


print("Done embedings")

documents = []
for _, row in df.iterrows():
    documents.append(
        Document(
            page_content=row['name'],          
                "id": row['id'],               
                # "category": row['category']   
            }
        )
    )
    
    
print("Vector_database")

## Build vector database
vector_db = FAISS.from_documents(documents=documents, embedding=custom_embeddings)

print("Read file")
df_ans = pd.read_excel(r"/hdd/sv10/svc/docker-svc/jupyter/data/Team_DC/long/Retrival/get_brand_ereport_handle.xlsx", sheet_name="Sheet1")

def normalized(text):
    return str(text).lower().strip()

df_ans["product_name_normalized"] = df_ans["product_name_normalized"].apply(normalized)
# df_ans["category"] = df_ans["category"].apply(normalized)


print("Write json file")
dict_direct = {}

with open("dict_cate_group.jsonl", "w", encoding="utf-8") as f:
    for idx, rows in tqdm(df_ans.iterrows(), desc="Process", total=len(df_ans)):
        query = str(rows['product_name_normalized'])
        # category = str(rows['category'])
        
        results_with_scores = vector_db.similarity_search_with_score(
            query=query,
            k=5,
            # filter={"category": category}
        )
        
        page_content_list = [doc.page_content for doc, score in results_with_scores]
        
        if not page_content_list:
            page_content_list = []
        
        dict_direct[query] = page_content_list
        
        json_line = json.dumps({query: page_content_list}, ensure_ascii=False)
        f.write(json_line + "\n")
        
        
        
     
print("Write csv file")
   
jsonl_path = r"/hdd/sv10/svc/docker-svc/jupyter/data/Team_DC/long/Retrival/dict_cate_group.jsonl"
csv_path = r"/hdd/sv10/svc/docker-svc/jupyter/data/Team_DC/long/Retrival/dict_cate_group.csv"

with open(csv_path, mode="w", encoding="utf-8", newline='') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=["key", "value"])
    writer.writeheader()

    with open(jsonl_path, mode="r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())  # Đọc từng dòng JSON
            for key, value in record.items():
                writer.writerow({"key": key, "value": value})  # Ghi từng dòng vào CSV
                
                
print("Done")
