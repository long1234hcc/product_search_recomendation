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

# Khởi tạo embeddings với HuggingFaceEmbeddings từ mô hình SentenceTransformer
# custom_embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-m3')
custom_embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
# custom_embeddings = HuggingFaceEmbeddings(model_name='Alibaba-NLP/gte-Qwen2-7B-instruct')
# custom_embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')


print("Done embedings")

# Tạo danh sách Document với embeddings từ mô hình tùy chỉnh
documents = []
for _, row in df.iterrows():
    documents.append(
        Document(
            page_content=row['name'],          # Nội dung văn bản chính để tạo embeddings
            metadata={
                "id": row['id'],               # ID sản phẩm
                # "category": row['category']    # Ngành hàng của sản phẩm
            }
        )
    )
    
    
print("Vector_database")

## Build vector database
vector_db = FAISS.from_documents(documents=documents, embedding=custom_embeddings)

print("Read file")
# Giả sử df_ans đã được đọc từ file Excel và normalized
df_ans = pd.read_excel(r"/hdd/sv10/svc/docker-svc/jupyter/data/Team_DC/long/Retrival/get_brand_ereport_handle.xlsx", sheet_name="Sheet1")

# Hàm chuẩn hóa chuỗi
def normalized(text):
    return str(text).lower().strip()

df_ans["product_name_normalized"] = df_ans["product_name_normalized"].apply(normalized)
# df_ans["category"] = df_ans["category"].apply(normalized)


print("Write json file")
# Tạo dictionary để lưu kết quả
dict_direct = {}

# Mở file JSON Lines để lưu dữ liệu
with open("dict_cate_group.jsonl", "w", encoding="utf-8") as f:
    # Lặp qua từng dòng trong DataFrame với thanh tiến trình
    for idx, rows in tqdm(df_ans.iterrows(), desc="Process", total=len(df_ans)):
        query = str(rows['product_name_normalized'])
        # category = str(rows['category'])
        
        # Tạo bộ truy vấn với điểm similarity
        results_with_scores = vector_db.similarity_search_with_score(
            query=query,
            k=5,
            # filter={"category": category}
        )
        
        # Trích xuất page_content từ kết quả
        page_content_list = [doc.page_content for doc, score in results_with_scores]
        
        # Nếu page_content_list không tồn tại, gán nó là list trống
        if not page_content_list:
            page_content_list = []
        
        # Lưu kết quả vào dict_direct với query làm key
        dict_direct[query] = page_content_list
        
        # Ghi vào file JSON Lines
        json_line = json.dumps({query: page_content_list}, ensure_ascii=False)
        f.write(json_line + "\n")
        
        
        
     
print("Write csv file")
   
# Đường dẫn tới file
jsonl_path = r"/hdd/sv10/svc/docker-svc/jupyter/data/Team_DC/long/Retrival/dict_cate_group.jsonl"
csv_path = r"/hdd/sv10/svc/docker-svc/jupyter/data/Team_DC/long/Retrival/dict_cate_group.csv"

# Mở file CSV để ghi
with open(csv_path, mode="w", encoding="utf-8", newline='') as csv_file:
    # Khởi tạo writer và ghi header
    writer = csv.DictWriter(csv_file, fieldnames=["key", "value"])
    writer.writeheader()

    # Đọc từng dòng JSON và ghi trực tiếp vào CSV
    with open(jsonl_path, mode="r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())  # Đọc từng dòng JSON
            for key, value in record.items():
                writer.writerow({"key": key, "value": value})  # Ghi từng dòng vào CSV
                
                
print("Done")
