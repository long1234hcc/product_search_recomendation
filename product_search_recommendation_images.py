import numpy as np
import pandas as pd
import faiss
import torch
from transformers import CLIPProcessor, CLIPModel
import requests
from PIL import Image
import io

clip_model = CLIPModel.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
clip_processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")

def preprocess_image(image_path):
    try:
        response = requests.get(image_path)
        image = Image.open(io.BytesIO(response.content)) 
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

# Embeding
def add_embeddings_to_faiss(urls, faiss_index, batch_size=8):
    for i in range(0, len(urls), batch_size):
        batch_urls = urls[i:i+batch_size]
        images = []
        
        for url in batch_urls:
            image = preprocess_image(url)
            if image is not None:
                images.append(image)
        
        if images:
            inputs = clip_processor(images=images, return_tensors="pt", padding=True)
            with torch.no_grad():
                batch_features = clip_model.get_image_features(**inputs).cpu().numpy()
                
                if batch_features.shape[1] != faiss_index.d:
                    print(f"Warning: Batch feature dimension {batch_features.shape[1]} does not match FAISS index dimension {faiss_index.d}")
                else:
                    faiss_index.add(batch_features)

# Search similar FAISS
def search_faiss(query_embedding, faiss_index, k=5):
    D, I = faiss_index.search(query_embedding, k)  # D: khoảng cách, I: chỉ số ảnh tương tự
    return D, I

file_path = r'/hdd/sv10/svc/docker-svc/jupyter/data/Team_DC/long/Retrival/retrieval_images/xit_khoang_final.xlsx'
df = pd.read_excel(file_path, sheet_name="Sheet1")
df = df.head(20)

df['url_thumbnail'] = df['url_thumbnail'].apply(str)

dim = 1280  # Kích thước vector embedding cho CLIP-ViT-bigG-14 (1280)
faiss_index = faiss.IndexFlatL2(dim)  # Index sử dụng metric L2 (Euclidean distance)

res = faiss.StandardGpuResources()  # Tạo GPU resources
gpu_index = faiss.index_cpu_to_gpu(res, 0, faiss_index)  # Chuyển index sang GPU

add_embeddings_to_faiss(df['url_thumbnail'].tolist(), gpu_index, batch_size=8)

query_url = "https://cf.shopee.vn/file/bd273309b222e7c3f6ad46b06ff37c0a"
query_image = preprocess_image(query_url)
if query_image:
    inputs = clip_processor(images=query_image, return_tensors="pt", padding=True)
    with torch.no_grad():
        query_embedding = clip_model.get_image_features(**inputs).cpu().numpy().reshape(1, -1)

    D, I = search_faiss(query_embedding, gpu_index, k=5)

    print(f"Top 5 similar images:")
    for i in range(len(I[0])):
        print(f"Index: {I[0][i]}, Distance: {D[0][i]}")
        print(f"URL: {df.iloc[I[0][i]]['url_thumbnail']}")
else:
    print("Query image could not be processed.")
