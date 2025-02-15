import numpy as np
import pandas as pd
import faiss
import torch
from transformers import CLIPProcessor, CLIPModel
import requests
from PIL import Image
import io

# Khởi tạo mô hình và processor CLIP từ LAION
clip_model = CLIPModel.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
clip_processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")

# Hàm để tải và xử lý ảnh từ URL
def preprocess_image(image_path):
    try:
        response = requests.get(image_path)
        image = Image.open(io.BytesIO(response.content))  # Mở ảnh từ byte
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

# Hàm để trích xuất đặc trưng từ ảnh (embedding) theo batch và thêm vào FAISS
def add_embeddings_to_faiss(urls, faiss_index, batch_size=8):
    # Chia các URL thành các batch
    for i in range(0, len(urls), batch_size):
        batch_urls = urls[i:i+batch_size]
        images = []
        
        # Tải và xử lý các ảnh trong batch
        for url in batch_urls:
            image = preprocess_image(url)
            if image is not None:
                images.append(image)
        
        # Nếu batch không rỗng, tiến hành tạo embeddings và thêm vào FAISS
        if images:
            # Xử lý batch ảnh và tạo embeddings
            inputs = clip_processor(images=images, return_tensors="pt", padding=True)
            with torch.no_grad():
                # Trích xuất đặc trưng từ batch ảnh
                batch_features = clip_model.get_image_features(**inputs).cpu().numpy()
                
                # Kiểm tra kích thước của batch_features
                if batch_features.shape[1] != faiss_index.d:
                    print(f"Warning: Batch feature dimension {batch_features.shape[1]} does not match FAISS index dimension {faiss_index.d}")
                else:
                    # Thêm các embedding vào FAISS index
                    faiss_index.add(batch_features)

# Hàm tìm kiếm ảnh tương tự trong FAISS
def search_faiss(query_embedding, faiss_index, k=5):
    # Sử dụng FAISS để tìm kiếm các ảnh tương tự
    D, I = faiss_index.search(query_embedding, k)  # D: khoảng cách, I: chỉ số ảnh tương tự
    return D, I

# Đọc dữ liệu từ file Excel
file_path = r'/hdd/sv10/svc/docker-svc/jupyter/data/Team_DC/long/Retrival/retrieval_images/xit_khoang_final.xlsx'
df = pd.read_excel(file_path, sheet_name="Sheet1")
df = df.head(20)

# Chuẩn hóa dữ liệu, nếu cần
df['url_thumbnail'] = df['url_thumbnail'].apply(str)

# Cập nhật kích thước vector embedding từ CLIP (1280 thay vì 768)
dim = 1280  # Kích thước vector embedding cho CLIP-ViT-bigG-14 (1280)
faiss_index = faiss.IndexFlatL2(dim)  # Index sử dụng metric L2 (Euclidean distance)

# Đưa index lên GPU (nếu bạn có GPU)
res = faiss.StandardGpuResources()  # Tạo GPU resources
gpu_index = faiss.index_cpu_to_gpu(res, 0, faiss_index)  # Chuyển index sang GPU

# Lấy và thêm embeddings trực tiếp vào FAISS index
add_embeddings_to_faiss(df['url_thumbnail'].tolist(), gpu_index, batch_size=8)

# Tìm kiếm ảnh tương tự qua URL thumbnail
query_url = "https://cf.shopee.vn/file/bd273309b222e7c3f6ad46b06ff37c0a"
query_image = preprocess_image(query_url)
if query_image:
    inputs = clip_processor(images=query_image, return_tensors="pt", padding=True)
    with torch.no_grad():
        query_embedding = clip_model.get_image_features(**inputs).cpu().numpy().reshape(1, -1)

    # Thực hiện tìm kiếm trên FAISS
    D, I = search_faiss(query_embedding, gpu_index, k=5)

    # In ra các kết quả tìm kiếm
    print(f"Top 5 similar images:")
    for i in range(len(I[0])):
        print(f"Index: {I[0][i]}, Distance: {D[0][i]}")
        print(f"URL: {df.iloc[I[0][i]]['url_thumbnail']}")
else:
    print("Query image could not be processed.")
