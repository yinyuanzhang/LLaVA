from transformers import LlavaForConditionalGeneration, CLIPImageProcessor, LlamaTokenizer
from PIL import Image  # 导入PIL的Image模块
import torch
import os
import numpy as np
import faiss
import time

# 加载LLAVA模型和预处理器
model_name = "liuhaotian/llava-v1.5-7b"
model = LlavaForConditionalGeneration.from_pretrained(model_name)
processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

def get_image_embedding(image_path):
    image = Image.open(image_path)  # 使用PIL打开图片
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    return outputs.cpu().numpy()

def measure_retrieval_time(index, query_embeddings):
    start_time = time.time()
    D, I = index.search(query_embeddings, k=5)  # 检索最近邻的k个结果
    end_time = time.time()
    return end_time - start_time

if __name__ == "__main__":
    print("begin ")
    
    # 图片目录
    image_dir = "/users/zyy/autodl-tmp/playground/data/coco/train2017"
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]

    # 获取所有图片的embeddings
    embeddings = []
    for img_file in image_files[:1000]:  # 只取前1000张图片作为例子
        embedding = get_image_embedding(img_file)
        embeddings.append(embedding)
    embeddings = np.concatenate(embeddings, axis=0)

    # 定义索引参数
    embedding_dim = embeddings.shape[1]  # 假设每个embedding的维度
    nlist = 100  # number of cluster centers/centroids
    quantizer = faiss.IndexFlatL2(embedding_dim)  # the quantizer is a flat index
    index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist, faiss.METRIC_L2)

    # 训练索引
    index.train(embeddings)

    # 添加图片embedding到索引中
    index.add(embeddings)

    # 测试不同规模的数据集
    for size in [10, 100, 1000]:
        subset_embeddings = embeddings[:size]
        
        # 创建查询向量，这里我们用subset中的第一个向量作为查询
        query_embedding = subset_embeddings[0].reshape(1, embedding_dim)
        
        retrieval_time = measure_retrieval_time(index, query_embedding)
        print(f"Size {size} images: Retrieval time = {retrieval_time:.4f} seconds")  # 修正了打印语句中的多余点
