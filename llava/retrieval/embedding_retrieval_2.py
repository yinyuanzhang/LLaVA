from transformers import CLIPImageProcessor, CLIPVisionModel
import torch
import torch.nn as nn
from PIL import Image
import os
import numpy as np
import faiss
import time

# 加载预训练的CLIP视觉模型以获取配置
clip_model_name = "openai/clip-vit-large-patch14-336"
vision_model = CLIPVisionModel.from_pretrained(clip_model_name)
config = vision_model.config

# 提取配置参数
patch_size = config.patch_size
in_channels = config.num_channels
embed_dim = config.hidden_size

# 自定义 patch embedding 层
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_channels, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.embedding_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

    def forward(self, images):
        patches = self.embedding_layer(images)  # shape: [B, C, H/p, W/p]
        patch_embeds = patches.flatten(2).transpose(1, 2)  # shape: [B, num_patches, embed_dim]
        return patch_embeds

# 初始化 patch embedding 层
patch_embed = PatchEmbedding(patch_size, in_channels, embed_dim)

# 图像预处理
processor = CLIPImageProcessor.from_pretrained(clip_model_name)

def get_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs['pixel_values']  # shape: [1, 3, 336, 336]

    with torch.no_grad():
        patch_embeddings = patch_embed(pixel_values)  # shape: [1, num_patches, embed_dim]

    return patch_embeddings.squeeze(0).cpu().numpy()  # shape: [num_patches, embed_dim]

def measure_retrieval_time(index, query_embeddings):
    start_time = time.time()
    D, I = index.search(query_embeddings, k=5)
    print("检索到的距离矩阵（D）：")
    print(D)
    print("检索到的索引矩阵（I）：")
    print(I)
    end_time = time.time()
    return end_time - start_time

if __name__ == "__main__":
    # 可选项：'flat' 或 'avg'
    embedding_type = 'flat'

    # 图片目录
    image_dir = "/users/zyy/autodl-tmp/playground/data/coco/train2017"
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]

    # 获取前1000张图片的 embeddings
    train_embeddings = []
    for img_file in image_files[:1000]:  # 训练集：前1000张图片
        embedding = get_image_embedding(img_file)  # shape: [num_patches, embed_dim]
        if embedding_type == 'flat':
            train_embeddings.append(embedding.flatten())
        elif embedding_type == 'avg':
            train_embeddings.append(embedding.mean(axis=0))

    embeddings_np = np.array(train_embeddings).astype('float32')

    # 使用第1001张图作为查询向量（不加入索引）
    if len(image_files) >= 1001:
        query_img_path = image_files[1000]
        query_patch_emb = get_image_embedding(query_img_path)
        if embedding_type == 'flat':
            query_emb = query_patch_emb.flatten().reshape(1, -1)
        elif embedding_type == 'avg':
            query_emb = query_patch_emb.mean(axis=0).reshape(1, -1)
        query_emb = query_emb.astype('float32')
    else:
        raise IndexError("Not enough images to use the 1001st as query.")

    # 测试不同规模的数据集
    for size in [10, 100, 1000]:
        print(f"\nTesting with {size} images...")

        # 步骤一：提取当前规模的嵌入子集
        subset_embeddings = embeddings_np[:size]

        # 步骤二：为当前规模重建 Faiss 索引
        embedding_dim = subset_embeddings.shape[1]
        nlist = min(100, size // 5)  # 确保 nlist 不超过训练样本数
        quantizer = faiss.IndexFlatL2(embedding_dim)
        index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist, faiss.METRIC_L2)

        # 如果 size 太小，可能无法训练 IndexIVFFlat，需要跳过或换用 IndexFlat
        if size > nlist * 2:
            index.train(subset_embeddings)
        else:
            print(f"Warning: Not enough data to train IVF index with size={size}, using Flat index instead.")
            index = faiss.IndexFlatL2(embedding_dim)

        index.add(subset_embeddings)

        # 步骤三：进行检索并测量时间
        retrieval_time = measure_retrieval_time(index, query_emb)
        print(f"Size {size} images: Retrieval time = {retrieval_time:.4f} seconds")