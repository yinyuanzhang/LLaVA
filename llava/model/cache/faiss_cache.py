# faiss_cache.py

import faiss
import torch
import numpy as np
import os

class FaissCache:
    def __init__(self, key_dim: int, cache_file_path: str = "background_cache.faiss"):
        """
        Args:
            key_dim (int): 用于Faiss搜索的特征键的维度.
            cache_file_path (str): Faiss索引和缓存数据的保存路径.
        """
        self.key_dim = key_dim
        self.cache_file_path = cache_file_path
        self.index_file = os.path.join(self.cache_file_path, "cache.index")
        self.data_file = os.path.join(self.cache_file_path, "cache_data.pt")

        if os.path.exists(self.index_file):
            print(f"Loading existing Faiss index from {self.index_file}")
            self.index = faiss.read_index(self.index_file)
            self.cached_data = torch.load(self.data_file)
        else:
            print("Initializing new Faiss index.")
            self.index = faiss.IndexFlatL2(self.key_dim)
            # 存储完整的背景逻辑Token和其对应的position_ids
            self.cached_data = []
            os.makedirs(self.cache_file_path, exist_ok=True)

    def add_feature(self, key: torch.Tensor, bg_tokens: torch.Tensor, bg_position_ids: torch.Tensor):
        """
        向缓存中添加新的背景特征及其精确的position_ids.
        
        Args:
            key (torch.Tensor): [1, key_dim] 的搜索键.
            bg_tokens (torch.Tensor): [N, D] 的背景逻辑Token特征.
            bg_position_ids (torch.Tensor): [3, N] 的背景逻辑Token对应的position_ids切片.
        """
        if bg_tokens.shape[0] == 0:
            print("Warning: Attempted to add empty background tokens to cache. Skipping.")
            return

        key_np = key.detach().cpu().numpy().astype('float32')
        self.index.add(key_np)
        
        # 将逻辑Token和对应的position_ids一起保存
        self.cached_data.append({
            'tokens': bg_tokens.detach().cpu(),
            'position_ids': bg_position_ids.detach().cpu() # 修正：存储position_ids
        })
        print(f"Added new background to cache. Total items: {self.index.ntotal}")

    def search_feature(self, query_key: torch.Tensor, distance_threshold: float) -> tuple:
        """
        在缓存中搜索相似的背景特征.
        
        Args:
            query_key (torch.Tensor): [1, key_dim] 的查询键.
            distance_threshold (float): 距离阈值，小于此值视为命中.
        Returns:
            一个元组 (bg_tokens, bg_position_ids) 或 (None, None)
        """
        if self.index.ntotal == 0:
            return None, None

        query_key_np = query_key.detach().cpu().numpy().astype('float32')
        distances, indices = self.index.search(query_key_np, k=2)
        
        best_distance = distances[0][0]
        best_index = indices[0][0]

        if best_distance < distance_threshold:
            print(f"Cache HIT! Distance: {best_distance:.4f} (Threshold: {distance_threshold})")
            cached_item = self.cached_data[best_index]
            # 修正：返回tokens和position_ids
            return cached_item['tokens'], cached_item['position_ids']
        else:
            print(f"Cache MISS! Min Distance: {best_distance:.4f} (Threshold: {distance_threshold})")
            return None, None
            
    def save(self):
        """保存索引和缓存数据到文件"""
        if self.index.ntotal > 0:
            print(f"Saving cache to {self.cache_file_path}...")
            faiss.write_index(self.index, self.index_file)
            torch.save(self.cached_data, self.data_file)