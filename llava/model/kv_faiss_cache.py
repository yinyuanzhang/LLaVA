# kv_faiss_cache.py - 专门用于CacheBlend KV cache存储的Faiss缓存

import faiss
import torch
import numpy as np
import os
from typing import Optional, Tuple, List, Dict

class KVFaissCache:
    def __init__(self, key_dim: int, cache_file_path: str = "cacheblend_kv_cache", cache_type: str = "background"):
        """
        Args:
            key_dim (int): 用于Faiss搜索的特征键的维度
            cache_file_path (str): 缓存文件的基础路径
            cache_type (str): 缓存类型 ("background" 或 "foreground")
        """
        self.key_dim = key_dim
        self.cache_type = cache_type
        self.cache_file_path = cache_file_path
        
        # 为不同类型创建不同的文件路径
        cache_dir = os.path.join(cache_file_path, cache_type)
        self.index_file = os.path.join(cache_dir, "kv_cache.index")
        self.data_file = os.path.join(cache_dir, "kv_cache_data.pt")

        if os.path.exists(self.index_file) and os.path.exists(self.data_file):
            print(f"Loading existing {cache_type} KV cache from {self.index_file}")
            self.index = faiss.read_index(self.index_file)
            self.cached_data = torch.load(self.data_file)
        else:
            print(f"Initializing new {cache_type} KV cache.")
            self.index = faiss.IndexFlatL2(self.key_dim)  # L2距离索引
            self.cached_data = []
            os.makedirs(cache_dir, exist_ok=True)

    def add_kv_cache(self, query_key: torch.Tensor, kv_cache_list: List[Dict], num_tokens: int, position_ids: torch.Tensor):
        """
        向缓存中添加新的KV cache数据
        
        Args:
            query_key (torch.Tensor): [1, key_dim] 归一化的查询特征
            kv_cache_list (List[Dict]): 每层的KV cache，格式: [{"key": tensor, "value": tensor}, ...]
            num_tokens (int): 该patch的token数量
        """
        if num_tokens == 0:
            print(f"Warning: Attempted to add empty {self.cache_type} KV cache. Skipping.")
            return
            
        # 验证kv_cache_list的有效性
        valid_layers = [kv for kv in kv_cache_list if kv is not None]
        if len(valid_layers) == 0:
            print(f"Warning: No valid KV cache layers for {self.cache_type}. Skipping.")
            return

        # 添加特征到索引
        key_np = query_key.detach().cpu().numpy().astype('float32')
        self.index.add(key_np)
        
        # 存储KV cache数据
        cache_entry = {
            'kv_cache': [kv.copy() if kv is not None else None for kv in kv_cache_list],  # 深拷贝
            'num_tokens': num_tokens,
            'position_ids': position_ids.clone().cpu(),  # 存储position_ids
            'timestamp': torch.tensor(0.0)  # 可用于LRU等策略
        }
        self.cached_data.append(cache_entry)
        
        print(f"Added new {self.cache_type} KV cache. Tokens: {num_tokens}, Total items: {self.index.ntotal}")

    def search_kv_cache(self, query_key: torch.Tensor, similarity_threshold: float) -> Tuple[Optional[List[Dict]], Optional[int], Optional[torch.Tensor]]:
        """
        在缓存中搜索相似的KV cache
        
        Args:
            query_key (torch.Tensor): [1, key_dim] 归一化的查询特征
            similarity_threshold (float): 相似度阈值，距离小于此值视为命中
            
        Returns:
            Tuple[Optional[List[Dict]], Optional[int]]: (匹配的KV cache列表, token数量) 或 (None, None)
        """
        if self.index.ntotal == 0:
            return None, None, None

        query_key_np = query_key.detach().cpu().numpy().astype('float32')
        distances, indices = self.index.search(query_key_np, k=1)  # 搜索最近的1个
        
        best_distance = distances[0][0]
        best_index = indices[0][0]

        if best_distance <= similarity_threshold:
            print(f"{self.cache_type.capitalize()} cache HIT! Distance: {best_distance:.4f} (Threshold: {similarity_threshold})")
            cached_item = self.cached_data[best_index]
            return cached_item['kv_cache'], cached_item['num_tokens'], cached_item.get('position_ids')
        else:
            print(f"{self.cache_type.capitalize()} cache MISS! Distance: {best_distance:.4f} (Threshold: {similarity_threshold})")
            return None, None, None

    def save(self):
        """保存索引和缓存数据到文件"""
        if self.index.ntotal > 0:
            print(f"Saving {self.cache_type} KV cache to {self.cache_file_path}...")
            faiss.write_index(self.index, self.index_file)
            torch.save(self.cached_data, self.data_file)
            print(f"{self.cache_type.capitalize()} KV cache saved successfully.")

    def get_stats(self) -> Dict:
        """获取缓存统计信息"""
        return {
            "cache_type": self.cache_type,
            "total_entries": self.index.ntotal,
            "key_dimension": self.key_dim,
            "cache_size_mb": len(self.cached_data) * 0.1 if self.cached_data else 0  # 粗略估计
        }


class CacheBlendKVController:
    """CacheBlend KV缓存控制器，管理背景和前景的独立缓存"""
    
    def __init__(self, key_dim: int, cache_base_path: str = "cacheblend_kv_cache"):
        self.key_dim = key_dim
        self.cache_base_path = cache_base_path
        
        # 创建背景和前景的独立缓存
        self.bg_cache = KVFaissCache(key_dim, cache_base_path, "background")
        self.fg_cache = KVFaissCache(key_dim, cache_base_path, "foreground")
        
        print(f"CacheBlend KV Controller initialized with key_dim={key_dim}")

    def add_patch_cache(self, bg_feature: Optional[torch.Tensor], fg_feature: Optional[torch.Tensor], 
                       bg_kv_cache: Optional[List[Dict]], fg_kv_cache: Optional[List[Dict]],
                       bg_tokens: int, fg_tokens: int,
                       bg_position_ids: Optional[torch.Tensor] = None, 
                       fg_position_ids: Optional[torch.Tensor] = None):
        """
        添加背景和前景的patch缓存
        
        Args:
            bg_feature (Optional[torch.Tensor]): 背景的归一化特征 [1, D]
            fg_feature (Optional[torch.Tensor]): 前景的归一化特征 [1, D]
            bg_kv_cache (Optional[List[Dict]]): 背景的KV cache
            fg_kv_cache (Optional[List[Dict]]): 前景的KV cache
            bg_tokens (int): 背景token数量
            fg_tokens (int): 前景token数量
        """
        if bg_feature is not None and bg_kv_cache is not None and bg_tokens > 0:
            self.bg_cache.add_kv_cache(bg_feature, bg_kv_cache, bg_tokens, bg_position_ids)
            
        if fg_feature is not None and fg_kv_cache is not None and fg_tokens > 0:
            self.fg_cache.add_kv_cache(fg_feature, fg_kv_cache, fg_tokens, fg_position_ids)

    def search_patch_cache(self, bg_feature: Optional[torch.Tensor], fg_feature: Optional[torch.Tensor], 
                          similarity_threshold: float) -> Tuple[
                              Optional[List[Dict]], Optional[int], Optional[torch.Tensor], 
                              Optional[List[Dict]], Optional[int], Optional[torch.Tensor]
                          ]:
        """
        搜索背景和前景的patch缓存
        
        Args:
            bg_feature (Optional[torch.Tensor]): 背景查询特征
            fg_feature (Optional[torch.Tensor]): 前景查询特征  
            similarity_threshold (float): 相似度阈值
            
        Returns:
            Tuple: (bg_kv_cache, bg_tokens, fg_kv_cache, fg_tokens)
        """
        bg_kv_cache, bg_tokens, bg_pos_ids = None, None, None
        fg_kv_cache, fg_tokens, fg_pos_ids = None, None, None
        
        if bg_feature is not None:
            bg_kv_cache, bg_tokens, bg_pos_ids = self.bg_cache.search_kv_cache(bg_feature, similarity_threshold)
            
        if fg_feature is not None:
            fg_kv_cache, fg_tokens, fg_pos_ids = self.fg_cache.search_kv_cache(fg_feature, similarity_threshold)
            
        return bg_kv_cache, bg_tokens, bg_pos_ids, fg_kv_cache, fg_tokens, fg_pos_ids

    def save_all(self):
        """保存所有缓存"""
        self.bg_cache.save()
        self.fg_cache.save()

    def get_stats(self) -> Dict:
        """获取整体缓存统计信息"""
        bg_stats = self.bg_cache.get_stats()
        fg_stats = self.fg_cache.get_stats()
        
        return {
            "background": bg_stats,
            "foreground": fg_stats,
            "total_entries": bg_stats["total_entries"] + fg_stats["total_entries"]
        }