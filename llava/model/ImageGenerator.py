import os
import torch
import faiss
import numpy as np
import json
import torch.nn.functional as F

class BackgroundFeatureCache:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(BackgroundFeatureCache, cls).__new__(cls)
        return cls._instance

    def __init__(self, cache_dir="~/background_feature_cache", faiss_key_dim=576, device="cpu"):
        if self._initialized:
            return

        self.cache_dir = os.path.expanduser(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.device = device
        self.faiss_index = None
        self.feature_storage_map = {}
        self.next_faiss_id = 0
        self.faiss_dim = faiss_key_dim

        self.faiss_index_path = os.path.join(self.cache_dir, "background_faiss_index.bin")
        self.metadata_path = os.path.join(self.cache_dir, "cache_metadata.json")

        self._load_cache_and_index()
        self._initialized = True

    def _load_cache_and_index(self):
        """从磁盘加载已有的缓存和Faiss索引和元数据。"""
        # --- 尝试加载现有索引 ---
        loaded_index = None
        if os.path.exists(self.faiss_index_path) and os.path.exists(self.metadata_path):
            print(f"Loading background feature cache and Faiss index from {self.cache_dir}...")
            try:
                temp_index = faiss.read_index(self.faiss_index_path)
                # 关键检查: 确保加载的索引是预期的 IndexIDMap 类型且维度匹配其内部索引
                # 注意：这里检查的是 IndexIDMap，因为我们现在总是用它封装
                # 并且要检查其内部的 Index 类型是否是 IndexFlatL2
                if isinstance(temp_index, faiss.IndexIDMap) and \
                   isinstance(temp_index.index, faiss.Index) and \
                   temp_index.index.d == self.faiss_dim:
                    loaded_index = temp_index
                    print(f"Faiss IndexIDMap(IndexFlatL2) loaded successfully with dimension: {loaded_index.index.d}.")
                else:
                    print(f"Warning: Loaded Faiss index is not of expected type IndexIDMap(IndexFlatL2) or dimension mismatch. Rebuilding index.")
                    # 打印更详细的加载索引类型
                    if loaded_index is not None:
                        print(f"Loaded index type: {type(temp_index).__name__}")
                        if isinstance(temp_index, faiss.IndexIDMap):
                            print(f"Inner index type: {type(temp_index.index).__name__}, dim: {temp_index.index.d}")
                    loaded_index = None # 加载失败或不匹配，重置为None以触发重新初始化

            except Exception as e:
                print(f"Failed to load Faiss index: {e}. Starting anew.")
                loaded_index = None

        # --- 根据加载结果初始化或重新初始化 faiss_index ---
        if loaded_index is not None:
            self.faiss_index = loaded_index
        else:
            print(f"Initializing new Faiss IndexIDMap(IndexFlatL2) with dimension: {self.faiss_dim}.")
            # --- 核心修改：使用 IndexIDMap 封装 IndexFlatL2 ---
            self.faiss_index = faiss.IndexIDMap(faiss.IndexFlatL2(self.faiss_dim))

        # --- 加载元数据 ---
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
                self.next_faiss_id = metadata.get("next_faiss_id", 0)
                # 重要: 检查元数据中的 faiss_dim 是否与当前配置一致
                if metadata.get("faiss_dim") != self.faiss_dim:
                    print("Warning: Metadata faiss_dim mismatch. Clearing feature_storage_map and resetting IDs.")
                    self.feature_storage_map = {}
                    self.next_faiss_id = 0 # 如果维度不匹配，重置 ID
                else:
                    self.feature_storage_map = metadata.get("feature_storage_map", {})

            # 清理 feature_storage_map 中指向不存在文件的条目
            for faiss_id_str, filename in list(self.feature_storage_map.items()):
                file_path = os.path.join(self.cache_dir, filename)
                if not os.path.exists(file_path):
                    print(f"Warning: Cache value file {file_path} not found, removing from index metadata.")
                    del self.feature_storage_map[faiss_id_str]

        print(f"Loaded {len(self.feature_storage_map)} background feature metadata entries for values.")


    def _save_cache_and_index(self):
        """保存当前缓存元数据和Faiss索引到磁盘。"""
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, self.faiss_index_path)

            metadata = {
                "next_faiss_id": self.next_faiss_id,
                "faiss_dim": self.faiss_dim,
                "feature_storage_map": self.feature_storage_map
            }
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f)
            print(f"Successfully saved Faiss index and cache metadata to {self.cache_dir}.")

    def add_feature(self, key_feature: torch.Tensor, value_feature: torch.Tensor) -> int:
        """
        将固定维度的Key特征添加到Faiss索引，并将可变维度的Value特征保存到本地。
        Args:
            key_feature (torch.Tensor): [1, faiss_key_dim] 形状的固定维度特征向量（已归一化）。
            value_feature (torch.Tensor): [1, N_actual_bg_tokens, embedding_dim] 形状的原始背景特征 (已剥离无效token)。
        Returns:
            int: 添加的特征在Faiss中的ID。
        """
        key_vector_np = key_feature.cpu().numpy().reshape(1, -1)

        # if key_vector_np.shape[1] != self.faiss_dim:
        #     raise ValueError(f"Key feature dimension {key_vector_np.shape[1]} mismatch with Faiss index dimension {self.faiss_dim}. Expected {self.faiss_dim}.")

        self.faiss_index.add_with_ids(key_vector_np, np.array([self.next_faiss_id], dtype=np.int64))

        value_filename = f"bg_value_feature_{self.next_faiss_id}.pt"
        value_filepath = os.path.join(self.cache_dir, value_filename)
        torch.save(value_feature.cpu(), value_filepath) # 保存剥离无效token后的背景特征

        self.feature_storage_map[str(self.next_faiss_id)] = value_filename
        self.next_faiss_id += 1
        return self.next_faiss_id - 1 # 返回当前添加的ID

    def search_feature(self, query_key_feature: torch.Tensor, top_k=1, distance_threshold=0.01) -> tuple[torch.Tensor | None, float | None]:
        """
        在Faiss索引中搜索最相似的Key特征，并返回对应的Value特征。
        Args:
            query_key_feature (torch.Tensor): [1, faiss_key_dim] 形状的查询Key向量（已归一化）。
            top_k (int): 返回最相似的K个结果 (通常是1)。
            distance_threshold (float): L2距离阈值。
        Returns:
            tuple: (相似背景特征Value张量, 相似度距离) 或 (None, None) 如果未找到。
                   Value张量形状为 [1, N_actual_bg_tokens, embedding_dim] (已剥离无效token)。
        """
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            print("Faiss index is empty or not initialized.")
            return None, None

        query_vector_np = query_key_feature.cpu().numpy().reshape(1, -1)

        if query_vector_np.shape[1] != self.faiss_dim:
            print(f"Warning: Query key feature dimension {query_vector_np.shape[1]} mismatch with Faiss index dimension {self.faiss_dim}. Cannot search. Expected {self.faiss_dim}.")
            return None, None

        distances, indices = self.faiss_index.search(query_vector_np, top_k)

        print(f"Nearest neighbor distance found: {distances[0][0]:.6f} (Threshold: {distance_threshold:.6f})")
        # 搜索结果的 ID 也需要从 IndexIDMap 封装的索引中获取
        # IndexIDMap.search() 的 indices 返回的直接就是原始 ID
        if distances[0][0] <= distance_threshold:
            faiss_id = indices[0][0]
            value_filename = self.feature_storage_map.get(str(faiss_id))
            if value_filename:
                value_filepath = os.path.join(self.cache_dir, value_filename)
                if os.path.exists(value_filepath):
                    cached_value_feature = torch.load(value_filepath, map_location=query_key_feature.device)
                    print(f"Found similar background (ID: {faiss_id}), distance: {distances[0][0]:.6f}. Reusing cached value feature.")
                    return cached_value_feature, distances[0][0]
                else:
                    print(f"Warning: Cached value file {value_filepath} not found for ID {faiss_id}, unable to reuse.")
        print("No sufficiently similar background found, or cache file missing. Will compute new features.")
        return None, None

    def close(self):
        """在程序结束时调用，确保所有缓存都被保存。"""
        self._save_cache_and_index()
        print("Background feature cache system closed.")