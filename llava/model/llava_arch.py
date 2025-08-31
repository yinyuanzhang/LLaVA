#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_BACKGROUND_OBJECT_TOKEN


from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from llava.mm_utils import get_anyres_image_grid_shape
import copy
from transformers import AutoTokenizer
from .ImageGenerator import BackgroundFeatureCache
from .CacheStatisticsCollector import CacheStatisticsCollector
import torch.nn.functional as F

class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)
        self.method_type = getattr(config, 'method_type', 'native')
        self.cache_mode = getattr(config, 'cache_mode', 'read-only')

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )
        
        # self.init_build_prefusion=False
        # self.load_prefusion_layers=False
        # if self.image_cache:
        #     self.build_prefusion(config)
        #     self.init_build_prefusion=True
            # self.load_prefusion()


        # 初始化缓存系统（如果支持缓存的模式）
        if self.method_type in ["segmentation-cache", "fuzzy-cache"] and self.cache_mode in ["write-only", "read-load"]:
            try:
                import faiss
                import os
                from llava.mm_utils import get_model_name_from_path
                
                dataset_name = getattr(config, 'dataset', 'default_dataset')
                model_path = getattr(config, 'model_path', 'unknown_model')
                model_name = get_model_name_from_path(model_path)
                print(f"Final dataset_name: {dataset_name}")
                print(f"Model path: {model_path}")
                print(f"Model name: {model_name}")
                print(f"Method type: {self.method_type}")
                print(f"Cache mode: {self.cache_mode}")
                
                base_cache_path = "faiss"
                dynamic_cache_path = os.path.join(base_cache_path, self.method_type, model_name, dataset_name)
                
                self.background_cache = BackgroundFeatureCache(
                    cache_dir=dynamic_cache_path,
                    faiss_key_dim = 4096,
                    device=self.device # 缓存加载时指定设备
                )
                print(f"Background caching system initialized at: {dynamic_cache_path}")
            except ImportError:
                print("Faiss not installed. Background caching will be disabled.")
                self.background_cache = None

        # 初始化统计收集器（如果需要）
        self.stats_collector = None
        if self.method_type in ["segmentation-cache", "fuzzy-cache"] and self.cache_mode in ["read-only", "read-load"]:
            self.stats_collector = CacheStatisticsCollector()
            print(f"Cache statistics collector initialized for {self.cache_mode} mode.")



    def build_prefusion(self,config):
        self.prefusion_layer_num= getattr(config,'prefusion_layer_num', 4)

        self.prefusion_layers=nn.ModuleList([LlamaDecoderLayer(self.base_model.config,layer_idx=i) for i in range(self.prefusion_layer_num)])
        if self.base_model.device.type != 'meta':
            self.prefusion_layers.to(self.base_model.device).to(self.base_model.dtype)
            


    def load_prefusion(self):
        for i in range(self.prefusion_layer_num):
            # 获取原模型对应层的权重
            src_layer = self.base_model.layers[i]
            # 加载到新层
            self.prefusion_layers[i].load_state_dict(src_layer.state_dict())

        for p in self.prefusion_layers.parameters():
            p.requires_grad = True

        self.load_prefusion_layers=True    

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    # def get_background_object_vision_tower(self):
    #     background_object_vision_tower = getattr(self, 'background_object_vision_tower', None)
    #     if background_object_vision_tower is None and self.config.image_cache:
    #         self.config.mm_vision_select_layer = -2
    #         background_object_vision_tower = build_vision_tower(self.config)            
    #         self.background_object_vision_tower = background_object_vision_tower
    #     if type(background_object_vision_tower) is list:
    #         background_object_vision_tower = background_object_vision_tower[0]
    #     return background_object_vision_tower
        
    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        # if model_args.image_cache and self.get_background_object_vision_tower() is None:
        #     background_object_vision_tower = build_vision_tower(model_args)            
        #     self.background_object_vision_tower = background_object_vision_tower

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

        # if not self.load_prefusion_layers and model_args.image_cache:
        #     if getattr(model_args, 'pretrain_prefusion', None):
        #         model_weights = torch.load(model_args.pretrain_prefusion, map_location='cpu')
        #         for name, module in self.prefusion_layers.named_parameters():
        #             module.data=model_weights[f"{name}"].data.type_as(module.data)
        #             module.requires_grad = True
        #         print(f"load pretrain_prefusion from {model_args.pretrain_prefusion}")
        #         self.load_prefusion_layers=True


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    # def _pad_or_truncate_tokens(self, tokens, target_len, embedding_dim):
    #         """
    #         Helper function to pad or truncate tokens to a target length.
    #         Assumes tokens are [batch_size, current_len, embedding_dim]
    #         """
    #         batch_size, current_len, _ = tokens.shape
    #         if current_len == target_len:
    #             return tokens
    #         elif current_len < target_len:
    #             padding_needed = target_len - current_len
    #             # Pad with zeros
    #             padding = torch.zeros(batch_size, padding_needed, embedding_dim, device=tokens.device, dtype=tokens.dtype)
    #             return torch.cat([tokens, padding], dim=1)
    #         else: # current_len > target_len
    #             # Truncate
    #             return tokens[:, :target_len, :]

    # Helper function to pad/truncate tokens
    def _pad_or_truncate_tokens(self, tokens: torch.Tensor, target_length: int, embedding_dim: int):
        current_length = tokens.shape[1]
        if current_length == target_length:
            return tokens
        elif current_length < target_length:
            # Pad with zeros
            padding = torch.zeros(tokens.shape[0], target_length - current_length, embedding_dim, device=tokens.device)
            return torch.cat([tokens, padding], dim=1)
        else:
            # Truncate
            return tokens[:, :target_length, :]
                    
    def encode_background_and_object_images_back_cache(self, images, masks):
        """
        编码背景&目标图像，并优化缓存逻辑。
        此方法将统计**分离后的**背景和目标有效token数量，并交给统计类处理。
        同时，它会记录缓存的命中/未命中情况。
        """        
        self.cache_mode = getattr(self.get_model(), "cache_mode", "read-only")
        
        background_object_visual_tower = self.get_model().get_vision_tower().to(images.device)
        background_features, object_features, background_attention_mask, object_attention_mask = background_object_visual_tower(images, masks)
        background_features = self.get_model().mm_projector(background_features).to(images.device)
        object_features = self.get_model().mm_projector(object_features).to(object_features.device)

        batch_size, _, embedding_dim = background_features.shape

        valid_background_mask = background_attention_mask.bool().unsqueeze(-1).to(images.device)
        valid_object_mask = object_attention_mask.bool().unsqueeze(-1).to(images.device)
        
        current_bg_valid_count = valid_background_mask.squeeze(-1).sum(dim=1).item()
        current_obj_valid_count = valid_object_mask.squeeze(-1).sum(dim=1).item()

        assert (current_bg_valid_count + current_obj_valid_count == 576), "总有效token数应为576"


        # --- Pass these counts to the statistics collector if it exists and is read-only or read-load ---
        if self.cache_mode in ["read-only", "read-load"]:
            self.get_model().stats_collector.collect_stats(current_bg_valid_count, current_obj_valid_count)


        # --- 缓存逻辑开始 ---
        reused_background_features_final = None 
        
        bg_flat_calculated = background_features[
            valid_background_mask.repeat(1, 1, embedding_dim)
        ].reshape(batch_size, -1, embedding_dim)

        # ----------------------------------------------------
        # 缓存逻辑优化核心：根据 self.cache_mode 进行控制
        # ----------------------------------------------------
        is_cache_search_attempted = False # Flag to track if a search was performed
        cache_hit_status = False          # Flag to track if the search resulted in a hit

        if self.cache_mode == "read-only":
            pass # No caching, no print
            
        elif self.cache_mode != "write-only" and bg_flat_calculated.shape[1] == 0:
            print("背景有效token数为0，跳过背景缓存处理。")
            
        elif self.cache_mode == "write-only":
            print("缓存模式为 write-only，将计算的背景特征写入缓存。")
            with torch.no_grad():
                # bg_padded_for_key = self._pad_or_truncate_tokens(
                #     bg_flat_calculated,
                #     576,
                #     embedding_dim
                # ) 
                # faiss_key_feature = torch.max(bg_padded_for_key, dim=2).values
                # assert batch_size == 1, "Faiss cache logic assumes batch_size == 1"
                # query_key_for_search = faiss_key_feature.squeeze(0).unsqueeze(0)

                if bg_flat_calculated.dim() < 2:
                    # 根据实际情况调整unsqueeze，确保存在 num_tokens 维度
                    if bg_flat_calculated.dim() == 1: # (4096,) -> (1, 1, 4096)
                        bg_flat_calculated = bg_flat_calculated.unsqueeze(0).unsqueeze(0)
                    elif bg_flat_calculated.dim() == 0:
                        raise ValueError("bg_flat_calculated cannot be a scalar.")

                query_key_for_search = torch.mean(bg_flat_calculated, dim=1) 
                query_key_for_search = F.normalize(query_key_for_search, p=2, dim=1)  # L2归一化

                
                if self.get_model().background_cache:
                    self.get_model().background_cache.add_feature(query_key_for_search, bg_flat_calculated.clone().detach())
                    # Note: write-only doesn't count as a "search attempt" for hit rate
                else:
                    print("警告: 缓存系统未初始化，无法写入。")
        
        elif self.cache_mode == "read-load":
            is_cache_search_attempted = True # Mark that a search is being attempted
            with torch.no_grad():
                # bg_padded_for_key = self._pad_or_truncate_tokens(
                #     bg_flat_calculated,
                #     576,
                #     embedding_dim
                # )
                # faiss_key_feature = torch.max(bg_padded_for_key, dim=2).values
                # assert batch_size == 1, "Faiss cache logic assumes batch_size == 1"
                # query_key_for_search = faiss_key_feature.squeeze(0).unsqueeze(0)
                
                if bg_flat_calculated.dim() < 2:
                    # 根据实际情况调整unsqueeze，确保存在 num_tokens 维度
                    if bg_flat_calculated.dim() == 1: # (4096,) -> (1, 1, 4096)
                        bg_flat_calculated = bg_flat_calculated.unsqueeze(0).unsqueeze(0)
                    elif bg_flat_calculated.dim() == 0:
                        raise ValueError("bg_flat_calculated cannot be a scalar.")

                query_key_for_search = torch.mean(bg_flat_calculated, dim=1) 
                query_key_for_search = F.normalize(query_key_for_search, p=2, dim=1)  # L2归一化


                if self.get_model().background_cache:
                    reused_background_features_final, _ = self.get_model().background_cache.search_feature( # Capture hit status
                        query_key_for_search,
                        distance_threshold=0.1 # 归一化后使用更小的阈值
                    )
                    if reused_background_features_final is not None:
                        cache_hit_status = True 
                    else:
                        cache_hit_status = False     
                else:
                    print("警告: 缓存系统未初始化，无法搜索。")
                    reused_background_features_final = None
                    cache_hit_status = False # No cache, so no hit
                
            if reused_background_features_final is not None:
                print("使用缓存的背景特征。")
            else:
                print("缓存未命中，使用新计算的背景特征，不写入缓存。")
        else:
            print(f"警告：未知的缓存模式 '{self.cache_mode}'。将不进行任何缓存操作。")

        # --- Record cache outcome if a search was attempted ---
        if is_cache_search_attempted:
            self.get_model().stats_collector.record_cache_outcome(cache_hit_status)


        # --- 根据是否复用背景特征，决定最终使用的特征 ---
        if reused_background_features_final is not None:
            background_features_to_use_in_concat = reused_background_features_final.to(images.device)
            background_valid_final = torch.tensor([background_features_to_use_in_concat.shape[1]], device=self.device)
        else:
            background_features_to_use_in_concat = bg_flat_calculated
            background_valid_final = torch.tensor([current_bg_valid_count], device=self.device)

        bg_flat_final = background_features_to_use_in_concat 

        object_features_to_use_in_concat = object_features
        object_attention_mask_to_use_in_concat = object_attention_mask

        valid_object_mask_final = object_attention_mask_to_use_in_concat.bool().unsqueeze(-1).to(images.device)
        obj_flat_final = object_features_to_use_in_concat[valid_object_mask_final.squeeze(-1)].reshape(batch_size, -1, embedding_dim)
        object_valid_final = torch.tensor([current_obj_valid_count], device=self.device)

        total_valid_tokens = (background_valid_final + object_valid_final).item()
        if total_valid_tokens != 576:
            print(f"警告：总有效 token 数应为576，但实际得到 {total_valid_tokens}。")

        concatenated_features = torch.cat(
            (bg_flat_final.squeeze(0), obj_flat_final.squeeze(0)), 
            dim=0 
        ).unsqueeze(0) 
        
        return concatenated_features
    
    def encode_object_only_images(self, images, masks):
        """
        编码目标图像，仅使用目标部分，舍弃背景。
        基于 segmentation-cache 的逻辑，但只返回目标特征。
        """
        background_object_visual_tower = self.get_model().get_vision_tower().to(images.device)
        background_features, object_features, background_attention_mask, object_attention_mask = background_object_visual_tower(images, masks)
        
        # 只处理目标特征
        object_features = self.get_model().mm_projector(object_features).to(object_features.device)
        
        batch_size, _, embedding_dim = object_features.shape
        valid_object_mask = object_attention_mask.bool().unsqueeze(-1).to(images.device)
        
        # 提取有效的目标特征
        obj_flat = object_features[valid_object_mask.repeat(1, 1, embedding_dim)].reshape(batch_size, -1, embedding_dim)
        
        return obj_flat
    
    def encode_fuzzy_cache_images(self, images, masks):
        """
        模糊缓存模式：使用完整图像编码，但支持缓存复用。
        和 native 一样使用完整图片进行编码，但复用思路和 segmentation-cache 一样。
        """
        self.cache_mode = getattr(self.get_model(), "cache_mode", "read-only")
        
        # 使用原生方式编码完整图像
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        
        batch_size, num_tokens, embedding_dim = image_features.shape
        
        # 将完整的图像特征作为一个整体进行缓存处理
        reused_features_final = None
        
        # 缓存逻辑（类似 segmentation-cache，但处理完整特征）
        is_cache_search_attempted = False
        cache_hit_status = False
        
        if self.cache_mode == "read-only":
            pass # No caching
            
        elif self.cache_mode == "write-only":
            print("缓存模式为 write-only，将完整图像特征写入缓存。")
            with torch.no_grad():
                # 对 token 维度进行平均池化，得到 [batch_size, embedding_dim] 的特征作为检索 key
                query_key_for_search = torch.mean(image_features, dim=1)  # [1, 576, 4096] -> [1, 4096]
                query_key_for_search = F.normalize(query_key_for_search, p=2, dim=1)  # L2归一化
                
                if self.get_model().background_cache:
                    # 将完整的图像特征作为 value 存储
                    image_features_flattened = image_features.view(batch_size, -1, embedding_dim)
                    self.get_model().background_cache.add_feature(query_key_for_search, image_features_flattened.clone().detach())
                else:
                    print("警告: 缓存系统未初始化，无法写入。")
        
        elif self.cache_mode == "read-load":
            is_cache_search_attempted = True
            with torch.no_grad():
                query_key_for_search = torch.mean(image_features, dim=1)  # [1, 576, 4096] -> [1, 4096]
                query_key_for_search = F.normalize(query_key_for_search, p=2, dim=1)  # L2归一化
                
                if self.get_model().background_cache:
                    reused_features_final, _ = self.get_model().background_cache.search_feature(
                        query_key_for_search,
                        distance_threshold=0.1  # 归一化后使用更小的阈值
                    )
                    if reused_features_final is not None:
                        cache_hit_status = True
                        print("使用缓存的完整图像特征。")
                    else:
                        cache_hit_status = False
                        print("缓存未命中，使用新计算的完整图像特征。")
                else:
                    print("警告: 缓存系统未初始化，无法搜索。")
                    cache_hit_status = False
        
        # 记录缓存结果
        if is_cache_search_attempted:
            self.get_model().stats_collector.record_cache_outcome(cache_hit_status)
            
        # 决定最终使用的特征
        if reused_features_final is not None:
            return reused_features_final.to(images.device)
        else:
            return image_features
    

    # # 统一在此处进行封装，即均是调用 get_vision_tower
    # def encode_background_and_object_images(self, images, masks=None, inference_mode="both"):
    #     """
    #     编码背景&目标图像，并根据推理模式进行特征拼接。

    #     Args:
    #         images (torch.Tensor): 输入图像数据。
    #         masks (torch.Tensor, optional): (batch_size, 2) 张量，每行 [背景有效token数, 目标有效token数]。
    #                                          用于辅助 VisionTower 生成 attention mask。
    #         inference_mode (str): 推理模式，可选值：
    #                               - "both": 同时使用背景和目标特征（默认）。
    #                               - "background_only": 只使用背景特征。
    #                               - "object_only": 只使用目标特征。
    #                                 如果目标为空（object_valid为0），则即使选择"object_only"也回退到背景。
    #     Returns:
    #         torch.Tensor: 拼接后的图像特征。
    #     """        
    #     background_object_visual_tower = self.get_model().get_vision_tower().to(images.device)

    #     # 这一行保持不变，符合你的API要求
    #     background_features, object_features, background_attention_mask, object_attention_mask = background_object_visual_tower(images, masks)

    #     # ----------------------------------------------------------------------
    #     # 注意：这里的 background_features 和 object_features 是 VisionTower 的原始输出
    #     # 它可能包含填充的0或无效token。
    #     # mm_projector 应该作用在有效部分或者整个序列上，如果它处理的是整个序列，
    #     # 那么后续再根据 attention_mask 提取有效特征。
    #     # 我将假设 mm_projector 作用在完整的特征张量上。
    #     # ----------------------------------------------------------------------

    #     # 对原始输出的特征进行 mm_projector 投影
    #     background_features_projected = self.get_model().mm_projector(background_features)
    #     object_features_projected = self.get_model().mm_projector(object_features)

    #     batch_size, _, embedding_dim = background_features_projected.shape # embedding_dim 在投影后获取

    #     # --- 验证有效token数 (保持不变) ---
    #     # 确保 attention_mask 是 bool 类型并增加一个维度以进行广播
    #     valid_background_mask = background_attention_mask.bool().unsqueeze(-1)
    #     valid_object_mask = object_attention_mask.bool().unsqueeze(-1)

    #     # 计算每个样本的有效 token 数量 (保持不变)
    #     background_valid = valid_background_mask.squeeze(-1).sum(dim=1)
    #     object_valid = valid_object_mask.squeeze(-1).sum(dim=1)
        
    #     # 验证总有效token数（如果你的 VisionTower 输出总token数固定为576）
    #     # total_tokens_per_sample = background_features_projected.shape[1] # 获取 VisionTower 输出的token总数
    #     # assert torch.all(background_valid + object_valid == total_tokens_per_sample), "总有效token数不正确"
    #     # 假设原始代码的 576 限制是正确的，且 VisionTower 返回的 mask 涵盖了所有 token。
    #     assert torch.all(background_valid + object_valid == 576), "总有效token数应为576"


    #     # --- 向量化提取有效token (保持不变，但提取的是投影后的特征) ---
    #     # 展平背景和目标特征，只提取 mask 标记为 True 的有效部分
    #     bg_flat = background_features_projected[valid_background_mask.repeat(1, 1, embedding_dim)].reshape(-1, embedding_dim)
    #     obj_flat = object_features_projected[valid_object_mask.repeat(1, 1, embedding_dim)].reshape(-1, embedding_dim)

    #     # 按样本分割 (保持不变)
    #     # bg_splits: list of tensors, each tensor is [num_bg_tokens_for_sample_i, embedding_dim]
    #     bg_splits = torch.split(bg_flat, background_valid.tolist())
    #     # obj_splits: list of tensors, each tensor is [num_obj_tokens_for_sample_i, embedding_dim]
    #     obj_splits = torch.split(obj_flat, object_valid.tolist())


    #     # --- 根据 inference_mode 决定如何处理特征 ---
    #     processed_features_list = [] # 用于存放每个样本最终的特征序列

    #     for i in range(batch_size):
    #         current_bg_feat = bg_splits[i] # [bg_count_i, embedding_dim]
    #         current_obj_feat = obj_splits[i] # [obj_count_i, embedding_dim]
    #         current_obj_valid_count = object_valid[i].item() # 实际的目标 token 数量

    #         current_sample_final_features = None

    #         if inference_mode == "both":
    #             current_sample_final_features = torch.cat([current_bg_feat, current_obj_feat], dim=0)
    #         elif inference_mode == "background_only":
    #             current_sample_final_features = current_bg_feat
    #         elif inference_mode == "object_only":
    #             if current_obj_valid_count > 0: # 如果目标有效 token 数大于0，则使用目标特征
    #                 current_sample_final_features = current_obj_feat
    #             else: # 如果目标为空，则回退到背景特征
    #                 # print(f"Warning: Object is empty for sample {i} (obj_valid={current_obj_valid_count}), falling back to background for 'object_only' mode.")
    #                 current_sample_final_features = current_bg_feat
    #         else:
    #             raise ValueError(f"Unknown inference_mode: {inference_mode}. Must be 'both', 'background_only', or 'object_only'.")
            
    #         processed_features_list.append(current_sample_final_features)

    #     # 堆叠拼接后的特征，需要处理长度不一致的情况 (保持不变的 padding 逻辑)
    #     max_len = max(feat.shape[0] for feat in processed_features_list)
        
    #     padded_features_list = []
    #     for feat in processed_features_list:
    #         padding_needed = max_len - feat.shape[0]
    #         if padding_needed > 0:
    #             # 填充0或其他pad token的embedding
    #             pad_tensor = torch.zeros(padding_needed, embedding_dim, device=feat.device, dtype=feat.dtype)
    #             padded_feat = torch.cat([feat, pad_tensor], dim=0)
    #         else:
    #             padded_feat = feat
    #         padded_features_list.append(padded_feat)

    #     concatenated_features = torch.stack(padded_features_list, dim=0) 
        
    #     return concatenated_features


    # # 统一在此处进行封装，即均是调用 get_vision_tower
    # def encode_background_and_object_images_back(self, images, masks):
    #     """
    #     编码背景&目标图像。
    #     """        
    #     background_object_visual_tower = self.get_model().get_vision_tower().to(images.device)
    #     background_features, object_features, background_attention_mask, object_attention_mask = background_object_visual_tower(images, masks)


    #     # 补充两者的模态之间的融合
        


    #     background_features = self.get_model().mm_projector(background_features).to(images.device)
    #     object_features = self.get_model().mm_projector(object_features).to(images.device)

    #     batch_size, _, embedding_dim = background_features.shape

    #     # --- 验证有效token数 ---
    #     valid_background_mask = background_attention_mask.bool().unsqueeze(-1).to(images.device)
    #     valid_object_mask = object_attention_mask.bool().unsqueeze(-1).to(images.device)
    #     background_valid = valid_background_mask.squeeze(-1).sum(dim=1)
    #     object_valid = valid_object_mask.squeeze(-1).sum(dim=1)
    #     assert torch.all(background_valid + object_valid == 576), "总有效token数应为576"

    #     # --- 向量化提取有效token ---
    #     # 展平背景和目标特征
    #     bg_flat = background_features[valid_background_mask.repeat(1, 1, embedding_dim)]  # [total_bg_tokens, 4096]
    #     obj_flat = object_features[valid_object_mask.repeat(1, 1, embedding_dim)]        # [total_obj_tokens, 4096]

    #     # 按样本分割
    #     bg_splits = torch.split(bg_flat, (background_valid * embedding_dim).tolist())
    #     obj_splits = torch.split(obj_flat, (object_valid * embedding_dim).tolist())

    #     # 拼接并重塑形状
        

    #     # flag_token = torch.zeros(1, embedding_dim).to(background_features.device).to(background_features.dtype)
    #     # DEFAULT_BACKGROUND_OBJECT_TOKEN_ID = 32000 
    #     # flag_token = self.get_model().get_input_embeddings()(torch.tensor([DEFAULT_BACKGROUND_OBJECT_TOKEN_ID], device=background_features.device))
        
    #     # concatenated_features = [
    #     #     torch.cat([bg.reshape(-1, embedding_dim), flag_token, obj.reshape(-1, embedding_dim)], dim=0)
    #     #     for bg, obj in zip(bg_splits, obj_splits)
    #     # ]
    #     concatenated_features = [
    #         torch.cat([bg.reshape(-1, embedding_dim), obj.reshape(-1, embedding_dim)], dim=0)
    #         for bg, obj in zip(bg_splits, obj_splits)
    #     ]

    #     # descriptions = [
    #     #     f"Left first {valid.item()} image background tokens, then {576 - valid.item()} image foreground tokens."
    #     #     for valid in background_valid
    #     # ]
    #     # if not hasattr(self, 'tokenizer') or self.tokenizer is None:
    #     #     self.tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5", use_fast=False)

    #     # # 处理每个样本
    #     # processed_features = []
    #     # for i, feat in enumerate(concatenated_features):
    #     #     # 检查是否需要拆分背景和前景
    #     #     if background_valid[i].item() != feat.size(0):  # 需要拆分背景和前景
    #     #         # 使用 tokenizer 对描述进行编码
    #     #         description_id = self.tokenizer(descriptions[i], return_tensors="pt").input_ids.to(self.model.device)
                
    #     #         # 获取 description_feature
    #     #         description_feature = self.get_model().embed_tokens(description_id).squeeze(0)  # [description_length, embedding_dim]
                
    #     #         # 分割背景和目标特征
    #     #         bg_feature = feat[:background_valid[i].item()]  # 背景特征
    #     #         obj_feature = feat[background_valid[i].item():]  # 目标特征
                
    #     #         # 将 description_feature 插入到背景和目标特征之间
    #     #         feat = torch.cat([bg_feature, obj_feature, description_feature], dim=0)  # 在序列维度上拼接
            
    #     #     # 将处理后的特征加入列表
    #     #     processed_features.append(feat)


    #     # 堆叠拼接后的特征
    #     concatenated_features = torch.stack(concatenated_features, dim=0) 
    #     return concatenated_features

        
    #     # # --- 生成掩码和位置ID ---
    #     # concatenated_attention_mask = torch.ones((batch_size, 577), dtype=torch.bool, device=device)

    #     # attention_mask = concatenated_attention_mask.int()

    #     # position_ids = attention_mask.long().cumsum(-1) - 1

    #     # attention_mask = concatenated_attention_mask.unsqueeze(1).unsqueeze(3) 
    #     # attention_mask = attention_mask.expand(-1, -1, -1, 577).bool()

    #     # position_ids = torch.arange(concatenated_attention_mask.shape[1], device=device).expand(batch_size, -1)  # [batch_size, 576]

    #     # # model 初始化    【这里增加条件判断】
    #     # # if not self.get_model().load_prefusion_layers:  # and model未load
    #     # #     self.get_model().load_prefusion()
        
    #     # orig_concatenated_features = concatenated_features.clone()

    #     # # 模态预融合
    #     # for layer in self.get_model().prefusion_layers:
    #     #     concatenated_features = layer(concatenated_features, attention_mask=attention_mask, position_ids=position_ids)[0]

    #     # # 提取经过融合处理的目标特征部分
    #     # final_features_list = []
    #     # for i in range(batch_size):
    #     #     start_idx = background_valid[i].item()
    #     #     end_idx = start_idx + object_valid[i].item()
    #     #     fused_object_features = concatenated_features[i, start_idx + 1:end_idx + 1]
    #     #     original_background_features = orig_concatenated_features[i, 0:start_idx]
    #     #     final_feature = torch.cat([original_background_features, fused_object_features], dim=0)  # [576, 4096]
    #     #     final_features_list.append(final_feature)

    #     # final_features = torch.stack(final_features_list, dim=0)  # [batch_size, 576, 4096]
    #     # assert all(final_features.shape[1] == 576 for _ in range(batch_size)), "特征数量不匹配"

    #     # return final_features

    
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, masks, image_sizes=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            if self.method_type == "segmentation-cache":
                # image_features = self.encode_background_and_object_images(images, masks, inference_mode = 'object_only')
                # image_features = self.encode_background_and_object_images_back(images, masks)
                image_features = self.encode_background_and_object_images_back_cache(images, masks)
            elif self.method_type == "object-only":
                image_features = self.encode_object_only_images(images, masks)
            elif self.method_type == "fuzzy-cache":
                image_features = self.encode_fuzzy_cache_images(images, masks)
            else:  # native method - don't pass masks
                image_features = self.encode_images(images)


        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        # 训练时添加 DEFAULT_BACKGROUND_OBJECT_TOKEN
        tokenizer.add_tokens([DEFAULT_BACKGROUND_OBJECT_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
