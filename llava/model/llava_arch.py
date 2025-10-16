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

################################################################################
# 新增导入：导入您的KV控制器类。
# 请确保此文件（例如来自您Qwen实现的kv_faiss_cache.py）位于Python可发现的路径中。
from .kv_faiss_cache import CacheBlendKVController
import os
################################################################################


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
                from llava.mm_utils import get_model_name_from_path
                
                dataset_name = getattr(config, 'dataset', 'default_dataset')
                model_path = getattr(config, '_name_or_path', 'unknown_model')
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


        ################################################################################
        ####### 新增代码块：为 CacheBlend 初始化 KV 控制器 #######
        if self.method_type == "cacheblend":
            from llava.mm_utils import get_model_name_from_path
            
            # 这些属性是在模型加载时，通过 model_args 传递给 config 的
            dataset_name = getattr(config, "dataset", "default_dataset")
            model_path = getattr(config, "_name_or_path", "unknown_model")
            model_name = get_model_name_from_path(model_path)
            
            # LLM的 hidden_size 是KV缓存特征正确的 key_dim
            key_dim = getattr(config, "hidden_size", 4096)
            
            # 您可以根据需要将其配置为从外部传入
            base_cache_path = "/data/zyy/LLaVA/faiss"
            dynamic_cache_path = os.path.join(base_cache_path, self.method_type, model_name, dataset_name)
            
            self.kv_controller = CacheBlendKVController(
                key_dim=key_dim,
                cache_base_path=dynamic_cache_path
            )
            print(f"CacheBlend: KV 控制器已在 '{dynamic_cache_path}' 路径下为模型 '{model_name}' (数据集 '{dataset_name}') 初始化。")
        ####### 代码块结束 #######
        ################################################################################



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

    ################################################################################
    ####### 新增辅助函数：此函数包含 'write-only' 模式的核心逻辑 #######
    def _perform_cacheblend_write_only_pass(self, images, masks):
        """
        在 'write-only' 模式下，此函数为背景和前景图像段执行独立的前向传播，
        以预先计算并保存它们的KV缓存。
        这是一个副作用操作，不会改变主计算流程。
        """
        print("CacheBlend 'write-only' 模式：正在执行独立的背景/前景KV缓存过程...")
        model = self.get_model()
        vision_tower = self.get_vision_tower()  # 到底调用哪个visual_tower,应该是在模型初始化的时候根据 method_type 决定的
        
        if not hasattr(model, 'kv_controller'):
            print("CacheBlend 'write-only' 警告：未找到 kv_controller。跳过缓存步骤。")
            return

        # 1. 获取分离的背景/前景特征 (embeddings)
        # 此逻辑假设 vision_tower 可以根据掩码返回分离的特征。
        # LLaVA 的 vision_tower 需要进行相应修改以支持此功能，或在此处实现分离逻辑。
        bg_feats, fg_feats, bg_attn_mask, fg_attn_mask = vision_tower(images, masks)
        bg_embeds = model.mm_projector(bg_feats)
        fg_embeds = model.mm_projector(fg_feats)

        bg_mask = bg_attn_mask.bool().unsqueeze(-1).expand_as(bg_embeds)
        fg_mask = fg_attn_mask.bool().unsqueeze(-1).expand_as(fg_embeds)

        bg_embeds_flat = bg_embeds[bg_mask].reshape(1, -1, model.config.hidden_size)
        fg_embeds_flat = fg_embeds[fg_mask].reshape(1, -1, model.config.hidden_size)

        # 2. 为Faiss索引派生特征键
        with torch.no_grad():
            bg_feature_key = F.normalize(bg_embeds_flat.mean(dim=1), p=2, dim=1) if bg_embeds_flat.shape[1] > 0 else None
            fg_feature_key = F.normalize(fg_embeds_flat.mean(dim=1), p=2, dim=1) if fg_embeds_flat.shape[1] > 0 else None

        # 3. 执行独立的前向传播并收集KV缓存
        bg_kv_cache_list, fg_kv_cache_list = None, None
        
        # --- 背景传播 ---
        if bg_embeds_flat.shape[1] > 0:
            # 我们直接调用基础模型（即Transformer层堆栈），以获取hidden_states和past_key_values
            bg_outputs = model(
                inputs_embeds=bg_embeds_flat,
                use_cache=True,
                return_dict=True
            )
            # 返回的 past_key_values 是每层 (key, value) 张量的元组
            bg_kv_cache_list = [{"key": kv[0].clone().cpu(), "value": kv[1].clone().cpu()} for kv in bg_outputs.past_key_values]

        # --- 前景传播 ---
        if fg_embeds_flat.shape[1] > 0:
            fg_outputs = model(
                inputs_embeds=fg_embeds_flat,
                use_cache=True,
                return_dict=True
            )
            fg_kv_cache_list = [{"key": kv[0].clone().cpu(), "value": kv[1].clone().cpu()} for kv in fg_outputs.past_key_values]

        # 4. 使用KV控制器保存收集到的缓存
        model.kv_controller.add_patch_cache(
            bg_feature=bg_feature_key,
            fg_feature=fg_feature_key,
            bg_kv_cache=bg_kv_cache_list,
            fg_kv_cache=fg_kv_cache_list,
            bg_tokens=bg_embeds_flat.shape[1],
            fg_tokens=fg_embeds_flat.shape[1],
            # 注意：在LLaVA中，position_ids的生成较晚且复杂。
            # 在'write-only'阶段，我们可能还没有它们。可以存储None或一个简单的arange作为占位符。
            bg_position_ids=torch.arange(bg_embeds_flat.shape[1]) if bg_embeds_flat.shape[1] > 0 else None,
            fg_position_ids=torch.arange(fg_embeds_flat.shape[1]) if fg_embeds_flat.shape[1] > 0 else None,
        )
        print(f"CacheBlend 'write-only': 背景/前景 KV 缓存过程完成。")
    ####### 函数结束 #######
    ################################################################################
    

    ################################################################################
    ####### 新增辅助函数：实现 'read-load' 模式的核心逻辑 #######
    
    def _search_and_prepare_cached_data(self, images, masks):
        """
        封装了 'read-load' 模式下的所有缓存交互：
        1. 提取BG/FG特征。
        2. 搜索相似的缓存。
        3. 如果命中，准备对齐的 old_kvs 以供后续层使用。
        4. 返回搜索结果和命中状态。
        """
        model = self.get_model()
        vision_tower = self.get_vision_tower()

        # 1. 提取视觉特征 (与 write-only 中类似)
        bg_feats, fg_feats, bg_attn_mask, fg_attn_mask = vision_tower(images, masks)
        bg_embeds = model.mm_projector(bg_feats)
        fg_embeds = model.mm_projector(fg_feats)
        
        bg_mask = bg_attn_mask.bool().unsqueeze(-1).expand_as(bg_embeds)
        fg_mask = fg_attn_mask.bool().unsqueeze(-1).expand_as(fg_embeds)

        bg_embeds_flat = bg_embeds[bg_mask].reshape(1, -1, model.config.hidden_size)
        fg_embeds_flat = fg_embeds[fg_mask].reshape(1, -1, model.config.hidden_size)

        with torch.no_grad():
            bg_feature_key = F.normalize(bg_embeds_flat.mean(dim=1), p=2, dim=1) if bg_embeds_flat.shape[1] > 0 else None
            fg_feature_key = F.normalize(fg_embeds_flat.mean(dim=1), p=2, dim=1) if fg_embeds_flat.shape[1] > 0 else None

        # 2. 在缓存中搜索
        # 假设 similarity_threshold 从 config 中获取
        similarity_threshold = getattr(model.config, 'similarity_threshold', 0.1)
        bg_kv, bg_tokens, bg_pos_ids, fg_kv, fg_tokens, fg_pos_ids = \
            model.kv_controller.search_patch_cache(bg_feature_key, fg_feature_key, similarity_threshold)

        cache_hit = (bg_kv is not None) or (fg_kv is not None)
        
        # 将 token 数量为 None 的情况处理为 0
        bg_tokens = bg_tokens if bg_tokens is not None else 0
        fg_tokens = fg_tokens if fg_tokens is not None else 0

        # 3. 如果命中，准备对齐的 old_kvs
        if cache_hit:
            # 初始化一个空的 old_kvs 列表，长度为模型层数
            num_layers = model.config.num_hidden_layers
            old_kvs = [[None, None] for _ in range(num_layers)]
            
            # 按 BG -> FG 的顺序将缓存的KV拼接到一个对齐的张量中
            # 注意：这里的图像总长度(expected_len)需要与当前输入的图像token数一致
            # 在调用此函数后，我们将根据重建的输入来确定这个长度
            # 这里我们只返回原始的、未对齐的缓存数据
            pass # 对齐逻辑将在主函数中处理，因为我们需要知道最终的输入长度

        cached_data = {
            "hit": cache_hit,
            "bg_kv": bg_kv, "bg_tokens": bg_tokens, "bg_pos_ids": bg_pos_ids,
            "fg_kv": fg_kv, "fg_tokens": fg_tokens, "fg_pos_ids": fg_pos_ids,
            "original_bg_len": bg_embeds_flat.shape[1]
        }
        return cached_data


    def _rebuild_inputs_from_cache(self, original_input_ids, original_inputs_embeds, cached_data):
        """
        根据缓存命中结果（可能是变长的），动态重建 inputs_embeds, input_ids 等。
        此版本已修正，以正确处理 LLaVA 的 "占位符替换" 机制。
        """
        model = self.get_model()
        device = original_inputs_embeds.device
        
        # 1. 解构原始输入，分离出文本嵌入和原始图像嵌入
        from llava.constants import IMAGE_TOKEN_INDEX
        image_token_idx = torch.where(original_input_ids == IMAGE_TOKEN_INDEX)[1][0].item()
        original_image_len = original_inputs_embeds.shape[1] - original_input_ids.shape[1] + 1
        
        pre_text_embeds = original_inputs_embeds[:, :image_token_idx, :]
        post_text_embeds = original_inputs_embeds[:, image_token_idx + original_image_len:, :]
        original_image_embeds = original_inputs_embeds[:, image_token_idx:image_token_idx + original_image_len, :]

        # 2. 根据 vision tower 的输出，分离原始图像嵌入为 BG 和 FG 部分
        #    这是为了在部分未命中时，能正确填充真实 embedding
        #    注意: original_bg_len 需要从 vision_tower 的处理结果中获取，这里假设它可以被传递
        original_bg_len = cached_data.get('original_bg_len', 0) # 这是一个需要您从vision tower处获取并传入的变量
        original_bg_embeds = original_image_embeds[:, :original_bg_len, :]
        original_fg_embeds = original_image_embeds[:, original_bg_len:, :]

        # 3. 根据缓存命中情况，决定各段最终使用的 embedding
        final_image_embeds_parts = []
        
        # --- 获取一个合法的、中性的 token embedding 作为占位符模板 ---
        pad_token_id = model.config.pad_token_id if model.config.pad_token_id is not None else 0
        placeholder_template = model.embed_tokens(torch.tensor([[pad_token_id]], device=device))

        # --- 处理背景部分 ---
        if cached_data['bg_kv'] is not None: # 背景命中
            cached_len = cached_data['bg_tokens']
            # 使用模板 embedding 创建一个正确长度的占位符
            placeholder = placeholder_template.expand(1, cached_len, -1)
            final_image_embeds_parts.append(placeholder)
        else: # 背景未命中
            final_image_embeds_parts.append(original_bg_embeds)

        # --- 处理前景部分 ---
        if cached_data['fg_kv'] is not None: # 前景命中
            cached_len = cached_data['fg_tokens']
            placeholder = placeholder_template.expand(1, cached_len, -1)
            final_image_embeds_parts.append(placeholder)
        else: # 前景未命中
            final_image_embeds_parts.append(original_fg_embeds)
        
        # 4. 拼接成新的图像嵌入和最终的 inputs_embeds
        new_image_embeds = torch.cat(final_image_embeds_parts, dim=1)
        new_image_len = new_image_embeds.shape[1]
        new_inputs_embeds = torch.cat([pre_text_embeds, new_image_embeds, post_text_embeds], dim=1)
        
        # 5. 根据新的长度，重建 input_ids, position_ids, 和 attention_mask
        new_image_ids = torch.full((1, new_image_len), IMAGE_TOKEN_INDEX, device=device, dtype=torch.long)
        pre_text_ids = original_input_ids[:, :image_token_idx]
        post_text_ids = original_input_ids[:, image_token_idx + 1:]
        new_input_ids = torch.cat([pre_text_ids, new_image_ids, post_text_ids], dim=1)
        
        new_seq_len = new_inputs_embeds.shape[1]
        new_attention_mask = torch.ones(1, new_seq_len, device=device, dtype=torch.bool)
        new_position_ids = torch.arange(0, new_seq_len, device=device, dtype=torch.long).unsqueeze(0)
        
        # 6. 准备与新图像长度对齐的 old_kvs
        aligned_old_kvs = self._align_cached_kvs(cached_data, new_image_len, device)

        return new_input_ids, new_position_ids, new_attention_mask, new_inputs_embeds, aligned_old_kvs



    def _align_cached_kvs(self, cached_data, expected_len, device):
        """将BG/FG缓存拼接成一个对齐的old_kvs列表"""
        model = self.get_model()
        num_layers = model.config.num_hidden_layers
        aligned_old_kvs = [[None, None] for _ in range(num_layers)]

        bg_kv, bg_len = cached_data['bg_kv'], cached_data['bg_tokens']
        fg_kv, fg_len = cached_data['fg_kv'], cached_data['fg_tokens']
        
        if not cached_data['hit']:
            return aligned_old_kvs

        for i in range(num_layers):
            key_shape = (1, model.config.num_key_value_heads, expected_len, model.config.hidden_size // model.config.num_attention_heads)
            value_shape = key_shape
            
            aligned_key = torch.zeros(key_shape, device=device, dtype=model.dtype)
            aligned_value = torch.zeros(value_shape, device=device, dtype=model.dtype)
            
            current_pos = 0
            if bg_kv and bg_len > 0:
                k, v = bg_kv[i]['key'].to(device), bg_kv[i]['value'].to(device)
                aligned_key[:, :, current_pos:current_pos+bg_len, :] = k
                aligned_value[:, :, current_pos:current_pos+bg_len, :] = v
                current_pos += bg_len
            
            if fg_kv and fg_len > 0:
                k, v = fg_kv[i]['key'].to(device), fg_kv[i]['value'].to(device)
                aligned_key[:, :, current_pos:current_pos+fg_len, :] = k
                aligned_value[:, :, current_pos:current_pos+fg_len, :] = v

            aligned_old_kvs[i] = [aligned_key, aligned_value]
            
        return aligned_old_kvs

    ####### 函数结束 #######
    ################################################################################

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
            model = self.get_model()
            image_features = None # 初始化
            
            # ################################################################################
            # ####### 修改后的主要逻辑注入点 #######
            # if getattr(model, 'method_type', None) == 'cacheblend':
            #     is_prefill = past_key_values is None
                
            #     # 只在 prefill 阶段执行 cacheblend 的特殊逻辑
            #     if is_prefill:
            #         if getattr(model, 'cache_mode', None) == 'write-only':
            #             # 执行 'write-only' 的副作用操作
            #             with torch.no_grad():
            #                 self._perform_cacheblend_write_only_pass(images, masks)
            #             # write-only后，继续走原生图像编码流程
            #             image_features = self.encode_images(images)

            #         elif getattr(model, 'cache_mode', None) == 'read-load':
            #             # 执行 'read-load' 逻辑
            #             print("CacheBlend 'read-load' 模式：启动缓存搜索与输入重建...")
            #             cached_data = self._search_and_prepare_cached_data(images, masks)
                        
            #             if cached_data['hit']:
            #                 # 如果命中，则需要重建整个输入序列
            #                 input_ids, position_ids, attention_mask, original_inputs_embeds, old_kvs = \
            #                     self._rebuild_inputs_from_cache(input_ids, original_inputs_embeds, cached_data)

            #                 # 将准备好的数据附加到模型实例上，供 Attention Wrapper 使用
            #                 if not hasattr(model, 'cache_fuse_metadata'):
            #                     model.cache_fuse_metadata = {}
                            
            #                 image_start_index = torch.where(input_ids == IMAGE_TOKEN_INDEX)[0][0].item()
            #                 image_len = cached_data['bg_tokens'] + cached_data['fg_tokens']
                            
            #                 model.cache_fuse_metadata.update({
            #                     "old_kvs": old_kvs,
            #                     "image_span": (image_start_index, image_len),
            #                     "bg_len": cached_data['bg_tokens'],
            #                     "fg_len": cached_data['fg_tokens'],
            #                     "is_bg_hit": cached_data['bg_kv'] is not None,
            #                     "is_fg_hit": cached_data['fg_kv'] is not None,
            #                 })
                            
            #                 # 在这种情况下，image_features 已经包含在 original_inputs_embeds 中
            #                 # 所以我们设置 image_features 为一个空张量，以跳过后续的标准拼接逻辑
            #                 image_features = torch.tensor([], device=original_inputs_embeds.device)
            #             else:
            #                 # 缓存未命中，按原生流程处理
            #                 print("CacheBlend 'read-load' 模式：缓存未命中，执行原生图像编码。")
            #                 image_features = self.encode_images(images)
            #         else:
            #             # 其他模式（如 read-only），按原生流程处理
            #             image_features = self.encode_images(images)
            #     else: # Decode 阶段，总是原生处理
            #         image_features = self.encode_images(images)

            # ####### 代码块结束 #######
            # ################################################################################


            ################################################################################
            ####### 简化后的逻辑注入点 #######
            # 在这里，我们只处理 'write-only' 模式，因为它是一个独立的副作用操作。
            # 'read-load' 的逻辑将被移动到更高层的 llava_llama.py 中。
            if getattr(model, 'method_type', None) == 'cacheblend':
                is_prefill = past_key_values is None
                if is_prefill and getattr(model, 'cache_mode', None) == 'write-only':
                    with torch.no_grad():
                        self._perform_cacheblend_write_only_pass(images, masks)
            ####### 代码块结束 #######
            ################################################################################



            # 如果 image_features 尚未被计算（即非cacheblend路径），则在这里计算
            if image_features is None:
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
