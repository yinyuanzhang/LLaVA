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
from .cache import FaissCache
import torch.nn.functional as F

################################################################################
# 新增导入：导入您的KV控制器类。
# 请确保此文件（例如来自您Qwen实现的kv_faiss_cache.py）位于Python可发现的路径中。
from .cache.kv_faiss_cache import CacheBlendKVController
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


        # 初始化缓存系统和统计收集器
        self._initialize_cache_systems(config)

        # # ----------------------------------------------------
        # # 【修复】在这里修改
        # # ----------------------------------------------------
        # if hasattr(self, 'query_key_extractor') and self.query_key_extractor is not None:
            
        #     # 【修改点】
        #     # 不仅要检查它是否存在 (hasattr)，还要检查它是否为 None
        #     if not hasattr(self, '_keep_in_fp32_modules') or self._keep_in_fp32_modules is None:
        #         # 如果它不存在，或者它是 None，就强制将其设置为空列表 []
        #         self._keep_in_fp32_modules = []
            
        #     # 现在 self._keep_in_fp32_modules 保证是一个列表 (List)
        #     # 下面这行代码 (llava_arch.py, line 79) 就安全了
        #     if "query_key_extractor" not in self._keep_in_fp32_modules:
        #         self._keep_in_fp32_modules.append("query_key_extractor")
        #         print("[INFO] Registered 'query_key_extractor' to be kept in float32.")
        # # ----------------------------------------------------

    def _initialize_cache_systems(self, config):
        """
        统一初始化所有缓存系统和统计收集器
        """
        # 初始化基础配置
        self.background_cache = None
        # 移除旧的stats_collector初始化
        self.stats_collector = None
        self.kv_controller = None
        self.use_lightweight_query_key = getattr(config, 'use_lightweight_query_key', False)
        self.query_key_extractor = None
        self.similarity_threshold = getattr(config, 'similarity_threshold', 0.1)

        # 获取公共配置
        common_config = self._get_common_cache_config(config)

        # 1. 初始化 segmentation-cache 和 fuzzy-cache 系统
        if self.method_type in ["segmentation-cache", "fuzzy-cache"]:
            self._initialize_faiss_cache_system(config, common_config)

        # 2. 初始化 CacheBlend KV 控制器
        elif self.method_type == "cacheblend":
            self._initialize_cacheblend_system(config, common_config)

        # 3. 初始化统计数据结构（与Qwen2.5-VL保持一致）
        if self.method_type in ["segmentation-cache", "fuzzy-cache", "cacheblend", "object-only"]:
            # 使用与Qwen2.5-VL完全一致的统计数据结构
            self.stats = {
                'no_cache': {'system_len': [], 'img_len': [], 'query_len': []},
                'with_cache': {'system_len': [], 'img_len': [], 'query_len': []},
                'no_cache_count': 0,
                'with_cache_count': 0,
                # 缓存命中率统计
                'cache_hits': 0,
                'cache_searches': 0,
            }
            print(f"Cache statistics (Qwen2.5-VL style) initialized for {self.cache_mode} mode.")

    def _get_common_cache_config(self, config):
        """
        提取所有缓存系统共用的配置信息
        """
        from llava.mm_utils import get_model_name_from_path

        dataset_name = getattr(config, 'dataset', 'default_dataset')
        model_path = getattr(config, '_name_or_path', 'unknown_model')
        model_name = get_model_name_from_path(model_path)

        # 确定key维度和extractor类型
        if self.use_lightweight_query_key:
            extractor_type = getattr(config, 'query_key_extractor_type', 'resnet18')
            if extractor_type in ["resnet18", "resnet34", "vgg11", "vgg13", "vgg16", "vgg19"]:
                key_dim = 512
            elif extractor_type in ["resnet50", "resnet101"]:
                key_dim = 2048
            else:
                key_dim = 512
        else:
            extractor_type = "vit"
            key_dim = getattr(config, "hidden_size", 4096)

        # 构建缓存路径
        base_cache_path = "/data/zyy/LLaVA/faiss"
        dynamic_cache_path = os.path.join(base_cache_path, self.method_type, model_name, dataset_name)

        # FineGym特殊处理：为不同阈值创建独立缓存路径
        if dataset_name == "finegym":
            threshold_str = f"threshold_{self.similarity_threshold:.1f}".replace(".", "_")
            dynamic_cache_path = os.path.join(dynamic_cache_path, extractor_type, threshold_str)
        else:
            dynamic_cache_path = os.path.join(dynamic_cache_path, extractor_type)

        return {
            'dataset_name': dataset_name,
            'model_name': model_name,
            'extractor_type': extractor_type,
            'key_dim': key_dim,
            'dynamic_cache_path': dynamic_cache_path
        }

    def _initialize_faiss_cache_system(self, config, common_config):
        """
        初始化 segmentation-cache 和 fuzzy-cache 的 Faiss 缓存系统
        """
        if self.cache_mode not in ["write-only", "read-load"]:
            return

        try:
            import faiss

            print(f"Initializing {self.method_type} cache system:")
            print(f"  Dataset: {common_config['dataset_name']}")
            print(f"  Model: {common_config['model_name']}")
            print(f"  Cache mode: {self.cache_mode}")

            # 配置轻量级query_key提取器
            if self.use_lightweight_query_key:
                from .lightweight_query_key_extractor import create_query_key_extractor

                self.query_key_extractor = create_query_key_extractor(
                    extractor_type=common_config['extractor_type'],
                    output_dim=None,  # 使用backbone原生特征维度
                    target_size=224,
                    patch_size=14,    # CLIP ViT patch size
                    spatial_merge_size=1  # LLaVA 不使用spatial merge，所以设为1
                )

                self.query_key_extractor.eval()
                print(f"  Lightweight query_key extractor: {common_config['extractor_type']} (protected with float32)")
                print(f"  Faiss key dimension: {common_config['key_dim']}")
            else:
                print(f"  Using VIT-based query_key (dim: {common_config['key_dim']})")

            # 初始化缓存
            self.background_cache = FaissCache(
                key_dim=common_config['key_dim'],
                cache_file_path=common_config['dynamic_cache_path']
            )
            print(f"  Cache initialized at: {common_config['dynamic_cache_path']}")

        except ImportError:
            print("Faiss not installed. Background caching will be disabled.")
            self.background_cache = None

    def _initialize_cacheblend_system(self, config, common_config):
        """
        初始化 CacheBlend KV 控制器系统
        """
        try:
            # 配置轻量级query_key提取器
            if self.use_lightweight_query_key:
                from .lightweight_query_key_extractor import create_query_key_extractor

                self.query_key_extractor = create_query_key_extractor(
                    extractor_type=common_config['extractor_type'],
                    output_dim=None,  # 使用backbone原生特征维度
                    target_size=224,
                    patch_size=14,    # CLIP ViT patch size
                    spatial_merge_size=1  # LLaVA 不使用spatial merge，所以设为1
                )

                self.query_key_extractor.eval()
                print(f"  Lightweight query_key extractor: {common_config['extractor_type']} (protected with float32)")
                print(f"  Faiss key dimension: {common_config['key_dim']}")
            else:
                print(f"  Using VIT-based query_key (dim: {common_config['key_dim']})")

            self.kv_controller = CacheBlendKVController(
                key_dim=common_config['key_dim'],
                cache_base_path=common_config['dynamic_cache_path']
            )
            print(f"CacheBlend: KV controller initialized at '{common_config['dynamic_cache_path']}' for model '{common_config['model_name']}' (dataset '{common_config['dataset_name']}')")

        except ImportError as e:
            print(f"CacheBlend initialization failed: {e}")
            self.kv_controller = None



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


    def print_and_reset_stats(self):
        """Prints average stats and resets for a new session."""
        print("--- Segmentation Cache Statistics ---")

        # 打印配置信息
        if hasattr(self, 'similarity_threshold'):
            print(f"Similarity Threshold: {self.similarity_threshold}")

        # 计算并显示缓存命中率
        if self.stats['cache_searches'] > 0:
            hit_rate = (self.stats['cache_hits'] / self.stats['cache_searches']) * 100
            print(f"Cache Hit Rate: {self.stats['cache_hits']}/{self.stats['cache_searches']} ({hit_rate:.2f}%)")
        else:
            print("Cache Hit Rate: No cache searches recorded")

        # 计算未采用缓存时的平均值
        no_cache_stats = self.stats['no_cache']
        if self.stats['no_cache_count'] > 0:
            avg_no_cache_sys = sum(no_cache_stats['system_len']) / self.stats['no_cache_count']
            avg_no_cache_img = sum(no_cache_stats['img_len']) / self.stats['no_cache_count']
            avg_no_cache_query = sum(no_cache_stats['query_len']) / self.stats['no_cache_count']
            print(f"No Cache (N={self.stats['no_cache_count']}):")
            print(f"  Avg System Len: {avg_no_cache_sys:.2f}")
            print(f"  Avg Image Len: {avg_no_cache_img:.2f}")
            print(f"  Avg Query Len: {avg_no_cache_query:.2f}")

        # 计算采用缓存时的平均值
        with_cache_stats = self.stats['with_cache']
        if self.stats['with_cache_count'] > 0:
            avg_with_cache_sys = sum(with_cache_stats['system_len']) / self.stats['with_cache_count']
            avg_with_cache_img = sum(with_cache_stats['img_len']) / self.stats['with_cache_count']
            avg_with_cache_query = sum(with_cache_stats['query_len']) / self.stats['with_cache_count']
            print(f"With Cache (N={self.stats['with_cache_count']}):")
            print(f"  Avg System Len: {avg_with_cache_sys:.2f}")
            print(f"  Avg Recomputed Image Len: {avg_with_cache_img:.2f}")
            print(f"  Avg Query Len: {avg_with_cache_query:.2f}")

        # 重置统计数据
        self.stats = {
            'no_cache': {'system_len': [], 'img_len': [], 'query_len': []},
            'with_cache': {'system_len': [], 'img_len': [], 'query_len': []},
            'no_cache_count': 0,
            'with_cache_count': 0,
            # 缓存命中率统计
            'cache_hits': 0,
            'cache_searches': 0,
        }
        print("--- Stats reset ---")
            

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

    # Helper: Convert CLIP pixel_values back to raw [0,1] images
    def _to_raw_images(self, images: torch.Tensor) -> torch.Tensor:
        vt = self.get_model().get_vision_tower()
        mean_list = getattr(getattr(vt, "image_processor", None), "image_mean", [0.48145466, 0.4578275, 0.40821073])
        std_list = getattr(getattr(vt, "image_processor", None), "image_std", [0.26862954, 0.26130258, 0.27577711])
        imgs = images.float() if images.dtype != torch.float32 else images
        mean = torch.tensor(mean_list, device=imgs.device, dtype=imgs.dtype).view(1, 3, 1, 1)
        std = torch.tensor(std_list, device=imgs.device, dtype=imgs.dtype).view(1, 3, 1, 1)
        return torch.clamp(imgs * std + mean, 0.0, 1.0)

    ################################################################################
    ####### CacheBlend 辅助函数：特征提取 #######
    ################################################################################
    def _extract_vision_features(self, image_embeds, final_is_foreground_mask,
                                raw_image_tensor=None, image_shape=None,
                                bg_token_count=None, fg_token_count=None):
        """从重排后的 embeds 中提取并归一化 BG/FG 特征，用于相似度搜索

        支持两种方法：
        1. VIT-based方法：使用重排后的image_embeds计算均值（原始方法）
        2. 轻量级方法：使用CNN backbone从原始图像中提取query_key（复用_extract_query_key_lightweight）

        Args:
            image_embeds: 重排后的图像embeddings [num_patches, hidden_size]
            final_is_foreground_mask: 前景/背景mask [num_patches]
            raw_image_tensor: CLIP预处理后的图像tensor [B, C, H, W] (轻量级方法需要)
            image_shape: 图像形状 (H, W) (轻量级方法需要)
            bg_token_count: 当前序列中BG token数量 (VIT方法必需)
            fg_token_count: 当前序列中FG token数量 (VIT方法必需)            

        Returns:
            bg_feature: 背景特征 [1, feature_dim]
            fg_feature: 前景特征 [1, feature_dim]
        """
        model = self.get_model()

        if image_embeds is None or image_embeds.size(0) == 0:
            return None, None

        # 检查是否使用轻量级extractor
        if model.use_lightweight_query_key and model.query_key_extractor is not None:
            # 轻量级方法：复用 _extract_query_key_lightweight 分别提取BG和FG特征
            if raw_image_tensor is not None and image_shape is not None:
                bg_feature = None
                fg_feature = None

                # 提取BG特征：使用背景mask（前景=False）
                if (~final_is_foreground_mask).any():  # 确保有背景
                    # 创建背景mask的full_patch_mask格式 (0=背景, 1=前景)
                    bg_full_patch_mask = final_is_foreground_mask.unsqueeze(0).int()
                    bg_feature = self._extract_query_key_lightweight(raw_image_tensor, bg_full_patch_mask)

                # 提取FG特征：使用前景mask（前景=True）
                if final_is_foreground_mask.any():  # 确保有前景
                    # 创建前景mask的full_patch_mask格式 (反转：0=前景→提取, 1=背景)
                    fg_full_patch_mask = (~final_is_foreground_mask).unsqueeze(0).int()
                    fg_feature = self._extract_query_key_lightweight(raw_image_tensor, fg_full_patch_mask)

                print(f"CacheBlend: Using lightweight query_key extractor for feature extraction")
                return bg_feature, fg_feature
            else:
                print(f"CacheBlend WARNING: Lightweight extractor enabled but raw_image_tensor or image_shape is None. Falling back to VIT-based method.")

        # VIT-based方法：根据重排后embeds的结构直接切分
        assert bg_token_count is not None and fg_token_count is not None, \
            "bg_token_count and fg_token_count are required for VIT-based feature extraction"

        if bg_token_count > 0:
            bg_embeds = image_embeds[:bg_token_count]
            bg_feature = torch.mean(bg_embeds, dim=0, keepdim=True).float()
            bg_feature = F.normalize(bg_feature, p=2, dim=1)
        else:
            bg_feature = None

        if fg_token_count > 0:
            fg_embeds = image_embeds[bg_token_count:bg_token_count+fg_token_count]
            fg_feature = torch.mean(fg_embeds, dim=0, keepdim=True).float()
            fg_feature = F.normalize(fg_feature, p=2, dim=1)
        else:
            fg_feature = None
        print(f"CacheBlend: Using VIT-based method for feature extraction")
        return bg_feature, fg_feature

    ################################################################################
    ####### CacheBlend write-only 模式核心逻辑 #######
    ################################################################################
    def _perform_cacheblend_write_only_pass(self, images, masks):
        """
        在 'write-only' 模式下，为背景和前景图像段执行独立的前向传播，
        以预先计算并保存它们的KV缓存。

        参考 Qwen2.5-VL 的实现 (cacheblend_qwen_generation.py:1457-1530)

        Args:
            images: 输入图像 [B, C, H, W]
            masks: 分割掩码 [B, H, W]
        """
        print("CacheBlend: 'write-only' prefill stage. Performing separated BG/FG inference...")

        model = self.get_model()
        vision_tower = self.get_vision_tower()

        if not hasattr(model, 'kv_controller'):
            print("CacheBlend 'write-only' WARNING: kv_controller not found. Skipping cache step.")
            return

        # 1. 获取分离的背景/前景特征 (参考 Qwen 实现)
        bg_feats, fg_feats, bg_attn_mask, fg_attn_mask, full_patch_mask, _ = vision_tower(images, masks)
        bg_embeds = model.mm_projector(bg_feats)
        fg_embeds = model.mm_projector(fg_feats)

        # 提取有效的背景和前景patches
        bg_mask = bg_attn_mask.bool().unsqueeze(-1).expand_as(bg_embeds)
        fg_mask = fg_attn_mask.bool().unsqueeze(-1).expand_as(fg_embeds)

        bg_embeds_flat = bg_embeds[bg_mask].reshape(1, -1, model.config.hidden_size)
        fg_embeds_flat = fg_embeds[fg_mask].reshape(1, -1, model.config.hidden_size)

        num_bg_tokens = bg_embeds_flat.shape[1]
        num_fg_tokens = fg_embeds_flat.shape[1]

        # 2. 提取用于相似度搜索的代表性特征（支持轻量级和VIT两种方法）
        # 构造final_is_foreground_mask (True表示前景)
        final_is_foreground_mask = (full_patch_mask[0] == 1)  # [num_patches] bool tensor

        # 合并所有image embeds用于特征提取
        all_image_embeds = torch.cat([bg_embeds_flat.squeeze(0), fg_embeds_flat.squeeze(0)], dim=0) if num_bg_tokens > 0 and num_fg_tokens > 0 else \
                           (bg_embeds_flat.squeeze(0) if num_bg_tokens > 0 else fg_embeds_flat.squeeze(0))

        # 提取特征（自动选择轻量级或VIT方法）
        raw_images = self._to_raw_images(images)
        bg_feature, fg_feature = self._extract_vision_features(
            all_image_embeds,
            final_is_foreground_mask,
            raw_image_tensor=raw_images,  # 使用像素域原图以配合轻量级抽取器
            image_shape=(images.shape[2], images.shape[3]),
            bg_token_count=num_bg_tokens,  # 新增：VIT方法需要
            fg_token_count=num_fg_tokens   # 新增：VIT方法需要            
        )

        # 3. 分别进行前向传播并收集KV缓存
        bg_kv_cache_list, fg_kv_cache_list = None, None
        bg_position_ids, fg_position_ids = None, None

        # --- 背景传播 ---
        if num_bg_tokens > 0:
            bg_outputs = model(
                inputs_embeds=bg_embeds_flat,
                use_cache=True,
                return_dict=True
            )
            # 提取KV缓存
            bg_kv_cache_list = [{"key": kv[0].clone().cpu(), "value": kv[1].clone().cpu()}
                               for kv in bg_outputs.past_key_values]
            # 生成position_ids
            bg_position_ids = torch.arange(num_bg_tokens, device=bg_embeds_flat.device)

        # --- 前景传播 ---
        if num_fg_tokens > 0:
            fg_outputs = model(
                inputs_embeds=fg_embeds_flat,
                use_cache=True,
                return_dict=True
            )
            fg_kv_cache_list = [{"key": kv[0].clone().cpu(), "value": kv[1].clone().cpu()}
                               for kv in fg_outputs.past_key_values]
            fg_position_ids = torch.arange(num_fg_tokens, device=fg_embeds_flat.device)

        # 4. 存储到KV控制器（参考 Qwen _collect_patch_kv_cache 方法）
        model.kv_controller.add_patch_cache(
            bg_feature=bg_feature,
            fg_feature=fg_feature,
            bg_kv_cache=bg_kv_cache_list,
            fg_kv_cache=fg_kv_cache_list,
            bg_tokens=num_bg_tokens,
            fg_tokens=num_fg_tokens,
            bg_position_ids=bg_position_ids,
            fg_position_ids=fg_position_ids,
            bg_embeds=bg_embeds_flat.squeeze(0) if num_bg_tokens > 0 else None,  # 存储原始embeds
            fg_embeds=fg_embeds_flat.squeeze(0) if num_fg_tokens > 0 else None
        )

        # 记录写操作统计
        # 移除旧的write统计调用
        # write操作统计已不再需要

        print(f"CacheBlend: Cache collection finished. BG: {num_bg_tokens} tokens, FG: {num_fg_tokens} tokens")
    ####### 函数结束 #######
    ################################################################################
    

    ################################################################################
    ####### 新增辅助函数：实现 'read-load' 模式的核心逻辑 #######
    
    def _search_similar_patches(self, bg_feature, fg_feature):
        """
        【直接复制自 Qwen2.5-VL】使用相似度搜索匹配的 patch 缓存
        """
        model = self.get_model()

        if not hasattr(model, 'kv_controller'):
            return None, None, None, None, None, None, None, None  # 返回8个None

        if not (bg_feature is not None or fg_feature is not None):
            return None, None, None, None, None, None, None, None  # 返回8个None

        bg_kv_cache, bg_tokens, bg_pos_ids, bg_embeds, \
        fg_kv_cache, fg_tokens, fg_pos_ids, fg_embeds = model.kv_controller.search_patch_cache(
            bg_feature=bg_feature,
            fg_feature=fg_feature,
            similarity_threshold=getattr(model.config, 'similarity_threshold', 0.1)
        )

        bg_hit = bg_kv_cache is not None
        fg_hit = fg_kv_cache is not None

        hit_status_str = f"BG: {'✓' if bg_hit else '✗'}, FG: {'✓' if fg_hit else '✗'}"
        if bg_hit or fg_hit:
            print(f"CacheBlend: Cache search result: Hit! Status: [{hit_status_str}]")
        else:
            print(f"CacheBlend: Cache search result: Miss. Status: [{hit_status_str}]")

        # 记录缓存统计 - 这里是关键的修复！
        overall_hit = bg_hit or fg_hit  # 只要有一个命中就算命中
        # 移除旧的cache outcome统计调用
        # 缓存结果统计已在主要逻辑中处理

        if hasattr(self.get_model(), 'stats'):
            self.get_model().stats['cache_searches'] += 1
            if overall_hit:
                self.get_model().stats['cache_hits'] += 1


        return bg_kv_cache, bg_tokens, bg_pos_ids, bg_embeds, fg_kv_cache, fg_tokens, fg_pos_ids, fg_embeds


    def _rebuild_inputs_for_variable_cache(
        self,
        original_input_ids,
        original_inputs_embeds,
        original_position_ids,
        original_attention_mask,
        current_bg_tokens,  # 修改：传入当前BG token数量
        current_fg_tokens,  # 修改：传入当前FG token数量
        cached_bg_embeds, cached_bg_tokens, cached_bg_pos_ids,
        cached_fg_embeds, cached_fg_tokens, cached_fg_pos_ids,
    ):
        """
        【直接复制自 Qwen2.5-VL】根据不同长度的缓存命中，动态地重构模型输入。
        """
        device = original_inputs_embeds.device
        from llava.constants import IMAGE_TOKEN_INDEX  # LLaVA适配：使用IMAGE_TOKEN_INDEX而不是image_token_id

        # 1. 解构原始输入
        image_token_mask = (original_input_ids[0] == IMAGE_TOKEN_INDEX)  # LLaVA适配
        image_indices = torch.where(image_token_mask)[0]

        pre_image_slice = slice(0, image_indices[0])
        post_image_slice = slice(image_indices[-1] + 1, original_input_ids.shape[1])

        # 提取非图像部分
        pre_image_embeds = original_inputs_embeds[:, pre_image_slice, :]
        pre_image_pos_ids = original_position_ids[..., pre_image_slice]

        post_image_embeds = original_inputs_embeds[:, post_image_slice, :]
        post_image_pos_ids = original_position_ids[..., post_image_slice]

        # 提取原始图像部分，用于未命中时回退
        original_image_embeds = original_inputs_embeds[:, image_indices, :]
        original_image_pos_ids = original_position_ids[..., image_indices]

        # 修改：按重排后的结构直接切分，而不是使用mask
        # 验证长度匹配
        expected_total = current_bg_tokens + current_fg_tokens
        actual_total = original_image_embeds.shape[1]
        assert expected_total == actual_total, \
            f"Token count mismatch: BG({current_bg_tokens}) + FG({current_fg_tokens}) = {expected_total} != {actual_total}"

        # 按BG+FG顺序直接切分
        original_bg_embeds = original_image_embeds[:, :current_bg_tokens, :]
        original_fg_embeds = original_image_embeds[:, current_bg_tokens:current_bg_tokens+current_fg_tokens, :]
        original_bg_pos_ids = original_image_pos_ids[..., :current_bg_tokens]
        original_fg_pos_ids = original_image_pos_ids[..., current_bg_tokens:current_bg_tokens+current_fg_tokens]

        # 2. 决策与选择
        # 背景部分
        if cached_bg_tokens > 0:
            new_bg_len = cached_bg_tokens
            new_bg_embeds = cached_bg_embeds.to(device).unsqueeze(0) # 确保有 batch 维度
            new_bg_pos_ids = cached_bg_pos_ids.to(device)
            # 【修复】确保缓存的position_ids维度与原始维度一致
            if new_bg_pos_ids.dim() == 1 and original_position_ids is not None and original_position_ids.dim() > 1:
                new_bg_pos_ids = new_bg_pos_ids.unsqueeze(0)  # [seq_len] -> [1, seq_len]
        else:
            new_bg_len = original_bg_embeds.shape[1]
            new_bg_embeds = original_bg_embeds
            new_bg_pos_ids = original_bg_pos_ids

        # 前景部分
        if cached_fg_tokens > 0:
            new_fg_len = cached_fg_tokens
            new_fg_embeds = cached_fg_embeds.to(device).unsqueeze(0) # 确保有 batch 维度
            new_fg_pos_ids = cached_fg_pos_ids.to(device)
            # 【修复】确保缓存的position_ids维度与原始维度一致
            if new_fg_pos_ids.dim() == 1 and original_position_ids is not None and original_position_ids.dim() > 1:
                new_fg_pos_ids = new_fg_pos_ids.unsqueeze(0)  # [seq_len] -> [1, seq_len]
        else:
            new_fg_len = original_fg_embeds.shape[1]
            new_fg_embeds = original_fg_embeds
            new_fg_pos_ids = original_fg_pos_ids

        # 3. 重新组装
        new_inputs_embeds = torch.cat([pre_image_embeds, new_bg_embeds, new_fg_embeds, post_image_embeds], dim=1)
        new_position_ids = torch.cat([pre_image_pos_ids, new_bg_pos_ids, new_fg_pos_ids, post_image_pos_ids], dim=-1)

        # 4. 重建 input_ids 和 attention_mask (这是必须的，因为序列总长度变了)
        pre_image_ids = original_input_ids[:, pre_image_slice]
        post_image_ids = original_input_ids[:, post_image_slice]
        new_image_ids = torch.full((1, new_bg_len + new_fg_len), IMAGE_TOKEN_INDEX, device=device, dtype=torch.long)  # LLaVA适配
        new_input_ids = torch.cat([pre_image_ids, new_image_ids, post_image_ids], dim=1)

        # 重建 attention mask, 一个简单的 causal mask
        new_seq_len = new_input_ids.shape[1]
        new_attention_mask = torch.ones(1, new_seq_len, device=device)

        return new_input_ids, new_inputs_embeds, new_position_ids, new_attention_mask, new_bg_len, new_fg_len

    def _prepare_cache_fusion_metadata(self, metadata, input_ids, inputs_embeds, cached_data):
        """
        【直接复制自 Qwen2.5-VL】准备 CacheBlend 核心算法需要的元数据
        """
        try:
            seq_len = inputs_embeds.shape[1]
            from llava.constants import IMAGE_TOKEN_INDEX  # LLaVA适配
            image_token_indices = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]  # LLaVA适配

            if len(image_token_indices) > 0:
                first_image_pos = image_token_indices[0].item()
                last_image_pos = image_token_indices[-1].item()

                system_len = first_image_pos
                image_len = last_image_pos - first_image_pos + 1
                query_len = seq_len - (last_image_pos + 1)

                # 准备与图像部分对齐的 old_kvs
                self._prepare_aligned_old_kvs(
                    cached_data['bg_kv_cache'], cached_data['fg_kv_cache'],
                    cached_data['bg_tokens'], cached_data['fg_tokens'],
                    expected_cache_len=image_len
                )

                bg_hit = cached_data['bg_kv_cache'] is not None and cached_data['bg_tokens'] > 0
                fg_hit = cached_data['fg_kv_cache'] is not None and cached_data['fg_tokens'] > 0

                metadata.update({
                    "system_prompt_len": system_len,
                    "cacheable_start": first_image_pos,
                    "cacheable_len": image_len,
                    "org_seq_len": seq_len,
                    # 新增以下精细化信息
                    "bg_tokens_len": cached_data['bg_tokens'],
                    "fg_tokens_len": cached_data['fg_tokens'],
                    "is_bg_hit": bg_hit,
                    "is_fg_hit": fg_hit,
                    # 【新增】CacheBlend 层级任务分配配置
                    "check_layers": [1],  # 第1层进行重要性计算，与Qwen2.5-VL保持一致
                    "recomp_ratio": 0.16,  # 重计算比例
                })

                print(f"CacheBlend: Alignment metadata prepared - system: {system_len}, cacheable: {image_len}, query: {query_len}")
            else:
                raise ValueError("No image tokens found for cache fusion.")
        except Exception as e:
            print(f"CacheBlend: Error preparing fusion metadata: {e}. Disabling cache reuse for this run.")

    def _prepare_aligned_old_kvs(self, cached_bg_kv, cached_fg_kv, cached_bg_tokens, cached_fg_tokens, expected_cache_len):
        """
        【直接复制自 Qwen2.5-VL】准备与当前输入图像部分对齐的 old_kvs
        """
        try:
            model = self.get_model()
            num_layers = getattr(model.config, 'num_hidden_layers', 32)
            device = next(model.parameters()).device
            dtype = next(model.parameters()).dtype

            num_heads = getattr(model.config, 'num_key_value_heads',
                               getattr(model.config, 'num_attention_heads', 32))
            head_dim = getattr(model.config, 'hidden_size', 4096) // getattr(model.config, 'num_attention_heads', 32)

            bg_hit = cached_bg_kv is not None and cached_bg_tokens > 0
            fg_hit = cached_fg_kv is not None and cached_fg_tokens > 0

            # 只有在完全命中的情况下，才检查总长度是否匹配
            if bg_hit and fg_hit and (cached_bg_tokens + cached_fg_tokens) != expected_cache_len:
                 print(f"CacheBlend WARNING: Full cache hit, but total cached tokens ({cached_bg_tokens + cached_fg_tokens}) "
                       f"do not match expected image length ({expected_cache_len}). Cache might be misaligned.")

            # 确保 cache_fuse_metadata 存在
            if not hasattr(self, 'cache_fuse_metadata'):
                self.cache_fuse_metadata = {}

            old_kvs = []

            for layer_idx in range(num_layers):
                aligned_key = torch.zeros(1, num_heads, expected_cache_len, head_dim, device=device, dtype=dtype)
                aligned_value = torch.zeros(1, num_heads, expected_cache_len, head_dim, device=device, dtype=dtype)

                current_pos = 0
                # 按 BG -> FG 的顺序填充（与重排后的 embeds 顺序一致）
                if bg_hit:
                    bg_k = cached_bg_kv[layer_idx]['key'].to(device=device, dtype=dtype)
                    bg_v = cached_bg_kv[layer_idx]['value'].to(device=device, dtype=dtype)

                    if bg_k.shape[2] != cached_bg_tokens:
                        print(f"CacheBlend WARNING: Layer {layer_idx} BG cache length mismatch. "
                              f"Metadata tokens: {cached_bg_tokens}, Tensor shape: {bg_k.shape[2]}. Using tensor shape.")

                    len_to_fill = min(bg_k.shape[2], expected_cache_len - current_pos)
                    aligned_key[:, :, current_pos:current_pos+len_to_fill, :] = bg_k[:, :, :len_to_fill, :]
                    aligned_value[:, :, current_pos:current_pos+len_to_fill, :] = bg_v[:, :, :len_to_fill, :]
                    current_pos += cached_bg_tokens # 指针按元数据长度移动，以正确放置FG

                if fg_hit:
                    if current_pos >= expected_cache_len:
                        print(f"CacheBlend WARNING: BG cache ({cached_bg_tokens} tokens) already meets or exceeds expected length ({expected_cache_len}). FG cache will be ignored.")
                    else:
                        fg_k = cached_fg_kv[layer_idx]['key'].to(device=device, dtype=dtype)
                        fg_v = cached_fg_kv[layer_idx]['value'].to(device=device, dtype=dtype)

                        if fg_k.shape[2] != cached_fg_tokens:
                            print(f"CacheBlend WARNING: Layer {layer_idx} FG cache length mismatch. "
                                  f"Metadata tokens: {cached_fg_tokens}, Tensor shape: {fg_k.shape[2]}. Using tensor shape.")

                        len_to_fill = min(fg_k.shape[2], expected_cache_len - current_pos)
                        aligned_key[:, :, current_pos:current_pos+len_to_fill, :] = fg_k[:, :, :len_to_fill, :]
                        aligned_value[:, :, current_pos:current_pos+len_to_fill, :] = fg_v[:, :, :len_to_fill, :]

                old_kvs.append([aligned_key, aligned_value])

            self.old_kvs = old_kvs

            hit_status = f"BG: {'✓' if bg_hit else '✗'}, FG: {'✓' if fg_hit else '✗'}"
            print(f"CacheBlend: Aligned old_kvs prepared for {expected_cache_len} tokens. Hit status: [{hit_status}]")
        except Exception as e:
            print(f"CacheBlend: Error preparing aligned old_kvs: {e}")
            # 清空 old_kvs 以安全回退到原生模式
            if hasattr(self, 'cache_fuse_metadata'):
                self.old_kvs = []

    ####### 以上函数结束 #######
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

    def _cache_write_background_features(self, images, bg_features, full_patch_mask, embedding_dim):
        """写入背景特征到缓存"""
        print("缓存模式为 write-only，将计算的背景特征写入缓存。")
        with torch.no_grad():
            query_key = self._extract_query_key_for_search(images, bg_features, full_patch_mask)

            if self.get_model().background_cache:
                bg_position_ids = torch.arange(bg_features.shape[1]).unsqueeze(0).expand(3, -1)
                self.get_model().background_cache.add_feature(
                    query_key,
                    bg_features.clone().detach(),
                    bg_position_ids
                )
                # 记录写操作统计
                # 移除旧的write统计调用
                # write操作统计已不再需要
            else:
                print("警告: 缓存系统未初始化，无法写入。")

    def _cache_search_background_features(self, images, bg_features, full_patch_mask):
        """搜索并返回缓存中的背景特征"""
        print("缓存模式为 read-load，正在搜索缓存...")
        with torch.no_grad():
            query_key = self._extract_query_key_for_search(images, bg_features, full_patch_mask)

            if self.get_model().background_cache:
                reused_features, _ = self.get_model().background_cache.search_feature(
                    query_key,
                    distance_threshold=self.get_model().similarity_threshold
                )
                if reused_features is not None:
                    print("使用缓存的背景特征。")
                else:
                    print("缓存未命中，使用新计算的背景特征。")
                return reused_features
            else:
                print("警告: 缓存系统未初始化，无法搜索。")
                return None

    def _extract_query_key_for_search(self, images, bg_features, full_patch_mask):
        """提取用于缓存搜索的query_key"""
        if self.get_model().use_lightweight_query_key and hasattr(self.get_model(), 'query_key_extractor') and self.get_model().query_key_extractor is not None:
            # 轻量级方法：使用CNN骨干网络
            return self._extract_query_key_lightweight(images, full_patch_mask)
        else:
            # VIT方法：基于VIT输出的均值
            return self._extract_query_key_vit(bg_features)

    def _extract_query_key_lightweight(self, images, full_patch_mask):
        """使用轻量级提取器提取query_key
        注意：这里的 `images` 是 CLIP 的 pixel_values（已按 CLIP 均值/方差标准化）。
        为了给 ResNet/VGG 提取器提供正确的输入，这里先用视觉塔的 image_processor.mean/std 进行精确的反归一化，
        将其还原到近似原始的 [0,1] 范围，再交由轻量级提取器进行 ImageNet 归一化与特征提取。
        """
        print("LLaVA: Using lightweight query_key extractor for feature extraction")

        # 获取前景掩码（True 表示前景）- 适配新版 lightweight_query_key_extractor
        # full_patch_mask: 0=背景, 1=前景，所以直接转换为bool即可
        final_is_foreground_mask = (full_patch_mask[0] == 1)  # [576] bool tensor

        # 为避免精度问题，统一转 float32
        images = images.float() if images.dtype != torch.float32 else images

        # 使用视觉塔上的实际 CLIP 均值/方差进行"精确反归一化"
        vt = self.get_model().get_vision_tower()
        if hasattr(vt, "image_processor") and hasattr(vt.image_processor, "image_mean") and hasattr(vt.image_processor, "image_std"):
            mean_list = vt.image_processor.image_mean
            std_list = vt.image_processor.image_std
        else:
            # 兜底（不应常用）：使用常见 CLIP 统计量
            mean_list = [0.48145466, 0.4578275, 0.40821073]
            std_list = [0.26862954, 0.26130258, 0.27577711]

        clip_mean = torch.tensor(mean_list, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
        clip_std = torch.tensor(std_list, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
        raw_images = images * clip_std + clip_mean
        raw_images = torch.clamp(raw_images, 0.0, 1.0)

        # 构建 grid_thw - 适配新版接口
        # grid_thw 格式: [1, 3] = [t, h_physical, w_physical]
        grid_thw = torch.tensor([[1, images.shape[2], images.shape[3]]], device=images.device)

        # 提取 query_key - 使用新接口
        query_key = self.get_model().query_key_extractor(
            raw_image_tensor=raw_images,
            final_is_foreground_mask=final_is_foreground_mask,
            grid_thw=grid_thw
        )
        return query_key

    def _extract_query_key_vit(self, bg_features):
        """使用VIT方法提取query_key"""
        print("LLaVA: Using VIT-based method for feature extraction")

        if bg_features.dim() < 2:
            if bg_features.dim() == 1:
                bg_features = bg_features.unsqueeze(0).unsqueeze(0)
            else:
                raise ValueError("bg_features dimension error")

        query_key = torch.mean(bg_features, dim=1)
        query_key = F.normalize(query_key, p=2, dim=1)
        return query_key
                    
    def encode_background_and_object_images_back_cache(self, images, masks):
        """
        编码背景&目标图像，并优化缓存逻辑。
        此方法将统计**分离后的**背景和目标有效token数量，并交给统计类处理。
        同时，它会记录缓存的命中/未命中情况。
        """
        # ============================================================================
        # 阶段1: 获取视觉特征并进行投影
        # ============================================================================
        self.cache_mode = getattr(self.get_model(), "cache_mode", "read-only")

        background_object_visual_tower = self.get_model().get_vision_tower().to(images.device)
        background_features, object_features, background_attention_mask, object_attention_mask, full_patch_mask, reorder_mapping = background_object_visual_tower(images, masks)

        # 存储重排映射信息，供position_ids构建使用
        self.get_model()._segmentation_reorder_mapping = reorder_mapping

        background_features = self.get_model().mm_projector(background_features).to(images.device)
        object_features = self.get_model().mm_projector(object_features).to(object_features.device)

        batch_size, _, embedding_dim = background_features.shape


        # ============================================================================
        # 阶段2: 统计背景/前景有效token数量
        # ============================================================================
        valid_background_mask = background_attention_mask.bool().unsqueeze(-1).to(images.device)
        valid_object_mask = object_attention_mask.bool().unsqueeze(-1).to(images.device)

        current_bg_valid_count = valid_background_mask.squeeze(-1).sum(dim=1).item()
        current_obj_valid_count = valid_object_mask.squeeze(-1).sum(dim=1).item()

        assert (current_bg_valid_count + current_obj_valid_count == 576), "总有效token数应为576"

        # 移除旧的统计调用，改用统一的stats结构
        # 背景和前景token数量统计已在原始序列统计中处理


        # ============================================================================
        # 阶段3: 提取背景特征
        # ============================================================================
        bg_flat_calculated = background_features[
            valid_background_mask.repeat(1, 1, embedding_dim)
        ].reshape(batch_size, -1, embedding_dim)


        # ============================================================================
        # 阶段4: 缓存处理（写入/搜索）
        # ============================================================================
        reused_background_features_final = None
        cache_hit_status = False

        if self.cache_mode == "read-only":
            cache_hit_status = False

        elif self.cache_mode != "write-only" and bg_flat_calculated.shape[1] == 0:
            print("背景有效token数为0，跳过背景缓存处理。")
            cache_hit_status = False

        elif self.cache_mode == "write-only":
            self._cache_write_background_features(images, bg_flat_calculated, full_patch_mask, embedding_dim)
            cache_hit_status = False

        elif self.cache_mode == "read-load" and self.method_type == "segmentation-cache":
            reused_background_features_final = self._cache_search_background_features(images, bg_flat_calculated, full_patch_mask)
            cache_hit_status = reused_background_features_final is not None

        else:
            print(f"警告：未知的缓存模式 '{self.cache_mode}'。将不进行任何缓存操作。")
            cache_hit_status = False

        # 记录缓存统计
        # 记录缓存统计（与Qwen2.5-VL一致）
        if hasattr(self.get_model(), 'stats'):
            self.get_model().stats['cache_searches'] += 1
            if cache_hit_status:
                self.get_model().stats['cache_hits'] += 1


        # ============================================================================
        # 阶段5: 选择最终使用的背景特征
        # ============================================================================
        if reused_background_features_final is not None:
            background_features_to_use = reused_background_features_final.to(images.device)
            background_valid_final = background_features_to_use.shape[1]
        else:
            background_features_to_use = bg_flat_calculated
            background_valid_final = current_bg_valid_count


        # ============================================================================
        # 阶段6: 提取目标特征
        # ============================================================================
        valid_object_mask_final = object_attention_mask.bool().unsqueeze(-1).to(images.device)
        object_features_flat = object_features[valid_object_mask_final.repeat(1, 1, embedding_dim)].reshape(batch_size, -1, embedding_dim)


        # ============================================================================
        # 阶段7: 拼接背景和目标特征
        # ============================================================================
        total_valid_tokens = background_valid_final + current_obj_valid_count
        if total_valid_tokens != 576:
            print(f"警告：总有效 token 数应为576，但实际得到 {total_valid_tokens}。")

        concatenated_features = torch.cat(
            (background_features_to_use.squeeze(0), object_features_flat.squeeze(0)),
            dim=0
        ).unsqueeze(0)

        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
        # 记录segmentation-cache的with_cache统计（修复：根据缓存命中情况正确计算img_len）
        if hasattr(self.get_model(), 'stats'):
            # 根据Qwen2.5-VL的设计，正确计算实际需要重计算的token数量
            if cache_hit_status:
                # 缓存命中：只需要重计算前景token（背景从缓存复用）
                actual_img_len = current_obj_valid_count
                print(f"Segmentation Cache: 缓存命中，只重计算前景token: {actual_img_len}")
            else:
                # 缓存未命中：需要重计算所有图像token（背景+前景）
                actual_img_len = background_valid_final + current_obj_valid_count
                print(f"Segmentation Cache: 缓存未命中，重计算所有token: {actual_img_len}")

            # 使用默认的system和query长度（需要从原始序列中计算）
            # 这里简化处理，实际应该从之前记录的原始序列信息中获取
            if self.get_model().stats['no_cache_count'] > 0:
                last_system_len = self.get_model().stats['no_cache']['system_len'][-1] if self.get_model().stats['no_cache']['system_len'] else 0
                last_query_len = self.get_model().stats['no_cache']['query_len'][-1] if self.get_model().stats['no_cache']['query_len'] else 0

                self.get_model().stats['with_cache']['system_len'].append(last_system_len)
                self.get_model().stats['with_cache']['img_len'].append(actual_img_len)
                self.get_model().stats['with_cache']['query_len'].append(last_query_len)
                self.get_model().stats['with_cache_count'] += 1
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        return concatenated_features


    def encode_object_only_images(self, images, masks):
        """
        编码目标图像，仅使用目标部分，舍弃背景。
        基于 segmentation-cache 的逻辑，但只返回目标特征。
        """
        background_object_visual_tower = self.get_model().get_vision_tower().to(images.device)
        background_features, object_features, background_attention_mask, object_attention_mask, _, _ = background_object_visual_tower(images, masks)
        
        # 只处理目标特征
        object_features = self.get_model().mm_projector(object_features).to(object_features.device)
        
        batch_size, _, embedding_dim = object_features.shape
        valid_object_mask = object_attention_mask.bool().unsqueeze(-1).to(images.device)
        
        # 提取有效的目标特征
        obj_flat = object_features[valid_object_mask.repeat(1, 1, embedding_dim)].reshape(batch_size, -1, embedding_dim)

        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
        # 记录object-only的with_cache统计（只计算前景token）
        if hasattr(self.get_model(), 'stats'):
            # Object-only模式：只使用前景token，所以img_len就是前景token数量
            actual_fg_len = valid_object_mask.squeeze(-1).sum(dim=1).item()
            print(f"Object-only: 仅使用前景token，重计算token数: {actual_fg_len}")

            # 获取原始序列信息
            if self.get_model().stats['no_cache_count'] > 0:
                last_system_len = self.get_model().stats['no_cache']['system_len'][-1] if self.get_model().stats['no_cache']['system_len'] else 0
                last_query_len = self.get_model().stats['no_cache']['query_len'][-1] if self.get_model().stats['no_cache']['query_len'] else 0

                self.get_model().stats['with_cache']['system_len'].append(last_system_len)
                self.get_model().stats['with_cache']['img_len'].append(actual_fg_len)  # 仅前景token
                self.get_model().stats['with_cache']['query_len'].append(last_query_len)
                self.get_model().stats['with_cache_count'] += 1
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

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
                        # 与 segmentation-cache 对齐：使用 CLI 传入的 similarity_threshold
                        distance_threshold=self.get_model().similarity_threshold
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
            pass
            # 记录缓存统计（与Qwen2.5-VL一致）
        if hasattr(self.get_model(), 'stats'):
            self.get_model().stats['cache_searches'] += 1
            if cache_hit_status:
                self.get_model().stats['cache_hits'] += 1
            
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
                            from llava.mm_utils import get_anyres_image_grid_shape
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

            # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
            # 记录原始序列统计（与Qwen2.5-VL保持一致）
            if hasattr(self.get_model(), 'stats') and images is not None:
                original_img_token_indices = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
                if original_img_token_indices.numel() > 0:
                    original_system_len = original_img_token_indices[0].item()
                    original_img_len = original_img_token_indices.numel()
                    original_query_len = input_ids.shape[1] - (original_system_len + original_img_len)
                else:
                    original_system_len = input_ids.shape[1]
                    original_img_len = 0
                    original_query_len = 0

                self.get_model().stats['no_cache']['system_len'].append(original_system_len)
                self.get_model().stats['no_cache']['img_len'].append(original_img_len)
                self.get_model().stats['no_cache']['query_len'].append(original_query_len)
                self.get_model().stats['no_cache_count'] += 1
            # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

            # 检查是否为CacheBlend的write-only模式
            if getattr(self.get_model(), 'method_type', None) == 'cacheblend' and \
               getattr(self.get_model(), 'cache_mode', None) == 'write-only':
                # 执行CacheBlend write-only逻辑
                self._perform_cacheblend_write_only_pass(images, masks)

            # 如果 image_features 尚未被计算（即非cacheblend 的 write-only路径），则在这里计算
            if image_features is None:
                if self.method_type == "segmentation-cache" or (self.method_type == "cacheblend" and self.get_model().cache_mode == "read-load"):
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
                    # 记录图像tokens的准确起始位置（在添加图像特征之前）
                    image_start_position = sum(x.shape[0] for x in cur_new_input_embeds)

                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

                    # 存储图像位置信息（用于segmentation_cache的position_ids构建）
                    if not hasattr(self.get_model(), '_image_token_positions'):
                        self.get_model()._image_token_positions = []
                    self.get_model()._image_token_positions.append(image_start_position)
            
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

        # 位置优化的调试代码
        # # 检查是否需要修正position_ids（segmentation_cache模式）
        # if (getattr(self, 'method_type', None) == "segmentation-cache" and
        #     hasattr(self.get_model(), '_segmentation_reorder_mapping')):

        #     # 构建正确的position_ids
        #     corrected_position_ids = self._build_segmentation_position_ids(
        #         _position_ids, past_key_values, new_input_embeds
        #     )

        #     # 清理临时映射信息
        #     delattr(self.get_model(), '_segmentation_reorder_mapping')

        #     return None, corrected_position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

        # native模式保持原逻辑
        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def _build_segmentation_position_ids(self, original_position_ids, past_key_values, inputs_embeds):
        """构建segmentation_cache的正确position_ids"""
        device = inputs_embeds.device
        batch_size, seq_len = inputs_embeds.shape[:2]

        # 1. 计算past_key_values偏移
        past_length = 0
        if past_key_values is not None:
            print("目前不支持处理 past_key_values 不为 None的情况")
            if hasattr(past_key_values, 'get_usable_length'):
                past_length = past_key_values.get_usable_length(seq_len)
            else:
                past_length = past_key_values[0][0].shape[2]

        # 2. 创建基础position_ids
        base_position_ids = torch.arange(past_length, seq_len + past_length, dtype=torch.long, device=device)
        position_ids = base_position_ids.unsqueeze(0).expand(batch_size, -1)

        # 3. 获取重排映射
        reorder_mapping = self.get_model()._segmentation_reorder_mapping  # [total_valid_tokens] tensor

        # 4. 获取准确的图像tokens起始位置
        if hasattr(self.get_model(), '_image_token_positions'):
            image_start_pos = self.get_model()._image_token_positions[0]  # 假设只处理第一张图
            # 清理临时位置信息
            delattr(self.get_model(), '_image_token_positions')
        else:
            raise ValueError("Cannot find accurate image token positions for segmentation_cache")

        # 5. 根据重排映射调整图像部分的position_ids
        for i, original_patch_idx in enumerate(reorder_mapping):
            current_pos = image_start_pos + i
            if current_pos < seq_len:
                # 使用原始patch的真实位置 + 基础偏移
                position_ids[0, current_pos] = past_length + image_start_pos + original_patch_idx.item()

        return position_ids

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
