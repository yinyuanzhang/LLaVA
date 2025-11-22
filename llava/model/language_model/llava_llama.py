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


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from ..cacheblend_llama import CacheBlendLlamaModel


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class CacheBlendLlavaLlamaModel(LlavaMetaModel, CacheBlendLlamaModel):
    """
    CacheBlend 版本的 LlavaLlamaModel，整合多模态功能和 CacheBlend 架构
    采用标准多重继承：LlavaMetaModel + CacheBlendLlamaModel

    设计原则：
    - LlavaMetaModel: 提供多模态功能
    - CacheBlendLlamaModel: 提供CacheBlend核心逻辑
    - 本类: 只负责参数转换和接口适配
    """
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super().__init__(config)

    def _prepare_cacheblend_kwargs(self):
        """
        将 LLaVA 的 cache_fuse_metadata 转换为 CacheBlend 需要的 kwargs 格式
        这是LLaVA特有的适配逻辑，不应该重复实现CacheBlend核心逻辑
        """
        if not hasattr(self, 'cache_fuse_metadata'):
            return {}

        metadata = self.cache_fuse_metadata

        # 转换为 CacheBlend 标准格式（模仿 Qwen2.5-VL）
        cacheblend_kwargs = {
            "cacheblend_status": 0,  # 默认状态，将在 DecoderLayer 中动态调整
            "cacheblend_metadata": metadata,
            "cacheblend_old_kvs": getattr(self, 'old_kvs', [])
        }

        return cacheblend_kwargs

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """
        LLaVA版本的CacheBlend forward

        职责：
        1. 处理LLaVA特有的参数转换 (_prepare_cacheblend_kwargs)
        2. 委托给CacheBlendLlamaModel处理核心CacheBlend逻辑

        避免重复实现decoder loop和稀疏化传播逻辑
        """
        # 🎯 LLaVA特有逻辑：参数格式转换
        cacheblend_kwargs = self._prepare_cacheblend_kwargs()

        # 🔄 将LLaVA格式的参数注入到kwargs中
        kwargs.update(cacheblend_kwargs)

        # ✅ 委托给CacheBlendLlamaModel处理所有CacheBlend核心逻辑
        # 这样避免了重复实现decoder loop、稀疏化传播等逻辑
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs
        )


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config, *model_args):
        if hasattr(model_args[0], "method_type"):
            config.method_type = model_args[0].method_type
            self.method_type = model_args[0].method_type
        if hasattr(model_args[0], "cache_mode"):
            config.cache_mode = model_args[0].cache_mode
        if hasattr(model_args[0], "dataset"):
            config.dataset = model_args[0].dataset

        # 【新增】轻量级query_key参数传递
        if hasattr(model_args[0], "use_lightweight_query_key"):
            config.use_lightweight_query_key = model_args[0].use_lightweight_query_key
        if hasattr(model_args[0], "query_key_extractor_type"):
            config.query_key_extractor_type = model_args[0].query_key_extractor_type
        if hasattr(model_args[0], "similarity_threshold"):
            config.similarity_threshold = model_args[0].similarity_threshold

        # 【新增】灵活路由参数传递
        if hasattr(model_args[0], "is_flexible_route"):
            config.is_flexible_route = model_args[0].is_flexible_route

        super(LlamaForCausalLM, self).__init__(config)

        # 【核心修复】根据 method_type 和 cache_mode 选择模型架构
        if getattr(config, 'method_type', None) == 'cacheblend':
            print("CacheBlend: 初始化 CacheBlend 架构。")
            self.model = CacheBlendLlavaLlamaModel(config)
        else:
            print("CacheBlend: 初始化原生 LlavaLlamaModel 架构。")
            self.model = LlavaLlamaModel(config)

        # 【新增】初始化 CacheBlend 需要的 old_kvs
        num_layers = getattr(config, 'num_hidden_layers', 0)
        self.old_kvs = [[None, None] for _ in range(num_layers)]

        # 【新增】初始化 cache_fuse_metadata（模仿 Qwen2.5-VL）
        self.cache_fuse_metadata = {
            "check_layers": [1],
            "recomp_ratio": 1.0,
            "check": False,
            "collect": False,
            "imp_indices": None,
        }

        # 【新增】添加 cache_mode 管理（模仿 Qwen2.5-VL）
        self.cache_mode = getattr(config, 'cache_mode', 'write-only')  # 从 config 中获取，默认 write-only

        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model


    def _update_cacheblend_state(self):
        """根据当前cache_mode更新CacheBlend的运行状态（模仿 Qwen2.5-VL）"""
        if self.cache_mode == "read-load":
            self.cache_fuse_metadata["check"] = True
            self.cache_fuse_metadata["collect"] = False
        elif self.cache_mode == "write-only":
            self.cache_fuse_metadata["check"] = False
            self.cache_fuse_metadata["collect"] = True
        else:
            self.cache_fuse_metadata["check"] = False
            self.cache_fuse_metadata["collect"] = False

        self.cache_fuse_metadata["imp_indices"] = None

        # 【修复】同步到 CacheBlend 模型
        if hasattr(self.model, '_prepare_cacheblend_kwargs'):
            self.model.cache_fuse_metadata = self.cache_fuse_metadata


    ################################################################################
    ####### 优化后的 read-load 逻辑处理函数，直接复制 Qwen2.5-VL 的流程 #######
    def _handle_cacheblend_read_load(
        self,
        input_ids, position_ids, attention_mask,
        images, masks, image_sizes
    ):
        """
        【优化版】处理 cacheblend 的 read-load 逻辑，参考 Qwen2.5-VL 的实现流程

        核心改进：
        1. 统一使用 _extract_vision_features 进行特征提取（支持轻量级方法）
        2. 使用 _search_similar_patches 进行缓存搜索
        3. 使用 _rebuild_inputs_for_variable_cache 进行输入重建
        4. 使用 _prepare_cache_fusion_metadata 准备元数据
        """

        # 1. 先获取完整的inputs_embeds（这是必要的，因为需要与缓存进行对比） todo: 这里返回的 inputs_embeds是按照 BG-FG 的拼接顺序的feature
        (
            _, _, _,
            _, inputs_embeds, _
        ) = self.prepare_inputs_labels_for_multimodal(
            input_ids, position_ids, attention_mask, None, None,
            images, masks, image_sizes
        )

        # 2. 【修正】获取真正的图像特征和前景掩码（参考write-only模式的实现）
        model = self.get_model()
        vision_tower = self.get_vision_tower()

        # 【关键修正】使用与write-only模式相同的视觉处理流程来获取真正的掩码信息
        with torch.no_grad():
            # 获取分离的背景/前景特征和掩码信息（与write-only模式一致）
            bg_feats, fg_feats, bg_attn_mask, fg_attn_mask, full_patch_mask, _ = vision_tower(images, masks)
            bg_embeds = model.mm_projector(bg_feats)
            fg_embeds = model.mm_projector(fg_feats)

            # 提取有效的背景和前景patches
            bg_mask = bg_attn_mask.bool().unsqueeze(-1).expand_as(bg_embeds)
            fg_mask = fg_attn_mask.bool().unsqueeze(-1).expand_as(fg_embeds)

            bg_embeds_flat = bg_embeds[bg_mask].reshape(1, -1, model.config.hidden_size)
            fg_embeds_flat = fg_embeds[fg_mask].reshape(1, -1, model.config.hidden_size)

            # 【关键】构造真正的final_is_foreground_mask (True表示前景)
            final_is_foreground_mask = (full_patch_mask[0] == 1)  # [num_patches] bool tensor

            # 合并所有image embeds用于特征提取（与write-only模式一致）
            num_bg_tokens = bg_embeds_flat.shape[1]
            num_fg_tokens = fg_embeds_flat.shape[1]

            if num_bg_tokens > 0 and num_fg_tokens > 0:
                all_image_embeds = torch.cat([bg_embeds_flat.squeeze(0), fg_embeds_flat.squeeze(0)], dim=0)
            elif num_bg_tokens > 0:
                all_image_embeds = bg_embeds_flat.squeeze(0)
            else:
                all_image_embeds = fg_embeds_flat.squeeze(0)

        print("CacheBlend 'read-load' 模式：开始使用统一的特征提取和缓存搜索...")

        # 3. 【优化】使用统一的特征提取函数（使用真正的图像嵌入和掩码）
        raw_images = self._to_raw_images(images)
        bg_feature, fg_feature = self._extract_vision_features(
            all_image_embeds,
            final_is_foreground_mask,
            raw_image_tensor=raw_images,  # 使用像素域原图以配合轻量级抽取器
            image_shape=(images.shape[2], images.shape[3]),
            bg_token_count=num_bg_tokens,  # 新增：VIT方法需要
            fg_token_count=num_fg_tokens   # 新增：VIT方法需要            
        )

        # 【新增】根据配置决定是否使用FG缓存
        if not getattr(self.config, "use_fg_cache", False):
            fg_feature = None
            print("CacheBlend: Foreground cache disabled, FG will be fully recomputed.")
            
        # 4. 【优化】使用统一的缓存搜索函数
        bg_kv_cache, bg_tokens, bg_pos_ids, bg_embeds, \
        fg_kv_cache, fg_tokens, fg_pos_ids, fg_embeds = self._search_similar_patches(bg_feature, fg_feature)

        cache_hit = bg_kv_cache is not None or fg_kv_cache is not None

        # 处理None情况
        bg_tokens = bg_tokens if bg_tokens is not None else 0
        fg_tokens = fg_tokens if fg_tokens is not None else 0

        if cache_hit:
            print("CacheBlend 'read-load' 模式：缓存命中！正在使用变长缓存重建输入...")

            # ！！todo: 这里的position_ids 暂时使用虚拟的、手动构造的
            # 手动构造 position_ids：创建一个从0到序列长度-1的递增序列   【todo:这里也是需要进行更改的】
            if position_ids is None:
                device = inputs_embeds.device
                seq_len = inputs_embeds.shape[1]
                batch_size = inputs_embeds.shape[0]
                # 构造标准的 2D position_ids: (batch_size, seq_len)
                position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
                print(f"CacheBlend: 手动构造 position_ids，形状：{position_ids.shape}")

            # 【新增】扩充input_ids使其与inputs_embeds对应，【这里是llava与Qwen25vl实现上有差异的点】
            expanded_input_ids = self._expand_image_tokens_in_input_ids(
                input_ids, num_bg_tokens + num_fg_tokens
            )

            # 5. 【优化】使用Qwen2.5-VL的变长缓存重建逻辑
            new_input_ids, new_inputs_embeds, new_position_ids, new_attention_mask, \
            new_bg_len, new_fg_len = self._rebuild_inputs_for_variable_cache(
                expanded_input_ids, inputs_embeds, position_ids, attention_mask,
                num_bg_tokens,  # 修改：传入当前BG token数量，而不是final_is_foreground_mask
                num_fg_tokens,  # 修改：传入当前FG token数量
                bg_embeds, bg_tokens, bg_pos_ids,
                fg_embeds, fg_tokens, fg_pos_ids
            )

            # 6. 【优化】准备缓存融合元数据
            cached_data = {
                'bg_kv_cache': bg_kv_cache, 'bg_tokens': new_bg_len,
                'fg_kv_cache': fg_kv_cache, 'fg_tokens': new_fg_len,
            }

            self._prepare_cache_fusion_metadata(self.cache_fuse_metadata, new_input_ids, new_inputs_embeds, cached_data)

            # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
            # 记录CacheBlend的with_cache统计（修复：按照Qwen2.5-VL的正确逻辑）
            if hasattr(self.model, 'stats'):
                # 根据Qwen2.5-VL CacheBlend设计，计算实际需要重计算的token数量
                recomp_ratio = 0.16  # 与Qwen2.5-VL保持一致的重计算比例

                # 检查BG/FG缓存命中情况
                bg_hit = bg_tokens > 0 if bg_tokens is not None else False
                fg_hit = fg_tokens > 0 if fg_tokens is not None else False

                # 分别计算BG和FG的重计算token数量
                recomp_bg_len = new_bg_len if not bg_hit else int(new_bg_len * recomp_ratio)
                recomp_fg_len = new_fg_len if not fg_hit else int(new_fg_len * recomp_ratio)

                recomp_img_len = recomp_bg_len + recomp_fg_len

                # 从原始序列信息中获取system和query长度
                if self.model.stats['no_cache_count'] > 0:
                    last_system_len = self.model.stats['no_cache']['system_len'][-1] if self.model.stats['no_cache']['system_len'] else 0
                    last_query_len = self.model.stats['no_cache']['query_len'][-1] if self.model.stats['no_cache']['query_len'] else 0

                    self.model.stats['with_cache']['system_len'].append(last_system_len)
                    self.model.stats['with_cache']['img_len'].append(recomp_img_len)  # 实际重计算的token数
                    self.model.stats['with_cache']['query_len'].append(last_query_len)
                    self.model.stats['with_cache_count'] += 1

                    print(f"CacheBlend: BG命中={bg_hit}, FG命中={fg_hit}, 重计算token: BG={recomp_bg_len}, FG={recomp_fg_len}, 总计={recomp_img_len}")
            # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

            return new_input_ids, new_position_ids, new_attention_mask, new_inputs_embeds
        else:
            print("CacheBlend 'read-load' 模式：缓存未命中，执行原生流程。")
            # 即使未命中，也要设置基本的元数据以确保所有层都知道当前状态
            self.cache_fuse_metadata["check"] = False

            # 【新增】Flexible routing: 缓存未命中时切换到 Native 模式
            is_flexible_route = True  # 先固定为 True，后续可改为参数

            if is_flexible_route:
                print("CacheBlend: 启用 Flexible routing，切换到 Native 编码")

                # 使用 Native 模式重新编码图像
                native_image_features = self.encode_images(images)  # [1, 576, hidden_size]

                # 替换 inputs_embeds 中的图像部分
                from llava.constants import IMAGE_TOKEN_INDEX

                # 定位图像 tokens 的位置
                image_token_mask = (input_ids[0] == IMAGE_TOKEN_INDEX)
                image_indices = torch.where(image_token_mask)[0]

                if len(image_indices) > 0:
                    first_image_pos = image_indices[0].item()
                    last_image_pos = image_indices[-1].item()

                    # 提取非图像部分的 embeddings
                    pre_image_embeds = inputs_embeds[:, :first_image_pos, :]
                    post_image_embeds = inputs_embeds[:, last_image_pos+1:, :]

                    # 重建 inputs_embeds: system + native_image + query
                    native_inputs_embeds = torch.cat([
                        pre_image_embeds,
                        native_image_features,  # 使用 Native 编码的图像特征
                        post_image_embeds
                    ], dim=1)

                    print(f"CacheBlend: Native 编码完成，序列长度: {native_inputs_embeds.shape[1]}")

                    return input_ids, position_ids, attention_mask, native_inputs_embeds
                else:
                    print("CacheBlend WARNING: 未找到图像 tokens，使用原始 inputs_embeds")

            # 如果不启用 flexible routing，或者找不到图像 tokens，使用原始逻辑
            return input_ids, position_ids, attention_mask, inputs_embeds

            # 这是原本代码
            return input_ids, position_ids, attention_mask, inputs_embeds
    ####### 函数结束 #######
    ################################################################################

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        masks: Optional[torch.Tensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                masks,
                image_sizes
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:

        ################################################################################
        ####### 在不改变原生结构的基础上，注入 CacheBlend 的 read-load 逻辑 #######
        
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        model = self.get_model()

        # 步骤1：专门为 cacheblend 的 read-load 模式创建一个分支
        if getattr(model, 'method_type', None) == 'cacheblend' and \
           self.cache_mode == 'read-load' and \
           images is not None:

            # 【新增】检测新图像输入并更新 CacheBlend 状态（模仿 Qwen2.5-VL）
            is_new_generation = images is not None
            if is_new_generation:
                self.cache_fuse_metadata = {
                    "check_layers": [1],
                    "recomp_ratio": 1.0,
                    "check": False,
                    "collect": False,
                    "imp_indices": None,
                }
                                
                # 如果是新的图片输入，重置/更新 CacheBlend 状态
                self._update_cacheblend_state()
                for i in range(len(self.old_kvs)):
                    self.old_kvs[i] = [None, None]

            # 在这个分支中，我们调用辅助函数来完成所有 read-load 的输入准备工作
            (
                inputs,
                position_ids,
                attention_mask,
                inputs_embeds
            ) = self._handle_cacheblend_read_load(
                inputs, position_ids, attention_mask,
                images, masks, image_sizes
            )
            position_ids = None    # 这里是为了解决bug，专门设置的。
            attention_mask = None
            
            # --- CacheBlend 支持：设置模型的缓存元数据，不传递给 generate ---
            # 【修复】直接设置模型的缓存参数，避免传递给 generate 导致验证错误(内部参数校验不通过)
            if hasattr(self.model, '_prepare_cacheblend_kwargs'):
                self.model.cache_fuse_metadata = self.cache_fuse_metadata.copy()
                self.model.old_kvs = self.old_kvs.copy()

            # 【修复】不传递 CacheBlend 特定参数给 super().generate()
            return super().generate(
                position_ids=position_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                **kwargs  # 只传递标准参数
            )

        # 步骤2：将原生的 if 分支变为 elif，处理所有其他多模态情况
        elif images is not None:
            # 对于 write-only 或 native 模式，执行原始的 prepare_inputs... 流程
            # （write-only 的额外逻辑会在这里的 prepare_inputs_labels_for_multimodal 内部实现）
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                masks,
                image_sizes=image_sizes
            )

        # 步骤3：原生代码的 else 分支保持不变，处理纯文本输入
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        # --- CacheBlend 支持：如果模型已替换为 CacheBlend 版本，设置额外参数 ---
        # 【修复】直接设置模型参数，而不是传递给 generate
        if isinstance(self.model, type(self.model)) and hasattr(self.model, '_prepare_cacheblend_kwargs'):
            # 这是我们的 CacheBlend 模型，需要设置额外参数
            if hasattr(self, 'cache_fuse_metadata'):
                self.model.cache_fuse_metadata = self.cache_fuse_metadata
                self.model.old_kvs = getattr(self, 'old_kvs', [])

        # 最终，所有分支都会准备好 inputs_embeds, position_ids, attention_mask
        # 然后统一传递给 super().generate()（不包含CacheBlend特定参数）
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs  # 只传递标准参数
        )
        ####### 修改结束 #######
        ################################################################################

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)

        # 位置优化的调试代码
        # # 针对自定义 position_ids 的关键修改
        # is_decode_phase = past_key_values is not None
        # is_segmentation_mode = getattr(self, 'method_type', None) == 'segmentation-cache'
        # if is_decode_phase and is_segmentation_mode:
        #     kwargs.pop("position_ids", None)

        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

    def _expand_image_tokens_in_input_ids(self, input_ids, total_image_tokens):
        """
        将input_ids中的单个IMAGE_TOKEN_INDEX扩充为多个IMAGE_TOKEN_INDEX，
        使其与展开后的inputs_embeds长度对应。

        Args:
            input_ids: 原始input_ids，包含一个IMAGE_TOKEN_INDEX占位符 [1, seq_len]
            total_image_tokens: 图像展开后的总token数量（BG + FG）

        Returns:
            扩充后的input_ids，长度为 原长度 - 1 + total_image_tokens

        Example:
            原始: [1, 2, 3, -200, 4, 5] (length=6)
            total_image_tokens = 576
            结果: [1, 2, 3, -200, -200, ..., -200, 4, 5] (length=6-1+576=581)
        """
        from llava.constants import IMAGE_TOKEN_INDEX

        if input_ids is None:
            return None

        # 找到IMAGE_TOKEN_INDEX的位置
        image_token_mask = (input_ids[0] == IMAGE_TOKEN_INDEX)
        image_indices = torch.where(image_token_mask)[0]

        if len(image_indices) == 0:
            # 没有找到图像token，直接返回原input_ids
            print("CacheBlend WARNING: 没有找到IMAGE_TOKEN_INDEX，返回原始input_ids")
            return input_ids

        if len(image_indices) > 1:
            # 如果有多个图像token，目前只处理第一个
            print(f"CacheBlend WARNING: 发现{len(image_indices)}个IMAGE_TOKEN_INDEX，只处理第一个")

        # 取第一个图像token的位置
        image_pos = image_indices[0].item()

        # 分割序列为三部分：前缀、图像占位符、后缀
        device = input_ids.device
        dtype = input_ids.dtype

        prefix = input_ids[:, :image_pos]  # 图像前的部分
        suffix = input_ids[:, image_pos + 1:]  # 图像后的部分

        # 创建扩充的图像占位符
        expanded_image_tokens = torch.full(
            (1, total_image_tokens),
            IMAGE_TOKEN_INDEX,
            device=device,
            dtype=dtype
        )

        # 重新拼接
        expanded_input_ids = torch.cat([prefix, expanded_image_tokens, suffix], dim=1)

        print(f"CacheBlend: 扩充input_ids - 原长度: {input_ids.shape[1]}, 新长度: {expanded_input_ids.shape[1]}")
        print(f"  图像位置: {image_pos}, 扩充token数: {total_image_tokens}")

        return expanded_input_ids

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
