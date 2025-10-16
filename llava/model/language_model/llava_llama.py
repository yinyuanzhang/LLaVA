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


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


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
                
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    ################################################################################
    ####### 新增方法：用于在模型加载后注入 CacheBlend Attention #######
    def enable_cacheblend_attention(self):
        """
        遍历所有解码器层，并将原生的 LlamaAttention 替换为我们的 CacheBlendLlamaAttention 包装器。
        这个方法应该在模型权重（包括LoRA）完全加载和合并后调用。
        """
        if getattr(self.config, 'method_type', None) != 'cacheblend':
            return
            
        try:
            # 动态导入，避免循环依赖
            from ..cacheblend_llama import CacheBlendLlamaAttention
            from types import SimpleNamespace

            # 创建一个轻量级的上下文对象，用于在各层间共享信息
            # 注意：这里传递的是 self (即 LlavaLlamaForCausalLM 实例)
            # 包装器可以通过 self.ctx.model.get_model() 访问到 LlavaLlamaModel
            ctx = SimpleNamespace(
                model=self, 
                config=self.config,
            )
            
            print("正在启用 CacheBlend Attention 包装器...")
            for i, layer in enumerate(self.model.layers):
                layer.self_attn = CacheBlendLlamaAttention(
                    attn=layer.self_attn, 
                    parent_ctx=ctx, 
                    layer_idx=i
                )
            print(f"CacheBlend: 成功为 {len(self.model.layers)} 个层注入了 Attention 包装器。")
        except Exception as e:
            import traceback
            print(f"CacheBlend Attention 注入失败: {e}")
            traceback.print_exc()
    ####### 方法结束 #######
    ################################################################################

    ################################################################################
    ####### 新增辅助函数：处理 read-load 逻辑 #######
    def _handle_cacheblend_read_load(
        self,
        input_ids, position_ids, attention_mask,
        images, masks, image_sizes
    ):
        """
        处理 cacheblend 的 read-load 逻辑。
        它接收原始输入，返回可能被重建的新输入。
        """

        # 在'read-load'模式下，我们首先需要一个完整的`inputs_embeds`来进行比较和重建。
        # 因此，我们先调用一次`prepare_inputs_labels_for_multimodal`来生成它。
        (
            _, _, _, _, original_inputs_embeds, _
        ) = self.prepare_inputs_labels_for_multimodal(
            input_ids, position_ids, attention_mask, None, None,
            images, masks, image_sizes
        )
        
        # 现在我们拥有了'original_inputs_embeds'，可以安全地执行后续操作
        print("CacheBlend 'read-load' 模式：启动缓存搜索...")
        cached_data = self._search_and_prepare_cached_data(images, masks)
        
        if cached_data['hit']:
            print("CacheBlend 'read-load' 模式：缓存命中！正在重建输入...")
            # 如果命中，则重建整个输入序列。
            (
                new_input_ids, new_position_ids, new_attention_mask, 
                new_inputs_embeds, old_kvs
            ) = self._rebuild_inputs_from_cache(input_ids, original_inputs_embeds, cached_data)

            # 将准备好的数据附加到模型实例上，供 Attention Wrapper 使用
            if not hasattr(self, 'cache_fuse_metadata'):
                self.cache_fuse_metadata = {}
            
            from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
            image_token_indices = torch.where(new_input_ids == IMAGE_TOKEN_INDEX)[1]
            image_start_index = image_token_indices[0].item()
            image_len = len(image_token_indices)
            
            self.cache_fuse_metadata.update({
                "old_kvs": old_kvs,
                "image_span": (image_start_index, image_len),
                "bg_len": cached_data['bg_tokens'],
                "fg_len": cached_data['fg_tokens'],
                "is_bg_hit": cached_data['bg_kv'] is not None,
                "is_fg_hit": cached_data['fg_kv'] is not None,
            })
            
            # 返回重建后的输入
            return new_input_ids, new_position_ids, new_attention_mask, new_inputs_embeds
        else:
            # 缓存未命中，返回原始输入
            print("CacheBlend 'read-load' 模式：缓存未命中，执行原生流程。")
            return input_ids, position_ids, attention_mask, original_inputs_embeds
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
            return_dict=return_dict
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
           getattr(model, 'cache_mode', None) == 'read-load' and \
           images is not None:
            
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
        
        # 步骤2：将原生的 if 分支变为 elif，处理所有其他多模态情况
        elif images is not None:
            # 对于 write-only 或 native 模式，执行原始的 prepare_inputs... 流程
            # （write-only 的副作用会在这里的 prepare_inputs... 内部被触发）
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

        # 最终，所有分支都会准备好 inputs_embeds, position_ids, attention_mask
        # 然后统一传递给 super().generate()
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
        ####### 修改结束 #######
        ################################################################################

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
