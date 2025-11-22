# 文件名: llava/model/cacheblend_llama.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from transformers.models.llama.modeling_llama import LlamaAttention, LlamaDecoderLayer, LlamaModel, repeat_kv, apply_rotary_pos_emb
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from typing import Optional, List, Union, Tuple
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.utils.import_utils import is_torch_fx_available
from transformers.utils import logging
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_13


logger = logging.get_logger(__name__)

# This makes `_prepare_4d_causal_attention_mask` a leaf function in the FX graph.
# It means that the function will not be traced through and simply appear as a node in the graph.
if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx

    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)


### 步骤 1: 创建一个包含 CacheBlend 核心逻辑的 Mixin 类 ###
class CacheBlendAttentionMixin:
    """
    一个 Mixin 类，封装了 CacheBlend 的核心算法。
    它可以被 "混入" 到任何 Attention 实现中。

    参考 Qwen2.5-VL 的实现模式，提取通用的 CacheBlend 逻辑。
    """

    def _calculate_importance_and_sparsify_query(self, query_states, value_states, cacheblend_metadata, cacheblend_old_kv):
        """
        在 check 层（通常是第1层）执行。
        1. 对比图像部分的 V 和缓存的 V，计算出需要重计算的图像 token 索引。
        2. 结合必须重计算的 system 和 query 部分，生成最终的稀疏化索引 `imp_indices`。
        3. 使用 `imp_indices` 稀疏化 query_states。
        """
        old_key, old_value = cacheblend_old_kv if cacheblend_old_kv else (None, None)

        # 检查缓存是否有效
        if old_key is None or old_value is None:
            cacheblend_metadata["imp_indices"] = None
            return query_states # 无法比较，返回原始 query

        # 1. 获取元数据
        system_len = cacheblend_metadata.get("system_prompt_len", 0)
        cacheable_start = cacheblend_metadata.get("cacheable_start", 0)
        cacheable_len = cacheblend_metadata.get("cacheable_len", 0)
        recomp_ratio = cacheblend_metadata.get('recomp_ratio', 0.16)

        # 获取精细化的BG/FG信息
        bg_len = cacheblend_metadata.get("bg_tokens_len", 0)
        fg_len = cacheblend_metadata.get("fg_tokens_len", 0)
        is_bg_hit = cacheblend_metadata.get("is_bg_hit", False)
        is_fg_hit = cacheblend_metadata.get("is_fg_hit", False)

        # 2. 精确切片，获取完整的当前图像V值和带padding的缓存V值
        current_image_part_v = value_states[:, :, cacheable_start : cacheable_start + cacheable_len, :]
        cached_image_part_v = old_value.to(current_image_part_v.device)

        if current_image_part_v.shape[2] != cached_image_part_v.shape[2]:
            print(f"CacheBlend WARNING: Image length mismatch. Cannot perform importance calculation.")
            cacheblend_metadata["imp_indices"] = None
            return query_states

        # 3. 计算所有图像token的差异
        temp_diff = torch.sum((current_image_part_v - cached_image_part_v)**2, dim=[1, 3]).squeeze(0)
        image_indices_to_recompute = []

        # 4. 单独处理背景 (BG) 区域
        if bg_len > 0:
            if not is_bg_hit: # 规则1: BG未命中，100%重计算
                indices = torch.arange(0, bg_len, device=query_states.device, dtype=torch.long)
                image_indices_to_recompute.append(indices)
            else: # 规则2: BG命中，按比例计算topk
                topk_num_bg = max(1, int(bg_len * recomp_ratio))
                temp_diff_bg = temp_diff[0:bg_len]
                topk_indices_in_bg = torch.topk(temp_diff_bg, k=min(topk_num_bg, bg_len)).indices
                image_indices_to_recompute.append(topk_indices_in_bg)

        # 5. 单独处理前景 (FG) 区域
        if fg_len > 0:
            fg_start_index = bg_len
            if not is_fg_hit: # 规则1: FG未命中，100%重计算
                indices = torch.arange(fg_start_index, fg_start_index + fg_len, device=query_states.device, dtype=torch.long)
                image_indices_to_recompute.append(indices)
            else: # 规则2: FG命中，按比例计算topk
                topk_num_fg = max(1, int(fg_len * recomp_ratio))
                temp_diff_fg = temp_diff[fg_start_index : fg_start_index + fg_len]
                topk_indices_in_fg = torch.topk(temp_diff_fg, k=min(topk_num_fg, fg_len)).indices
                # 将FG内部的索引转换为相对于整个图像部分的索引
                topk_indices_in_image_part = topk_indices_in_fg + fg_start_index
                image_indices_to_recompute.append(topk_indices_in_image_part)

        # 6. 合并所有需要重计算的图像token相对索引
        if image_indices_to_recompute:
            image_indices_relative = torch.cat(image_indices_to_recompute)
            image_indices_relative = torch.unique(image_indices_relative)
        else:
            image_indices_relative = torch.tensor([], device=query_states.device, dtype=torch.long)

        # 7. 构建完整的绝对索引列表
        system_indices = torch.arange(0, system_len, device=query_states.device, dtype=torch.long)
        image_indices_absolute = image_indices_relative + cacheable_start
        query_start = cacheable_start + cacheable_len
        total_seq_len = value_states.shape[2]
        query_indices = torch.arange(query_start, total_seq_len, device=query_states.device, dtype=torch.long)

        all_indices = torch.cat([system_indices, image_indices_absolute, query_indices])
        all_indices = torch.unique(all_indices, sorted=True)

        # 8. 稀疏化Query并保存索引
        query_states = query_states[:, :, all_indices, :]
        cacheblend_metadata["imp_indices"] = all_indices

        return query_states

    def _fuse_and_reconstruct_kv(self, query_states, key_states, value_states, cacheblend_metadata, cacheblend_old_kv):
        """
        在深层执行。
        1. 接收的是稀疏化的 K 和 V。
        2. 将稀疏 K/V 中属于图像的部分，更新（融合）到旧的完整图像缓存中。
        3. 重建一个完整的 K/V，其中 system 和 query 部分来自稀疏 K/V，图像部分来自融合后的缓存。
        """
        imp_indices = cacheblend_metadata.get("imp_indices")
        if imp_indices is None:
            return query_states, key_states, value_states # 不应发生，但作为安全保护

        old_key_img, old_value_img = cacheblend_old_kv if cacheblend_old_kv else (None, None)
        if old_key_img is None or old_value_img is None:
            return query_states, key_states, value_states

        # 1. 获取序列结构元数据
        cacheable_start = cacheblend_metadata.get("cacheable_start", 0)
        cacheable_len = cacheblend_metadata.get("cacheable_len", 0)

        # 2. 找到 imp_indices 中哪些属于图像部分，并计算它们的相对位置
        image_mask = (imp_indices >= cacheable_start) & (imp_indices < cacheable_start + cacheable_len)
        image_indices_rel = imp_indices[image_mask] - cacheable_start # 用于更新旧缓存的相对索引

        if image_indices_rel.numel() > 0:
            # 3. 从当前传入的稀疏 K/V 中，提取出新计算的图像部分
            image_sparse_positions = torch.where(image_mask)[0]
            current_image_key_sparse = key_states[:, :, image_sparse_positions, :]
            current_image_value_sparse = value_states[:, :, image_sparse_positions, :]

            # 4. 核心KV融合：将新计算的部分更新到旧缓存中
            old_key_img = old_key_img.to(current_image_key_sparse.device, current_image_key_sparse.dtype)
            old_value_img = old_value_img.to(current_image_value_sparse.device, current_image_value_sparse.dtype)
            old_key_img[:, :, image_indices_rel, :] = current_image_key_sparse
            old_value_img[:, :, image_indices_rel, :] = current_image_value_sparse

        # 5. 重建完整的 K/V 张量
        total_seq_len = cacheblend_metadata["org_seq_len"]
        new_key_states = torch.zeros(
            key_states.shape[0], key_states.shape[1], total_seq_len, key_states.shape[3],
            device=key_states.device, dtype=key_states.dtype
        )
        new_value_states = torch.zeros_like(new_key_states)

        # 6. 从稀疏 K/V 中分离出 system 和 query 部分
        system_mask = imp_indices < cacheable_start
        query_mask = imp_indices >= (cacheable_start + cacheable_len)

        # 7. 填充重建后的 K/V
        # a. 填充 system 和 query 部分
        system_abs_indices = imp_indices[system_mask]
        query_abs_indices = imp_indices[query_mask]
        new_key_states[:, :, system_abs_indices, :] = key_states[:, :, system_mask, :]
        new_value_states[:, :, system_abs_indices, :] = value_states[:, :, system_mask, :]
        new_key_states[:, :, query_abs_indices, :] = key_states[:, :, query_mask, :]
        new_value_states[:, :, query_abs_indices, :] = value_states[:, :, query_mask, :]

        # b. 填充融合后的图像部分
        new_key_states[:, :, cacheable_start:cacheable_start+cacheable_len, :] = old_key_img
        new_value_states[:, :, cacheable_start:cacheable_start+cacheable_len, :] = old_value_img

        return query_states, new_key_states, new_value_states

    def _apply_cacheblend_logic(
        self,
        query_states,
        key_states,
        value_states,
        cacheblend_status,
        cacheblend_metadata,
        cacheblend_old_kv,
    ):
        """主分发函数：根据 status 调用不同的处理逻辑"""
        if cacheblend_status == 0:
            return query_states, key_states, value_states

        elif cacheblend_status == 1:
            query_states = self._calculate_importance_and_sparsify_query(
                query_states, value_states, cacheblend_metadata, cacheblend_old_kv
            )
            # 在 status=1 时，K和V保持完整，因为下一层需要它们进行融合
            return query_states, key_states, value_states

        elif cacheblend_status == 2:
            return self._fuse_and_reconstruct_kv(
                query_states, key_states, value_states, cacheblend_metadata, cacheblend_old_kv
            )

        return query_states, key_states, value_states


class CacheBlendLlamaAttention(CacheBlendAttentionMixin, LlamaAttention):
    """
    CacheBlend 版本的 LlamaAttention，采用继承模式以保持权重路径一致。

    继承模式的优势：
    - 权重路径保持一致（self.q_proj.weight 而不是 self.attn.q_proj.weight）
    - from_pretrained 可以自动加载权重，无需手动复制
    - 代码结构更简洁，维护性更好

    核心逻辑：
    - Prefill/Layer1: 执行重要性选择 (计算 imp_indices); 不进行KV融合。
    - Prefill/Layer2+: 构建融合后的KV (对命中段复用缓存; 用新KV覆盖重要token),
      并为重要token重计算输出; 返回融合后的 present_kv。
    - Decode: 回退到原生的 Attention 逻辑。
    """
    def __init__(self, config, layer_idx: int):
        # 调用父类初始化，确保所有权重被正确创建
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx

    def _should_fallback(self, hidden_states: torch.Tensor, past_key_value: Optional[Cache] = None, **kwargs) -> bool:
        """
        判断是否应回退到原生 attention。
        只有在 read-load 模式且缓存命中时，才执行 CacheBlend 自定义逻辑。
        其他所有情况（Decode 阶段、write-only 模式等）都回退到原生 attention。
        """
        # 1. 检查是否为 Decode 阶段 (序列长度为1)
        if hidden_states.shape[1] == 1:
            return True

        # 2. 检查 past_key_value 内部是否已有数据（Decode 阶段的标志）
        if past_key_value is not None and past_key_value.get_seq_length(self.layer_idx) > 0:
            return True

        # 3. 从 kwargs 中获取 CacheBlend 相关信息
        cacheblend_metadata = kwargs.get("cacheblend_metadata", {})

        # 4. 如果 CacheBlend 未启用，回退
        if not cacheblend_metadata.get("check", False):
            return True

        # 5. 检查是否有必要的缓存数据
        cacheblend_old_kvs = kwargs.get("cacheblend_old_kvs", [])
        if not cacheblend_old_kvs or self.layer_idx >= len(cacheblend_old_kvs):
            return True

        old_kv = cacheblend_old_kvs[self.layer_idx]
        if not old_kv or old_kv[0] is None:
            return True

        # 6. 如果所有条件满足，执行 CacheBlend 逻辑
        return False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        """
        CacheBlend 版本的 forward，采用继承模式
        """
        # ============== 安全检查：是否需要回退到原生实现 ==============
        if self._should_fallback(hidden_states, past_key_value, **kwargs):
            return super().forward(
                hidden_states, attention_mask, position_ids,
                past_key_value, output_attentions, use_cache,
                cache_position=cache_position, **kwargs
            )

        # ============== 从这里开始是CacheBlend自定义逻辑 ==============
        cacheblend_metadata = kwargs.get("cacheblend_metadata", {})
        cacheblend_status = kwargs.get("cacheblend_status", 0)
        cacheblend_old_kv = kwargs.get("cacheblend_old_kv", [None, None])

        # ============== 步骤1：QKV投影 (与原生相同) ==============
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # ============== 步骤2：应用RoPE ==============
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        rope_seq_len = position_ids.max().item() + 1
        if rope_seq_len < kv_seq_len:
                        rope_seq_len = kv_seq_len
        cos, sin = self.rotary_emb(value_states, seq_len=rope_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # ============== 步骤3：应用 CacheBlend 逻辑 ==============
        query_states, key_states, value_states = self._apply_cacheblend_logic(
            query_states, key_states, value_states,
            cacheblend_status, cacheblend_metadata, cacheblend_old_kv
        )

        # ============== 步骤4：处理历史KV缓存 ==============
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # ============== 步骤5：执行注意力计算 ==============
        # 重复 K/V 头 (GQA/MQA)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # 计算注意力权重
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # # 应用注意力掩码
        # if attention_mask is not None:
        #     # 检查 attention mask 的维度是否匹配
        #     if attn_weights.size() != attention_mask.size():
        #         # 动态构建稀疏场景下的 Mask
        imp_indices = cacheblend_metadata.get("imp_indices")
        if imp_indices is not None and attn_weights.shape[2] == len(imp_indices):
            new_q_len, kv_len = attn_weights.shape[2], attn_weights.shape[3]
            kv_range = torch.arange(kv_len, device=imp_indices.device)
            causal_mask_bool = imp_indices.unsqueeze(1) < kv_range.unsqueeze(0)

            custom_mask = torch.zeros(
                (new_q_len, kv_len), dtype=query_states.dtype, device=query_states.device
            )
            custom_mask.masked_fill_(causal_mask_bool, torch.finfo(query_states.dtype).min)
            attention_mask = custom_mask.unsqueeze(0).unsqueeze(0).expand_as(attn_weights)
        else:
            pass

        attn_weights = attn_weights + attention_mask

        # 应用 Softmax 和 Dropout
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # 计算最终输出
        attn_output = torch.matmul(attn_weights, value_states)

        # ============== 步骤6：输出投影 ==============
        attn_output = attn_output.transpose(1, 2).contiguous()
        current_seq_len = attn_output.shape[1]  # 适应稀疏或完整序列
        attn_output = attn_output.reshape(bsz, current_seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value



class CacheBlendLlamaDecoderLayer(LlamaDecoderLayer):
    """
    CacheBlend 版本的 LlamaDecoderLayer，采用继承模式以保持权重路径一致。

    主要修改：
    1. 在 __init__ 中直接替换 self.self_attn 为 CacheBlend 版本
    2. 在 forward 中添加 CacheBlend 状态管理逻辑
    3. 处理稀疏化时的残差连接维度匹配问题
    """

    def __init__(self, config, layer_idx: int):
        # 首先调用父类初始化，创建所有原生模块
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx

        # 关键修改：直接替换 self.self_attn 为我们的 CacheBlend 版本
        # 这样权重路径保持一致，from_pretrained 可以自动加载权重
        self.self_attn = CacheBlendLlamaAttention(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        复制 Qwen2.5-VL 的 CacheBlend 逻辑，但保持 Llama 的返回格式
        """

        # 从 kwargs 中提取 CacheBlend 相关参数
        cacheblend_metadata = kwargs.get("cacheblend_metadata", {})

        # 如果全局 check 标志为 False，则直接以原生模式运行
        if not cacheblend_metadata.get("check", False):
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs
            )

        # --- CacheBlend 状态管理开始（复制自 Qwen2.5-VL）---
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # 提炼自 Qwen2.5-VL 的核心状态判断逻辑
        cacheblend_status = 0
        is_decode_phase = hidden_states.shape[1] == 1

        if not is_decode_phase:
            is_check_layer = self.layer_idx in cacheblend_metadata.get("check_layers", [1])
            has_imp_indices = cacheblend_metadata.get("imp_indices") is not None

            if is_check_layer:
                cacheblend_status = 1  # 重要性计算层
            elif has_imp_indices:
                cacheblend_status = 2  # KV融合层
            else:
                # 显式地设置第0层的状态
                cacheblend_status = 0  # 第0层标准处理

        # 设置 CacheBlend 参数给 attention 层
        kwargs["cacheblend_status"] = cacheblend_status
        kwargs["cacheblend_old_kv"] = kwargs.get("cacheblend_old_kvs", [])[self.layer_idx] if kwargs.get("cacheblend_old_kvs") else [None, None]

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs  # 传递所有额外参数
        )

        # --- 关键修复：处理残差连接的维度匹配问题（复制自 Qwen2.5-VL）---
        if not is_decode_phase and cacheblend_metadata.get("imp_indices") is not None:
            imp_indices = cacheblend_metadata["imp_indices"]
            # 检查是否真的发生了稀疏化
            if residual.shape[1] > hidden_states.shape[1] and len(imp_indices) == hidden_states.shape[1]:
                # 使用 gather 操作来确保残差连接的对齐
                residual = torch.gather(residual, 1, imp_indices.unsqueeze(0).unsqueeze(-1).expand(-1, -1, residual.shape[2]))

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class CacheBlendLlamaModel(nn.Module):
    """
    CacheBlend 版本的 LlamaModel，复制 Qwen2.5-VL 的验证实现
    主要修改：
    1. 使用 CacheBlendLlamaDecoderLayer 替换原生层
    2. 在 forward 中处理稀疏化的位置信息传播
    3. 确保与 Llama 接口兼容性
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 复制原生 LlamaModel 的基本组件
        from transformers.models.llama.modeling_llama import LlamaModel
        original_model = LlamaModel(config)

        # 复制基本属性
        self.padding_idx = original_model.padding_idx
        self.vocab_size = original_model.vocab_size
        self.embed_tokens = original_model.embed_tokens

        # 关键修改：使用我们的 CacheBlend DecoderLayer
        self.layers = nn.ModuleList([
            CacheBlendLlamaDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])

        self._use_sdpa = False
        self._use_flash_attention_2 = original_model._use_flash_attention_2
        self.norm = original_model.norm

        self.gradient_checkpointing = original_model.gradient_checkpointing

        # self.rotary_emb = original_model.rotary_emb  # 不知道是否有用

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
        复制 Qwen2.5-VL 的 forward 逻辑，包括稀疏化传播处理
        """

        # 从原生 LlamaModel 复制的标准预处理
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs
            )

            hidden_states = layer_outputs[0]


            ### 修改开始(与原生的 LlamaModel forward的唯一区别)：稀疏化传播处理  ###
            # --- 关键修复：稀疏化传播处理（复制自 Qwen2.5-VL）---
            # 检查 imp_indices 是否刚被创建 (通过比较 hidden_states 长度)
            imp_indices = kwargs.get("cacheblend_metadata", {}).get("imp_indices")
                        
            if imp_indices is not None and hidden_states.shape[1] == len(imp_indices):
                # 检查是否已经更新过位置编码，避免重复操作
                if not kwargs.get("cacheblend_metadata", {}).get("_position_updated", False):
                    print(f"Model Loop: Sparsification detected. Updating position information for subsequent layers.")

                    # 1. 稀疏化 position_ids
                    if position_ids is not None:
                        position_ids = position_ids[..., imp_indices]

                    # 2. 【关键修复】删除 attention_mask 稀疏化！
                    # 原因：CacheBlend 中稀疏的 query 需要与完整的 key/value 交互
                    # attention_mask 应该在每层的 attention 内部动态构建

                    # 3. 设置标志位，防止重复操作
                    kwargs["cacheblend_metadata"]["_position_updated"] = True

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )