# 文件名: llava/model/cacheblend_llama.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from transformers.models.llama.modeling_llama import LlamaAttention, repeat_kv, apply_rotary_pos_emb
from typing import Optional, List, Union, Tuple


class CacheBlendLlamaAttention(nn.Module):
    """
    围绕 LlamaAttention 的包装器，用于实现 CacheBlend 算法:
    - Prefill/Layer0: 执行重要性选择 (计算 imp_indices); 不进行KV融合。
    - Prefill/Layer1+: 构建融合后的KV (对命中段复用缓存; 用新KV覆盖重要token),
      并为重要token重计算输出; 返回融合后的 present_kv。
    - Decode: 回退到原生的 Attention 逻辑。
    """
    def __init__(self, attn: LlamaAttention, parent_ctx, layer_idx: int):
        super().__init__()

        print(f">>>>> [CacheBlend DEBUG] SUCCESSFULLY INITIALIZED NEW WRAPPER V2 FOR LAYER {layer_idx} <<<<<")


        self.attn = attn
        self.ctx = parent_ctx # 轻量级上下文，避免在named_modules中递归
        self.layer_idx = layer_idx
        # 暴露原生 attention 的必要属性
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj
        self.rotary_emb = getattr(attn, 'rotary_emb', None)
        self.num_heads, self.num_key_value_heads = attn.num_heads, attn.num_key_value_heads
        self.head_dim = attn.head_dim
        self.hidden_size = attn.hidden_size

    def _should_fallback(self, hidden_states: torch.Tensor, past_key_value: Optional[object] = None) -> bool:
        """
        判断是否应回退到原生 attention。
        只有在 CacheBlend 的 Prefill 阶段且数据完备时，才执行自定义逻辑。
        其他所有情况（包括所有 Decode 阶段）都回退。
        """
        # 1. 检查是否为 Decode 阶段 (最可靠的标志是 past_key_value 中已经有内容)
        if hidden_states.shape[1] == 1:
            return True

        # 2. 如果是 Prefill 阶段，检查是否满足 CacheBlend 的执行条件
        model = self.ctx.model
        if getattr(model.config, 'method_type', None) != 'cacheblend':
            return True
        
        # 3. 如果是 read-load 模式，检查所需数据是否已准备好
        if getattr(model.config, 'cache_mode', None) == 'read-load':
            meta = getattr(model, 'cache_fuse_metadata', {})
            # 如果元数据为空，或没有 old_kvs，说明是 miss 或数据准备失败，应回退
            if not meta or "old_kvs" not in meta or not meta["old_kvs"] or meta["old_kvs"][self.layer_idx][0] is None:
                return True
        
        # 只有在 read-load 命中或 write-only 的 Prefill 阶段，才不回退
        return False

    def _compute_qkv(self, hidden_states: torch.Tensor):
        """计算 Q, K, V 并应用旋转位置编码。"""
        bsz, q_len, _ = hidden_states.size()
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        return q, k, v

    def _compute_imp_indices(self, v_new, metadata):
        """
        Layer 0 执行: 对比新旧V值，计算并返回需要重计算的重要token索引。
        """
        old_k, old_v = metadata["old_kvs"][self.layer_idx]
        img_start, img_len = metadata["image_span"]
        bg_len, fg_len = metadata["bg_len"], metadata["fg_len"]
        is_bg_hit, is_fg_hit = metadata["is_bg_hit"], metadata["is_fg_hit"]
        recomp_ratio = getattr(self.ctx.config, 'recomp_ratio', 0.16)
        
        # 1. 提取当前输入的图像V值和缓存的图像V值
        v_new_img = v_new[:, :, img_start:img_start+img_len, :]
        v_old_img = old_v.to(v_new.device)

        if v_new_img.shape[2] != v_old_img.shape[2]:
            print(f"Layer {self.layer_idx} 警告: V 长度不匹配，将重计算所有图像 token。")
            return torch.arange(img_start, img_start + img_len, device=v_new.device)

        # 2. 计算L2差异
        diff = (v_new_img - v_old_img).pow(2).sum(dim=[1, 3]).squeeze(0)
        
        # 3. 根据命中情况和比例选择重要索引
        img_indices_to_recompute = []
        if bg_len > 0:
            if not is_bg_hit:
                img_indices_to_recompute.append(torch.arange(0, bg_len, device=diff.device))
            else:
                k = max(1, int(bg_len * recomp_ratio))
                topk_indices = torch.topk(diff[:bg_len], k=min(k, bg_len)).indices
                img_indices_to_recompute.append(topk_indices)

        if fg_len > 0:
            fg_start_offset = bg_len
            if not is_fg_hit:
                img_indices_to_recompute.append(torch.arange(fg_start_offset, fg_start_offset + fg_len, device=diff.device))
            else:
                k = max(1, int(fg_len * recomp_ratio))
                topk_indices_in_fg = torch.topk(diff[fg_start_offset:], k=min(k, fg_len)).indices
                img_indices_to_recompute.append(topk_indices_in_fg + fg_start_offset)

        # 4. 合并并转换为绝对索引
        if not img_indices_to_recompute:
            important_img_indices = torch.tensor([], device=diff.device, dtype=torch.long)
        else:
            important_img_indices = torch.cat(img_indices_to_recompute).unique() + img_start
        
        return important_img_indices

    def _build_fused_kv(self, k_new, v_new, metadata):
        """
        Layer 1+ 执行: 以新计算的完整 K/V 为基础，
        用缓存的旧 K/V 替换其中的图像部分，构建一个融合后的完整 K/V。
        """
        # 1. 从元数据中获取缓存的、仅包含图像部分的 old_kvs 和图像位置
        old_k_img, old_v_img = metadata["old_kvs"][self.layer_idx]
        img_start, img_len = metadata["image_span"]
        
        # 确保 old_k_img 不为空
        if old_k_img is None:
            return k_new, v_new

        # 2. 创建一个新 K/V 的副本作为融合的基础
        fused_k = k_new.clone()
        fused_v = v_new.clone()
        
        # 3. 核心操作：将缓存的图像KV "贴" 到新KV的正确位置上
        #    确保长度匹配，防止越界
        len_to_replace = min(img_len, old_k_img.shape[2])
        img_end = img_start + len_to_replace
        
        fused_k[:, :, img_start:img_end, :] = old_k_img[:, :, :len_to_replace, :].to(fused_k.device)
        fused_v[:, :, img_start:img_end, :] = old_v_img[:, :, :len_to_replace, :].to(fused_v.device)
        
        return fused_k, fused_v

    def _sparse_attention(self, q, k, v, attention_mask, imp_indices):
        """
        [简化后] 假定输入的 Q, K, V 已完全准备好 (RoPE 已应用, K/V 已 repeat)
        只执行核心的 matmul 和 scatter 操作
        """
        # 1. RoPE 和 repeat_kv 已被移除！

        # 2. 执行稀疏 Q x 完整 K/V 的注意力计算
        q_sparse = q[:, :, imp_indices, :]
        
        attn_weights = torch.matmul(q_sparse, k.transpose(2, 3)) / (self.head_dim**0.5)

        if attention_mask is not None:
            # (mask 逻辑保持不变, 但需要仔细检查)
            # ...
            sparse_mask = attention_mask[:, :, imp_indices, :]
            attn_weights = attn_weights + sparse_mask

        attn_weights = nn.Softmax(dim=-1)(attn_weights.float()).to(q.dtype)
        attn_output_sparse = torch.matmul(attn_weights, v)
        
        # 3. 将稀疏输出 "scatter" 回全尺寸张量 (逻辑保持不变)
        bsz, num_heads, _, head_dim = q.shape
        full_seq_len = q.shape[2]
        attn_output = torch.zeros((bsz, num_heads, full_seq_len, head_dim), device=q.device, dtype=q.dtype)
        attn_output[:, :, imp_indices, :] = attn_output_sparse
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, full_seq_len, self.hidden_size)
        
        return self.o_proj(attn_output)
    

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[List[torch.FloatTensor]] = None,
        output_attentions: bool = False,
        use_cache: Optional[bool] = None,
    ):
        if self._should_fallback(hidden_states, past_key_value):
            return self.attn.forward(hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache)

        metadata = self.ctx.model.cache_fuse_metadata
        
        # 步骤 1: 计算原始 Q, K, V
        q_raw, k_raw, v_new = self._compute_qkv(hidden_states)

        # 步骤 2: **立即应用 RoPE**
        kv_seq_len = k_raw.shape[-2]
        cos, sin = self.rotary_emb(v_new, seq_len=kv_seq_len)
        q, k_new = apply_rotary_pos_emb(q_raw, k_raw, cos, sin, position_ids)
        
        # 现在 k_new 是旋转过的了，v_new 不变

        # 步骤 3: 根据层级准备 k_final (已旋转) 和 v_final
        if self.layer_idx == 0:
            text_len = metadata["image_span"][0]
            text_indices = torch.arange(text_len, device=q.device)
            img_indices = self._compute_imp_indices(v_new, metadata)
            
            imp_indices = torch.cat([text_indices, img_indices]).unique()
            metadata["imp_indices"] = imp_indices
            
            k_final, v_final = k_new, v_new
        else: 
            k_final, v_final = self._build_fused_kv(k_new, v_new, metadata)
            imp_indices = metadata["imp_indices"]

        assert k_final.shape == v_final.shape, "CRITICAL: K and V shapes do not match before caching!"

        # 步骤 4: **将旋转过的、紧凑的 k_final 存入缓存**
        if use_cache and past_key_value is not None:
             past_key_value.update(k_final, v_final, self.layer_idx)
        
        # 步骤 5: **仅在此处为 Attention 计算准备 K, V (GQA/MQA)**
        k_for_attn = repeat_kv(k_final, self.num_heads // self.num_key_value_heads)
        v_for_attn = repeat_kv(v_final, self.num_heads // self.num_key_value_heads)

        # 步骤 6: 执行稀疏注意力（传入已完全准备好的张量）
        attn_output = self._sparse_attention(q, k_for_attn, v_for_attn, attention_mask, imp_indices)

        # 步骤 7: 返回结果
        return attn_output, None, past_key_value