import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from transformers.models.clip.modeling_clip import CLIPVisionEmbeddings
from ultralytics import YOLO
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np

from transformers.models.clip.modeling_clip import CLIPVisionTransformer, CLIPEncoder, BaseModelOutputWithPooling
from typing import Any, Optional, Tuple, Union
from torch.nn.utils.rnn import pad_sequence
import os

class CLIPVisionTransformerWithBackgroundObject(CLIPVisionTransformer):
    def __init__(self, config: CLIPVisionConfig, args):
        super().__init__(config)        

        # 1. 获取消融实验配置
        self.use_reset_position_ids = getattr(args, "use_reset_position_ids", False)
        # 2. 获取 window_size 配置
        self.window_size = getattr(args, "window_size", 56)

        mode_msg = "Ablation: Reset Position IDs (0,1,2...)" if self.use_reset_position_ids else "Default: Original Position IDs"
        print(f"Using CLIPVisionTransformerWithBackgroundObject.")
        print(f"Current Mode: {mode_msg} (CLS token enabled internally for both modes)")
        print(f"Window Size: {self.window_size}")
        print(f"Reset Position IDs: {self.use_reset_position_ids}")
        
        self.config = config

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        masks: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:

        # 回退逻辑：如果没有 mask，表现如原生 CLIP
        if masks is None:
            return super().forward(
                pixel_values=pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # ============================ PHASE 1: Embedding & Pre-processing ============================
        # 获取基础 Embedding (Patch + Position + CLS)
        hidden_states = self.embeddings(pixel_values)

        if self.use_reset_position_ids:
            # 【消融模式】：我们需要纯净的 Patch 特征
            # 1. 减去原始位置编码
            seq_len = hidden_states.shape[1]
            pos_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
            orig_pos_embeds = self.embeddings.position_embedding(pos_ids)
            hidden_states = hidden_states - orig_pos_embeds
            
            # 注意：消融模式下，Pre-LayerNorm 延迟到 Padding 之后做
        else:
            # 【默认模式】：保持原有逻辑，立即做 LayerNorm
            # 此时 hidden_states 里包含了：[CLS(含Pos=0), Patch(含Pos=X)...]
            hidden_states = self.pre_layrnorm(hidden_states)

        # ============================ PHASE 2: Mask Processing ============================
        window_size = self.window_size  # 使用可配置的window_size
        masks_float = masks[:, 0, :, :].float()
        mask_4d = masks_float.unsqueeze(1)
        pool = torch.nn.MaxPool2d(kernel_size=window_size, stride=window_size)
        window_mask = pool(mask_4d)
        window_mask = (window_mask.squeeze(1) > 0).int()
        
        num_patches_per_window = window_size // self.config.patch_size
        patch_mask = window_mask.repeat_interleave(num_patches_per_window, dim=1).repeat_interleave(num_patches_per_window, dim=2)
                
        batch_size, seq_len, embed_dim = hidden_states.shape
        patch_mask = patch_mask.view(batch_size, -1)
        assert torch.all((patch_mask == 0) | (patch_mask == 1)), "Mask values must be either 0 or 1"

        # ============================ PHASE 3: Indexing (UNIFIED) ============================
        # 核心修改：无论什么模式，我们都显式加入 CLS Token (Index 0)
        # 这样保证了 Transformer 内部计算的语义完整性
        
        cls_index = torch.tensor([0], device=pixel_values.device, dtype=torch.long)

        background_indices = []
        target_indices = []
        full_reorder_mapping = []

        for i in range(batch_size):
            # 找到 Patch 索引
            bg_patches = torch.where(patch_mask[i] == 0)[0] + 1
            fg_patches = torch.where(patch_mask[i] == 1)[0] + 1
            
            # 【统一逻辑】拼接 CLS: [0, bg1, bg2...]
            background_indices.append(torch.cat([cls_index, bg_patches]))
            target_indices.append(torch.cat([cls_index, fg_patches]))

            # Mapping 保持仅 Patch 的逻辑 (用于可视化/debug，不含CLS)
            reordered_indices = torch.cat([bg_patches - 1, fg_patches - 1])
            full_reorder_mapping.append(reordered_indices)

        # 提取嵌入
        background_embeddings = [
            hidden_states[i, background_indices[i]] for i in range(batch_size)
        ]
        object_embeddings = [
            hidden_states[i, target_indices[i]] for i in range(batch_size)
        ]

        # ============================ PHASE 4: Ablation Logic (Reset Pos) ============================
        if self.use_reset_position_ids:
            pos_layer = self.embeddings.position_embedding
            
            for i in range(batch_size):
                # --- 处理背景 ---
                # background_embeddings[i] 现在是 [CLS, Patch1, ...]
                # new_bg_pos 是 [0, 1, ...]
                # 结果：CLS 获得 Pos=0 (正确), Patch1 获得 Pos=1 (正确)
                n_bg = background_embeddings[i].shape[0]
                new_bg_pos = torch.arange(n_bg, device=pixel_values.device)
                background_embeddings[i] = background_embeddings[i] + pos_layer(new_bg_pos)
                
                # --- 处理前景 ---
                n_obj = object_embeddings[i].shape[0]
                new_obj_pos = torch.arange(n_obj, device=pixel_values.device)
                object_embeddings[i] = object_embeddings[i] + pos_layer(new_obj_pos)

        # Padding
        background_embeddings_padded = pad_sequence(background_embeddings, batch_first=True)
        object_embeddings_padded = pad_sequence(object_embeddings, batch_first=True)

        # ============================ PHASE 5: Delayed LayerNorm (Ablation only) ============================
        if self.use_reset_position_ids:
            # 在加上新位置编码后，执行 LN
            background_embeddings_padded = self.pre_layrnorm(background_embeddings_padded)
            object_embeddings_padded = self.pre_layrnorm(object_embeddings_padded)

        # ============================ PHASE 6: Encoder Forward ============================
        max_background_len = background_embeddings_padded.size(1)
        max_object_len = object_embeddings_padded.size(1)

        background_lengths = torch.tensor([len(x) for x in background_embeddings], device=pixel_values.device)
        object_lengths = torch.tensor([len(x) for x in object_embeddings], device=pixel_values.device)

        background_attention_mask = (
            torch.arange(max_background_len, device=pixel_values.device).unsqueeze(0) < background_lengths.unsqueeze(1)
        ).to(torch.bool)

        object_attention_mask = (
            torch.arange(max_object_len, device=pixel_values.device).unsqueeze(0) < object_lengths.unsqueeze(1)
        ).to(torch.bool)

        # 构建 Attention Mask
        background_attention_mask_2d = background_attention_mask.unsqueeze(2) & background_attention_mask.unsqueeze(1)
        object_attention_mask_2d = object_attention_mask.unsqueeze(2) & object_attention_mask.unsqueeze(1)
        background_attention_mask_4d = background_attention_mask_2d.unsqueeze(1)
        object_attention_mask_4d = object_attention_mask_2d.unsqueeze(1)  

        # 此时输入 Encoder 的数据：
        # - 默认模式：[CLS(原Pos), Patch(原Pos)...]
        # - 消融模式：[CLS(新Pos=0), Patch(新Pos=1,2...)...]
        # 二者结构完全一致，非常完美！
        background_outputs = self.encoder(
            inputs_embeds=background_embeddings_padded,
            attention_mask=background_attention_mask_4d,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        object_outputs = self.encoder(
            inputs_embeds=object_embeddings_padded,
            attention_mask=object_attention_mask_4d,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # ============================ PHASE 7: Output Slicing (Trimming CLS) ============================
        # 核心承诺：无论什么模式，在返回给下游之前，无条件切除 CLS Token (Index 0)
        # 确保输出数量严格等于 Patch 数量 (576)
        
        # 1. 切除 last_hidden_state [:, 1:, :]
        background_outputs.last_hidden_state = background_outputs.last_hidden_state[:, 1:, :]
        object_outputs.last_hidden_state = object_outputs.last_hidden_state[:, 1:, :]

        # 2. 切除 attention_mask [:, 1:] (用于下游 token 计数验证)
        background_attention_mask = background_attention_mask[:, 1:]
        object_attention_mask = object_attention_mask[:, 1:]

        # 3. 切除 hidden_states tuple (如果存在)
        if background_outputs.hidden_states is not None:
            background_outputs.hidden_states = tuple(h[:, 1:, :] for h in background_outputs.hidden_states)
            object_outputs.hidden_states = tuple(h[:, 1:, :] for h in object_outputs.hidden_states)

        # 4. pooler_output 不需要切，它本身就是 CLS 的投影结果 (保留它不影响下游 patch 逻辑)

        # ============================ PHASE 8: Return ============================
        if return_dict:
            return [
                (
                    background_outputs.hidden_states[-2],
                    background_outputs[1:],
                    background_attention_mask, # 已切除 CLS，长度正确
                    patch_mask,
                    full_reorder_mapping[0]
                ),
                (
                    object_outputs.hidden_states[-2],
                    object_outputs[1:],
                    object_attention_mask, # 已切除 CLS，长度正确
                    patch_mask,
                    full_reorder_mapping[0]
                )
            ]

        return (
            BaseModelOutputWithPooling(
                last_hidden_state=background_outputs[0],
                pooler_output=background_outputs.pooler_output,
                hidden_states=background_outputs.hidden_states,
                attentions=background_outputs.attentions,
                attention_mask=background_attention_mask
            ),
            BaseModelOutputWithPooling(
                last_hidden_state=object_outputs[0],
                pooler_output=object_outputs.pooler_output,
                hidden_states=object_outputs.hidden_states,
                attentions=object_outputs.attentions,
                attention_mask=object_attention_mask
            ),
            patch_mask,
            full_reorder_mapping[0]
        )

class YOLOInference:
    def __init__(self, model_path="yolov8l.pt"):
        """
        初始化 YOLO 推理工具。
        :param model_path: YOLO 模型路径。
        :param device: 运行设备（如 'cuda' 或 'cpu'）。
        """
        # 初始化YOLOv8模型（自动下载或指定本地路径）
        self.model = YOLO(model_path).eval()
        self.target_size = (352, 352)

    def preprocess(self, image_tensor: torch.Tensor):
        """
        预处理输入图像：通过在右下角 padding 将分辨率调整为目标大小。
        :param image_tensor: 输入图像张量，形状为 (B, C, H, W)。
        :return: 预处理后的图像张量，形状为 (B, C, target_h, target_w)。
        """
        batch_size, channels, height, width = image_tensor.shape
        target_height, target_width = self.target_size

        # 计算需要填充的像素数
        pad_height = max(0, target_height - height)
        pad_width = max(0, target_width - width)

        # 右下角填充
        pad_top = 0
        pad_bottom = pad_height
        pad_left = 0
        pad_right = pad_width

        # 使用 torch.nn.functional.pad 添加 padding
        padded_image = torch.nn.functional.pad(
            image_tensor,
            (pad_left, pad_right, pad_top, pad_bottom),  # (left, right, top, bottom)
            mode="constant",
            value=0  # 使用零填充
        )
        return padded_image, (pad_top, pad_bottom, pad_left, pad_right)
    
    def postprocess(self, results, original_size, padding_info):
        """
        后处理：裁剪掉 padding 区域，并将检测结果恢复为原始分辨率。
        :param results: YOLO 推理结果列表，每个元素是一个包含 boxes 和 masks 的对象。
        :param original_size: 原始图像的分辨率 (H, W)。
        :param padding_info: 预处理时的 padding 信息 (pad_top, pad_bottom, pad_left, pad_right)。
        :return: 恢复后的 YOLO 推理结果。
        """
        pad_top, pad_bottom, pad_left, pad_right = padding_info
        original_height, original_width = original_size

        processed_results = []
        for result in results:
            # 初始化一个全零掩码，表示背景
            mask = torch.zeros((original_height, original_width), dtype=torch.bool)

            # 处理检测框（boxes）
            if hasattr(result, "boxes") and result.boxes is not None:
                boxes = result.boxes.xyxy.clone()  # 克隆为普通张量

                # 调整检测框坐标，去除 padding 的影响
                boxes[:, [0, 2]] -= pad_left  # 调整 x 坐标
                boxes[:, [1, 3]] -= pad_top   # 调整 y 坐标

                # 确保检测框坐标在有效范围内
                boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]], min=0, max=original_width)
                boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]], min=0, max=original_height)

                # 基于检测框生成掩码
                for box in boxes:
                    x1, y1, x2, y2 = box.int().tolist()
                    mask[y1:y2, x1:x2] = True  # 将检测框覆盖的区域标记为前景

                # 更新结果对象中的 masks 属性
                result.masks = mask.unsqueeze(0)  # 添加批次维度

                processed_results.append(result)

        return processed_results

    def infer(self, image):
        """
        使用 YOLO 模型对输入图像进行推理。
        :param image_tensor: 输入图像张量，形状为 (C, H, W)。
        :return: YOLO 推理结果。
        """
        
        # 确保输入张量在正确的设备上
        image_tensor = image.to(self.device)

        # 预处理：Padding 到目标分辨率
        original_size = (image_tensor.shape[2], image_tensor.shape[3])  # 原始分辨率
        padded_image, padding_info = self.preprocess(image_tensor)

        # 使用 YOLO 进行推理
        with torch.no_grad():
            results = self.model(padded_image)

        # 后处理：裁剪掉 padding 区域，并恢复到原始分辨率
        processed_results = self.postprocess(results, original_size, padding_info)

        return processed_results
    
    def to(self, device):
        """
        将模型迁移到指定设备。
        :param device: 目标设备（如 'cuda' 或 'cpu'）。
        """
        self.device = device
        self.model.to(device)
        return self  # 返回自身以支持链式调用    

    
class MyCLIPVisionEmbeddings(CLIPVisionEmbeddings):
    def __init__(self, config: CLIPVisionConfig, args):
        super().__init__(config)

        # # 初始化 YOLO 推理工具
        # self.yolo_inference = YOLOInference(model_path="yolov8n-seg.pt")


        
    def generate_patch_mask(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        使用 YOLOv8 模型为目标区域生成 patch 级别的二值掩码。
        :param pixel_values: 输入图像张量，形状为 (batch_size, num_channels, height, width)。
        :return: 二值掩码张量，形状为 (batch_size, num_patches)。
        """
        batch_size, _, height, width = pixel_values.shape
        patch_height, patch_width = self.patch_size, self.patch_size
        # num_patches_h = height // patch_height
        # num_patches_w = width // patch_width


        device = pixel_values.device
        if self.yolo_inference.model.device != device:
            self.yolo_inference.to(device)

        pixel_values = torch.clamp(pixel_values, max=1)

        # 直接对整个 batch 进行推理
        with torch.no_grad():
            results = self.yolo_inference.infer(pixel_values)

        # 处理每个样本的检测结果
        masks = []
        for i in range(batch_size):
            # todo:给出原本的图片
            detections = results[i].masks  
            
            # todo:给出整体的图像mask的结果
            mask = torch.tensor(detections, dtype=torch.float32, device=device).unsqueeze(0)
            mask = F.avg_pool2d(mask, kernel_size=self.patch_size, stride=self.patch_size)
            mask = (mask > 0.1).squeeze(0).squeeze(0)  # 形状为 (num_patches_h, num_patches_w)
            
            # todo:给出整体的图像下采样后mask的结果

            masks.append(mask.flatten())
            
            # todo: 这里给出一次验证(一次就行)，验证1.image 2. detections 3. mask 主要是需要在图片上观察上述的mask是否有效。

        # 堆叠所有样本的掩码
        return torch.stack(masks, dim=0)
    
    def forward(self, pixel_values: torch.FloatTensor) -> dict:
        """
        Args:
            pixel_values (torch.Tensor): 输入图像张量，形状为 (batch_size, num_channels, height, width)。
        Returns:
            dict: 包含以下内容：
                - embeddings (torch.Tensor): 合并后的嵌入张量，形状为 (batch_size, num_patches + 1, embed_dim)。
                - background_indices (list): 每个样本中背景 token 的索引列表。
                - target_indices (list): 每个样本中目标 token 的索引列表。
        """
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)

        # 动态生成 mask
        mask = self.generate_patch_mask(pixel_values)  # 形状为 (batch_size, height, width)

        # 分离背景和目标的索引
        background_indices_list = []
        target_indices_list = []

        for i in range(batch_size):
            cur_mask = mask[i].flatten()  # 当前样本的 mask，展平为 (num_patches,)
            assert torch.all((cur_mask == 0) | (cur_mask == 1)), "Mask values must be either 0 or 1"
            background_indices = torch.where(cur_mask == 0)[0]  # 背景 token 索引
            target_indices = torch.where(cur_mask == 1)[0]      # 目标 token 索引

            background_indices_list.append(background_indices + 1)   # 这里的 +1 是因为存在分类embedding
            target_indices_list.append(target_indices + 1) 


        # 返回结果
        return {
            "embeddings": embeddings,
            "background_indices": background_indices_list,
            "object_indices": target_indices_list
        }
        

class MyCLIPVisionModel(CLIPVisionModel):

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        # 使用父类的方法加载预训练权重
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        return model
    
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        masks: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, CLIPVisionModel

        >>> model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # MyCLIPVisionModel is only used in segmentation mode, so always pass masks
        return self.vision_model(
            pixel_values=pixel_values,
            masks=masks,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    

class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.args = args

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_matching_state_dict(self, target_model, source_state_dict):
        target_state = target_model.state_dict()
        
        # 定义要加载的部分及其对应的键前缀
        parts_to_load = {
            'background_object_encoder': 'encoder.',
            'embeddings': 'embeddings.',
            'pre_layrnorm': 'pre_layernorm.'
        }
        
        with torch.no_grad():
            for target_key_prefix, source_key_prefix in parts_to_load.items():
                matching_keys = {k[len(source_key_prefix):]: v for k, v in source_state_dict.items() if k.startswith(source_key_prefix)}
                
                if matching_keys:
                    # 确保目标模型中有相应的键
                    for key, value in matching_keys.items():
                        target_key = f"{target_key_prefix}.{key}" if target_key_prefix else key
                        if target_key in target_state:
                            target_state[target_key].copy_(value)
                        else:
                            print(f"Warning: Target key '{target_key}' not found in target model.")
                else:
                    print(f"Warning: No keys found for prefix '{source_key_prefix}' in source state dict.")
        
        # 加载修改后的状态字典回到目标模型
        target_model.load_state_dict(target_state)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)


        vision_tower_output_path = './checkpoints/clip-vit-large-patch14-336.pth'
        
        if hasattr(self.args, 'method_type') and self.args.method_type in ["segmentation-cache", "object-only", "background-only", "cacheblend"]:
            # Segmentation/object-only mode: use MyCLIPVisionModel that supports masks
            # 保存模型的状态字典(先保存，再加载)
            if not os.path.exists(vision_tower_output_path):
                temp_vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
                state_dict = temp_vision_tower.vision_model.state_dict()
                torch.save(state_dict, vision_tower_output_path)
                del temp_vision_tower
                torch.cuda.empty_cache()
            
            self.vision_tower = MyCLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
            
            config = self.vision_tower.config
            bak_obj_vision_model = CLIPVisionTransformerWithBackgroundObject(config, self.args).to(self.vision_tower.vision_model.embeddings.class_embedding.device)
        
            torch.cuda.empty_cache()
            bak_obj_vision_model.load_state_dict(torch.load(vision_tower_output_path), strict=True)


            # bak_obj_vision_model.load_state_dict(self.vision_tower.vision_model.state_dict(), strict=True)

            # 这里逻辑是先加载全部参数，再进行deepspeed分配。 考虑将上面改为 device_map = 'auto',应该就没这个问题了。
            # todo: 这里应该添加 加载模型的操作 [vision_tower的每一层模型参数如何load]
            if False:        
                # original_state_dict = self.vision_tower.state_dict()
                # self.load_state_dict(original_state_dict, strict=False)
                original_state_dict = torch.load(vision_tower_output_path)

                with torch.no_grad():
                    # 加载 background_encoder
                    bg_encoder_sd = {
                        k.replace("encoder.", ""): v
                        for k, v in original_state_dict.items()
                        if k.startswith("encoder.")
                    }
                    # 严格检查参数匹配
                    bak_obj_vision_model.background_object_encoder.load_state_dict(bg_encoder_sd, strict=True)
                                        
                    # 加载 embeddings（直接匹配）
                    embeddings_sd = {
                        k.replace("embeddings.", ""): v  # 删除前缀
                        for k, v in original_state_dict.items()
                        if k.startswith("embeddings.")
                    }
                    bak_obj_vision_model.embeddings.load_state_dict(embeddings_sd, strict=True)
                    
                    # 加载 pre_layrnorm（注意键名拼写一致性）
                    pre_layrnorm_sd = {
                        k.replace("pre_layrnorm.", ""): v  # 删除前缀
                        for k, v in original_state_dict.items()
                        if k.startswith("pre_layrnorm.")
                    }
                    bak_obj_vision_model.pre_layrnorm.load_state_dict(pre_layrnorm_sd, strict=True)

            self.vision_tower.vision_model = bak_obj_vision_model
        elif hasattr(self.args, 'method_type') and self.args.method_type == "fuzzy-cache":
            # Fuzzy-cache mode: use standard CLIPVisionModel (like native) but with caching support
            self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        else:
            # Native mode: use standard CLIPVisionModel (no masks support)
            self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)

        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    def my_feature_select(self, image_forward_outs):
        image_features = image_forward_outs[0]
        if self.select_feature == 'patch':
            if image_features.dim() == 1:
                image_features = image_features.unsqueeze(0)
            image_features = image_features[:, 0:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    ################################## MODIFICATION START ##################################
    # 目的: 将原 forward 方法拆分为两个功能明确的独立方法，并将原 forward 改造为调度器。

    @torch.no_grad()
    def forward_native(self, images):
        """执行原生（无分割）的图像特征提取。"""
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @torch.no_grad()
    def forward_segmented(self, images, masks):
        """执行支持背景-前景分割的特征提取。"""
        if masks is None:
            raise ValueError("forward_segmented requires masks to be provided.")

        # self.vision_tower 是 MyCLIPVisionModel, 它会调用 CLIPVisionTransformerWithBackgroundObject
        image_forward_outs = self.vision_tower(
            pixel_values=images.to(device=self.device, dtype=self.dtype),
            masks=masks.to(device=self.device, dtype=self.dtype),
            output_hidden_states=True # 虽然内部实现没用这个参数，但保持接口一致性
        )

        # 根据返回格式正确提取 patch_mask、特征和重排映射
        if isinstance(image_forward_outs, list):
            # return_dict=True 的情况：返回 list 包含 2 个 tuple
            # 每个 tuple 格式：(hidden_states, other_outputs, attention_mask, patch_mask, reorder_mapping)
            bg_output = image_forward_outs[0]
            obj_output = image_forward_outs[1]

            background_features = self.my_feature_select([bg_output[0]]).to(images.dtype)
            background_attention_mask = bg_output[2].to(images.dtype)
            object_features = self.my_feature_select([obj_output[0]]).to(images.dtype)
            object_attention_mask = obj_output[2].to(images.dtype)
            full_patch_mask = bg_output[3]  # patch_mask 在第4个位置
            reorder_mapping = bg_output[4]  # 新增：重排映射在第5个位置

        elif isinstance(image_forward_outs, tuple) and len(image_forward_outs) == 4:
            # return_dict=False 的情况：返回 tuple 包含 4 个元素
            # (bg_BaseModelOutput, obj_BaseModelOutput, patch_mask, reorder_mapping)
            background_features = self.my_feature_select(image_forward_outs[0]).to(images.dtype)
            background_attention_mask = image_forward_outs[0][-1].to(images.dtype)
            object_features = self.my_feature_select(image_forward_outs[1]).to(images.dtype)
            object_attention_mask = image_forward_outs[1][-1].to(images.dtype)
            full_patch_mask = image_forward_outs[2]  # patch_mask 在第3个位置
            reorder_mapping = image_forward_outs[3]  # 新增：重排映射在第4个位置

        else:
            raise ValueError(f"Unexpected return format from vision_tower: {type(image_forward_outs)} with length {len(image_forward_outs) if hasattr(image_forward_outs, '__len__') else 'unknown'}")

        return background_features, object_features, background_attention_mask, object_attention_mask, full_patch_mask, reorder_mapping

    @torch.no_grad()
    def forward(self, images, masks=None):
        """
        主 forward 方法，作为智能调度器。
        根据 masks 参数是否提供，来决定调用原生模式还是分割模式。
        """
        is_segmentation_capable = hasattr(self.args, 'method_type') and \
                                  self.args.method_type in ["segmentation-cache", "object-only", "background-only", "cacheblend"]

        if is_segmentation_capable and masks is not None:
            return self.forward_segmented(images, masks)
        else:
            return self.forward_native(images)
    ################################### MODIFICATION END ###################################

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2



class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)

        self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError('Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git')
        self.multiscale_forward = multiscale_forward

        # change resize/crop size in preprocessing to the largest image size in s2_scale
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.image_processor.size['shortest_edge'] = self.s2_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.image_processor.size['shortest_edge'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size)

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)
