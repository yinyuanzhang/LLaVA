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

class CLIPVisionTransformerWithBackgroundObject(CLIPVisionTransformer):
    def __init__(self, config: CLIPVisionConfig, args):
        super().__init__(config)  

        # 替换现有的 embeddings [目标检测、 mask标记]
        self.embeddings = MyCLIPVisionEmbeddings(config, args)

        # 添加两个独立的 Transformer 编码器
        self.background_object_encoder = CLIPEncoder(config)

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        """
        Args:
            pixel_values (torch.Tensor): 输入图像张量，形状为 (batch_size, num_channels, height, width)。
        Returns:
            BaseModelOutputWithPooling: 包含最后一层隐藏状态和池化输出的结果。
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 1. embedding 2. 根据index进行重组 -> list 3. 根据list，进行最大padding，同时记录mask 

        # 获取嵌入表示
        embedding_result = self.embeddings(pixel_values)
        hidden_states = embedding_result["embeddings"]
        hidden_states = self.pre_layrnorm(hidden_states)

        # 分离背景和目标的 token
        background_indices_list = embedding_result["background_indices"]
        target_indices_list = embedding_result["object_indices"]

        batch_size, seq_len, embed_dim = hidden_states.shape
        background_embeddings = []
        object_embeddings = []

        for i in range(batch_size):
            background_embeddings.append(hidden_states[i, background_indices_list[i]])
            object_embeddings.append(hidden_states[i, target_indices_list[i]])

        # 填充背景和目标的 token 到相同长度
        from torch.nn.utils.rnn import pad_sequence

        max_background_len = max(len(x) for x in background_embeddings)
        max_object_len = max(len(x) for x in object_embeddings)

        background_embeddings_padded = pad_sequence(
            [x for x in background_embeddings], batch_first=True, padding_value=0
        )
        object_embeddings_padded = pad_sequence(
            [x for x in object_embeddings], batch_first=True, padding_value=0
        )

        # 生成背景和目标的掩码
        background_attention_mask = (torch.arange(max_background_len).unsqueeze(0) < torch.tensor([len(x) for x in background_embeddings]).unsqueeze(1)).to(torch.int32).to(pixel_values.device)
        object_attention_mask = (torch.arange(max_object_len).unsqueeze(0) < torch.tensor([len(x) for x in object_embeddings]).unsqueeze(1)).to(torch.int32).to(pixel_values.device)

        # 分别通过背景和目标的编码器
        background_outputs = self.background_object_encoder(
            inputs_embeds=background_embeddings_padded,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        object_outputs = self.background_object_encoder(
            inputs_embeds=object_embeddings_padded,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 如果返回元组形式的结果
        if return_dict:
            return [
                (
                    background_outputs.hidden_states[-2],  # 背景的 last_hidden_state
                    background_outputs[1:],  # 背景的其他输出（如 hidden_states 和 attentions）
                    background_attention_mask  # 背景的掩码
                ),
                (
                    object_outputs.hidden_states[-2],  # 目标的 last_hidden_state
                    object_outputs[1:],  # 目标的其他输出（如 hidden_states 和 attentions）
                    object_attention_mask  # 目标的掩码
                )
            ]

        # 返回两个 BaseModelOutputWithPooling 对象
        return (
            BaseModelOutputWithPooling(
                last_hidden_state=background_outputs[0],
                pooler_output=background_outputs.pooler_output,
                hidden_states=background_outputs.hidden_states,
                attentions=background_outputs.attentions,
                attention_mask=background_attention_mask  # 背景的掩码
            ),
            BaseModelOutputWithPooling(
                last_hidden_state=object_outputs[0],
                pooler_output=object_outputs.pooler_output,
                hidden_states=object_outputs.hidden_states,
                attentions=object_outputs.attentions,
                attention_mask=object_attention_mask  # 目标的掩码
            )
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

        # 初始化 YOLO 推理工具
        self.yolo_inference = YOLOInference(model_path="yolov8n-seg.pt")


        
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


        # 动态迁移 yolo_inference 到输入张量的设备
        device = pixel_values.device
        self.yolo_inference.to(device)

        pixel_values = torch.clamp(pixel_values, min=0, max=1)

        # 直接对整个 batch 进行推理
        with torch.no_grad():
            results = self.yolo_inference.infer(pixel_values)

        # 处理每个样本的检测结果
        masks = []
        for i in range(batch_size):
            detections = results[i].masks  # 获取第 i 个样本的分割掩码

            # 下采样到 patch 级别
            mask = torch.tensor(detections, dtype=torch.float32, device=device).unsqueeze(0)
            mask = F.avg_pool2d(mask, kernel_size=self.patch_size, stride=self.patch_size)
            mask = (mask > 0.1).squeeze(0).squeeze(0)  # 形状为 (num_patches_h, num_patches_w)
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

        # 使用卷积生成所有 patches 的 embeddings
        all_patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # 形状为 (batch_size, embed_dim, grid_h, grid_w)
        all_patch_embeds = all_patch_embeds.flatten(2).transpose(1, 2)  # 形状为 (batch_size, num_patches, embed_dim)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, all_patch_embeds], dim=1)
        final_embeddings = embeddings + self.position_embedding(self.position_ids)

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

            background_indices_list.append(background_indices + 1)
            target_indices_list.append(target_indices + 1)


        # 返回结果
        return {
            "embeddings": final_embeddings,
            "background_indices": background_indices_list,
            "object_indices": target_indices_list
        }
        


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
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        
        vision_tower_output_path = './checkpoints/clip-vit-large-patch14-336.pth'

        # 保存模型的状态字典(先保存，再加载)
        # state_dict = self.vision_tower.vision_model.state_dict()
        # torch.save(state_dict, vision_tower_output_path)


        if self.args.image_cache:
            config = self.vision_tower.config
            bak_obj_vision_model = CLIPVisionTransformerWithBackgroundObject(config, self.args).to(self.vision_tower.vision_model.embeddings.class_embedding.device)
        
            # 这里逻辑是先加载全部参数，再进行deepspeed分配。 考虑将上面改为 device_map = 'auto',应该就没这个问题了。
            # todo: 这里应该添加 加载模型的操作 [vision_tower的每一层模型参数如何load]
            if True:        
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
    
    @torch.no_grad()
    def forward(self, images):
        if self.args.image_cache:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            background_features = self.my_feature_select(image_forward_outs[0]).to(images.dtype)
            background_attention_mask = image_forward_outs[0][-1].to(images.dtype)
            object_features = self.my_feature_select(image_forward_outs[1]).to(images.dtype)
            object_attention_mask = image_forward_outs[1][-1].to(images.dtype)
            
            return background_features, object_features, background_attention_mask, object_attention_mask

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
