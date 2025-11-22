"""
轻量级 Query Key 提取器 - 通用模块

用于从原始图像和前景/背景mask中提取query_key，用于缓存查询。
支持 Qwen2-VL 和 Qwen2.5-VL 两个版本。
支持ResNet和VGG预训练模型。

使用方法:
    from qwen2_vl.lightweight_query_key_extractor import create_query_key_extractor

    extractor = create_query_key_extractor(
        extractor_type="resnet18",
        output_dim=1280,
        patch_size=14,
        spatial_merge_size=2  # Qwen2-VL和Qwen2.5-VL默认值均为2
    )

    query_key = extractor(raw_image_tensor, final_is_foreground_mask, grid_thw)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.transforms import Normalize
from typing import Optional, Tuple


class LightweightQueryKeyExtractor(nn.Module):
    """
    轻量级 Query Key 提取器基类

    核心功能：
    1. 接收原始图像 [B, C, H, W] 和 logical patch级别的mask
    2. 将mask映射回pixel级别
    3. 提取背景区域并padding
    4. 使用轻量级backbone提取特征
    5. 输出归一化的query_key
    """

    def __init__(
        self,
        output_dim: Optional[int] = None,
        target_size: int = 224,
        patch_size: int = 14,
        spatial_merge_size: int = 2,
    ):
        """
        Args:
            output_dim: query_key的输出维度。如果为None，则直接使用backbone的原生特征维度（推荐）
            target_size: padding后的目标图像尺寸
            patch_size: VIT的patch size (Qwen2-VL和Qwen2.5-VL默认为14)
            spatial_merge_size: VIT的spatial merge size (Qwen2-VL和Qwen2.5-VL默认为2，表示2×2个physical patches合并成1个logical patch)
        """
        super().__init__()
        self.output_dim = output_dim
        self.target_size = target_size
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size

        # ImageNet normalization (for ResNet/VGG)
        self.normalize = Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def logical_mask_to_pixel_mask(
        self,
        final_is_foreground_mask: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """
        将logical patch级别的mask转换为pixel级别的mask

        Args:
            final_is_foreground_mask: [num_logical_patches] bool tensor
            grid_thw: [1, 3] 图像的(t, h_physical, w_physical)网格信息 - 注意是physical patch维度

        Returns:
            pixel_mask: [H, W] bool tensor (True=前景, False=背景)
        """
        # 确保mask在正确的设备上（与grid_thw相同）
        device = grid_thw.device
        final_is_foreground_mask = final_is_foreground_mask.to(device)

        # 提取物理网格尺寸
        t, h_physical, w_physical = grid_thw[0].tolist()

        # 转换为逻辑网格尺寸 (physical patches → logical patches)
        # spatial_merge_size 是每个维度上合并的physical patches数量
        # 例如: spatial_merge_size=2 表示 2×2=4个physical patches合并成1个logical patch
        # Qwen2-VL 和 Qwen2.5-VL 默认都是 spatial_merge_size=2
        h_logical = h_physical // self.spatial_merge_size
        w_logical = w_physical // self.spatial_merge_size

        # Reshape to 2D grid (logical patch level)
        mask_2d = final_is_foreground_mask.reshape(h_logical, w_logical)

        # 每个logical patch对应的pixel数
        logical_patch_pixel_size = self.spatial_merge_size * self.patch_size

        # Upsample to pixel level
        pixel_mask = mask_2d.repeat_interleave(logical_patch_pixel_size, dim=0)\
                            .repeat_interleave(logical_patch_pixel_size, dim=1)

        return pixel_mask

    def llava_mask_to_pixel_mask(
        self,
        final_is_foreground_mask: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """
        将 LLaVA 的 patch 级别 mask 转换为 pixel 级别 mask

        LLaVA 的 mask 处理流程 (clip_encoder.py:89-103):
        1. 原始 mask → MaxPool2d(window_size=56) → window_mask
        2. window_mask → repeat_interleave(num_patches_per_window=4) → patch_mask

        这里需要反向操作: patch_mask → pixel_mask

        Args:
            final_is_foreground_mask: [num_patches] bool tensor (True=前景, False=背景)
            grid_thw: [1, 3] 网格信息 (t, h_physical, w_physical)

        Returns:
            pixel_mask: [H, W] bool tensor (True=前景, False=背景)
        """
        _, H, W = grid_thw[0].tolist()

        # 计算 patch 网格尺寸
        h_patches = H // self.patch_size  # 例如 336 // 14 = 24
        w_patches = W // self.patch_size

        # Reshape 到 2D 网格
        mask_2d = final_is_foreground_mask.reshape(h_patches, w_patches)

        # Upsample 到 pixel 级别
        pixel_mask = mask_2d.repeat_interleave(self.patch_size, dim=0)\
                            .repeat_interleave(self.patch_size, dim=1)

        return pixel_mask
    
    def extract_and_pad_background(
        self,
        image_tensor: torch.Tensor,
        pixel_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        提取背景区域并padding成规则形状

        策略：
        1. 找到背景的bounding box
        2. 提取背景区域，前景部分置0
        3. Resize并padding到target_size

        Args:
            image_tensor: [C, H, W] 原始图像 (已归一化到[0,1])
            pixel_mask: [H, W] bool tensor (True=前景, False=背景)

        Returns:
            padded_bg: [C, target_size, target_size] padding后的背景图像
        """
        bg_mask = ~pixel_mask  # 背景为True

        # 找到背景的bounding box
        rows = torch.any(bg_mask, dim=1)
        cols = torch.any(bg_mask, dim=0)

        if not rows.any() or not cols.any():
            # 如果没有背景区域，返回全零图像
            if self.target_size is None:
                # target_size为None时，返回最小的32x32图像（ResNet最小输入尺寸）
                return torch.zeros(
                    image_tensor.shape[0],
                    32,
                    32,
                    device=image_tensor.device,
                    dtype=image_tensor.dtype
                )
            else:
                # target_size有具体值时
                return torch.zeros(
                    image_tensor.shape[0],
                    self.target_size,
                    self.target_size,
                    device=image_tensor.device,
                    dtype=image_tensor.dtype
                )

        row_indices = torch.where(rows)[0]
        col_indices = torch.where(cols)[0]
        rmin, rmax = row_indices[0].item(), row_indices[-1].item()
        cmin, cmax = col_indices[0].item(), col_indices[-1].item()

        # 提取bounding box区域
        bg_region = image_tensor[:, rmin:rmax+1, cmin:cmax+1].clone()

        # 将前景区域设为0（masked掉）
        bg_region_mask = bg_mask[rmin:rmax+1, cmin:cmax+1]
        bg_region = bg_region * bg_region_mask.unsqueeze(0).float()

        # --- 情况B: 动态尺寸 (Target Size is None) ---
        if self.target_size is None:
            # 【安全检查】ResNet 下采样倍数为 32。
            # 如果高或宽小于 32，会导致最后特征图变为 0，引发 RuntimeError。
            # 因此，如果尺寸太小，我们进行最小限度的放大。
            _, h, w = bg_region.shape
            if h < 32 or w < 32:
                scale_factor = max(32 / h, 32 / w)
                # 稍微多放一点余量，向上取整
                new_h = max(32, int(h * scale_factor))
                new_w = max(32, int(w * scale_factor))
                
                bg_region = F.interpolate(
                    bg_region.unsqueeze(0), 
                    size=(new_h, new_w), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
            
            return bg_region
            
        # Resize到target_size，保持aspect ratio
        _, h, w = bg_region.shape
        if h > w:
            new_h = self.target_size
            new_w = max(1, int(w * self.target_size / h))
        else:
            new_w = self.target_size
            new_h = max(1, int(h * self.target_size / w))

        # Resize
        bg_resized = F.interpolate(
            bg_region.unsqueeze(0),
            size=(new_h, new_w),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        # Padding到target_size (center padding)
        pad_h = self.target_size - new_h
        pad_w = self.target_size - new_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        padded_bg = F.pad(
            bg_resized,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode='constant',
            value=0
        )

        return padded_bg

    def forward(
        self,
        raw_image_tensor: torch.Tensor,
        final_is_foreground_mask: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """
        提取query_key

        Args:
            raw_image_tensor: [B, C, H, W] 原始图像 (归一化前，范围[0,1])
            final_is_foreground_mask: [num_logical_patches] VIT输出的mask
            grid_thw: [1, 3] 网格信息 (t, h_physical, w_physical) - 注意是physical patch维度

        Returns:
            query_key: [1, output_dim] 归一化的查询向量
        """
        # 1. 将logical mask转换为pixel mask
        pixel_mask = self.llava_mask_to_pixel_mask(
            final_is_foreground_mask,
            grid_thw
        )

        # 2. 提取并padding背景区域
        # 假设batch_size=1
        image = raw_image_tensor[0]  # [C, H, W]
        padded_bg = self.extract_and_pad_background(image, pixel_mask)

        # 3. Normalize（用于ResNet/VGG）
        normalized_bg = self.normalize(padded_bg)

        # 4. 添加batch维度
        normalized_bg = normalized_bg.unsqueeze(0)  # [1, C, H, W]

        # 5. 转换数据类型以匹配backbone权重
        # 获取backbone的第一个参数的dtype（通常是第一个卷积层的权重）
        backbone_dtype = next(self.backbone.parameters()).dtype
        normalized_bg = normalized_bg.to(dtype=backbone_dtype)

        # 6. 提取特征（由子类实现）
        features = self.extract_features(normalized_bg)

        # 7. 投影到output_dim（如果需要）
        if self.output_dim is not None:
            query_key = self.project_to_output_dim(features)
        else:
            # 直接使用backbone原生特征（推荐）
            query_key = features

        # 8. L2归一化
        query_key = F.normalize(query_key, p=2, dim=-1)   # 既要考虑抽取模型的精度；又要考虑缓存库faiss支持的精度(faiss也可以支持bf16)；到底该如何决定？标准的实现应该是什么样的？

        # 9. 转换为float32以兼容FAISS（FAISS不支持bfloat16）
        query_key = query_key.float()

        return query_key
    
    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        """子类需要实现此方法"""
        raise NotImplementedError

    def project_to_output_dim(self, features: torch.Tensor) -> torch.Tensor:
        """子类需要实现此方法"""
        raise NotImplementedError


class ResNetQueryKeyExtractor(LightweightQueryKeyExtractor):
    """基于ResNet的query_key提取器"""

    def __init__(
        self,
        output_dim: Optional[int] = None,
        target_size: int = 224,
        patch_size: int = 14,
        spatial_merge_size: int = 2,
        resnet_version: str = "resnet18",
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__(output_dim, target_size, patch_size, spatial_merge_size)

        # 加载预训练ResNet
        if resnet_version == "resnet18":
            backbone = models.resnet18(pretrained=pretrained)
            self.backbone_dim = 512
        elif resnet_version == "resnet34":
            backbone = models.resnet34(pretrained=pretrained)
            self.backbone_dim = 512
        elif resnet_version == "resnet50":
            backbone = models.resnet50(pretrained=pretrained)
            self.backbone_dim = 2048
        elif resnet_version == "resnet101":
            backbone = models.resnet101(pretrained=pretrained)
            self.backbone_dim = 2048
        else:
            raise ValueError(f"Unsupported ResNet version: {resnet_version}")

        # 移除最后的全连接层
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        # 冻结backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 投影层（仅当指定output_dim时使用）
        if output_dim is not None:
            self.projection = nn.Linear(self.backbone_dim, output_dim)
        else:
            self.projection = None


    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        """
        使用ResNet提取特征

        Args:
            image: [B, C, H, W] 归一化后的图像

        Returns:
            features: [B, backbone_dim]
        """
        with torch.set_grad_enabled(self.training):
            features = self.backbone(image)  # [B, backbone_dim, 1, 1]
            features = features.flatten(1)   # [B, backbone_dim]
        return features

    def project_to_output_dim(self, features: torch.Tensor) -> torch.Tensor:
        """投影到输出维度"""
        if self.projection is not None:
            return self.projection(features)
        else:
            return features


class VGGQueryKeyExtractor(LightweightQueryKeyExtractor):
    """基于VGG的query_key提取器"""

    def __init__(
        self,
        output_dim: Optional[int] = None,
        target_size: int = 224,
        patch_size: int = 14,
        spatial_merge_size: int = 2,
        vgg_version: str = "vgg16",
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__(output_dim, target_size, patch_size, spatial_merge_size)

        # 加载预训练VGG
        if vgg_version == "vgg11":
            backbone = models.vgg11(pretrained=pretrained)
        elif vgg_version == "vgg13":
            backbone = models.vgg13(pretrained=pretrained)
        elif vgg_version == "vgg16":
            backbone = models.vgg16(pretrained=pretrained)
        elif vgg_version == "vgg19":
            backbone = models.vgg19(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported VGG version: {vgg_version}")

        # 只使用特征提取部分
        self.backbone = backbone.features
        self.backbone_dim = 512

        # 冻结backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 投影层（仅当指定output_dim时使用）
        if output_dim is not None:
            self.projection = nn.Linear(self.backbone_dim, output_dim)
        else:
            self.projection = None


    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        """
        使用VGG提取特征

        Args:
            image: [B, C, H, W] 归一化后的图像

        Returns:
            features: [B, backbone_dim]
        """
        with torch.set_grad_enabled(self.training):
            features = self.backbone(image)       # [B, 512, H', W']
            features = self.global_pool(features) # [B, 512, 1, 1]
            features = features.flatten(1)        # [B, 512]
        return features

    def project_to_output_dim(self, features: torch.Tensor) -> torch.Tensor:
        """投影到输出维度"""
        if self.projection is not None:
            return self.projection(features)
        else:
            return features


def create_query_key_extractor(
    extractor_type: str = "resnet18",
    output_dim: Optional[int] = None,
    target_size: int = 224,
    patch_size: int = 14,
    spatial_merge_size: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> LightweightQueryKeyExtractor:
    """
    工厂函数：创建query_key提取器

    Args:
        extractor_type: 提取器类型
            - "resnet18", "resnet34", "resnet50", "resnet101"
            - "vgg11", "vgg13", "vgg16", "vgg19"
        output_dim: 输出维度。如果为None，则使用backbone原生特征维度（推荐）
            - ResNet18/34: 512维
            - ResNet50/101: 2048维
            - VGG: 512维
        target_size: padding后的目标尺寸
        patch_size: VIT patch size (Qwen2-VL和Qwen2.5-VL默认为14)
        spatial_merge_size: VIT spatial merge size (Qwen2-VL和Qwen2.5-VL默认为2，表示2×2个physical patches合并成1个logical patch)
        pretrained: 是否使用预训练权重
        freeze_backbone: 是否冻结backbone

    Returns:
        extractor: QueryKeyExtractor实例
    """
    if extractor_type.startswith("resnet"):
        return ResNetQueryKeyExtractor(
            output_dim=output_dim,
            target_size=target_size,
            patch_size=patch_size,
            spatial_merge_size=spatial_merge_size,
            resnet_version=extractor_type,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
    elif extractor_type.startswith("vgg"):
        return VGGQueryKeyExtractor(
            output_dim=output_dim,
            target_size=target_size,
            patch_size=patch_size,
            spatial_merge_size=spatial_merge_size,
            vgg_version=extractor_type,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
    else:
        raise ValueError(f"Unknown extractor type: {extractor_type}")


if __name__ == "__main__":
    # 测试代码
    print("=== Testing Lightweight Query Key Extractors ===\n")

    # 创建测试数据
    batch_size = 1
    raw_image = torch.rand(batch_size, 3, 560, 560)  # 原始图像 [0,1]
    num_logical_patches = 100
    final_mask = torch.rand(num_logical_patches) > 0.5  # 随机mask
    grid_thw = torch.tensor([[1, 10, 10]])  # 10x10的logical patch grid

    # 测试ResNet18
    print("1. Testing ResNet18 Extractor")
    print("-" * 50)
    resnet_extractor = create_query_key_extractor(
        extractor_type="resnet18",
        output_dim=1280,
        patch_size=14,
        spatial_merge_size=2  # Qwen2-VL和Qwen2.5-VL默认值均为2
    )
    resnet_extractor.eval()

    with torch.no_grad():
        query_key = resnet_extractor(raw_image, final_mask, grid_thw)

    print(f"  Input image shape: {raw_image.shape}")
    print(f"  Final mask shape: {final_mask.shape}")
    print(f"  Grid THW: {grid_thw}")
    print(f"  Query key shape: {query_key.shape}")
    print(f"  Query key norm: {torch.norm(query_key, dim=-1).item():.4f} (should be ~1.0)")
    print()

    # 测试VGG16
    print("2. Testing VGG16 Extractor")
    print("-" * 50)
    vgg_extractor = create_query_key_extractor(
        extractor_type="vgg16",
        output_dim=1280,
        patch_size=14,
        spatial_merge_size=2  # Qwen2-VL和Qwen2.5-VL默认值均为2
    )
    vgg_extractor.eval()

    with torch.no_grad():
        query_key = vgg_extractor(raw_image, final_mask, grid_thw)

    print(f"  Query key shape: {query_key.shape}")
    print(f"  Query key norm: {torch.norm(query_key, dim=-1).item():.4f} (should be ~1.0)")
    print()

    print("✓ All tests passed!")




