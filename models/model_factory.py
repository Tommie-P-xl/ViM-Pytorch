"""
模型工厂模块：负责根据配置文件创建主干网络和完整分类模型
支持 ResNet18 / ResNet50 / ResNet101 / EfficientNet-B0
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Tuple


# ==============================================================================
# 特征提取封装：在 forward 中同时返回特征和 logit
# ==============================================================================

class FeatureExtractorWrapper(nn.Module):
    """
    对标准分类网络的封装，使其能够同时输出：
    - 分类 logit（最终输出）
    - 全连接层之前的特征向量（用于 ViM 算法）
    - 暴露分类头的权重矩阵 W 和偏置 b（用于 ViM 算法）
    """

    def __init__(self, backbone: nn.Module, classifier: nn.Linear):
        """
        参数:
            backbone: 去除分类头后的特征提取网络
            classifier: 最后一层全连接分类头（nn.Linear）
        """
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(
        self, x: torch.Tensor, return_feature: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        参数:
            x: 输入张量
            return_feature: 若为 True，同时返回特征向量
        返回:
            return_feature=False → logit
            return_feature=True  → (logit, feature)
        """
        feature = self.backbone(x)
        logit = self.classifier(feature)

        if return_feature:
            return logit, feature
        return logit

    def get_fc_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取分类头的权重矩阵 W 和偏置 b（numpy 格式，用于 ViM 算法）
        返回: (W, b) 均为 numpy.ndarray
        """
        w = self.classifier.weight.detach().cpu().numpy()
        b = self.classifier.bias.detach().cpu().numpy()
        return w, b


# ==============================================================================
# 模型工厂函数
# ==============================================================================

def build_model(cfg: dict) -> FeatureExtractorWrapper:
    """
    根据配置文件构建模型
    
    参数:
        cfg: 完整配置字典
    返回:
        FeatureExtractorWrapper 实例
    """
    model_cfg = cfg["model"]
    backbone_name: str = model_cfg["backbone"]
    pretrained: str = model_cfg.get("pretrained", "imagenet")
    single_channel: bool = model_cfg.get("single_channel_input", True)
    dropout_rate: float = model_cfg.get("dropout", 0.0)
    num_classes: int = cfg["_num_known_classes"]  # 由外部注入

    use_pretrained = (pretrained == "imagenet")

    # ---------- 构建主干网络 ----------
    if backbone_name == "resnet18":
        base_model = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None
        )
        feature_dim = 512

    elif backbone_name == "resnet50":
        base_model = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2 if use_pretrained else None
        )
        feature_dim = 2048

    elif backbone_name == "resnet101":
        base_model = models.resnet101(
            weights=models.ResNet101_Weights.IMAGENET1K_V2 if use_pretrained else None
        )
        feature_dim = 2048

    elif backbone_name == "efficientnet_b0":
        base_model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if use_pretrained else None
        )
        feature_dim = 1280

    else:
        raise ValueError(
            f"不支持的主干网络: '{backbone_name}'，"
            f"可选: resnet18 | resnet50 | resnet101 | efficientnet_b0"
        )

    # ---------- 修改第一层卷积以适配单通道输入 ----------
    if single_channel:
        base_model = _adapt_to_single_channel(base_model, backbone_name, use_pretrained)

    # ---------- 拆分 backbone 与 classifier ----------
    if backbone_name.startswith("resnet"):
        # 去除原始分类头，用恒等映射替代，保留全局平均池化
        backbone = nn.Sequential(
            *list(base_model.children())[:-1],  # 去掉 fc 层
            nn.Flatten()
        )
        if dropout_rate > 0:
            backbone = nn.Sequential(backbone, nn.Dropout(p=dropout_rate))

    elif backbone_name == "efficientnet_b0":
        # EfficientNet 结构：features + avgpool + classifier
        backbone = nn.Sequential(
            base_model.features,
            base_model.avgpool,
            nn.Flatten(),
        )
        if dropout_rate > 0:
            backbone = nn.Sequential(backbone, nn.Dropout(p=dropout_rate))

    # ---------- 构建新的分类头 ----------
    classifier = nn.Linear(feature_dim, num_classes)
    nn.init.xavier_uniform_(classifier.weight)
    nn.init.zeros_(classifier.bias)

    model = FeatureExtractorWrapper(backbone, classifier)

    print(f"\n{'='*60}")
    print(f"  模型构建信息")
    print(f"{'='*60}")
    print(f"  主干网络      : {backbone_name}")
    print(f"  预训练权重    : {pretrained}")
    print(f"  输入通道数    : {'1（单通道）' if single_channel else '3（三通道）'}")
    print(f"  特征维度      : {feature_dim}")
    print(f"  已知类别数    : {num_classes}")
    print(f"  Dropout 率    : {dropout_rate}")
    print(f"{'='*60}\n")

    return model


def _adapt_to_single_channel(
    model: nn.Module,
    backbone_name: str,
    use_pretrained: bool
) -> nn.Module:
    """
    将模型第一层卷积由 3 通道修改为 1 通道
    若使用预训练权重，对原始三通道权重取均值作为单通道权重（知识迁移）
    """
    if backbone_name.startswith("resnet"):
        old_conv = model.conv1
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        if use_pretrained:
            # 三通道权重按通道均值合并为单通道
            new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
        model.conv1 = new_conv

    elif backbone_name == "efficientnet_b0":
        old_conv = model.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        if use_pretrained:
            new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
        model.features[0][0] = new_conv

    return model
