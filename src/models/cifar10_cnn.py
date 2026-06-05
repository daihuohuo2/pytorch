"""CIFAR-10 分类模型定义。

这个文件是项目的核心模型文件，存放在 src/models/ 下，
供 scripts/（训练）和 demo/（攻击演示）统一导入，
避免重复定义和 sys.path 操作。
"""

import torch
import torch.nn as nn


# CIFAR-10 的 10 个类别名，顺序和数据集标签编号 0~9 对应。
CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


class SimpleCifar10CNN(nn.Module):
    """用于 CIFAR-10 图像分类的基础 CNN 模型。

    这个模型的目标不是追求最高准确率，而是给后续对抗攻击实验提供一个
    能正常分类 CIFAR-10 图片的目标模型。

    输入:
        shape = (batch_size, 3, 32, 32)
        3 表示 RGB 三个颜色通道，32x32 是图片尺寸。

    输出:
        shape = (batch_size, 10)
        每张图片输出 10 个分数，对应 CIFAR-10 的 10 个类别。
        注意：这里输出的是 logits，没有经过 Softmax。
        torchattacks 和 CrossEntropyLoss 都直接接受 logits 输入。
    """

    def __init__(self) -> None:
        super().__init__()

        # features 是特征提取模块，负责从图片中提取边缘、纹理、形状等视觉特征。
        # 通道数逐层增加（32 → 64 → 128），让模型能表达越来越复杂的图像概念。
        self.features = nn.Sequential(
            # 第一组卷积：3 通道输入 → 32 通道特征，空间尺寸保持 32x32
            # padding=1 保证 3x3 卷积后宽高不变
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),   # 批归一化：稳定训练，加快收敛
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),      # 32x32 → 16x16，缩小特征图，降低计算量
            nn.Dropout(0.2),      # 训练时随机丢弃 20% 神经元，抑制过拟合

            # 第二组卷积：32 通道 → 64 通道，空间尺寸保持 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),      # 16x16 → 8x8
            nn.Dropout(0.3),

            # 第三组卷积：64 通道 → 128 通道，空间尺寸保持 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),      # 8x8 → 4x4
        )

        # classifier 是分类器模块，负责把卷积提取的特征转换成 10 个类别分数。
        self.classifier = nn.Sequential(
            # 展平：把 (128, 4, 4) 的多维特征图压成一维向量，长度 128*4*4=2048
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),  # 全连接层，压缩到 256 维
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 10),            # 最终映射到 10 个类别分数
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：图片经特征提取后送入分类器，返回 logits。"""
        x = self.features(x)
        return self.classifier(x)
