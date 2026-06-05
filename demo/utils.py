"""demo 脚本共用的工具函数。

包含：
- CIFAR-10 类别名
- 图片可视化函数
- 准确率计算函数
- 保存对抗样本的函数
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── CIFAR-10 类别名 ───────────────────────────────────────────────────────────
# 顺序和数据集标签编号 0~9 严格对应
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


# ── 准确率计算函数 ─────────────────────────────────────────────────────────────
def evaluate_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    attack=None,
) -> float:
    """计算模型在给定数据集上的准确率。

    Args:
        model:  要评估的模型，必须已调用 model.eval()。
        loader: 数据加载器。
        device: 运行设备（cpu 或 cuda）。
        attack: 可选，torchattacks 的攻击对象。
                传入时先对每批图片生成对抗样本，再评估准确率。

    Returns:
        准确率，取值范围 [0, 1]。
    """
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if attack is not None:
            # 生成对抗样本，attack 内部自动处理梯度，外部不需要额外设置
            images = attack(images, labels)

        with torch.no_grad():
            logits = model(images)                              # (N, 10) 类别分数
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return correct / total


# ── 图片可视化函数 ─────────────────────────────────────────────────────────────
def show_images(
    images: torch.Tensor,
    labels: torch.Tensor,
    preds: torch.Tensor,
    title: str = "",
    save_path: Path | None = None,
    max_cols: int = 8,
) -> None:
    """把一批图片排成网格展示，并标注真实标签和预测标签。

    Args:
        images:    图片张量，shape (N, 3, 32, 32)，像素值范围 [0, 1]。
        labels:    真实标签，shape (N,)。
        preds:     模型预测标签，shape (N,)。
        title:     整张图的标题。
        save_path: 保存路径。为 None 时不保存。
        max_cols:  每行最多展示几张图。
    """
    n = len(images)
    cols = min(n, max_cols)
    rows = (n + cols - 1) // cols                              # 向上取整

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.4))
    if title:
        fig.suptitle(title, fontsize=12)

    # 保证 axes 始终是二维数组，方便统一索引
    axes = axes.reshape(rows, cols) if rows > 1 else axes.reshape(1, cols)

    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        if i < n:
            # permute(1,2,0)：PyTorch 的 (C,H,W) → matplotlib 的 (H,W,C)
            img = images[i].cpu().permute(1, 2, 0).numpy().clip(0, 1)
            ax.imshow(img)
            true_name = CIFAR10_CLASSES[labels[i].item()]
            pred_name = CIFAR10_CLASSES[preds[i].item()]
            # 预测正确绿色，预测错误红色
            color = "green" if preds[i] == labels[i] else "red"
            ax.set_title(f"真:{true_name}\n预:{pred_name}", fontsize=7, color=color)
        ax.axis("off")

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")

    plt.close(fig)


# ── 保存对抗样本的函数 ─────────────────────────────────────────────────────────
def save_adversarial_examples(
    clean: torch.Tensor,
    adv: torch.Tensor,
    labels: torch.Tensor,
    save_path: Path,
    attack_name: str = "attack",
) -> None:
    """把干净样本和对抗样本一起保存成 .pt 文件。

    保存格式是一个字典，方便后续实验（如迁移攻击）直接加载使用，
    不需要重新生成对抗样本。

    Args:
        clean:       原始干净图片，shape (N, 3, 32, 32)。
        adv:         对抗样本，shape (N, 3, 32, 32)。
        labels:      真实标签，shape (N,)。
        save_path:   保存的 .pt 文件路径。
        attack_name: 攻击方法名称，作为字典的 key 区分不同攻击的样本。
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "clean": clean.cpu(),
        attack_name: adv.cpu(),
        "labels": labels.cpu(),
    }

    # 如果文件已存在，合并而不是覆盖，这样可以把多个攻击的结果存在同一个文件里
    if save_path.exists():
        existing = torch.load(save_path, map_location="cpu")
        existing.update(data)
        data = existing

    torch.save(data, save_path)
    print(f"对抗样本已保存：{save_path}  (keys: {list(data.keys())})")

