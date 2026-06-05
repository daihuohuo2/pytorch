"""
白盒定向攻击演示：White-box Targeted Attack on CIFAR-10

普通白盒攻击 vs 定向攻击的区别：
  普通攻击：只需让模型预测出"任意错误类别"即可。
  定向攻击：必须让模型预测出"攻击者指定的目标类别"才算成功。

本脚本流程（对应博客第 7 步）：
    7.1 加载模型和数据
    7.2 构造 PGD 攻击器
    7.3 切换到 targeted by label 模式
    7.4 构造目标标签：(真实标签 + 1) % 10
    7.5 生成目标攻击样本
    7.6 检查是否攻击到目标类别
    7.7 可视化对比并保存
"""

import sys
from pathlib import Path

# ── 路径设置 ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 必须先导入 torchattacks（pip 安装版），再把 PROJECT_ROOT 加入 sys.path。
# 原因：项目根目录下有同名的 torchattacks/ 占位文件夹，顺序反了会导入空壳。
import torch
import torch.nn as nn
import torchattacks                          # 必须在 sys.path 加入项目根目录之前导入
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cifar10_cnn import SimpleCifar10CNN, CIFAR10_CLASSES

# ── 路径配置 ──────────────────────────────────────────────────────────────────
MODEL_PATH  = PROJECT_ROOT / "models" / "cifar10_cnn.pth"
DATA_DIR    = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
ADV_DIR     = PROJECT_ROOT / "outputs" / "adversarial_examples"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
ADV_DIR.mkdir(parents=True, exist_ok=True)

# ── 攻击参数 ──────────────────────────────────────────────────────────────────
N_SAMPLES  = 1000
BATCH_SIZE = 100

# 定向攻击通常需要比普通攻击更大的 eps 和更多步数，
# 因为"让模型错成指定类别"比"让模型出错"更难。
PGD_EPS   = 8 / 255
PGD_ALPHA = 2 / 255
PGD_STEPS = 20    # 比白盒攻击多一倍步数，给定向攻击更多迭代机会

# ── 加载模型 ──────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{device}\n")

checkpoint = torch.load(MODEL_PATH, map_location=device)
model = SimpleCifar10CNN().to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"模型已加载：{MODEL_PATH}")
print(f"训练时测试集准确率：{checkpoint['test_accuracy']:.4f}\n")

# ── 加载数据 ──────────────────────────────────────────────────────────────────
transform = transforms.ToTensor()
test_set = datasets.CIFAR10(root=str(DATA_DIR), train=False, download=True, transform=transform)
subset = Subset(test_set, range(N_SAMPLES))
loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False)
print(f"演示样本数：{N_SAMPLES} 张\n")

# ── 7.2 构造 PGD 攻击器 ───────────────────────────────────────────────────────
# 和普通白盒攻击一样，先构造 PGD，后续通过 set_mode 切换成定向模式。
pgd = torchattacks.PGD(model, eps=PGD_EPS, alpha=PGD_ALPHA, steps=PGD_STEPS)

# ── 7.3 切换到 targeted by label 模式 ─────────────────────────────────────────
# 调用这个方法后，atk(images, labels) 的第二个参数变成"目标标签"，
# 不再是真实标签。攻击目标从"让模型出错"变成"让模型预测成目标标签"。
pgd.set_mode_targeted_by_label()

# ── 评估函数 ──────────────────────────────────────────────────────────────────
def evaluate_targeted(
    model: nn.Module,
    loader: DataLoader,
    attack,
) -> tuple[float, float]:
    """同时统计普通错误率和定向攻击成功率。

    Returns:
        (robust_acc, targeted_success_rate)
        robust_acc：对抗样本上模型仍预测正确的比例（越低攻击越强）。
        targeted_success_rate：对抗样本预测结果恰好等于目标标签的比例（越高定向越准）。
    """
    correct = 0        # 和真实标签一致的数量
    targeted_hit = 0   # 和目标标签一致的数量
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        # 目标标签：把每个样本的真实类别循环加 1
        # 例：真实标签 = [cat(3), dog(5), ...] → 目标标签 = [deer(4), frog(6), ...]
        target_labels = (labels + 1) % 10

        # 生成定向对抗样本
        adv_images = attack(images, target_labels)

        with torch.no_grad():
            logits = model(adv_images)
            preds = logits.argmax(dim=1)

        correct     += (preds == labels).sum().item()
        targeted_hit += (preds == target_labels).sum().item()
        total += labels.size(0)

    return correct / total, targeted_hit / total


# ── 7.4-7.5-7.6 生成并评估定向攻击 ───────────────────────────────────────────
print("正在生成定向对抗样本，请稍候（步数较多，耗时比普通白盒攻击长）...\n")
robust_acc, targeted_success = evaluate_targeted(model, loader, pgd)

print("── 定向攻击结果 ──────────────────────────────────────")
print(f"  对抗样本上的准确率（越低说明攻击越强）  : {robust_acc:.4f}")
print(f"  定向攻击成功率（预测 = 目标类别的比例）: {targeted_success:.4f}")
print(f"  目标标签规则：真实标签 → (真实标签+1)%10")
print("─────────────────────────────────────────────────────\n")

# ── 7.7 可视化 ────────────────────────────────────────────────────────────────
# 取第一批次详细可视化：展示"真实标签 → 目标标签 → 实际预测"
first_images, first_labels = next(iter(loader))
first_images  = first_images.to(device)
first_labels  = first_labels.to(device)
target_labels = (first_labels + 1) % 10

adv_images = pgd(first_images, target_labels)

with torch.no_grad():
    clean_preds = model(first_images).argmax(dim=1)
    adv_preds   = model(adv_images).argmax(dim=1)

# 筛选"定向攻击成功"的样本：预测结果 == 目标标签，最多展示 8 张
targeted_success_mask = (adv_preds == target_labels)
show_idx = targeted_success_mask.nonzero(as_tuple=True)[0][:8]

if len(show_idx) == 0:
    print("当前批次没有定向攻击成功的样本，跳过可视化。")
else:
    n = len(show_idx)
    fig, axes = plt.subplots(2, n, figsize=(n * 2.5, 5.5))
    fig.suptitle("Targeted Attack: Original vs Adversarial", fontsize=12)

    for col, idx in enumerate(show_idx):
        # 上行：原图
        ax_orig = axes[0, col] if n > 1 else axes[0]
        img_orig = first_images[idx].cpu().permute(1, 2, 0).numpy().clip(0, 1)
        ax_orig.imshow(img_orig)
        ax_orig.axis("off")
        true_name   = CIFAR10_CLASSES[first_labels[idx].item()]
        clean_name  = CIFAR10_CLASSES[clean_preds[idx].item()]
        ax_orig.set_title(f"true:{true_name}\npred:{clean_name}", fontsize=7,
                          color="green" if clean_preds[idx] == first_labels[idx] else "red")
        if col == 0:
            ax_orig.set_ylabel("Original", fontsize=9)

        # 下行：对抗样本
        ax_adv = axes[1, col] if n > 1 else axes[1]
        img_adv = adv_images[idx].cpu().permute(1, 2, 0).numpy().clip(0, 1)
        ax_adv.imshow(img_adv)
        ax_adv.axis("off")
        target_name = CIFAR10_CLASSES[target_labels[idx].item()]
        adv_name    = CIFAR10_CLASSES[adv_preds[idx].item()]
        # 定向成功：实际预测 == 目标标签，用绿色
        color = "green" if adv_preds[idx] == target_labels[idx] else "orange"
        ax_adv.set_title(f"target:{target_name}\npred:{adv_name}", fontsize=7, color=color)
        if col == 0:
            ax_adv.set_ylabel("Adversarial", fontsize=9)

    plt.tight_layout()
    save_path = FIGURES_DIR / "targeted_attack_comparison.png"
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"可视化图已保存：{save_path}")

# ── 保存对抗样本张量 ──────────────────────────────────────────────────────────
adv_save = {
    "clean":          first_images[:100].cpu(),
    "targeted_pgd":   adv_images[:100].cpu(),
    "true_labels":    first_labels[:100].cpu(),
    "target_labels":  target_labels[:100].cpu(),
}
adv_path = ADV_DIR / "targeted_adv_examples.pt"
torch.save(adv_save, adv_path)
print(f"对抗样本张量已保存：{adv_path}")
