"""
白盒攻击演示：FGSM 和 PGD 攻击 CIFAR-10 上训练好的 CNN 模型。

什么是白盒攻击：
    攻击者能完全访问目标模型的结构和参数，可以直接对输入图片求梯度。
    这是最理想的攻击条件，也是衡量模型鲁棒性下界的标准方式。

本脚本流程：
    1. 加载训练好的 SimpleCifar10CNN 模型
    2. 加载 CIFAR-10 测试集（取前 1000 张，快速演示）
    3. 分别用 FGSM 和 PGD 生成对抗样本
    4. 对比干净样本 vs 对抗样本的准确率
    5. 可视化部分对抗样本并保存
"""

import sys
from pathlib import Path

# ── 路径设置 ──────────────────────────────────────────────────────────────────
# parents[1] 就是项目根目录 adversarial-attacks-pytorch-lab/。
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 把项目根目录加入搜索路径，才能 import src.models 包。
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torchattacks                          # pip 安装的正式版对抗攻击库
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")                        # 不弹出窗口，直接把图保存成文件
import matplotlib.pyplot as plt

# SimpleCifar10CNN 和 CIFAR10_CLASSES 统一定义在 src/models/cifar10_cnn.py。
from src.models.cifar10_cnn import SimpleCifar10CNN, CIFAR10_CLASSES


# ── 路径配置 ──────────────────────────────────────────────────────────────────
MODEL_PATH  = PROJECT_ROOT / "models" / "cifar10_cnn.pth"
DATA_DIR    = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
ADV_DIR     = PROJECT_ROOT / "outputs" / "adversarial_examples"

# 确保输出目录存在，不存在就自动创建
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
ADV_DIR.mkdir(parents=True, exist_ok=True)

# ── 攻击参数 ──────────────────────────────────────────────────────────────────
N_SAMPLES  = 1000    # 演示时只取测试集前 N 张，节省时间
BATCH_SIZE = 100

# eps：允许对每个像素施加的最大扰动幅度（图片范围是 0~1）。
# 8/255 ≈ 0.031 是学术论文常用基准值，肉眼几乎看不出差异但足以欺骗模型。
FGSM_EPS = 8 / 255

# PGD 比 FGSM 多两个参数：
#   alpha：每步步长，通常取 2/255
#   steps：迭代步数，步数越多攻击越强，但也越慢
PGD_EPS   = 8 / 255
PGD_ALPHA = 2 / 255
PGD_STEPS = 10

# ── 加载模型 ──────────────────────────────────────────────────────────────────
# 优先使用 GPU，没有 GPU 就用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{device}\n")

# map_location=device 确保不管训练时用 GPU 还是 CPU，都能正确加载权重
checkpoint = torch.load(MODEL_PATH, map_location=device)

# 创建模型实例，加载权重，切换到评估模式。
# model.eval() 会关闭 Dropout，BatchNorm 使用训练时统计好的均值/方差。
model = SimpleCifar10CNN().to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"模型已加载：{MODEL_PATH}")
print(f"训练时测试集准确率：{checkpoint['test_accuracy']:.4f}\n")

# ── 加载数据 ──────────────────────────────────────────────────────────────────
# 必须和训练时一致：只用 ToTensor()，不加 Normalize，像素范围 [0, 1]。
# 如果多加了 Normalize，就需要额外告诉攻击器标准化参数，否则攻击会出错。
transform = transforms.ToTensor()
test_set = datasets.CIFAR10(root=str(DATA_DIR), train=False, download=True, transform=transform)

# Subset 截取数据集的子集，range(N_SAMPLES) 表示取索引 0 ~ N_SAMPLES-1 的样本
subset = Subset(test_set, range(N_SAMPLES))
loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False)
print(f"演示样本数：{N_SAMPLES} 张\n")

# ── 评估函数 ──────────────────────────────────────────────────────────────────
def evaluate(model: nn.Module, loader: DataLoader, attack=None) -> float:
    """计算准确率。传入 attack 时先生成对抗样本再评估。"""
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        if attack is not None:
            # attack(images, labels) 会：
            #   1. 临时开启梯度计算（无论外部是否 no_grad）
            #   2. 对 images 中每个像素求损失的梯度
            #   3. 沿梯度方向施加扰动，返回对抗样本（形状和原图完全一样）
            images = attack(images, labels)
        # 评估阶段不需要梯度，no_grad 节省显存和计算量
        with torch.no_grad():
            logits = model(images)                               # 前向传播，得到 (N, 10) 类别分数
            correct += (logits.argmax(dim=1) == labels).sum().item()  # 取分数最高的类别作为预测
            total += labels.size(0)
    return correct / total

# ── 干净样本基准准确率 ────────────────────────────────────────────────────────
clean_acc = evaluate(model, loader, attack=None)
print(f"干净样本准确率：{clean_acc:.4f}")

# ── FGSM 攻击 ─────────────────────────────────────────────────────────────────
# FGSM：Fast Gradient Sign Method（快速梯度符号法）
# 公式：adv_x = x + eps * sign(grad_x Loss(model(x), y))
# 只走一步：计算损失对输入图片的梯度，取符号（+1 或 -1），乘以 eps 加到原图上。
# 优点：极快；缺点：一步攻击效果不如迭代攻击强。
fgsm = torchattacks.FGSM(model, eps=FGSM_EPS)
fgsm_acc = evaluate(model, loader, attack=fgsm)
print(f"FGSM 攻击后准确率（eps={FGSM_EPS:.4f}）：{fgsm_acc:.4f}  下降 {clean_acc - fgsm_acc:.4f}")

# ── PGD 攻击 ──────────────────────────────────────────────────────────────────
# PGD：Projected Gradient Descent（投影梯度下降）
# 思路：重复做多步 FGSM，每步后把扰动"投影"回 eps 球（clip 到 [x-eps, x+eps]），
#       防止累积扰动超出允许范围。
# 步数越多攻击越强，是目前最常用的强白盒攻击基准。
pgd = torchattacks.PGD(model, eps=PGD_EPS, alpha=PGD_ALPHA, steps=PGD_STEPS)
pgd_acc = evaluate(model, loader, attack=pgd)
print(f"PGD  攻击后准确率（eps={PGD_EPS:.4f}, steps={PGD_STEPS}）：{pgd_acc:.4f}  下降 {clean_acc - pgd_acc:.4f}")

# ── 结果汇总 ──────────────────────────────────────────────────────────────────
print("\n── 攻击效果汇总 ──────────────────────────────────")
print(f"  干净样本准确率         : {clean_acc:.4f}")
print(f"  FGSM 攻击后准确率      : {fgsm_acc:.4f}  (下降 {clean_acc - fgsm_acc:.4f})")
print(f"  PGD  攻击后准确率      : {pgd_acc:.4f}  (下降 {clean_acc - pgd_acc:.4f})")
print("─────────────────────────────────────────────────\n")

# ── 可视化对抗样本 ────────────────────────────────────────────────────────────
# 只取第一个 batch（100 张）做可视化，节省时间
first_images, first_labels = next(iter(loader))
first_images = first_images.to(device)
first_labels = first_labels.to(device)

# 分别生成 FGSM 和 PGD 对抗样本
fgsm_images = fgsm(first_images, first_labels)
pgd_images  = pgd(first_images, first_labels)

def get_pred(model, images):
    """对一批图片推理，返回每张图片预测的类别编号。"""
    with torch.no_grad():
        return model(images).argmax(dim=1)

clean_preds = get_pred(model, first_images)
fgsm_preds  = get_pred(model, fgsm_images)
pgd_preds   = get_pred(model, pgd_images)

# 筛选"原来预测正确、但被 PGD 成功欺骗"的样本展示，最多取 6 张。
# 这类样本最有说服力：人眼几乎看不出差别，但模型已经判断出错。
orig_correct = (clean_preds == first_labels)   # 布尔张量：哪些原来预测对了
pgd_fooled   = (pgd_preds  != first_labels)    # 布尔张量：哪些被 PGD 骗了
show_idx     = (orig_correct & pgd_fooled).nonzero(as_tuple=True)[0][:6]

if len(show_idx) == 0:
    print("当前批次没有被成功攻击的样本，跳过可视化。")
else:
    n = len(show_idx)
    # 3 行（原图 / FGSM 对抗样本 / PGD 对抗样本）x n 列（每列一张图）
    fig, axes = plt.subplots(3, n, figsize=(n * 2.5, 7))
    fig.suptitle("白盒攻击可视化（每列一张图）", fontsize=13)

    row_titles = ["原图", "FGSM 对抗样本", "PGD 对抗样本"]
    imgs_list  = [first_images, fgsm_images, pgd_images]
    preds_list = [clean_preds,  fgsm_preds,  pgd_preds]

    for row, (row_title, imgs, preds) in enumerate(zip(row_titles, imgs_list, preds_list)):
        for col, idx in enumerate(show_idx):
            ax = axes[row, col] if n > 1 else axes[row]
            # permute(1,2,0)：PyTorch 的 (C,H,W) → matplotlib 的 (H,W,C)
            # clip(0,1)：防止 PGD 数值误差导致超出显示范围
            img = imgs[idx].cpu().permute(1, 2, 0).numpy().clip(0, 1)
            ax.imshow(img)
            ax.axis("off")
            true_name = CIFAR10_CLASSES[first_labels[idx].item()]
            pred_name = CIFAR10_CLASSES[preds[idx].item()]
            # 预测正确显示绿色，预测错误显示红色
            color = "green" if preds[idx] == first_labels[idx] else "red"
            ax.set_title(f"真:{true_name}\n预:{pred_name}", fontsize=8, color=color)
            if col == 0:
                ax.set_ylabel(row_title, fontsize=10)

    plt.tight_layout()
    save_path = FIGURES_DIR / "whitebox_attack_comparison.png"
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"可视化图已保存：{save_path}")

# ── 保存对抗样本张量 ──────────────────────────────────────────────────────────
# 把对抗样本保存成 .pt 文件，后续迁移攻击实验可以直接加载，不用重新生成。
adv_save = {
    "clean":  first_images[:100].cpu(),
    "fgsm":   fgsm_images[:100].cpu(),
    "pgd":    pgd_images[:100].cpu(),
    "labels": first_labels[:100].cpu(),
}
adv_path = ADV_DIR / "whitebox_adv_examples.pt"
torch.save(adv_save, adv_path)
print(f"对抗样本张量已保存：{adv_path}")
