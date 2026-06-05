import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# 当前脚本位于:
#   adversarial-attacks-pytorch-lab/scripts/train_cifar10_cnn.py
#
# parents[1] 是项目根目录 adversarial-attacks-pytorch-lab/。
# 无论从哪个终端目录运行这个脚本，数据和模型都会保存到本项目里。
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 把项目根目录加入搜索路径，这样才能 import src.models 包
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# SimpleCifar10CNN 和 CIFAR10_CLASSES 定义在 src/models/cifar10_cnn.py，
# 集中管理，供 scripts/ 和 demo/ 统一导入，避免重复定义。
from src.models.cifar10_cnn import SimpleCifar10CNN, CIFAR10_CLASSES



def get_loaders(data_dir: Path, batch_size: int) -> tuple[DataLoader, DataLoader]:
    """加载 CIFAR-10 数据集，并返回训练集和测试集的数据加载器。"""

    # ToTensor 会把图片转成 PyTorch 张量，并把像素值从 0~255 缩放到 0~1。
    # 当前模型没有额外做标准化，这样后面接 torchattacks 做基础攻击更直接。
    transform = transforms.ToTensor()

    # train=True 表示训练集，共 50000 张图片。
    # download=True 表示如果本地没有数据集，就自动下载。
    train_set = datasets.CIFAR10(
        root=str(data_dir),
        train=True,
        download=True,
        transform=transform,
    )

    # train=False 表示测试集，共 10000 张图片。
    # 测试集只用于评估，不参与参数更新。
    test_set = datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=True,
        transform=transform,
    )

    # DataLoader 是 PyTorch 的数据加载器，负责按批次读取数据。
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """训练一轮模型。

    一轮训练表示把整个训练集完整遍历一次。
    返回值是这一轮的平均损失和平均准确率。
    """

    # 切换到训练模式。
    # Dropout 会启用，BatchNorm 会使用当前批次的数据更新统计量。
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        # 模型在哪个设备上，数据也必须放到同一个设备上。
        images = images.to(device)
        labels = labels.to(device)

        # 清空上一批数据留下的梯度。
        optimizer.zero_grad(set_to_none=True)

        # 前向传播：模型根据图片输出类别分数。
        logits = model(images)

        # 计算分类损失。
        # CrossEntropyLoss 适合“输出类别分数、标签是类别编号”的多分类任务。
        loss = criterion(logits, labels)

        # 反向传播：计算每个参数应该怎么调整。
        loss.backward()

        # 根据梯度更新模型参数。
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size

        # logits.argmax(dim=1) 会取每张图片分数最高的类别作为预测结果。
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """评估模型。

    评估阶段不更新模型参数，只计算损失和准确率。
    """

    # 切换到评估模式。
    # Dropout 会关闭，BatchNorm 会使用训练阶段保存好的统计量。
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return total_loss / total, correct / total


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    例子:
        python train_cifar10_cnn.py --epochs 1
        python train_cifar10_cnn.py --batch-size 64 --lr 0.0005
    """

    parser = argparse.ArgumentParser(description="训练一个基础 CIFAR-10 CNN 图像分类模型。")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--model-out",
        type=Path,
        default=PROJECT_ROOT / "models" / "cifar10_cnn.pth",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 确保数据目录和模型保存目录存在。
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)

    # 优先使用显卡；如果没有显卡，就使用 CPU。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)
    if torch.cuda.is_available():
        print("显卡:", torch.cuda.get_device_name(0))

    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size)

    model = SimpleCifar10CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 只保存测试集准确率最高的那一版模型。
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        if test_acc > best_acc:
            best_acc = test_acc

            # state_dict 是 PyTorch 推荐的模型参数保存方式。
            # 保存这些元信息，是为了后面做对抗攻击实验时方便确认模型配置。
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": CIFAR10_CLASSES,
                    "image_shape": (3, 32, 32),
                    "input_range": (0.0, 1.0),
                    "test_accuracy": best_acc,
                },
                args.model_out,
            )

        print(
            f"轮次 {epoch:02d}/{args.epochs} "
            f"训练损失={train_loss:.4f} 训练准确率={train_acc:.4f} "
            f"测试损失={test_loss:.4f} 测试准确率={test_acc:.4f} "
            f"最佳准确率={best_acc:.4f}"
        )

    print("最佳模型已保存到:", args.model_out)


if __name__ == "__main__":
    main()
