# Adversarial Attacks PyTorch Lab

这是我的 PyTorch 对抗攻击复盘项目，也是后续学习深度学习、大模型安全、鲁棒性评估的长期学习仓库。

当前项目从 CIFAR-10 图像分类开始，先训练一个基础 CNN 模型，再围绕白盒攻击流程复现 PGD/FGSM 一类对抗样本生成方法。项目目标不是追求最高榜单成绩，而是把“模型、数据、攻击、评估、可视化”这条链路吃透。

复盘参考：

- https://daihuohuo2.github.io/myblog/2026/06/01/adversarial-attacks-pytorch-%E5%A4%8D%E7%9B%98/
- https://github.com/Harry24k/adversarial-attacks-pytorch

## 项目目标

这个仓库主要用于记录和复盘：

- PyTorch 基础图像分类模型训练；
- CIFAR-10 数据加载和评估；
- 白盒对抗攻击流程；
- PGD / FGSM 等攻击方法的核心思想；
- 干净准确率、鲁棒准确率、攻击成功率的计算；
- 对抗样本和扰动的可视化；
- 后续大模型安全学习中的实验代码和笔记。

## 当前状态

已经完成：

- 基础环境检查脚本；
- CIFAR-10 CNN 训练脚本；
- 本地模型权重加载逻辑；
- 白盒 PGD 攻击 demo；
- 对抗样本可视化保存；
- 适合 GitHub 的项目目录和忽略规则。

注意：数据集、模型权重、实验输出图片默认不会提交到 GitHub。它们会保留在本地。

## 目录结构

```text
adversarial-attacks-pytorch-lab/
├─ data/                         # 本地数据集，不提交实际数据
├─ demo/                         # 可直接运行的复盘 demo
│  ├─ White-box Attack on CIFAR10.py
│  ├─ White-box Targeted Attack on CIFAR10.py
│  ├─ Transfer Attack on CIFAR10.py
│  └─ utils.py
├─ experiments/                  # 后续扩展实验入口
├─ models/                       # 本地模型权重，不提交 .pth 文件
├─ outputs/                      # 本地实验输出，不提交生成结果
│  ├─ adversarial_examples/
│  ├─ figures/
│  ├─ logs/
│  └─ metrics/
├─ scripts/                      # 环境检查、模型训练等脚本
│  ├─ check_env.py
│  └─ train_cifar10_cnn.py
├─ src/                          # 可复用源码
│  └─ models/
│     └─ cifar10_cnn.py
├─ torchattacks/                 # 用于复盘攻击库结构的本地代码
│  ├─ attack.py
│  └─ attacks/
│     ├─ fgsm.py
│     └─ pgd.py
├─ requirements.txt
└─ README.md
```

## 环境准备

本项目使用 Python 和 PyTorch。建议在虚拟环境中安装依赖：

```powershell
python -m pip install -r requirements.txt
```

当前主要依赖：

```text
torch
torchvision
torchattacks
robustbench
matplotlib
numpy
```

如果使用本地已有环境，可以先运行：

```powershell
python scripts/check_env.py
```

## 训练基础模型

对抗攻击需要一个可以被攻击的分类模型。本项目提供了一个基础 CIFAR-10 CNN：

```powershell
python scripts/train_cifar10_cnn.py
```

训练完成后会在本地生成：

```text
models/cifar10_cnn.pth
```

这个文件是模型权重，默认不会提交到 GitHub。

## 白盒攻击流程

白盒攻击的基本链路：

```text
准备环境
  ↓
导入 torch / torchattacks / robustbench
  ↓
加载数据 images, labels
  ↓
加载模型 model
  ↓
检查干净样本准确率 clean accuracy
  ↓
选择攻击算法 atk
  ↓
设置攻击参数 eps / alpha / steps / random_start
  ↓
如有标准化，调用 set_normalization_used
  ↓
生成对抗样本 adv_images = atk(images, labels)
  ↓
用 model(adv_images) 测量鲁棒准确率 / 攻击成功率
  ↓
可视化或保存 adv_images
```

当前可运行 demo：

```powershell
python "demo/White-box Attack on CIFAR10.py"
```

该脚本会完成：

- 加载 CIFAR-10 测试样本；
- 加载本地训练好的 CNN；
- 计算干净样本准确率；
- 使用 PGD 生成对抗样本；
- 计算鲁棒准确率和攻击成功率；
- 保存原图、对抗样本和扰动图。

输出位置：

```text
outputs/adversarial_examples/
```

## 常用指标

| 指标 | 含义 |
|---|---|
| clean accuracy | 模型在原始图片上的准确率 |
| robust accuracy | 模型在对抗样本上的准确率 |
| attack success rate | 攻击成功率，通常可以理解为 1 - robust accuracy |
| eps | 最大扰动范围 |
| alpha | PGD 每一步更新的步长 |
| steps | PGD 迭代次数 |

## 后续计划

- 复盘 FGSM 的单步梯度攻击；
- 复盘 PGD 的多步迭代攻击；
- 增加 targeted attack；
- 增加 transfer attack；
- 把评估指标保存为 CSV/JSON；
- 对不同 eps、steps 做对比实验；
- 逐步迁移到更复杂模型和大模型安全方向。

## 说明

这个仓库是学习型项目。代码会优先保证流程清楚、注释可读、便于复盘，而不是一开始就追求复杂工程化。
