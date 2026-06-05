import importlib
import platform

import torch


def show_package(name: str) -> None:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        print(f"{name}: 未通过 ({exc})")
        return

    version = getattr(module, "__version__", "unknown")
    print(f"{name}: 通过 ({version})")


def main() -> None:
    print("Python 版本:", platform.python_version())
    print("PyTorch 版本:", torch.__version__)
    print("是否可以使用 CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA 设备:", torch.cuda.get_device_name(0))

    for package in ["torchvision", "numpy", "matplotlib", "torchattacks", "robustbench"]:
        show_package(package)


if __name__ == "__main__":
    main()
