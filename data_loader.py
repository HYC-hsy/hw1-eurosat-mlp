"""
数据加载与预处理模块
- 加载 EuroSAT_RGB 数据集
- 划分训练集、验证集、测试集
- 数据归一化与展平
"""

import os
import numpy as np
from PIL import Image


CLASS_NAMES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
    'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
    'River', 'SeaLake'
]


def load_eurosat(data_dir, img_size=64):
    """加载 EuroSAT 数据集，返回图像数组和标签数组"""
    images = []
    labels = []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
            fpath = os.path.join(class_dir, fname)
            img = Image.open(fpath).convert('RGB')
            img = img.resize((img_size, img_size))
            img_array = np.array(img, dtype=np.float32)
            images.append(img_array)
            labels.append(class_idx)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels


def preprocess(images, mean=None, std=None):
    """
    归一化到 [0, 1] 并做 per-channel 标准化，最后展平为二维数组 (N, D)
    参考 mmdetection 的数据预处理: 减均值除标准差
    """
    # 归一化到 [0, 1]
    images = images.astype(np.float64) / 255.0

    # 计算或使用给定的 per-channel 均值和标准差
    if mean is None:
        # 分通道计算避免 float32 精度问题
        mean = np.array([images[:, :, :, c].mean() for c in range(3)])
    if std is None:
        std = np.array([images[:, :, :, c].std() for c in range(3)])

    # per-channel 标准化
    images = (images - mean.reshape(1, 1, 1, 3)) / (std.reshape(1, 1, 1, 3) + 1e-8)

    # 转回 float32 节省内存
    images = images.astype(np.float32)

    # 展平: (N, H, W, C) -> (N, H*W*C)
    N = images.shape[0]
    images_flat = images.reshape(N, -1)
    return images_flat, mean, std


def train_val_test_split(images, labels, train_ratio=0.7, val_ratio=0.15, seed=42):
    """划分训练集、验证集、测试集"""
    np.random.seed(seed)
    N = len(labels)
    indices = np.random.permutation(N)

    train_end = int(N * train_ratio)
    val_end = int(N * (train_ratio + val_ratio))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return (images[train_idx], labels[train_idx],
            images[val_idx], labels[val_idx],
            images[test_idx], labels[test_idx])


def get_data(data_dir, img_size=64):
    """完整的数据加载流程"""
    print("Loading dataset...")
    images, labels = load_eurosat(data_dir, img_size)
    print(f"Total samples: {len(labels)}, Image shape: {images.shape[1:]}")

    # 先划分数据集 (在原始图像上划分)
    X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test = train_val_test_split(images, labels)

    # 用训练集计算 per-channel 均值和标准差，再对所有集合做标准化
    X_train, mean, std = preprocess(X_train_raw)
    X_val, _, _ = preprocess(X_val_raw, mean=mean, std=std)
    X_test, _, _ = preprocess(X_test_raw, mean=mean, std=std)

    print(f"Flattened feature dim: {X_train.shape[1]}")
    print(f"Per-channel mean: {mean}, std: {std}")
    print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

    return X_train, y_train, X_val, y_val, X_test, y_test
