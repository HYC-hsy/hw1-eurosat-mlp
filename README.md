# HW1: 从零构建三层神经网络分类器 — EuroSAT 地表覆盖图像分类

基于 NumPy 从零实现三层 MLP 分类器，在 EuroSAT 遥感图像数据集上进行训练和评估。

## 环境依赖

- Python 3.10+
- NumPy
- Pillow (PIL)
- matplotlib
- tqdm

### 安装

```bash
# 创建 conda 环境（推荐）
conda create -n hw1_mlp python=3.10
conda activate hw1_mlp

# 安装依赖
pip install numpy pillow matplotlib tqdm
```

## 项目结构

```
├── data_loader.py          # 数据加载与预处理（per-channel 标准化）
├── model.py                # 模型定义（MLP + ReLU/Sigmoid/Tanh + 反向传播）
├── trainer.py              # 训练循环（SGD+Momentum, WarmupMultiStepLR, L2正则化）
├── test_eval.py            # 测试评估（准确率、混淆矩阵、错例分析）
├── hyperparam_search.py    # 超参数搜索（网格搜索 + 随机搜索）
├── visualize.py            # 可视化（Loss/Acc曲线、权重可视化、混淆矩阵）
├── main.py                 # 主程序入口（v1 基础版）
├── train_v2.py             # 改进版训练脚本（v2, 含Dropout+数据增强）
├── figures/                # v1 可视化结果
├── figures_v2/             # v2 可视化结果
├── hw1_report.pdf          # 实验报告
└── README.md
```

## 数据集

EuroSAT_RGB 数据集（需自行下载放置在项目同级目录）：
```
../EuroSAT_RGB/
├── AnnualCrop/       (3000 张)
├── Forest/           (3000 张)
├── HerbaceousVegetation/ (3000 张)
├── Highway/          (2500 张)
├── Industrial/       (2500 张)
├── Pasture/          (2000 张)
├── PermanentCrop/    (2500 张)
├── Residential/      (3000 张)
├── River/            (2500 张)
└── SeaLake/          (3000 张)
```

## 运行方式

### 1. 训练基础版模型（v1）

```bash
python main.py --mode train --epochs 50
```

### 2. 训练改进版模型（v2，推荐）

```bash
python train_v2.py
```

### 3. 测试评估（加载已训练权重）

```bash
python main.py --mode test
```

### 4. 完整流程（训练 + 测试）

```bash
python main.py --mode all --epochs 50
```

### 5. 超参数搜索

```bash
python main.py --mode search
```

### 6. 自定义参数

```bash
python main.py --mode train \
    --lr 0.01 \
    --hidden_dim1 512 \
    --hidden_dim2 256 \
    --weight_decay 1e-4 \
    --activation relu \
    --epochs 50 \
    --batch_size 128
```

## 模型权重

训练好的最优模型权重下载地址：

- v1 权重：[best_model_v1.npz](https://drive.google.com/file/d/13PD1Zi1ixc5-eJC1sUgi121pmZozREDQ/view?usp=drive_link) → 下载后放置到 `checkpoints/best_model.npz`
- v2 权重：[best_model_v2.npz](https://drive.google.com/file/d/1J3HsDvH2eUPhK67lY_5sj6lJLToq22FK/view?usp=drive_link) → 下载后放置到 `checkpoints_v2/best_model.npz`

## 实验结果

| 版本 | 验证准确率 | 测试准确率 | 特点 |
|------|-----------|-----------|------|
| v1 基础版 | 65.48% | 64.89% | 512/256 隐藏层，无正则化 |
| v2 改进版 | 70.35% | **68.91%** | 1024/512 + Dropout(0.4) + 数据增强 |

## 设计参考

参考 [mmdetection](https://github.com/open-mmlab/mmdetection) 的以下设计：
- SGD 优化器配置（momentum=0.9, weight_decay）
- WarmupMultiStepLR 学习率策略
- 混淆矩阵归一化百分比可视化
- 超参数网格搜索模式
