"""
训练循环模块
- SGD 优化器 (支持 L2 正则化 / Weight Decay)
- 学习率衰减策略
- 训练循环 (含验证集评估与最优模型保存)
"""

import numpy as np
import os
from model import MLP
from tqdm import tqdm


class SGD:
    """
    SGD 优化器 (带动量)
    参考 mmdetection: optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
    """
    def __init__(self, model, lr=0.01, momentum=0.9, weight_decay=0.0):
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        # 初始化动量缓存
        self.velocity_W = [np.zeros_like(layer.W) for layer in model.layers]
        self.velocity_b = [np.zeros_like(layer.b) for layer in model.layers]

    def step(self):
        """
        更新参数 (带动量 + weight decay)
        参考 mmdetection: weight_decay 只作用于 W，不作用于 bias
        """
        for i, layer in enumerate(self.model.layers):
            # Weight decay 只对 W 生效 (不对 bias 做正则化)
            grad_W = layer.dW + self.weight_decay * layer.W
            grad_b = layer.db

            # 动量更新
            self.velocity_W[i] = self.momentum * self.velocity_W[i] + grad_W
            self.velocity_b[i] = self.momentum * self.velocity_b[i] + grad_b
            layer.W -= self.lr * self.velocity_W[i]
            layer.b -= self.lr * self.velocity_b[i]


class WarmupMultiStepLR:
    """
    带 Warmup 的 MultiStepLR 学习率衰减策略
    参考 mmdetection:
    - dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500)
    - dict(type='MultiStepLR', milestones=[8, 11], gamma=0.1)
    前 warmup_epochs 个 epoch 线性增长学习率，之后按 milestones 衰减
    """
    def __init__(self, optimizer, milestones, gamma=0.1, warmup_epochs=5):
        self.optimizer = optimizer
        self.milestones = sorted(milestones)
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        self.initial_lr = optimizer.lr

    def step(self, epoch):
        """根据 epoch 更新学习率"""
        if epoch < self.warmup_epochs:
            # 线性 warmup: 从 initial_lr * 0.01 增长到 initial_lr
            warmup_factor = 0.01 + (1.0 - 0.01) * (epoch / self.warmup_epochs)
            self.optimizer.lr = self.initial_lr * warmup_factor
        else:
            # MultiStepLR 衰减
            factor = 1.0
            for m in self.milestones:
                if epoch >= m:
                    factor *= self.gamma
            self.optimizer.lr = self.initial_lr * factor


class MultiStepLR:
    """
    MultiStepLR 学习率衰减策略
    参考 mmdetection: dict(type='MultiStepLR', milestones=[8, 11], gamma=0.1)
    在指定的 milestone epoch 处将学习率乘以 gamma
    """
    def __init__(self, optimizer, milestones, gamma=0.1):
        self.optimizer = optimizer
        self.milestones = sorted(milestones)
        self.gamma = gamma
        self.initial_lr = optimizer.lr

    def step(self, epoch):
        """根据 epoch 更新学习率"""
        factor = 1.0
        for m in self.milestones:
            if epoch >= m:
                factor *= self.gamma
        self.optimizer.lr = self.initial_lr * factor


class StepLR:
    """
    StepLR 学习率衰减策略
    每 step_size 个 epoch，学习率乘以 gamma
    """
    def __init__(self, optimizer, step_size=10, gamma=0.5):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.initial_lr = optimizer.lr

    def step(self, epoch):
        """根据 epoch 更新学习率"""
        factor = self.gamma ** (epoch // self.step_size)
        self.optimizer.lr = self.initial_lr * factor


def compute_accuracy(model, X, y, batch_size=512):
    """计算准确率"""
    N = len(y)
    correct = 0
    for i in range(0, N, batch_size):
        X_batch = X[i:i+batch_size]
        y_batch = y[i:i+batch_size]
        preds = model.predict(X_batch)
        correct += np.sum(preds == y_batch)
    return correct / N


def compute_loss_full(model, X, y, weight_decay=0.0, batch_size=512):
    """计算整个数据集上的平均损失"""
    N = len(y)
    total_loss = 0.0
    num_batches = 0
    for i in range(0, N, batch_size):
        X_batch = X[i:i+batch_size]
        y_batch = y[i:i+batch_size]
        logits = model.forward(X_batch)
        loss = model.compute_loss(logits, y_batch, weight_decay)
        total_loss += loss
        num_batches += 1
    return total_loss / num_batches


def train(model, X_train, y_train, X_val, y_val,
          lr=0.01, momentum=0.9, weight_decay=1e-4, epochs=50, batch_size=128,
          lr_milestones=None, lr_gamma=0.1, save_dir='checkpoints', verbose=True):
    """
    训练循环
    参考 mmdetection 的训练配置:
    - SGD with momentum=0.9, weight_decay=0.0001
    - MultiStepLR with milestones and gamma=0.1
    返回训练历史 (loss, accuracy 曲线)
    """
    os.makedirs(save_dir, exist_ok=True)

    if lr_milestones is None:
        lr_milestones = [30, 40]

    optimizer = SGD(model, lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = WarmupMultiStepLR(optimizer, milestones=lr_milestones, gamma=lr_gamma, warmup_epochs=5)

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'train_acc': [],
    }

    best_val_acc = 0.0
    best_params = None
    N = len(y_train)

    for epoch in range(epochs):
        # 更新学习率
        scheduler.step(epoch)

        # 打乱训练数据
        perm = np.random.permutation(N)
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]

        epoch_loss = 0.0
        num_batches = 0

        # batch 进度条
        n_batches_total = (N + batch_size - 1) // batch_size
        pbar = tqdm(range(0, N, batch_size), total=n_batches_total,
                    desc=f"Epoch {epoch+1}/{epochs}", leave=False,
                    bar_format='{l_bar}{bar:30}{r_bar}')

        for i in pbar:
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            # 前向传播
            logits = model.forward(X_batch)
            loss = model.compute_loss(logits, y_batch, weight_decay)

            # 反向传播
            model.backward()

            # 参数更新
            optimizer.step()

            epoch_loss += loss
            num_batches += 1

            # 更新进度条显示
            pbar.set_postfix(loss=f"{loss:.4f}", lr=f"{optimizer.lr:.5f}")

        # 记录训练损失
        avg_train_loss = epoch_loss / num_batches
        history['train_loss'].append(avg_train_loss)

        # 验证集评估
        val_loss = compute_loss_full(model, X_val, y_val, weight_decay)
        val_acc = compute_accuracy(model, X_val, y_val)
        train_acc = compute_accuracy(model, X_train, y_train)

        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['train_acc'].append(train_acc)

        # 保存最优模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = model.get_params()
            np.savez(os.path.join(save_dir, 'best_model.npz'), **best_params)
            save_mark = " *"
        else:
            save_mark = ""

        if verbose:
            print(f"Epoch {epoch+1}/{epochs} | LR: {optimizer.lr:.6f} | "
                  f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}{save_mark}")

    print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")
    return history, best_params
