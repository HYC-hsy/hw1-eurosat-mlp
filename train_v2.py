"""
改进版训练脚本 - 针对过拟合问题
改进点:
1. 加入 Dropout
2. 加大 weight_decay
3. 加入简单数据增强 (随机水平翻转)
4. 更大的隐藏层 + 更强正则化
"""

import sys
import numpy as np
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import get_data, CLASS_NAMES, load_eurosat, preprocess, train_val_test_split
from model import MLP, Linear, get_activation, SoftmaxCrossEntropy
from trainer import SGD, WarmupMultiStepLR, compute_accuracy, compute_loss_full
from test_eval import run_test
from visualize import (plot_training_curves, visualize_weights,
                       visualize_errors, visualize_confusion_matrix)
from tqdm import tqdm


# ============ Dropout 层 ============

class Dropout:
    """Dropout 正则化层"""
    def __init__(self, p=0.5):
        self.p = p  # drop 概率
        self.mask = None
        self.training = True

    def forward(self, x):
        if self.training:
            self.mask = (np.random.rand(*x.shape) > self.p).astype(np.float32) / (1.0 - self.p)
            return x * self.mask
        return x

    def backward(self, grad_output):
        return grad_output * self.mask


# ============ 带 Dropout 的 MLP ============

class MLPv2:
    """
    改进版三层神经网络 (带 Dropout):
    Input -> Linear1 -> Act1 -> Dropout1 -> Linear2 -> Act2 -> Dropout2 -> Linear3 -> Softmax
    """
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, num_classes,
                 activation='relu', dropout_rate=0.3):
        init_mode = 'he' if activation.lower() == 'relu' else 'xavier'

        self.fc1 = Linear(input_dim, hidden_dim1, init_mode=init_mode)
        self.act1 = get_activation(activation)
        self.drop1 = Dropout(p=dropout_rate)
        self.fc2 = Linear(hidden_dim1, hidden_dim2, init_mode=init_mode)
        self.act2 = get_activation(activation)
        self.drop2 = Dropout(p=dropout_rate)
        self.fc3 = Linear(hidden_dim2, num_classes, init_mode=init_mode)
        self.loss_fn = SoftmaxCrossEntropy()

        self.layers = [self.fc1, self.fc2, self.fc3]

    def train_mode(self):
        self.drop1.training = True
        self.drop2.training = True

    def eval_mode(self):
        self.drop1.training = False
        self.drop2.training = False

    def forward(self, x):
        x = self.fc1.forward(x)
        x = self.act1.forward(x)
        x = self.drop1.forward(x)
        x = self.fc2.forward(x)
        x = self.act2.forward(x)
        x = self.drop2.forward(x)
        x = self.fc3.forward(x)
        return x

    def compute_loss(self, logits, labels, weight_decay=0.0):
        ce_loss = self.loss_fn.forward(logits, labels)
        l2_loss = 0.0
        if weight_decay > 0:
            for layer in self.layers:
                l2_loss += 0.5 * weight_decay * np.sum(layer.W ** 2)
        return ce_loss + l2_loss

    def backward(self):
        grad = self.loss_fn.backward()
        grad = self.fc3.backward(grad)
        grad = self.drop2.backward(grad)
        grad = self.act2.backward(grad)
        grad = self.fc2.backward(grad)
        grad = self.drop1.backward(grad)
        grad = self.act1.backward(grad)
        grad = self.fc1.backward(grad)

    def predict(self, x):
        self.eval_mode()
        logits = self.forward(x)
        self.train_mode()
        return np.argmax(logits, axis=1)

    def get_params(self):
        params = {}
        for i, layer in enumerate(self.layers):
            params[f'W{i+1}'] = layer.W.copy()
            params[f'b{i+1}'] = layer.b.copy()
        return params

    def set_params(self, params):
        for i, layer in enumerate(self.layers):
            layer.W = params[f'W{i+1}'].copy()
            layer.b = params[f'b{i+1}'].copy()


# ============ 数据增强 ============

def augment_batch(X_batch, img_size=64):
    """简单数据增强: 随机水平翻转"""
    N = X_batch.shape[0]
    X_aug = X_batch.copy()
    for i in range(N):
        if np.random.rand() > 0.5:
            # 水平翻转: reshape -> flip -> flatten
            img = X_aug[i].reshape(img_size, img_size, 3)
            img = img[:, ::-1, :]
            X_aug[i] = img.reshape(-1)
    return X_aug


# ============ 训练 ============

def train_v2(model, X_train, y_train, X_val, y_val,
             lr=0.01, momentum=0.9, weight_decay=1e-3, epochs=80, batch_size=128,
             lr_milestones=None, lr_gamma=0.1, save_dir='checkpoints_v2',
             augment=True, verbose=True):
    """改进版训练循环"""
    os.makedirs(save_dir, exist_ok=True)

    if lr_milestones is None:
        lr_milestones = [40, 60]

    optimizer = SGD(model, lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = WarmupMultiStepLR(optimizer, milestones=lr_milestones, gamma=lr_gamma, warmup_epochs=5)

    history = {
        'train_loss': [], 'val_loss': [], 'val_acc': [], 'train_acc': [],
    }

    best_val_acc = 0.0
    best_params = None
    N = len(y_train)

    for epoch in range(epochs):
        scheduler.step(epoch)
        model.train_mode()

        perm = np.random.permutation(N)
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]

        epoch_loss = 0.0
        num_batches = 0
        n_batches_total = (N + batch_size - 1) // batch_size

        pbar = tqdm(range(0, N, batch_size), total=n_batches_total,
                    desc=f"Epoch {epoch+1}/{epochs}", leave=False,
                    bar_format='{l_bar}{bar:30}{r_bar}')

        for i in pbar:
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            # 数据增强
            if augment:
                X_batch = augment_batch(X_batch)

            logits = model.forward(X_batch)
            loss = model.compute_loss(logits, y_batch, weight_decay)
            model.backward()
            optimizer.step()

            epoch_loss += loss
            num_batches += 1
            pbar.set_postfix(loss=f"{loss:.4f}", lr=f"{optimizer.lr:.5f}")

        avg_train_loss = epoch_loss / num_batches
        history['train_loss'].append(avg_train_loss)

        # 评估 (eval mode)
        model.eval_mode()
        val_loss = compute_loss_full(model, X_val, y_val, weight_decay)
        val_acc = compute_accuracy(model, X_val, y_val)
        train_acc = compute_accuracy(model, X_train, y_train)
        model.train_mode()

        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['train_acc'].append(train_acc)

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


# ============ 主程序 ============

if __name__ == '__main__':
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'EuroSAT_RGB')
    SAVE_DIR = 'checkpoints_v2'
    FIGURE_DIR = 'figures_v2'

    # 加载数据
    X_train, y_train, X_val, y_val, X_test, y_test = get_data(DATA_DIR)

    # 配置
    config = {
        'lr': 0.01,
        'momentum': 0.9,
        'hidden_dim1': 1024,
        'hidden_dim2': 512,
        'weight_decay': 5e-3,
        'activation': 'relu',
        'dropout_rate': 0.4,
        'epochs': 80,
        'batch_size': 128,
        'lr_milestones': [40, 60],
        'lr_gamma': 0.1,
    }

    print("\n" + "=" * 60)
    print("Training Configuration (v2 - with Dropout & Augmentation):")
    print("=" * 60)
    for k, v in config.items():
        print(f"  {k}: {v}")
    print("=" * 60 + "\n")

    input_dim = X_train.shape[1]
    model = MLPv2(input_dim, config['hidden_dim1'], config['hidden_dim2'],
                  10, config['activation'], config['dropout_rate'])

    history, best_params = train_v2(
        model, X_train, y_train, X_val, y_val,
        lr=config['lr'], momentum=config['momentum'],
        weight_decay=config['weight_decay'], epochs=config['epochs'],
        batch_size=config['batch_size'], lr_milestones=config['lr_milestones'],
        lr_gamma=config['lr_gamma'], save_dir=SAVE_DIR,
        augment=True, verbose=True
    )

    # 可视化
    plot_training_curves(history, save_dir=FIGURE_DIR)
    model.set_params(best_params)
    visualize_weights(model, save_dir=FIGURE_DIR)

    # 测试
    model.set_params(best_params)
    model.eval_mode()
    from test_eval import evaluate, confusion_matrix, print_confusion_matrix, per_class_accuracy, error_analysis
    accuracy, preds = evaluate(model, X_test, y_test)
    print(f"\nTest Accuracy: {accuracy:.4f} ({int(accuracy * len(y_test))}/{len(y_test)})")

    cm = confusion_matrix(y_test, preds, 10)
    print_confusion_matrix(cm, CLASS_NAMES)
    per_class_accuracy(cm, CLASS_NAMES)
    errors = error_analysis(model, X_test, y_test)

    visualize_confusion_matrix(cm, CLASS_NAMES, save_dir=FIGURE_DIR)
    visualize_errors(errors, X_test, save_dir=FIGURE_DIR)

    print("\nDone! Results saved to:", SAVE_DIR, "and", FIGURE_DIR)
