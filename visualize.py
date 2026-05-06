"""
可视化模块
- Loss 曲线 (训练集 + 验证集)
- Accuracy 曲线
- 权重可视化
- 错例可视化
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def plot_training_curves(history, save_dir='figures'):
    """绘制训练过程中的 Loss 和 Accuracy 曲线"""
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    # Loss 曲线
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy 曲线
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_dir}/training_curves.png")


def visualize_weights(model, img_size=64, save_dir='figures', num_neurons=16):
    """
    可视化第一层隐藏层权重
    将权重恢复为图像尺寸 (H, W, 3) 进行展示
    """
    os.makedirs(save_dir, exist_ok=True)

    W1 = model.fc1.W  # shape: (input_dim, hidden_dim1) = (64*64*3, hidden_dim1)
    # 每一列是一个神经元的权重

    n_show = min(num_neurons, W1.shape[1])
    rows = int(np.ceil(np.sqrt(n_show)))
    cols = int(np.ceil(n_show / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten() if n_show > 1 else [axes]

    for i in range(n_show):
        w = W1[:, i].reshape(img_size, img_size, 3)
        # 归一化到 [0, 1] 用于显示
        w_min, w_max = w.min(), w.max()
        if w_max - w_min > 0:
            w_normalized = (w - w_min) / (w_max - w_min)
        else:
            w_normalized = np.zeros_like(w)
        axes[i].imshow(w_normalized)
        axes[i].set_title(f'Neuron {i}', fontsize=8)
        axes[i].axis('off')

    # 隐藏多余的子图
    for i in range(n_show, len(axes)):
        axes[i].axis('off')

    plt.suptitle('First Hidden Layer Weight Visualization', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'weight_visualization.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Weight visualization saved to {save_dir}/weight_visualization.png")


def visualize_errors(errors, X_test_raw, save_dir='figures'):
    """
    可视化分类错误的样本
    X_test_raw: 展平后的测试数据 (N, D), 值在 [0, 1]
    """
    os.makedirs(save_dir, exist_ok=True)

    n_errors = min(len(errors), 10)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for i in range(n_errors):
        err = errors[i]
        idx = err['index']
        img = X_test_raw[idx].reshape(64, 64, 3)
        # 确保值在 [0, 1]
        img = np.clip(img, 0, 1)
        axes[i].imshow(img)
        axes[i].set_title(f"True: {err['true_label']}\nPred: {err['pred_label']}", fontsize=8)
        axes[i].axis('off')

    for i in range(n_errors, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Misclassified Samples', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'error_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Error analysis saved to {save_dir}/error_analysis.png")


def visualize_confusion_matrix(cm, class_names, save_dir='figures'):
    """
    可视化混淆矩阵
    参考 mmdetection/tools/analysis_tools/confusion_matrix.py 的 plot_confusion_matrix
    使用归一化百分比显示
    """
    os.makedirs(save_dir, exist_ok=True)

    # 归一化为百分比
    per_label_sums = cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = cm.astype(np.float32) / (per_label_sums + 1e-12) * 100

    num_classes = len(class_names)
    fig, ax = plt.subplots(figsize=(0.5 * num_classes + 4, 0.5 * num_classes * 0.8 + 3), dpi=150)
    cmap = plt.get_cmap('plasma')
    im = ax.imshow(cm_normalized, cmap=cmap)
    plt.colorbar(mappable=im, ax=ax)

    title_font = {'weight': 'bold', 'size': 12}
    ax.set_title('Normalized Confusion Matrix', fontdict=title_font)
    label_font = {'size': 10}
    plt.ylabel('Ground Truth Label', fontdict=label_font)
    plt.xlabel('Prediction Label', fontdict=label_font)

    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # 在格子中显示百分比数值
    for i in range(num_classes):
        for j in range(num_classes):
            val = cm_normalized[i, j]
            ax.text(j, i, f'{int(val)}%',
                    ha='center', va='center',
                    color='w' if val > 50 else 'black',
                    size=7)

    ax.set_ylim(num_classes - 0.5, -0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_dir}/confusion_matrix.png")
