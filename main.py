"""
主程序 - 完整的训练与评估流程
用法:
    python main.py --mode train        # 训练模型
    python main.py --mode search       # 超参数搜索
    python main.py --mode test         # 测试评估
    python main.py --mode all          # 完整流程
"""

import argparse
import numpy as np
import os
import sys

from data_loader import get_data, CLASS_NAMES
from model import MLP
from trainer import train, compute_accuracy
from test_eval import run_test, load_model, confusion_matrix, print_confusion_matrix, \
    per_class_accuracy, error_analysis
from hyperparam_search import grid_search, random_search
from visualize import (plot_training_curves, visualize_weights,
                       visualize_errors, visualize_confusion_matrix)


# ============ 配置 ============
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'EuroSAT_RGB')
SAVE_DIR = 'checkpoints'
FIGURE_DIR = 'figures'
SEARCH_DIR = 'search_results'

# 默认超参数 (参考 mmdetection 的配置风格)
DEFAULT_CONFIG = {
    'lr': 0.01,
    'momentum': 0.9,
    'hidden_dim1': 512,
    'hidden_dim2': 256,
    'weight_decay': 1e-4,
    'activation': 'relu',
    'epochs': 50,
    'batch_size': 128,
    'lr_milestones': [30, 40],
    'lr_gamma': 0.1,
}


def run_train(X_train, y_train, X_val, y_val, config=None):
    """训练模型"""
    if config is None:
        config = DEFAULT_CONFIG

    input_dim = X_train.shape[1]
    num_classes = 10

    print("\n" + "=" * 60)
    print("Training Configuration:")
    print("=" * 60)
    for k, v in config.items():
        print(f"  {k}: {v}")
    print(f"  input_dim: {input_dim}")
    print("=" * 60 + "\n")

    model = MLP(input_dim, config['hidden_dim1'], config['hidden_dim2'],
                num_classes, config['activation'])

    history, best_params = train(
        model, X_train, y_train, X_val, y_val,
        lr=config['lr'],
        momentum=config['momentum'],
        weight_decay=config['weight_decay'],
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        lr_milestones=config['lr_milestones'],
        lr_gamma=config['lr_gamma'],
        save_dir=SAVE_DIR,
        verbose=True
    )

    # 绘制训练曲线
    plot_training_curves(history, save_dir=FIGURE_DIR)

    # 加载最优权重并可视化
    model.set_params(best_params)
    visualize_weights(model, save_dir=FIGURE_DIR)

    return model, history


def run_search(X_train, y_train, X_val, y_val):
    """超参数搜索"""
    input_dim = X_train.shape[1]

    param_grid = {
        'lr': [0.01, 0.05, 0.1],
        'hidden_dim1': [256, 512],
        'hidden_dim2': [128, 256],
        'weight_decay': [1e-4, 1e-3],
        'activation': ['relu', 'tanh'],
    }

    print("\n" + "=" * 60)
    print("Hyperparameter Grid Search")
    print("=" * 60)

    results = grid_search(
        X_train, y_train, X_val, y_val,
        param_grid=param_grid,
        input_dim=input_dim,
        epochs=30,
        batch_size=128,
        save_dir=SEARCH_DIR,
        verbose=False
    )

    # 保存搜索结果
    print("\n\nBest configuration found:")
    best = results[0]
    print(f"  Params: {best['params']}")
    print(f"  Val Acc: {best['best_val_acc']:.4f}")

    return results


def run_test_eval(X_test, y_test, config=None):
    """测试评估"""
    if config is None:
        config = DEFAULT_CONFIG

    input_dim = X_test.shape[1]
    weight_path = os.path.join(SAVE_DIR, 'best_model.npz')

    if not os.path.exists(weight_path):
        print(f"Error: Model weights not found at {weight_path}")
        print("Please run training first: python main.py --mode train")
        return

    accuracy, cm, errors = run_test(
        weight_path, X_test, y_test,
        input_dim=input_dim,
        hidden_dim1=config['hidden_dim1'],
        hidden_dim2=config['hidden_dim2'],
        num_classes=10,
        activation=config['activation']
    )

    # 可视化混淆矩阵
    visualize_confusion_matrix(cm, CLASS_NAMES, save_dir=FIGURE_DIR)

    # 可视化错误样本
    visualize_errors(errors, X_test, save_dir=FIGURE_DIR)

    return accuracy, cm, errors


def main():
    parser = argparse.ArgumentParser(description='EuroSAT MLP Classifier')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['train', 'search', 'test', 'all'],
                        help='Running mode: train, search, test, or all')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR,
                        help='Path to EuroSAT_RGB dataset')
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--hidden_dim1', type=int, default=None)
    parser.add_argument('--hidden_dim2', type=int, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--activation', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)

    args = parser.parse_args()

    # 更新配置
    config = DEFAULT_CONFIG.copy()
    if args.lr is not None:
        config['lr'] = args.lr
    if args.hidden_dim1 is not None:
        config['hidden_dim1'] = args.hidden_dim1
    if args.hidden_dim2 is not None:
        config['hidden_dim2'] = args.hidden_dim2
    if args.weight_decay is not None:
        config['weight_decay'] = args.weight_decay
    if args.activation is not None:
        config['activation'] = args.activation
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size

    # 加载数据
    X_train, y_train, X_val, y_val, X_test, y_test = get_data(args.data_dir)

    if args.mode == 'train' or args.mode == 'all':
        model, history = run_train(X_train, y_train, X_val, y_val, config)

    if args.mode == 'search':
        results = run_search(X_train, y_train, X_val, y_val)

    if args.mode == 'test' or args.mode == 'all':
        run_test_eval(X_test, y_test, config)

    print("\nDone!")


if __name__ == '__main__':
    main()
