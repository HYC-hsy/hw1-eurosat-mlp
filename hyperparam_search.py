"""
超参数查找模块
- 支持网格搜索和随机搜索
- 搜索学习率、隐藏层大小、正则化强度等
- 结果保存为 JSON 文件
"""

import numpy as np
import os
import json
import itertools
from model import MLP
from trainer import train, compute_accuracy


def grid_search(X_train, y_train, X_val, y_val, param_grid, input_dim, num_classes=10,
                epochs=30, batch_size=128, save_dir='search_results', verbose=False):
    """
    网格搜索超参数
    param_grid: dict, 例如:
        {
            'lr': [0.01, 0.05, 0.1],
            'hidden_dim1': [256, 512],
            'hidden_dim2': [128, 256],
            'weight_decay': [1e-4, 1e-3],
            'activation': ['relu', 'tanh'],
        }
    """
    os.makedirs(save_dir, exist_ok=True)

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))

    print(f"Grid Search: {len(combinations)} combinations to try")
    print("=" * 70)

    results = []

    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        print(f"\n[{i+1}/{len(combinations)}] Testing: {params}")

        lr = params.get('lr', 0.01)
        hidden_dim1 = params.get('hidden_dim1', 512)
        hidden_dim2 = params.get('hidden_dim2', 256)
        weight_decay = params.get('weight_decay', 1e-4)
        activation = params.get('activation', 'relu')

        # 创建模型
        model = MLP(input_dim, hidden_dim1, hidden_dim2, num_classes, activation)

        # 训练
        history, best_params = train(
            model, X_train, y_train, X_val, y_val,
            lr=lr, weight_decay=weight_decay, epochs=epochs,
            batch_size=batch_size, save_dir=os.path.join(save_dir, f'run_{i}'),
            verbose=verbose
        )

        best_val_acc = max(history['val_acc'])
        results.append({
            'params': params,
            'best_val_acc': best_val_acc,
            'history': history,
        })

        print(f"  -> Best Val Acc: {best_val_acc:.4f}")

    # 排序结果
    results.sort(key=lambda x: x['best_val_acc'], reverse=True)

    print("\n" + "=" * 70)
    print("Grid Search Results (Top 5):")
    print("=" * 70)
    for i, r in enumerate(results[:5]):
        print(f"  #{i+1} Val Acc: {r['best_val_acc']:.4f} | Params: {r['params']}")

    # 保存搜索结果到 JSON
    results_summary = []
    for r in results:
        results_summary.append({
            'params': {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                       for k, v in r['params'].items()},
            'best_val_acc': float(r['best_val_acc']),
        })
    with open(os.path.join(save_dir, 'grid_search_results.json'), 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nResults saved to {save_dir}/grid_search_results.json")

    return results


def random_search(X_train, y_train, X_val, y_val, param_distributions, input_dim,
                  num_classes=10, n_iter=10, epochs=30, batch_size=128,
                  save_dir='search_results', verbose=False):
    """
    随机搜索超参数
    param_distributions: dict, 例如:
        {
            'lr': [0.001, 0.005, 0.01, 0.05, 0.1],
            'hidden_dim1': [256, 512, 1024],
            'hidden_dim2': [128, 256, 512],
            'weight_decay': [1e-5, 1e-4, 1e-3, 1e-2],
            'activation': ['relu', 'tanh'],
        }
    """
    os.makedirs(save_dir, exist_ok=True)

    print(f"Random Search: {n_iter} iterations")
    print("=" * 70)

    results = []

    for i in range(n_iter):
        # 随机采样参数
        params = {}
        for key, values in param_distributions.items():
            params[key] = values[np.random.randint(len(values))]

        print(f"\n[{i+1}/{n_iter}] Testing: {params}")

        lr = params.get('lr', 0.01)
        hidden_dim1 = params.get('hidden_dim1', 512)
        hidden_dim2 = params.get('hidden_dim2', 256)
        weight_decay = params.get('weight_decay', 1e-4)
        activation = params.get('activation', 'relu')

        # 创建模型
        model = MLP(input_dim, hidden_dim1, hidden_dim2, num_classes, activation)

        # 训练
        history, best_params = train(
            model, X_train, y_train, X_val, y_val,
            lr=lr, weight_decay=weight_decay, epochs=epochs,
            batch_size=batch_size, save_dir=os.path.join(save_dir, f'run_{i}'),
            verbose=verbose
        )

        best_val_acc = max(history['val_acc'])
        results.append({
            'params': params,
            'best_val_acc': best_val_acc,
            'history': history,
        })

        print(f"  -> Best Val Acc: {best_val_acc:.4f}")

    # 排序结果
    results.sort(key=lambda x: x['best_val_acc'], reverse=True)

    print("\n" + "=" * 70)
    print("Random Search Results (Top 5):")
    print("=" * 70)
    for i, r in enumerate(results[:5]):
        print(f"  #{i+1} Val Acc: {r['best_val_acc']:.4f} | Params: {r['params']}")

    # 保存搜索结果到 JSON
    results_summary = []
    for r in results:
        results_summary.append({
            'params': {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                       for k, v in r['params'].items()},
            'best_val_acc': float(r['best_val_acc']),
        })
    with open(os.path.join(save_dir, 'random_search_results.json'), 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nResults saved to {save_dir}/random_search_results.json")

    return results
