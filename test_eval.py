"""
测试评估模块
- 加载最优模型权重
- 计算测试集准确率
- 打印混淆矩阵
- 错例分析
"""

import numpy as np
import os
from model import MLP
from data_loader import CLASS_NAMES


def load_model(model, weight_path):
    """加载训练好的模型权重"""
    data = np.load(weight_path)
    params = {key: data[key] for key in data.files}
    model.set_params(params)
    print(f"Model loaded from {weight_path}")
    return model


def evaluate(model, X_test, y_test):
    """评估模型，返回准确率和预测结果"""
    preds = model.predict(X_test)
    accuracy = np.mean(preds == y_test)
    return accuracy, preds


def confusion_matrix(y_true, y_pred, num_classes=10):
    """计算混淆矩阵"""
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def print_confusion_matrix(cm, class_names):
    """打印混淆矩阵"""
    print("\n" + "=" * 80)
    print("Confusion Matrix")
    print("=" * 80)

    # 表头
    label = 'True \\ Pred'
    header = f"{label:<20}"
    for name in class_names:
        header += f"{name[:8]:>9}"
    print(header)
    print("-" * 80)

    # 每行
    for i, name in enumerate(class_names):
        row = f"{name:<20}"
        for j in range(len(class_names)):
            row += f"{cm[i, j]:>9}"
        print(row)
    print("=" * 80)


def per_class_accuracy(cm, class_names):
    """计算每个类别的准确率"""
    print("\nPer-class Accuracy:")
    print("-" * 40)
    for i, name in enumerate(class_names):
        total = cm[i].sum()
        correct = cm[i, i]
        acc = correct / total if total > 0 else 0
        print(f"  {name:<25} {acc:.4f} ({correct}/{total})")


def error_analysis(model, X_test, y_test, num_errors=10):
    """错例分析：找出分类错误的样本"""
    preds = model.predict(X_test)
    error_indices = np.where(preds != y_test)[0]

    print(f"\nTotal errors: {len(error_indices)} / {len(y_test)}")
    print(f"\nSample misclassified cases:")
    print("-" * 60)

    # 随机选取一些错误样本
    np.random.seed(42)
    if len(error_indices) > num_errors:
        selected = np.random.choice(error_indices, num_errors, replace=False)
    else:
        selected = error_indices

    errors = []
    for idx in selected:
        true_label = CLASS_NAMES[y_test[idx]]
        pred_label = CLASS_NAMES[preds[idx]]
        print(f"  Sample {idx}: True={true_label}, Predicted={pred_label}")
        errors.append({
            'index': idx,
            'true_label': true_label,
            'pred_label': pred_label,
            'true_class': y_test[idx],
            'pred_class': preds[idx],
        })

    return errors


def run_test(weight_path, X_test, y_test, input_dim, hidden_dim1, hidden_dim2,
             num_classes=10, activation='relu'):
    """完整的测试流程"""
    # 创建模型并加载权重
    model = MLP(input_dim, hidden_dim1, hidden_dim2, num_classes, activation)
    model = load_model(model, weight_path)

    # 评估
    accuracy, preds = evaluate(model, X_test, y_test)
    print(f"\nTest Accuracy: {accuracy:.4f} ({int(accuracy * len(y_test))}/{len(y_test)})")

    # 混淆矩阵
    cm = confusion_matrix(y_test, preds, num_classes)
    print_confusion_matrix(cm, CLASS_NAMES)
    per_class_accuracy(cm, CLASS_NAMES)

    # 错例分析
    errors = error_analysis(model, X_test, y_test)

    return accuracy, cm, errors
