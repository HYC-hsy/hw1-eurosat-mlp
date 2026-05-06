"""
模型定义模块
- 三层 MLP (输入层 -> 隐藏层1 -> 隐藏层2 -> 输出层)
- 支持 ReLU / Sigmoid / Tanh 激活函数
- 手动实现前向传播和反向传播
"""

import numpy as np


# ============ 激活函数 ============

class ReLU:
    """ReLU 激活函数"""
    def forward(self, x):
        self.mask = (x > 0).astype(np.float32)
        return x * self.mask

    def backward(self, grad_output):
        return grad_output * self.mask


class Sigmoid:
    """Sigmoid 激活函数"""
    def forward(self, x):
        # 数值稳定的 sigmoid
        self.out = np.where(x >= 0,
                            1.0 / (1.0 + np.exp(-x)),
                            np.exp(x) / (1.0 + np.exp(x)))
        return self.out

    def backward(self, grad_output):
        return grad_output * self.out * (1.0 - self.out)


class Tanh:
    """Tanh 激活函数"""
    def forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def backward(self, grad_output):
        return grad_output * (1.0 - self.out ** 2)


def get_activation(name):
    """根据名称获取激活函数实例"""
    activations = {
        'relu': ReLU,
        'sigmoid': Sigmoid,
        'tanh': Tanh,
    }
    if name.lower() not in activations:
        raise ValueError(f"Unsupported activation: {name}. Choose from {list(activations.keys())}")
    return activations[name.lower()]()


# ============ 线性层 ============

class Linear:
    """全连接线性层"""
    def __init__(self, in_features, out_features, init_mode='he'):
        """
        init_mode: 'he' for ReLU, 'xavier' for Sigmoid/Tanh
        参考 mmdetection 中对不同激活函数使用不同初始化策略
        """
        if init_mode == 'he':
            self.W = np.random.randn(in_features, out_features).astype(np.float32) * np.sqrt(2.0 / in_features)
        else:  # xavier
            self.W = np.random.randn(in_features, out_features).astype(np.float32) * np.sqrt(1.0 / in_features)
        self.b = np.zeros((1, out_features), dtype=np.float32)
        # 梯度
        self.dW = None
        self.db = None
        # 缓存输入
        self.input = None

    def forward(self, x):
        self.input = x
        return x @ self.W + self.b

    def backward(self, grad_output):
        # grad_output: (N, out_features)
        self.dW = self.input.T @ grad_output  # (in, out)
        self.db = np.sum(grad_output, axis=0, keepdims=True)  # (1, out)
        grad_input = grad_output @ self.W.T  # (N, in)
        return grad_input


# ============ Softmax + Cross-Entropy Loss ============

class SoftmaxCrossEntropy:
    """Softmax + 交叉熵损失 (合并计算以保证数值稳定)"""
    def forward(self, logits, labels):
        """
        logits: (N, C)
        labels: (N,) 整数标签
        """
        self.N = logits.shape[0]
        # 数值稳定的 softmax
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # 交叉熵损失
        self.labels = labels
        log_probs = -np.log(self.probs[np.arange(self.N), labels] + 1e-12)
        loss = np.mean(log_probs)
        return loss

    def backward(self):
        """返回 logits 的梯度"""
        grad = self.probs.copy()
        grad[np.arange(self.N), self.labels] -= 1.0
        grad /= self.N
        return grad


# ============ 三层 MLP 模型 ============

class MLP:
    """
    三层神经网络:
    Input -> Linear1 -> Activation1 -> Linear2 -> Activation2 -> Linear3 -> Softmax
    """
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, num_classes, activation='relu'):
        # 根据激活函数选择初始化方式: ReLU 用 He, Sigmoid/Tanh 用 Xavier
        init_mode = 'he' if activation.lower() == 'relu' else 'xavier'

        self.fc1 = Linear(input_dim, hidden_dim1, init_mode=init_mode)
        self.act1 = get_activation(activation)
        self.fc2 = Linear(hidden_dim1, hidden_dim2, init_mode=init_mode)
        self.act2 = get_activation(activation)
        self.fc3 = Linear(hidden_dim2, num_classes, init_mode=init_mode)
        self.loss_fn = SoftmaxCrossEntropy()

        self.layers = [self.fc1, self.fc2, self.fc3]  # 含参数的层

    def forward(self, x):
        """前向传播"""
        x = self.fc1.forward(x)
        x = self.act1.forward(x)
        x = self.fc2.forward(x)
        x = self.act2.forward(x)
        x = self.fc3.forward(x)
        return x

    def compute_loss(self, logits, labels, weight_decay=0.0):
        """计算交叉熵损失 (L2 正则化在 optimizer 中实现)"""
        ce_loss = self.loss_fn.forward(logits, labels)
        # L2 正则项仅用于 loss 记录/显示，实际梯度在 optimizer.step() 中处理
        l2_loss = 0.0
        if weight_decay > 0:
            for layer in self.layers:
                l2_loss += 0.5 * weight_decay * np.sum(layer.W ** 2)
        return ce_loss + l2_loss

    def backward(self):
        """反向传播 (纯交叉熵梯度，L2 正则化梯度在 optimizer 中处理)"""
        grad = self.loss_fn.backward()
        grad = self.fc3.backward(grad)
        grad = self.act2.backward(grad)
        grad = self.fc2.backward(grad)
        grad = self.act1.backward(grad)
        grad = self.fc1.backward(grad)

    def predict(self, x):
        """预测类别"""
        logits = self.forward(x)
        return np.argmax(logits, axis=1)

    def get_params(self):
        """获取所有参数 (用于保存)"""
        params = {}
        for i, layer in enumerate(self.layers):
            params[f'W{i+1}'] = layer.W.copy()
            params[f'b{i+1}'] = layer.b.copy()
        return params

    def set_params(self, params):
        """设置参数 (用于加载)"""
        for i, layer in enumerate(self.layers):
            layer.W = params[f'W{i+1}'].copy()
            layer.b = params[f'b{i+1}'].copy()
