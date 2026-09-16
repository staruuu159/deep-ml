import torch
import torch.nn as nn


def train_neuron(features: torch.Tensor,
                 labels: torch.Tensor,
                 initial_weights: torch.Tensor,
                 initial_bias: float,
                 learning_rate: float,
                 epochs: int) -> tuple[list[float], float, list[float]]:
    """
    模拟一个带 sigmoid 激活的单神经元，使用 MSE 损失和 SGD 进行反向传播训练。

    返回:
        (updated_weights, updated_bias, mse_values)，均四舍五入到 4 位小数
    """
    # 确保数据类型为 float32，并且不干扰外部 tensor 的梯度
    X = features.clone().detach().to(torch.float32)
    y = labels.clone().detach().to(torch.float32)

    # 可训练参数
    weight = initial_weights.clone().detach().to(torch.float32).requires_grad_(True)
    bias = torch.tensor(initial_bias, dtype=torch.float32, requires_grad=True)

    optimizer = torch.optim.SGD([weight, bias], lr=learning_rate)
    criterion = nn.MSELoss()

    mse_values = []

    for _ in range(epochs):
        optimizer.zero_grad()

        # 前向传播
        z = X @ weight + bias
        y_pred = torch.sigmoid(z)

        # 计算 MSE 损失（更新前）
        loss = criterion(y_pred, y)
        mse_values.append(round(loss.item(), 4))

        # 反向传播 + 参数更新
        loss.backward()
        optimizer.step()

    # 更新后的参数
    updated_weights = [round(w.item(), 4) for w in weight]
    updated_bias = round(bias.item(), 4)

    return updated_weights, updated_bias, mse_values


# ------------------- 测试 -------------------
if __name__ == "__main__":
    features = torch.tensor([[1.0, 2.0], [2.0, 1.0], [-1.0, -2.0]])
    labels = torch.tensor([1, 0, 0])
    initial_weights = torch.tensor([0.1, -0.2])
    initial_bias = 0.0
    learning_rate = 0.1
    epochs = 2

    result = train_neuron(features, labels, initial_weights,
                          initial_bias, learning_rate, epochs)
    print(result)