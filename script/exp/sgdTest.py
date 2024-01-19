
import numpy as np

# 定义激活函数
def activation(x):
    return 1 / (1 + np.exp(-x))

# 定义激活函数的导数
def activation_derivative(x):
    return x * (1 - x)

# 定义损失函数
def loss(y, y_pred):
    return -(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

# 定义损失函数的导数
def loss_derivative(y, y_pred):
    return -(y / y_pred) + (1 - y) / (1 - y_pred)

# 定义神经网络的结构
input_size = 2
hidden_size = 3
output_size = 1

# 初始化权重和偏置
weights1 = np.random.randn(input_size, hidden_size)
bias1 = np.zeros(hidden_size)
weights2 = np.random.randn(hidden_size, output_size)
bias2 = np.zeros(output_size)

# 设置训练数据
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])

# 设置学习率
learning_rate = 0.1

# 迭代次数
num_epochs = 1000

# 开始训练
for epoch in range(num_epochs):
    # 遍历所有训练数据
    for i in range(len(inputs)):
        # 前向传播
        hidden_layer_input = np.dot(inputs[i], weights1) + bias1
        hidden_layer_output = activation(hidden_layer_input)
        output_layer_input = np.dot(hidden_layer_output, weights2) + bias2
        output = activation(output_layer_input)

        # 计算损失
        loss_value = loss(targets[i], output)

        # 反向传播
        output_error = loss_derivative(targets[i], output)
        hidden_error = np.dot(output_error, weights2.T) * activation_derivative(hidden_layer_input)

        # 更新权重和偏置
        weights2 -= learning_rate * np.outer(hidden_layer_output, output_error)
        bias2 -= learning_rate * output_error
        weights1 -= learning_rate * np.outer(inputs[i], hidden_error)
        bias1 -= learning_rate * hidden_error

        # 每隔100次迭代打印一次损失
        if i % 100 == 0:
            print("Epoch {0}, Loss: {1}".format(epoch, loss_value))
