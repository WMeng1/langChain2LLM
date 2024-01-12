import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 加载MNIST数据集
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

# 对数据进行预处理
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 定义模型参数
input_size = X_train.shape[1]
output_size = 10
hidden_size = 64
learning_rate = 0.01

# 初始化模型参数
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros(hidden_size)
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros(output_size)


# 定义激活函数和损失函数
def relu(x):
    return np.maximum(0, x)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy(y, y_pred):
    return -np.sum(y * np.log(y_pred + 1e-8))


# 定义前向传播
def forward(x, W1, b1, W2, b2):
    z1 = np.dot(x, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    y_pred = softmax(z2)
    return y_pred


# 定义反向传播
def backward(x, y, y_pred, W1, b1, W2, b2):
    delta2 = y_pred - y
    dW2 = np.dot(a1.T, delta2)
    db2 = np.sum(delta2, axis=0)
    delta1 = np.dot(delta2, W2.T) * (a1 > 0)
    dW1 = np.dot(x.T, delta1)
    db1 = np.sum(delta1, axis=0)
    return dW1, db1, dW2, db2


# 定义训练函数
def train(X_train, y_train, epochs, batch_size):
    for epoch in range(epochs):
        # 随机打乱数据
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]

        for i in range(0, X_train.shape[0], batch_size):
            x_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            # 前向传播
            y_pred = forward(x_batch, W1, b1, W2, b2)

            # 计算损失
            loss = cross_entropy(y_batch, y_pred)

            # 反向传播
            dW1, db1, dW2, db2 = backward(x_batch, y_batch, y_pred, W1, b1, W2, b2)

            # 更新参数
            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1
            W2 -= learning_rate * dW2