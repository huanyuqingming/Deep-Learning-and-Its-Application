import jittor as jt
from jittor import init
from jittor import nn
import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 生成真实数据
x1 = 10 * jt.array(np.random.rand(1000, 1))
x2 = 10 * jt.array(np.random.rand(1000, 1))
x = jt.concat((x1, x2), dim=1)
y = 5 * x1 + 26 * x2 + 2004 + jt.array(np.random.normal(0, 0.01, size=(1000, 1)))

# 初始化模型
model = nn.Linear(2, 1)
init.gauss_(model.weight, 0, 100)
init.gauss_(model.bias, 0, 100)
loss_fn = nn.MSELoss()
optimizer = jt.optim.SGD(model.parameters(), lr=1e-2)

# 训练模型
losses = []
i = 0
pre_loss = float('inf')
convergence = 0

while convergence < 100:
    i = i + 1
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    optimizer.step(loss)
    losses.append(loss.data.sum())
    if abs(pre_loss - loss) < 1e-5:
        convergence = convergence + 1
    else:
        convergence = 0
    pre_loss = loss.data.sum()
    if (i+1) % 100 == 0:
        print(f'Epoch {i+1}, Loss: {loss.data.sum()}')

# 可视化损失函数
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# 可视化拟合结果
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 可视化真实数据
x_np = x.data
y_np = y.data
ax.scatter(x_np[:, 0], x_np[:, 1], y_np)

# 可视化拟合平面
a = model.weight[0, 0].data
b = model.weight[0, 1].data
c = model.bias[0].data
x_range = np.linspace(x_np[:, 0].min(), x_np[:, 0].max(), 100)
y_range = np.linspace(x_np[:, 1].min(), x_np[:, 1].max(), 100)
x_surf, y_surf = np.meshgrid(x_range, y_range)
z_surf = a * x_surf + b * y_surf + c
ax.plot_surface(x_surf, y_surf, z_surf, color='b', alpha=0.5)

ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('y')
ax.legend(['True Data', 'Fitted Plane'])

plt.show()
print(f'w1: {a}, w2: {b}, b: {c}')
