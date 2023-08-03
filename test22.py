import numpy as np
import matplotlib.pyplot as plt

# 定义复数函数
def complex_function(z):
    return z**2 + 2*z + 1

# 生成复平面上的网格点（增加网格点数量）
x = np.linspace(-2, 2, 200)
y = np.linspace(-2, 2, 200)
X, Y = np.meshgrid(x, y)
Z = X + 1j*Y

# 计算复数函数的值
result = complex_function(Z)

# 绘制复数函数等于1的等高线图
plt.figure(figsize=(6, 6))
plt.contour(X, Y, np.abs(result), levels=[1], colors='red')  # 使用绝对值表示等高线高度，设置等高线值为1，颜色为红色
plt.xlabel('Re(z)')
plt.ylabel('Im(z)')
plt.title('Contour Plot of Complex Function (|f(z)| = 1)')

plt.show()