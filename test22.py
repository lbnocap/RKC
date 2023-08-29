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
#plt.show()
# 创建两个数组

# 创建三个列向量
vector1 = np.array([[1], [2], [3]])
vector2 = np.array([[4], [5], [6]])
vector3 = np.array([[7], [8], [9]])

# 使用 np.vstack() 函数将列向量按行叠加
stacked_matrix = np.vstack((vector1, vector2, vector3))

print(2*np.cos(2*np.pi*1)/0.001)
print(0.035*0.7*1.3/(0.7*1.3+0.1)-2*np.sin(2*np.pi*0.01)*((-2*np.cos(2*np.pi*0.01))**2))

r=0.8*((1/0.4628)**(1/3))
print(r)