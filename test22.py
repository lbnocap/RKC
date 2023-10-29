import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd
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
pi=np.pi
bt=0.03
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

#print(2*np.cos(2*np.pi*1)/0.001)
#print(0.035*0.7*1.3/(0.7*1.3+0.1)-2*np.sin(2*np.pi*0.01)*((-2*np.cos(2*np.pi*0.01))**2))

r=0.8*((1/0.4628)**(1/3))
solu=np.zeros((1002,1))
solu1=np.zeros((1203,1))

x=np.linspace(0,1,500+1,dtype=float)
x1=np.linspace(0,1,400+1,dtype=float)
for i in range(0,500+1):
     solu[500+1+i]=np.exp(-((pi)**2) *bt*0.01)*np.cos(pi*(x[i]-0.01))
     solu[i]=np.exp(-((pi)**2) *bt*0.01)*np.sin(pi*(x[i]-0.01))

for i in range(0,401):
        solu1[i]=np.exp(-1)*np.sin(pi*x1[i])
        solu1[401+i]=0
        solu1[802+i]=1-np.exp(-1)*np.sin(pi*x1[i])
print(solu1)


#[2.76551848] [1.40166833] [0.28742099] [0.81988987] 2.0247827947216464 -0.9615580646630746 0.10579341717210078 0.7415737131783371
#[2.76551848] [1.40166833] [0.28742099] [0.81988987] 1.7655184769831167 -0.863850142301906 0.10184522681723429 0.6343141062448767
#[2.76551848] [1.40166833] [0.28742099] [0.81988987] 1.7655184769831167 -0.863850142301906 0.10184522681723429 0.6343141062448767
#[-0.60716771 -0.35291914  1.55158716  0.3518669   0.05663279]

err1=0.02932012
err2=0.00010726
print(np.log2(err2/err1))
array_of_matrices = np.empty((50, 1), dtype=object)
array_of_matrices[0,0]=np.array([0,1,2,3,4,55,26,12])




# 创建一个示例的多维数组，其中每个元素都是一个不同大小的一维数组
data = []
max_length = 202  # 设置一维数组的最大长度

for N in range(1, max_length):
    # 创建一维数组，长度为 max_length，并将其中的数据存储在适当的位置
    arr = np.zeros((max_length, 1))
    arr[:N, 0] = np.random.rand(N, 1).flatten()
    data.append(arr)
'''
# 将多维数组保存到HDF5文件
with h5py.File('example.h5', 'w') as hf:
    # 创建一个数据集，将多维数组存储在其中
    dataset = hf.create_dataset('my_data', data=np.array(data))

# 从HDF5文件中读取多维数组
with h5py.File('example.h5', 'r') as hf:
    loaded_data = hf['my_data'][:]
    
# 确认数据已成功加载
for i, item in enumerate(loaded_data):
    print(f"Element {i + 1}: {item.shape}")
'''

widetwoRKCv2= np.load('widetwostepRKCv2.npz', allow_pickle=True)

twoRKCv1= np.load('twostepRKCv1.npz', allow_pickle=True)


RKC2=np.load('RKC2.npz', allow_pickle=True)
cs=RKC2['cs']


a=(26/(25* pi**2))*(-np.sin(pi*(1-0.0001))-2*np.sin(pi*0)+np.sin(pi*0.0001))/(0.001**2)

print(np.log2(3.8115117e-05/9.71008659e-06))
print(np.log2(0.00039466/0.00242638))