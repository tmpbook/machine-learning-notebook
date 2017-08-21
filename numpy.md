### 读取数据
```python
world_alcohol = numpy.genfromtxt('world_alcohol.txt', delimiter=',', dtype=float)
```
### 创建数据
```python
vector = numpy.array([5, 10, 15, 20])
matrix = numpy.array([[5, 10, 15], [20, 25, 30], [35, 40, 45]])
```
### 查看数据`形状`
```python
world_alcohol.shape
verctor.shape
matrix.shape
```
### 查看`numpy.ndarray`数据类型
```python
print(matrix.dtype)
# int64
```
### 矩阵索引管道
```python
# 第二行第五个元素
world_alcohol[1, 4]
# 第三行第四个元素
world_alcohol[2, 3]
# 所有行的第五个元素，也就是第五列
world_alcohol[:, 5]
# 所有行的第四个元素，也就是第四列
world_alcohol[:, 3]
```
> 逗号前后也可以使用切片，先切行，比如[0:3]是取前两行，逗号后面对应的是列，比如[0:3]取的是前面取到的行的1、2列，所以[0:3, 0:3]取的是前两行的前两列
### 矩阵的比较
```python
vector = numpy.array([5, 10, 15, 20])
vector == 10
# array([False, True, False, False], dtype=bool)
# 注意，这里新生成了一个对象，存的都是bool类型的值
```
### 使用全bool类型的矩阵作为索引
```python
equal_to_ten = vertor == 10
# [False True False False]
vector[equal_to_ten]
# [10]

matrix = numpy.array([
    [5, 10, 15],
    [20, 25, 30],
    [35, 40, 45]
])
second_column_25 = (metrix[:, 1] == 25)
# [False True False]
matrix[second_column_25, :]
# [[20 25 30]]

vector = numpy.array([5, 10, 15, 20])
equal_to_ten_or_five = (vector == 10) | (vector == 5)
vector[equal_to_ten_or_five]
# [5, 10]
```
### 矩阵类型转换
```python
vector = numpy.array(['1,', '2', '3'])
vector.dtype
# <U1
vector.dtype.name
# str32
vector_float = vector.astype(float)
# [1., 2., 3.]
```
### 内置函数
```python
vector = numpy.array[1., 2., 3.]
vector.min()
# 1.0

matrix = ([
    [ 5, 20, 15, 20],
    [20, 15, 20, 25]
])
matrix.min() # 5
matrix.min(axis=1) # [5, 15]
matrix.min(axis=0) # [5, 15, 15, 20]
matrix.max() # 25
matrix.max(axis=1) # [20, 25]
matrix.max(axis=0) # [20, 20, 20, 25]
```
> axis的值为1，意思我们的操作是对每一行，0则是对每一列
### 生成矩阵并对其变换
```python
import numpy as np
np.arange(15)
[0 1 2 3 4 5 6 7 8 9 10 11 12 13 14]
a = np.arange(15).reshape(3, 5)
array([[0, 1, 2, 3, 4],
       [5, 6, 7, 8, 9],
       [10, 11, 12, 13, 14]])
a.shape
# (3, 5)
```
### 查看维度
```python
a.ndimo # 2
```
### 查看元素总数
```python
a.size
```
### 创建0，1填充的矩阵
```python
np.zeros((3, 4))
np.ones((2, 3, 4), dtype=np.int32)
```
### 创建固定间隔的序列
```python
np.arange(10, 30, 5)
```
### 创建矩阵填充随机数
```python
np.random.random((3, 4))
```
### 创建已知头尾步长相等的n个数
```python
from numpy import pi
np.linspace(0, 2 * pi, 100)
```
### 矩阵每个元素map函数
```python
np.sin(np.linspace(0, 2 * pi, 100))
```
### 矩阵的数学运算
```python
a = np.array([20, 30, 40, 50])
b = np.arange(4)
a - b # [20, 29, 38, 47]
a - 1 # [19, 29, 39, 49]
b**2  # [0, 1, 4, 9]
a < 35 # [True, True, False, False] 这个上面有提到过
```
> 不知道你有没有注意到，shape相同的时候，会按位计算，读者可以试试其他运算，比如`*` `/` `+`
### 矩阵相乘
```python
A = np.array([[1, 1],
              [0, 1]])
B = np.array([[2, 0],
              [3, 4]])
A * B
# [[2, 0],
#  [0, 4]]
```
### 数量积，点积，点乘，内积
```python
A.dot(B)
# [[5, 4],
#  [3, 4]]
```
> 另一种写法是`np.dot(A, B)`
