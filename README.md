## 运行环境：

CFS ，Fisher，ILFS，Laplacian，LASSO，Mutinffs：MATLAB
CRS，FCRS，RSLRS：Python3.0
Python依赖：numpy、pandas、sklearn、csv

## 数据集：

anneal、GINA、hepatitis、letter、lymphography、zoo、Amazon initial 50 30 10000为离散数据集，不需要对数据集进行离散化操作
Healthy_Older_People2、htru2、ionosphere、sensorReadings、vowel为连续数据集，需要对数据集进行离散化操作

## 代码说明：

对比实验包括CFS、Fisher、ILFS、Laplacian、LASSO、Mutinffs代码均来自Feature Selection Code Library (FSLib)(https://www.mathworks.cn/matlabcentral/fileexchange/56937-feature-selection-library)
CRS、FCRS、RSLRS：
**输入：**csv类型的数据集（注意连续数据需要启动数据离散化的代码）
**输出：**非冗余属性、代码的效率、代码的精度
**主要函数：**

def eql_class_split(not_redu, current_equ_classes, data)：划分等价类
def att_rdtion(data)：计算非冗余属性
def mean_std(a)：计算方差
def serarch(lis, num)：二分查找
def classifier(re, data, train_data)：分类器函数