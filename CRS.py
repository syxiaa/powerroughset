import numpy as np
import pandas as pd
import warnings
import time
import time
import random
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
# from lightgbm import LGBMClassifier
import time
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
import warnings

'''
划分等价类
atts：属性集
data：数据集
return：返回等价类，二维列表
'''


# 当前属性集的等价类划分
def eql_class_split(not_redu, current_equ_classes, data):
    ud_equ_classes = []
    for equ_class in current_equ_classes:  # 加入最重要的属性，更新等价类
        new_equ_classes = {}
        for sample in equ_class:
            if data[sample, not_redu] not in new_equ_classes:
                new_equ_classes[data[sample, not_redu]] = [sample]
            else:
                new_equ_classes[data[sample, not_redu]].append(sample)
        for keys in new_equ_classes:
            ud_equ_classes.append(new_equ_classes[keys])
    return ud_equ_classes


def att_rdtion(data):
    current_equ_class = [[i for i in range(len(data))]]  # current_equ_class用来保存当前的等价类，初始整个数据集为一个等价类

    redund_atts = []
    not_redund_atts = [] #保存非冗余属性
    if len(data) != 0:
        redund_atts = [i for i in range(len(data[0, :-2]))]  # 全部属性的索引列表
        num_re = 0   #当前等价类中正域的个数
        num_re_222=0  #用来保存上一次的等价类的个数
        while redund_atts:
            stemp_eql_class = {}  # 用来保存求正域时候产生的临时等价类
            prepare_att = -1
            max_num = 0
            num_re_222= num_re
            for index in redund_atts:  #计算出每个属性的属性重要度，选出属性重要度最高的属性
                num=0
                num_re_111=0  #当前属性下的正域个数
                # 求正域
                new_eql_class = eql_class_split(index, current_equ_class, data)  # new_eql_class用来保存加入一个属性后新的等价类
                stemp_eql_class[index] = new_eql_class.copy()
                for new_item in new_eql_class:
                    list11 = []  # 用来保存每个等价类的决策属性
                    for att_num in new_item:
                        list11.append(data[att_num][-2])
                    if len(set(list11)) == 1:
                        num_re_111 += len(new_item)
                num=num_re_111-num_re_222 #当前属性新增正域的数量
                if num>max_num:
                    max_num=num
                    num_re=num_re_111
                    prepare_att = index
            if prepare_att < 0:
                break
            redund_atts.remove(prepare_att)
            current_equ_class = stemp_eql_class[prepare_att].copy()
            not_redund_atts.append(prepare_att)
    print('not_redund_atts', not_redund_atts)
    return  not_redund_atts

def mean_std(a):
    # 计算一维数组的均值和标准差
    a = np.array(a)
    std = np.sqrt(((a - np.mean(a)) ** 2).sum() / (a.size - 1))
    return a.mean(), std

import KF
import csv

keys = ["anneal","hepatitis","letter","lymphography","zoo"]   #离散数据
classifiers = ['CART', 'LR', 'GBDT']
for key in keys:
    df = pd.read_csv("D:\\粗糙集代码\\我的数据集\\最终的数据集\\离散\\" + key + ".csv", header=None)
    data = df.values
    numberSample, numberAttribute = data.shape

    #离散化
    # for i in range(1, len(data[0])):
    #     start1_time = time.clock()
    #     # print(i)
    #     if len(set(data[:, i])) == 1:
    #         continue
    #     group = KF.Chi_Discretization(df, i, 0, max_interval=10, binning_method='chiMerge', feature_type=0)  # 分箱
    #     for j in range(len(data)):  # 数据替换 直接替换到原数据上
    #         k = 0
    #         while (True):
    #             # print(k)
    #             if (k == len(group) - 1):
    #                 data[j][i] = group[k]
    #                 # new_data[j][0]=data.values[j][0]
    #                 break
    #             if (data[j][i] < group[k + 1] and data[j][i] >= group[k]):
    #                 data[j][i] = group[k + 1]
    #                 # new_data[j][0] = data.values[j][0]
    #                 break
    #             else:
    #                 k += 1

    data = np.hstack((data[:, 1:], data[:, 0].reshape(numberSample, 1)))
    orderAttribute = np.array([i for i in range(0, numberSample)]).reshape(numberSample, 1)  # 创建一个列表保存数字序列从1到numberSample
    data = np.hstack((data, orderAttribute))
    # print(data[2])
    a = time.clock()
    re=att_rdtion(data)
    b = time.clock()
    print("时间开销===========>", b - a)

    result = []
    result.append(key)
    result.append(b - a)
    for c in range(len(classifiers)):
        classifier = classifiers[c]
        print(classifier)
        if classifier == 'BP':
            clf = MLPClassifier()
        elif classifier == 'KNN':
            clf = KNeighborsClassifier(n_neighbors=5)
        elif classifier == 'SVM':
            clf = SVC(kernel='rbf', gamma='auto')
        elif classifier == 'CART':
            clf = DecisionTreeClassifier()
        elif classifier == 'LR':
            clf = LogisticRegression(solver='liblinear')
        elif classifier == 'GBDT':
            clf = GradientBoostingClassifier()
        #

        if re != [i for i in range(len(data[0, :-1]))]:
            mat_data = data[:, re]
            start_time2 = time.clock()
            orderAttribute = data[:, -2]

            scores = cross_val_score(clf, mat_data, orderAttribute, cv=10)
            avg1, std1 = mean_std(scores)
            print("实验精度", avg1, "\t", std1)
            end_time2 = time.clock()
            run_time2 = end_time2 - start_time2
            result = []
            result.append(key)
            result.append(format(avg1, '.4f') + '±' + format(std1, '.4f'))
            with open(r'E:\\shiyanjieguo\\10.24\\经典_正域.csv', 'a+', encoding='utf-8-sig',
                      newline='') as f:
                writer = csv.writer(f)
                writer.writerow(result)
        else:
            print("本次未约简")
            break
