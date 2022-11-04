'''
求活动域时采用二分查找来初步确定属于那个等价类
上下同时删除
'''
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
import csv

warnings.filterwarnings("ignore")  # 忽略警告
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=np.inf)

'''
划分等价类
atts：属性集
data：数据集
return：返回等价类，二维列表
'''
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

'''
属性约简
'''
def att_rdtion(data):
    current_equ_class = [[i for i in range(len(data))]]  # current_equ_class用来保存当前的等价类，初始整个数据集为一个等价类
    not_redund_atts = []    #保存约简属性
    if len(data) != 0:
        redund_atts = [i for i in range(len(data[0, :-2]))]  # 全部属性的索引列表
        active_re = {}  # 保存每个属性的等价类

        for redund_att in redund_atts:
            result1 = eql_class_split(redund_att, current_equ_class, data)
            active_re[redund_att] = result1
        active_re_keys = active_re.keys()
        active_re_keys = list(active_re_keys)

        active_list111 = {}
        # 获取单个属性等价类的首字段
        for index in active_re_keys:
            list444 = []
            for item in active_re[index]:
                list444.append(item[0])
            active_list111[index] = list444
        #计算第一个属性，因为在计算第一个属性的时候非活动域为空，所以此时直接在整个数据集上计算
        max_num1 = 0
        prepare_att1 = -1
        for index_1 in active_re_keys:
            re_list = []
            num_re_111 = 0
            for new_item in active_re[index_1]:
                list11 = []  # 用来保存每个等价类的决策属性
                for att_num in new_item:
                    list11.append(data[att_num][-2])
                if len(set(list11)) == 1:
                    re_list.append(new_item)
                    num_re_111 += len(new_item)
            if num_re_111 > max_num1:
                max_num1 = num_re_111
                prepare_att1 = index_1
        redund_atts.remove(prepare_att1)
        active_re_keys.remove(prepare_att1)
        not_redund_atts.append(prepare_att1)
        current_equ_class = active_re[prepare_att1]
        #属性约简
        while redund_atts:
            stemp_eql_class = {}  # 用来保存求正域时候产生的临时等价类
            list_item = []
            for equ_class in current_equ_class:  # 求边界域
                list_split = []
                for index in equ_class:
                    list_split.append(data[index][-2])
                if len(set(list_split)) == 1:
                    list_item.append(equ_class)

            if len(list_item) > 0:
                for item in list_item:
                    current_equ_class.remove(item)
            board_area = current_equ_class.copy()  # 获得边界域
            prepare_att = -1
            max_num = 0

            for index in active_re_keys:
                lists = active_re[index]  # 每个属性下的等价类
                board_area_stemp = board_area.copy()  # 临时存储等价类，保存当前的活动域
                active_list = []  # 保存是非活动域的等价类

                if (2 * len(board_area_stemp) < len(active_re[index])):
                    num_re_111 = 0  # 当前属性下的新增正域个数
                    # 求正域
                    new_eql_class = eql_class_split(index, board_area_stemp, data)  # new_eql_class用来保存加入一个属性后新的等价类
                    stemp_eql_class[index] = new_eql_class.copy()
                    num_class = len(new_eql_class) - len(board_area_stemp)  # 新增等价类的个数
                    for new_item in new_eql_class:
                        list11 = []  # 用来保存每个等价类的决策属性
                        for att_num in new_item:
                            list11.append(data[att_num][-2])
                        if len(set(list11)) == 1:
                            num_re_111 += len(new_item)
                    if num_class != 0:       #计算属性的相对重要度
                        if num_re_111 / num_class> max_num:
                            max_num = num_re_111/ num_class
                            prepare_att = index
                else:
                    for item in board_area_stemp:  # 这层循环是用来求活动域的
                        num2 = serarch(active_list111[index], item[0])
                        k = 0
                        l = binarySearch(lists[num2], 0, len(lists[num2]) - 1, item[0])
                        r = binarySearch(lists[num2], 0, len(lists[num2]) - 1, item[len(item) - 1])

                        if len(item) > r - l + 1:
                            break
                        if l <= -1 or r <= -1:
                            break
                        else:
                            for i in range(1, len(item) - 1):
                                m = binarySearch(lists[num2], l, r, item[i])
                                if m > -1:
                                    k += 1
                                else:
                                    break
                        if k == len(item) - 2:  # 第一个和最后一个元素已经判断过了
                            active_list.append(item)
                            del lists[num2][l:r+1]
                    if len(active_list) > 0:
                        for item in active_list:
                            board_area_stemp.remove(item)
                    if len(board_area_stemp) == 0:
                        continue
                    num_re_111 = 0  # 当前属性下的新增正域个数
                    # 求正域
                    new_eql_class = eql_class_split(index, board_area_stemp, data)  # new_eql_class用来保存加入一个属性后新的等价类
                    stemp_eql_class[index] = new_eql_class.copy()

                    num_class = len(new_eql_class) - len(board_area_stemp)  # 新增等价类的个数
                    for new_item in new_eql_class:
                        list11 = []  # 用来保存每个等价类的决策属性
                        for att_num in new_item:
                            list11.append(data[att_num][-2])
                        if len(set(list11)) == 1:
                            num_re_111 += len(new_item)
                    #计算属性的相对重要度
                    if num_class!=0:
                        if num_re_111/ num_class > max_num:
                            max_num = num_re_111/ num_class
                            prepare_att = index
                            for iii in active_list:
                                stemp_eql_class[index].append(iii)
            if prepare_att < 0:  # prepare_att<0表示所有实行均不增加正域，停止循环
                break
            redund_atts.remove(prepare_att)
            current_equ_class = stemp_eql_class[prepare_att].copy()
            active_re_keys.remove(prepare_att)
            not_redund_atts.append(prepare_att)
    print('not_redund_atts=======>', sorted(not_redund_atts))
    return not_redund_atts


def binarySearch(arr, l, r, x):
    left = 0
    right = r
    while left <= right:  # 循环条件
        mid = (left + right) // 2  # 获取中间位置，数字的索引（序列前提是有序的）
        if x < arr[mid]:  # 如果查询数字比中间数字小，那就去二分后的左边找，
            right = mid - 1  # 来到左边后，需要将右变的边界换为mid-1
        elif x > arr[mid]:  # 如果查询数字比中间数字大，那么去二分后的右边找
            left = mid + 1  # 来到右边后，需要将左边的边界换为mid+1
        else:
            return mid  # 如果查询数字刚好为中间值，返回该值得索引
    return -2  #

'''
二分查找
lis：要查找的集合
num：要查找的数字
'''
def serarch(lis, num):
    left = 0
    right = len(lis) - 1
    while left <= right:
        # if (right - 1 == left):
        #     return left
        mid = left+(right-left) // 2
        if num < lis[mid]:  # 如果查询数字比中间数字小，那就去二分后的左边找，
            right = mid - 1  # 来到左边后，需要将右变的边界换为mid-1
        elif num > lis[mid]:  # 如果查询数字比中间数字大，那么去二分后的右边找
            left = mid + 1  # 来到右边后，需要将左边的边界换为mid+1
        else:
            return mid
    return left - 1

'''
计算一维数组的均值和标准差
'''
def mean_std(a):
    a = np.array(a)
    std = np.sqrt(((a - np.mean(a)) ** 2).sum() / (a.size - 1))
    return a.mean(), std

'''
粗糙概念树进行分类
re：约简结果
data：数据集
train_data：训练集
'''
def classifier(re, data, train_data):
    eql_class = [[i for i in range(len(data))]]
    for index in re:
        eql_class = eql_class_split(index, eql_class, data)
    atts_dict = judge_re(re, eql_class, data)
    atts_dict_keylist11 = sorted(list(atts_dict.keys()))

    right_num = 0  # 表示预测正确的样本数量
    for train_sample in train_data:
        list44 = []
        for index in re:
            list44.append(train_sample[index])
        atts_dict_keylist11.append(str(list44))
        atts_dict_keylist = sorted(atts_dict_keylist11).copy()

        train_location = sorted(atts_dict_keylist).index(str(list44))
        if train_location + 1 > len(atts_dict_keylist) - 1:
            train_decision = atts_dict[atts_dict_keylist[train_location - 1]]
        else:
            if atts_dict_keylist[train_location - 1] == atts_dict_keylist[train_location]:
                train_decision = atts_dict[atts_dict_keylist[train_location - 1]]
            elif atts_dict_keylist[train_location + 1] == atts_dict_keylist[train_location]:
                train_decision = atts_dict[atts_dict_keylist[train_location + 1]]
            else:
                for i in range(min(len(atts_dict_keylist[train_location]), len(atts_dict_keylist[train_location - 1]),
                                   len(atts_dict_keylist[train_location + 1]))):
                    if atts_dict_keylist[train_location - 1][i] == atts_dict_keylist[train_location][i] and \
                            atts_dict_keylist[train_location][i] == atts_dict_keylist[train_location + 1][i]:
                        continue
                    else:
                        if ord(atts_dict_keylist[train_location][i]) - ord(
                                atts_dict_keylist[train_location - 1][i]) < ord(
                                atts_dict_keylist[train_location + 1][i]) - ord(atts_dict_keylist[train_location][i]):
                            train_decision = atts_dict[atts_dict_keylist[train_location - 1]]
                        else:
                            train_decision = atts_dict[atts_dict_keylist[train_location + 1]]
        if train_decision == train_sample[-2]:
            right_num += 1
        atts_dict_keylist11.remove(str(list44))
    acc = right_num / len(train_data)
    print(acc)
    return acc

def judge_re(re,eql_class,data):
    atts_dict ={}      #用来保存每个属性集对应的决策属性
    # re_list = []
    # board_list = []
    for new_item in eql_class:
        list22 =[]
        for index in re:
            list22.append(data[new_item[0]][index])

        list11 = []  # 用来保存每个等价类的决策属性
        for att_num in new_item:
            list11.append(data[att_num][-2])
        atts_dict[str(list22)] =max(list11, key=list11.count)

    return atts_dict

classifiers = ['LR'] #所用的分类器
keys = ["Amazon_initial_50_30_10000"]  #所用的数据集
for key in keys:
    df = pd.read_csv("D:\\新数据集\\" + key + ".csv", header=None)
    data = df.values
    numberSample, numberAttribute = data.shape

    #数据的离散化处理
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

    orderAttribute = np.array([i for i in range(0, numberSample)]).reshape(numberSample,
                                                                           1)  # 创建一个列表保存数字序列从1到numberSample
    data = np.hstack((data, orderAttribute))
    a = time.clock()
    re = att_rdtion(data)
    b = time.clock()
    print("时间开销===========>", b - a)


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

        if re != [i for i in range(len(data[0, :-1]))]:
            random.shuffle(data)
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
            with open(r'E:\\shiyanjieguo\\7.3.csv', 'a+', encoding='utf-8-sig',
                      newline='') as f:
                writer = csv.writer(f)
                writer.writerow(result)
        else:
            print("本次未约简")
            break





