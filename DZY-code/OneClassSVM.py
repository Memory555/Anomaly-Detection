from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import pandas as pd
import os
import time

# 使用 make_blob () 函数生成具有随机数据点的数据集。
data, _ = make_blobs(n_samples=5000, centers=2,random_state=170,shuffle=False)

# 构造分类器
# nu :错误和支持度的下界 0.5默认,设定训练误差(0, 1]，表示异常点比例
# kernal：内核类型 rbf默认
colors = ['red','blue']
res = []
nus = [0.1,0.2,0.5,0.6,0.8]
kernels = ['rbf','linear','poly','sigmoid']
#先把图片名字存储成列表
name_list = []
for i in range(100):
    name_list.append(str(i))
step = 0
for nu in nus:
    for kernel in kernels:
        start = time.time()
        model = svm.OneClassSVM(nu=nu, kernel=kernel)
        # # 模型拟合
        labels = model.fit_predict(data)
        list1 = []
        list2 = []
        for i in range(len(labels)):
            if labels[i] >= 0:
                list1.append(i)
            else:
                list2.append(i)
        # 异常点的个数
        outliners = len(list2)
        end = time.time()
        timem = end - start
        res.append({'次数': step+1,'异常点比例': nu, '内核类型': kernel,'异常点个数': outliners,'单次运行时间': timem})
        plt.scatter(data[list1, 0], data[list1, 1], c='blue')
        plt.scatter(data[list2, 0], data[list2, 1], c='red')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Scatter Plot of Feature X and Y")
        # 指定图片保存路径
        figure_save_path = "files_fig_many_SVM"  # 这里创建了一个文件夹，如果依次创建不同文件夹，可以用name_list[i]
        if not os.path.exists(figure_save_path):
            os.makedirs(figure_save_path)  # 如果不存在目录figure_save_path，则创建
        plt.savefig(os.path.join(figure_save_path, name_list[step]))  # 分别命名图片
        step = step + 1
        # plt.show()
# 将迭代后的结果存储到数据框中
df = pd.DataFrame(res)
df.to_excel('data_OneClassSVM_1.xls',sheet_name='SVM_1')
print(df)