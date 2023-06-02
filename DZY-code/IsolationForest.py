# 加载模型所需要的的包
import os
import numpy   as np
import pandas  as pd
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs
import xlwt
import time

# 使用 make_blob () 函数生成具有随机数据点的数据集。
data, _ = make_blobs(n_samples=5000, centers=2,random_state=170,shuffle=False)

# #构建模型
# # n_estimators=100 ,构建100颗树
# # max_samples 构造一棵树使用的样本数，输入大于1的整数则使用该数字作为构造的最大样本数目，
# # contamination 多少比例的样本可以作为异常值
res = []
n_estimatorss = [1,50,100,200,1000]
max_sampless = [5000,20,200,1000]
contaminations = [float(0.1),float(0.2)]
#先把图片名字存储成列表
name_list = []
for i in range(100):
    name_list.append(str(i))
step = 0
for n_estimators in n_estimatorss:
    for max_samples in max_sampless:
        for contamination in contaminations:
            start = time.time()
            model = IsolationForest(n_estimators=n_estimators,
                                    max_samples=max_samples,
                                    contamination=contamination)
            # 模型拟合
            model.fit(data)
            # 预测 decision_function 可以得出 异常评分
            scores = model.decision_function(data)
            #  predict() 函数 可以得到模型是否异常的判断，-1为异常，1为正常
            labels = model.predict(data)
            # plt.figure()
            list1 = []
            list2 = []
            for i in range(len(labels)):
                if labels[i] > 0:
                    list1.append(i)
                else:
                    list2.append(i)
            # 异常点的个数
            outliners = len(list2)
            end = time.time()
            timem = end - start
            res.append({'次数': step+1,'生成树的数目': n_estimators, '每棵树的样本量': max_samples, '异常值比例': contamination,
                        '异常点个数': outliners,'单次运行时间': timem})
            plt.scatter(data[list1, 0], data[list1, 1], c='blue')
            plt.scatter(data[list2, 0], data[list2, 1], c='red')
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.title("Scatter Plot of Feature X and Y")
            # 指定图片保存路径
            figure_save_path = "files_fig_many_ISO"  # 这里创建了一个文件夹，如果依次创建不同文件夹，可以用name_list[i]
            if not os.path.exists(figure_save_path):
                os.makedirs(figure_save_path)  # 如果不存在目录figure_save_path，则创建
            plt.savefig(os.path.join(figure_save_path, name_list[step]))  # 分别命名图片
            step = step + 1
            # plt.show()
# 将迭代后的结果存储到数据框中
df = pd.DataFrame(res)
df.to_excel('data_IsolationForest_1.xls',sheet_name='IsolationForest_1')
print(df)
