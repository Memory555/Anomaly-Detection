# 加载模型所需要的的包
import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
import pandas as pd
from sklearn.metrics.cluster import adjusted_rand_score
import time

# 生成随机簇类数据，样本数为5000，类别为2
# 使用 make_blob () 函数生成具有随机数据点的数据集。
data, _ = make_blobs(n_samples=5000, centers=2,random_state=170,shuffle=False)

colors = ['blue','red']
# eps表示聚类点为中心划定邻域
# min_samples表示每个邻域内需要多少个样本点。
res = []
# 迭代不同的eps值
epss = [0.1,0.5,1,2]
#先把图片名字存储成列表
name_list = []
for i in range(100):
    name_list.append(str(i))
step = 0
for eps in epss:
    # 迭代不同的min_samples值
    for min_samples in range(2,5):
        start = time.time()
        model= DBSCAN(eps = eps, min_samples = min_samples)
        # 模型拟合
        model.fit(data)
        clusters = model.fit_predict(data)
        # 统计各参数组合下的聚类个数（-1表示异常点）
        n_clusters = len([i for i in set(model.labels_) if i != -1])
        # 异常点的个数
        outliners = np.sum(np.where(model.labels_ == -1, 1,0))
        end = time.time()
        timem = end - start
        res.append({'次数':step+1,'聚类中心':eps,'领域内样本数':min_samples,'聚类个数':n_clusters,'异常点个数':outliners,'单次运行时间': timem})
        plt.scatter(data[:, 0], data[:, 1], c=model.labels_, cmap=matplotlib.colors.ListedColormap(colors))
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Scatter Plot of Feature X and Y")
        # 指定图片保存路径
        figure_save_path = "files_fig_many_DBSCAN"  # 这里创建了一个文件夹，如果依次创建不同文件夹，可以用name_list[i]
        if not os.path.exists(figure_save_path):
            os.makedirs(figure_save_path)  # 如果不存在目录figure_save_path，则创建
        plt.savefig(os.path.join(figure_save_path, name_list[step]))  # 分别命名图片
        step = step + 1
        # plt.show()
# 将迭代后的结果存储到数据框中
df = pd.DataFrame(res)
df.to_excel('data_DBSCAN_1.xls', sheet_name='DBSCAN_1')
print(df)

