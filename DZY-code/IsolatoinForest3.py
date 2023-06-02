# 加载模型所需要的的包
import os
import numpy   as np
import pandas  as pd
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs
import xlwt
import numpy as np

# 信用卡欺诈是指故意使用伪造、作废的信用卡，冒用他人的信用卡骗取财物，或用本人信用卡进行恶意透支的行为,信用卡欺诈形式分为3种：
# 失卡冒用、假冒申请、伪造信用卡。欺诈案件中，有60%以上是伪造信用卡诈骗，其特点是团伙性质，从盗取卡资料、制造假卡、贩卖假卡，
# 到用假卡作案，牟取暴利。而信用卡欺诈检测是银行减少损失的重要手段。
#
# 该数据集包含欧洲持卡人于 2013 年 9 月通过信用卡进行的交易信息。此数据集显示的是两天内发生的交易，
# 在 284807 笔交易中，存在 492 起欺诈，数据集高度不平衡，正类（欺诈）仅占所有交易的 0.172%。
# 原数据集已做脱敏处理和PCA处理，匿名变量V1， V2， ...V28 是 PCA 获得的主成分，
# 唯一未经过 PCA 处理的变量是 Time 和 Amount。Time 是每笔交易与数据集中第一笔交易之间的间隔，
# 单位为秒；Amount 是交易金额。Class 是分类变量，在发生欺诈时为1，否则为0。项目要求根据现有数据集建立分类模型，
# 对信用卡欺诈行为进行检测。
#
# 注：PCA - "Principal Component Analysis" - 主成分分析，用于提取数据集的"主成分"特征，即对数据集进行降维处理。

# 工作空间设置
os.chdir('./')
os.getcwd()

# 数据读取
data = pd.read_csv('creditcard.csv')


# 数据集划分 + 简单的特征工程
data['Hour'] = data["Time"].apply(lambda x : divmod(x, 3600)[0])


X = data.drop(['Time','Class'],axis=1)
Y = data.Class


# 数据归一化
from sklearn.preprocessing import StandardScaler
sd        = StandardScaler()
column    = X.columns
X[column] = sd.fit_transform(X[column])


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.55,random_state=0)


# #构建模型
# # n_estimators=100 ,构建100颗树
# # max_samples 构造一棵树使用的样本数，输入大于1的整数则使用该数字作为构造的最大样本数目，
# # contamination 多少比例的样本可以作为异常值
res = []
n_estimatorss = [20,50,100,200,1000]
max_sampless = [5000,20,200,1000]
contaminations = [float(0.1),float(0.2)]
step = 0
for n_estimators in n_estimatorss:
    for max_samples in max_sampless:
        for contamination in contaminations:
            model= IsolationForest(n_estimators=n_estimators,
                                        max_samples=max_samples,
                                        contamination=contamination)
            # 训练集拟合
            model.fit(X_train)
            # 预测集预测
            labels = model.fit_predict(X_test)
            X_test['labels_'] = labels
            X_test['labels'] = Y_test
            # 真实异常值
            mnoraml, nnoraml = np.shape(X_test[X_test['labels'] == 1])
            # 预测异常值
            prenmoraml, prennoraml = np.shape(X_test[(X_test['labels'] == 1) & (X_test['labels_'] == -1)])
            # outliners = len(list2)
            nonomarl_rate = prenmoraml / mnoraml
            res.append({'次数': step + 1, 'n_estimator': n_estimators, 'max_samples': max_samples,
                        'contamination': contamination,
                        '测试集真实异常值':mnoraml,'测试集预测异常值':prenmoraml,'测试集准确率':nonomarl_rate})
            step = step + 1
            print("第", step, "次结束！")

# 将迭代后的结果存储到数据框中
df = pd.DataFrame(res)
df.to_excel('data_IsolationForest_3.xls',sheet_name='IsolationForest_3')
print(df)
