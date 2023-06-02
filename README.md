# Anomaly-Detection
此次设计通过对随机森林算法，DBSCAN算法，LOF算法，OneClassSVM算法四个算法在两类数据集上的结果进行分析得出一定的精度与适用性的结论。
# 设计目的和内容

## **设计目的**

 异常检测分为离群点检测（outlier detection） 以及奇异值检测（novelty detection） 两种。其中离群点检测：适用于训练数据中包含异常值的情况，离群点检测模型会尝试拟合训练数据最集中的区域，而忽略异常数据。奇异值检测：适用于训练数据不受异常值的污染，目标是去检测新样本是否是异常值。 在这种情况下，异常值也被称为奇异点。本设计将对孤立算法，DBSCAN算法，LOF算法，OneClassSVM算法四类异常检测算法的对比，给出上述算法的优缺分析与对比。

## **设计内容**

此次设计通过对随机森林算法，DBSCAN算法，LOF算法，OneClassSVM算法四个算法在两类数据集上的结果分析得出一定的结论。

# 数据集说明

数据集一：此数据集为使用make\_blob () 函数生成具有随机数据点的5000个样本数据。

数据集二：数据集包含欧洲持卡人于 2013 年 9 月通过信用卡进行的交易信息。此数据集显示的是两天内发生的交易，

在 284807 笔交易中，存在 492 起欺诈，数据集高度不平衡，正类（欺诈）仅占所有交易的 0.172%。

原数据集已做脱敏处理，匿名变量V1 ...V28已做PCA处理，唯一未经过 PCA 处理的变量是 Time 和 Amount。Time 是每笔交易与数据第一笔交易之间的间隔，单位为秒；Amount 是交易金额。Class 是分类变量，在发生欺诈时为1，否则为0。

注：PCA通过线性变换将原始数据变换为一组各维度线性无关的表示，可用于提取数据的主要特征分量，常用于高维数据的降维。

数据来源：数据集 Credit Card Fraud Detection 由比利时布鲁塞尔自由大学(ULB) - Worldline and the Machine Learning Group 提供。kaggle中下载途径：[https://www.kaggle.com/mlg-ulb/creditcardfraud](https://www.kaggle.com/mlg-ulb/creditcardfraud)

# 算法分析

## **优缺分析**

4.1.1 孤立森林算法

优：

孤立森林不需要根据距离和密度来衡量异常，因此孤立森林的时间复杂度是线性的，需要的内存也更少。

孤立森林的随机选取方法在处理大数据和高维数据效率更优。

缺：

孤立森林不适用于要求算法可靠性高的高维的数据。由于每次切数据空间都是随机选取一个维度，建完树后仍然有大量的维度信息没有被使用，导致算法可靠性降低。

孤立森林仅对全局稀疏点敏感，不擅长处理局部的相对稀疏点。因为局部异常被相似密度的正常聚类所掩盖，并且它们变得不容易被隔离。

4.1.2 DBSCAN算法

优：

DBSCAN 不需要预先声明聚类数量，其可以对任意形状的稠密数据集进行聚类。

可以在聚类的同时发现异常点，但对数据集中的异常点不敏感。

缺：

算法基于密度求解，需要对数据集中的每个样本都进行计算。

对于具有不同密度的簇，DBSCAN 算法的效果可能不是很好，当空间聚类的密度不均匀、样本分布较分散、聚类间距差相差很大时，聚类质量较差，需要为算法指定eps和MinPts参数。

调参复杂，主要需要对距离阈值eps，邻域样本数阈值MinPts联合调参，不同的参数组合对最后的聚类效果有较大影响，DBSCAN聚类算法对参数eps和MinPts的设置是非常敏感的，如果指定不当，该算法将造成聚类质量的下降。

作为异常检测时，并没有异常程度，只有一个标签。

4.1.3 LOF算法

优

LOF算法能在样本空间数据分布不均匀的情况下也可以准确发现离群点。

LOF算法适合于对不同密度的数据的异常检测。

缺：

检测的数据必须有明显的密度差异。

计算比较复杂。

应用场景有限，不适合高维度的海量数据。

4.1.4 OneClassSVM算法

优

最终决策函数只由少数的支持向量所确定,计算的复杂性取决于支持向量的数目,而不是样本空间的维数,这在某种意义上避免了"维数灾难"。

泛化性能比较好, 不容易过拟合。

可以在较少的数据下取得好的性能。

缺：

算法对大规模训练样本难以实施,对大规模训练样本难以实施。

应用场景有限,解决多分类问题存在困难。

## **结果分析**

3.2.1时间复杂度：

对于小样本数据：t(孤立森林算法)\>t(OneClassSVM算法)\>

t(DBSCAN算法)\>t(LOF算法)

对于高维度数据：t(OneClassSVM算法)\>t(LOF算法)\>

t(DBSCAN算法)\>t(孤立森林算法)

3.2.2预测准确率：

孤立森林算法，DBSCAN算法，SVMOneClass算法似乎都可以。

3.2.3参数的敏感程度：

LOF算法明显对参数的设定更为敏感，其他算法无法通过实验给出此结论

# 参考博客

[1] https://www.zhihu.com/question/280696035/answer/417091151. 数据挖掘中常见的「异常检测」算法有哪些？

[2] https://blog.csdn.net/weixin\_39822493/article/details/111586403. 异常数据4种剔除方法\_4种常见异常值检测算法实现

[3] https://blog.csdn.net/qq\_30031221/article/details/116494511. 异常检测方法——DBSCAN、孤立森林、OneClassSVM、LOF、同比环比、正态分布、箱线图

[4]https://blog.csdn.net/ewen\_lee/article/details/109892412. 孤立森林（Isolation Forest）从原理到实践

[5]https://blog.csdn.net/qq\_52785473/article/details/124293826. 异常检测之孤立森林算法详细解释且配上代码运行实例

[6] https://zhuanlan.zhihu.com/p/484495545. 孤立森林(isolation Forest)-一个通过瞎几把乱分进行异常检测的算法

[7] https://blog.csdn.net/sgzqc/article/details/122147329. 使用DBSCAN找出数据集中的异常值

[8] https://zhuanlan.zhihu.com/p/515268801. 聚类算法也可以异常检测？DBSCAN算法详解

[9] https://blog.csdn.net/Pysamlam/article/details/124013896. 异常检测算法之(LOF)-Local Outlier Factor

[10] https://www.csdn.net/tags/MtTaMg1sMDc3MjkwLWJsb2cO0O0O.html. 异常检测之LOF

[11] https://blog.csdn.net/juanjuanyou/article/details/121715138. 异常检测第二篇：一分类SVM（OneClassSVM）

[12] https://blog.51cto.com/u\_15127629/3319901. Python机器学习笔记：异常点检测算法——One Class SVM
