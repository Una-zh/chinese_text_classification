#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from sklearn.naive_bayes import MultinomialNB  # 导入多项式贝叶斯算法
from sklearn.neighbors import KNeighborsClassifier  # 导入KNN算法
from sklearn.svm import SVC  # 导入SVM算法
from sklearn import metrics
from Tools import readbunchobj
import time
import os

working_directory = os.getcwd()
cwd = working_directory + "/中文文本分类python3.6/chinese_text_classification-master/"

# 导入训练集
trainpath = cwd + "large_train_word_bag/tfdifspace.dat"
train_set = readbunchobj(trainpath)

# 导入测试集
testpath = cwd + "large_test_word_bag/testspace.dat"
test_set = readbunchobj(testpath)

'''
    训练分类器：输入词袋向量和分类标签，学习率alpha:0.001 alpha越小，迭代次数越多，精度越高。
    tdm中的tdm[i][j]代表第j个词在第i个文本中的TF-IDF值。
    每一行（tdm[i]）可以当作一个文本的TF-IDF词向量，与之对应的类别标签是label[i]。
'''
NB_start_time = time.time()
NBayes_clf = MultinomialNB(alpha=0.05).fit(train_set.tdm, train_set.label)  # 模型学习到了先验概率P(Y=c_k)和条件概率P(X=x|Y=c_k)
NB_cost_time = time.time() - NB_start_time

KNN_start_time = time.time()
KNN_clf = KNeighborsClassifier(n_neighbors=3).fit(train_set.tdm, train_set.label)
KNN_cost_time = time.time() - KNN_start_time

SVM_start_time = time.time()
SVM_clf = SVC(kernel='linear', C=1.0).fit(train_set.tdm, train_set.label)
SVM_cost_time = time.time() - SVM_start_time

# 预测分类结果
NBayes_predicted = NBayes_clf.predict(test_set.tdm)

KNN_predicted = KNN_clf.predict(test_set.tdm)

SVM_predicted = SVM_clf.predict(test_set.tdm)
'''
    zip()接受一系列可迭代的对象作为参数，将对象中对应的元素打包成一个个tuple（元组），然后返回由这些tuples组成的list（列表）。
    若传入参数的长度不等，则返回list的长度和参数中长度最短的对象相同。
'''
'''
for flabel, file_name, expct_cate in zip(test_set.label, test_set.filenames, NBayes_predicted):
    if flabel != expct_cate:
        print(file_name, ": 实际类别:", flabel, " -->预测类别:", expct_cate)
'''

print("预测完毕!!!")

# 计算分类精度：
def metrics_result(actual, predicted):  # actual为实际类别，predict为预测类别
    print('precision:{0:.3f}'.format(metrics.precision_score(actual, predicted, average='weighted')))
    print('recall:{0:0.3f}'.format(metrics.recall_score(actual, predicted, average='weighted')))
    print('F1-score:{0:.3f}'.format(metrics.f1_score(actual, predicted, average='weighted')))

print("\nNBayes predict:")
metrics_result(test_set.label, NBayes_predicted)
print("\nKNN predict:")
metrics_result(test_set.label,KNN_predicted)
print("\nSVM predict:")
metrics_result(test_set.label,SVM_predicted)
