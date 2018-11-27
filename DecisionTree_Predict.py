#!/usr/bin/python3
# @Author: Jing
# @Time: 2018/5/27, 15:21
# -*- coding: utf-8 -*-

from sklearn.tree import DecisionTreeClassifier # 导入决策树算法
from sklearn import metrics
from Tools import readbunchobj

cwd = "D:\Documents\学习\课程\机器学习 高志强\文本分类\中文文本分类python3.6/chinese_text_classification-master/"

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

clf = DecisionTreeClassifier()
