#!/usr/bin/python3
# @Author: Jing
# @Time: 2018/5/27, 15:21
# -*- coding: utf-8 -*-

from sklearn.tree import DecisionTreeClassifier # ����������㷨
from sklearn import metrics
from Tools import readbunchobj

cwd = "D:\Documents\ѧϰ\�γ�\����ѧϰ ��־ǿ\�ı�����\�����ı�����python3.6/chinese_text_classification-master/"

# ����ѵ����
trainpath = cwd + "large_train_word_bag/tfdifspace.dat"
train_set = readbunchobj(trainpath)

# ������Լ�
testpath = cwd + "large_test_word_bag/testspace.dat"
test_set = readbunchobj(testpath)

'''
    ѵ��������������ʴ������ͷ����ǩ��ѧϰ��alpha:0.001 alphaԽС����������Խ�࣬����Խ�ߡ�
    tdm�е�tdm[i][j]�����j�����ڵ�i���ı��е�TF-IDFֵ��
    ÿһ�У�tdm[i]�����Ե���һ���ı���TF-IDF����������֮��Ӧ������ǩ��label[i]��
'''

clf = DecisionTreeClassifier()
