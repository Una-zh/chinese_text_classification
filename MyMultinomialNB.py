#!/usr/bin/python3
# @Author: Jing
# @Time: 2018/6/13, 11:54
# -*- coding: UTF-8 -*-
import numpy
from Tools import readbunchobj
from sklearn import metrics
import time
from sklearn.neighbors import KNeighborsClassifier  # 导入KNN算法
from sklearn.svm import SVC  # 导入SVM算法
from sklearn.naive_bayes import BernoulliNB
import os


class MyMultinomialNB(object):
    '''
        This multinomial Naive Bayes classifier is suitable for classification with
        discrete tf-idf features.
    '''

    def fit(self, X, Y, numClass, alpha=1.0):
        numTrainDoc = X.shape[0]
        numWords = X.shape[1]
        X = X.toarray()
        self.cls_dict = {}  # 给每个类一个下标，从0开始，方便下面计算self.conditional_probability
        self.cls_weight = numpy.array([0.0]*numTrainDoc)  # 每个类的所有文档中所有词（vocabulary中的）的权重之和
        self.prior_probability = numpy.array([alpha]*numClass)  # 先验概率，共有numClass个

        '''
        # 伯努利模型：计算先验概率，每个类别的总文档数/全部文档数
        for each in Y:
            cls = Y[each]
            if cls in self.cls_dict.keys():
                index = self.cls_dict[cls]
                self.prior_probability[index] += 1
            else:
                self.cls_dict[cls] = cls_index
                self.prior_probability[cls_index] = 1
                cls_index += 1
        # self.prior_probability = self.prior_probability/(numTrainDoc + alpha * numClass)
        '''

        # 计算条件概率
        # self.conditional_probability先存储每个类的所有文档中的各个词的权重之和，将所有词的权重之和初始化为alpha，拉普拉斯平滑
        self.conditional_probability = [numpy.array([alpha] * numWords) for i in range(numClass)]
        # 每个类中所有词的总权重+类别数，“+类别数”的原因：等会做分母，拉普拉斯平滑
        sum_all_weight = [0.0 + float(alpha * numWords)] * numClass
        '''
            1.将tfidf矩阵中相同类别的文本向量相加，形成numClass*numWords的矩阵
            2.将每一个词的权重除以该类所有词的总权重（sum_all_weight[cls_index]）
        '''
        index_of_cls = 0  # self.cls_dict中每个类的下标
        for each in range(numTrainDoc):
            cls = Y[each]
            if cls not in self.cls_dict.keys():  # 如果类cls还未被加入到self.cls_dict中，则加入，并给其一个下标
                self.cls_dict[cls] = index_of_cls
                index_of_cls += 1
            cls_index = self.cls_dict[cls]
            self.conditional_probability[cls_index] += X[each]
            sum_all_weight[cls_index] += sum(X[each])
        # 计算（多项式模型）先验概率，每个类中所有特征项的权值之和/训练集中所有类的所有特征项的权值之和
        self.cls_weight = sum(self.conditional_probability)-alpha  # 每个类的所有文档中所有词（vocabulary中的）的权重之和。
        sum_all = numpy.sum(sum_all_weight)  # 求sum_all_weight矩阵的和，即所有文档的所有词的权重和，作为先验概率的分母。


        for each in range(numClass):
            # 计算（多项式模型）条件概率，每个词的权重/该类所有词的总权重sum_all_weight
            self.conditional_probability[each] = numpy.log(self.conditional_probability[each]/sum_all_weight[each])



    def predict(self, test_set):
        numTestDoc = test_set.shape[0]
        numClass = len(self.conditional_probability)
        test_set = test_set.toarray()
        self.predicted_class = []  # 一维数组，记录每个doc的类别预测值
        for each in range(numTestDoc):
            posterior_probability = numpy.array([0.0] * numClass)  # 一维数组，记录当前doc的每个类别的后验概率
            for cls in range(numClass):
                posterior_probability[cls] = sum(test_set[each]*self.conditional_probability[cls]) + numpy.log(self.prior_probability[cls])
            self.predicted_class.append(list(self.cls_dict.keys())[list(self.cls_dict.values()).index(numpy.argmax(posterior_probability))])
        return self.predicted_class

    def cross_validation(self):
        pass


def metrics_result(actual, predicted, train_cost_time):  # actual为实际类别，predicted为预测类别
    print('precision:{0:.3f}'.format(metrics.precision_score(actual, predicted, average='weighted')))
    print('recall:{0:0.3f}'.format(metrics.recall_score(actual, predicted, average='weighted')))
    print('F1-score:{0:.3f}'.format(metrics.f1_score(actual, predicted, average='weighted')))
    print('training time:{0:.3f}'.format(train_cost_time))
    # print('accuracy:{0:.3f}'.format(metrics.accuracy_score(actual, predicted)))


if __name__ == '__main__':
    working_directory = os.getcwd()
    cwd = working_directory + "/中文文本分类python3.6/chinese_text_classification-master/"
    # 导入训练集
    trainpath = cwd + "large_train_word_bag/tfdifspace.dat"
    train_set = readbunchobj(trainpath)

    # 导入测试集
    testpath = cwd + "large_test_word_bag/testspace.dat"
    test_set = readbunchobj(testpath)

    # 训练各分类器：自己实现的多项式贝叶斯、python自带的伯努利贝叶斯、K近邻、支持向量机
    MyNB_start_time = time.time()
    MyNBclassifier = MyMultinomialNB()
    MyNBclassifier.fit(X=train_set.tdm, Y=train_set.label, numClass=6, alpha=0.05)
    MyNB_cost_time = time.time() - MyNB_start_time

    BNB_start_time = time.time()
    BNBayes_clf = BernoulliNB(alpha=0.05).fit(train_set.tdm, train_set.label)  # 模型学习到了先验概率P(Y=c_k)和条件概率P(X=x|Y=c_k)
    BNB_cost_time = time.time() - BNB_start_time

    KNN_start_time = time.time()
    KNN_clf = KNeighborsClassifier(n_neighbors=3).fit(train_set.tdm, train_set.label)
    KNN_cost_time = time.time() - KNN_start_time

    SVM_start_time = time.time()
    SVM_clf = SVC(kernel='linear', C=1.0).fit(train_set.tdm, train_set.label)
    SVM_cost_time = time.time() - SVM_start_time

    # 预测分类结果
    MyNB_predicted = MyNBclassifier.predict(test_set.tdm)

    BNBayes_predicted = BNBayes_clf.predict(test_set.tdm)

    KNN_predicted = KNN_clf.predict(test_set.tdm)

    SVM_predicted = SVM_clf.predict(test_set.tdm)

    # 输出预测效果
    print("\nMyNBayes predict:")
    metrics_result(test_set.label, MyNB_predicted, MyNB_cost_time)

    print("\nBNBayes predict:")
    metrics_result(test_set.label, BNBayes_predicted, BNB_cost_time)

    print("\nKNN predict:")
    metrics_result(test_set.label, KNN_predicted, KNN_cost_time)

    print("\nSVM predict:")
    metrics_result(test_set.label, SVM_predicted, SVM_cost_time)

    '''
        MyNB_predicted = MyNBclassifier.predict(test_set.tdm)
        print("\nMyNBayes predict:")
        metrics_result(test_set.label, MyNB_predicted, MyNB_cost_time)
    '''

