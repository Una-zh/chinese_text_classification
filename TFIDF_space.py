#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from sklearn.datasets.base import Bunch
from sklearn.feature_extraction.text import TfidfVectorizer
import Tools
import os

# vector_space()函数用于创建TF-IDF词向量空间
def vector_space(stopword_path, bunch_path, space_path, train_tfidf_path=None):
    stpwrdlst = Tools.readfile(stopword_path).splitlines()  # 读取停用词
    bunch = Tools.readbunchobj(bunch_path)  # 读取分词后的词向量bunch对象
    tfidfspace = Bunch(target_name=bunch.target_name, label=bunch.label, filenames=bunch.filenames, tdm=[],
                       vocabulary={})  # 构建TF-IDF词向量空间对象
    '''
        target_name,label和filenames这几个成员都是我们自己定义的玩意儿，前面已经讲过不再赘述。 
        下面我们讲一下tdm和vocabulary（这俩玩意儿也都是我们自己创建的）： 
        tdm存放的是计算后得到的TF-IDF权重矩阵。请记住，我们后面分类器需要的东西，其实就是训练集的tdm和标签label，因此这个成员是 
        很重要的。 
        vocabulary是词典索引，例如 
        vocabulary={"我":0,"爱":1,"中国":2}，这里的数字对应的就是tdm矩阵的列 
        我们现在就是要构建一个词向量空间，因此在初始时刻，这个tdm和vocabulary自然都是空的。如果你在这一步将vocabulary赋值了一个 
        自定义的内容，那么，你是傻逼。 
    '''

    print("玩命儿创建词向量空间实例中...")

    if train_tfidf_path is not None:
        '''
            测试集数据与训练集数据要处在同一个词向量空间（vocabulary相同），即：忽略测试集中新出现的词，
            故有41行vocabulary=trainbunch.vocabulary。词典索引（可以理解为各坐标轴的名称）沿用训练集的。
        '''
        trainbunch = Tools.readbunchobj(train_tfidf_path)
        tfidfspace.vocabulary = trainbunch.vocabulary
        vectorizer = TfidfVectorizer(stop_words=stpwrdlst, sublinear_tf=True, max_df=0.45,min_df=0.1,
                                     vocabulary=trainbunch.vocabulary) # test跟train在一个词向量空间，忽略test中新出现的词
        tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)  # 此时tdm里面存储的就是TF-IDF权值矩阵(tdm[i][j]代表第j个词在第i个文本中的TF-IDF值)
        '''
            The formula that is used to compute the tf-idf of term t is
            tf-idf(d, t) = tf(t) * idf(d, t), and the idf is computed as
            idf(d, t) = log [ n / df(d, t) ] + 1 (if ``smooth_idf=False``),
            where n is the total number of documents and df(d, t) is the
            document frequency; the document frequency is the number of documents d
            that contain term t. The effect of adding "1" to the idf in the equation
            above is that terms with zero idf, i.e., terms  that occur in all documents
            in a training set, will not be entirely ignored.
            (Note that the idf formula above differs from the standard
            textbook notation that defines the idf as
            idf(d, t) = log [ n / (df(d, t) + 1) ]).
        
            If ``smooth_idf=True`` (the default), the constant "1" is added to the
            numerator and denominator of the idf as if an extra document was seen
            containing every term in the collection exactly once, which prevents
            zero divisions: idf(d, t) = log [ (1 + n) / (1 + df(d, t)) ] + 1.
            
            sublinear_tf : boolean, default=False
            Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf)
        '''

        '''
            关于TfidfVectorizer的参数，你只需要了解这么几个就可以了： 
            stop_words: 
            传入停用词，以后我们获得vocabulary_的时候，就会根据文本信息去掉停用词得到 
            vocabulary: 
            之前说过，不再解释。 
            sublinear_tf: 
            计算tf值采用亚线性策略。比如，我们以前算tf是词频，现在用1+log(tf)来充当词频。 
            smooth_idf: 
            计算idf的时候log(分子/分母)分母有可能是0，smooth_idf会采用log((1+分子)/(1+分母))的方式解决。默认已经开启，无需关心。 
            norm: 
            归一化，我们计算TF-IDF的时候，是用TF*IDF，TF可以是归一化的，也可以是没有归一化的，一般都是采用归一化的方法，默认开启. 
            max_df: 
            有些词，他们的文档频率太高了（一个词如果每篇文档都出现，那还有必要用它来区分文本类别吗？当然不用了呀），所以，我们可以 
            设定一个阈值，比如float类型0.5（取值范围[0.0,1.0]）,表示这个词如果在整个数据集中超过50%的文本都出现了，那么我们也把它列 
            为临时停用词。当然你也可以设定为int型，例如max_df=10,表示这个词如果在整个数据集中超过10的文本都出现了，那么我们也把它列 
            为临时停用词。 
            min_df: 
            与max_df相反，虽然文档频率越低，似乎越能区分文本，可是如果太低，例如10000篇文本中只有1篇文本出现过这个词，仅仅因为这1篇 
            文本，就增加了词向量空间的维度，太不划算。 
            当然，max_df和min_df在给定vocabulary参数时，就失效了。 
        '''

        '''
            与上面这2行代码等价的代码是： 
            vectorizer=CountVectorizer()#构建一个计算词频（TF）的玩意儿，当然这里面不只是可以做这些 
            transformer=TfidfTransformer()#构建一个计算TF-IDF的玩意儿 
            tfidf=transformer.fit_transform(vectorizer.fit_transform(corpus)) 
            #vectorizer.fit_transform(corpus)将文本corpus输入，得到词频矩阵 
            #将这个矩阵作为输入，用transformer.fit_transform(词频矩阵)得到TF-IDF权重矩阵 
         
            看名字你也应该知道： 
            Tfidf-Transformer + Count-Vectorizer  = Tfidf-Vectorizer 
            下面的代码一步到位，把上面的两个步骤一次性全部完成 
         
            值得注意的是，CountVectorizer()和TfidfVectorizer()里面都有一个成员叫做vocabulary_(后面带一个下划线) 
            这个成员的意义，与我们之前在构建Bunch对象时提到的自己定义的那个vocabulary的意思是一样的，只不过一个是私有成员，一个是外部输入，原则上应该保持一致。
            显然，我们在第19行中创建tfidfspace中定义的vocabulary就应该被赋值为这个vocabulary_ 
        '''

    else:
        vectorizer = TfidfVectorizer(stop_words=stpwrdlst, sublinear_tf=True, max_df=0.45, min_df=0.1)  # 该过程创造了词典索引
        tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)
        tfidfspace.vocabulary = vectorizer.vocabulary_
        print("TF-IDF词向量空间的维度（即特征词的个数）为" + str(len(tfidfspace.vocabulary)))

    Tools.writebunchobj(space_path, tfidfspace)
    print("TF-IDF词向量空间实例创建成功！！！")


if __name__ == '__main__':
    # 当前工作路径
    working_directory = os.getcwd()
    cwd = working_directory + "/中文文本分类python3.6/chinese_text_classification-master/"
    stopword_path = cwd + "large_train_word_bag/hlt_stop_words.txt"
    bunch_path = cwd + "large_train_word_bag/train_set.dat"
    space_path = cwd + "large_train_word_bag/tfdifspace.dat"
    vector_space(stopword_path, bunch_path, space_path)

    bunch_path = cwd + "large_test_word_bag/test_set.dat"
    space_path = cwd + "large_test_word_bag/testspace.dat"
    train_tfidf_path = cwd + "large_train_word_bag/tfdifspace.dat" # test用train的词向量空间（vocabulary）
    vector_space(stopword_path, bunch_path, space_path, train_tfidf_path)
