import jieba
import jieba.analyse
import numpy as np
import pandas as pd
import chardet
from collections import Counter
from sklearn.cluster import KMeans
# from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from matplotlib import pyplot as plt

def tfidf_vector(corpus_path):
    corpus_train = []
    # 利用train-corpus提取特征
    target_train = []
    for line in cut:
        target_train.append(str(line[0]))
        corpus_train.append(str(line[1]))
    print("build train-corpus done!!")
    count_v1 = CountVectorizer(max_df=0.4,stop_words=stopkey)
    counts_train = count_v1.fit_transform(corpus_train)

    word_dict = {}
    for index, word in enumerate(count_v1.get_feature_names()):
        word_dict[index] = word

    print("the shape of train is " + repr(counts_train.shape))
    tfidftransformer = TfidfTransformer()
    tfidf_train = tfidftransformer.fit(counts_train).transform(counts_train)
    return tfidf_train, word_dict


def cluster_kmeans(tfidf_train, word_dict, cluster_docs, cluster_keywords, num_clusters):  # K均值分类
    f_docs = open(cluster_docs, 'w+')
    km = KMeans(n_clusters=num_clusters)
    km.fit(tfidf_train)
    clusters = km.labels_.tolist() #每一个句子的所属类别,3456个
    cluster_dict = {}
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    temp = km.cluster_centers_
    doc = 1
    for cluster in clusters:
        f_docs.write(str(str(doc)) + ',' + str(cluster) + '\n')
        doc += 1
        if cluster not in cluster_dict:
            cluster_dict[cluster] = 1
        else:
            cluster_dict[cluster] += 1
    f_docs.close()
    cluster = 1

    f_clusterwords = open(cluster_keywords, 'w+')
    for ind in order_centroids:  # 每个聚类选 10 个词
        words = []
        for index in ind[:8]:
            words.append(word_dict[index])
        print(cluster, ','.join(words))
        f_clusterwords.write(str(cluster) + '\t' + ','.join(words) + '\n')
        cluster += 1
        print('*****' * 5)
    f_clusterwords.close()


#with open('fileNew.csv', 'rb') as f:
#    result = chardet.detect(f.read())

df_data = pd.read_csv("fileNew_mini.csv", names=['group_name','user_name', 'chat_time', 'content'])

df_data = df_data.drop([0])
user_name = df_data.user_name.values
#group_name = df_data.group_name.values
#time = pd.to_datetime(df_data.chat_time.values, format='%Y%m%d %H:%M')
content = df_data.content.values

#time_day = time.map(lambda x: x.strftime('%Y-%m-%d'))  # 此处大小写有影响
#time_hour = time.map(lambda x: x.strftime('%H'))
#
#res_time_hour = Counter(time_hour)
#res_time_day = Counter(time_day)
#res_user = Counter(user_name).most_common(25)

# 数据拼接
string = ''
for i in content:
    if i!=i:
        continue
    # 判断nan值
    # https: // blog.csdn.net / jpbirdy / article / details / 52333301
    string = string + i


# 去停用词, 取关键词
jieba.analyse.set_stop_words("stop_word.txt")
tags = jieba.analyse.extract_tags(string, topK=40)

## 文本聚类
stopkey=[line.strip() for line in open('stop_word.txt',encoding="utf-8").readlines()]

cut = []
count = 1
for i in content:
    if i!=i:
        continue
    l=[]
    l.append(count)
    count=count+1
    # l.append(list(set(jieba.cut(i))-set(stopkey)))
    l.append(jieba.lcut(i))
    cut.append(l)


cluster_docs = "./cluster_result_document.txt"
cluster_keywords = "./cluster_result_keyword.txt"
num_clusters = 4
tfidf_train,word_dict=tfidf_vector(cut)
# tfidf_train size : 3456x5986, 3456为总分析的句子数,5986为所有关键词数
cluster_kmeans(tfidf_train,word_dict,cluster_docs,cluster_keywords,num_clusters)


# plot
#
#fig, ax = plt.subplots()
#plt.xlabel("时间")
#plt.ylabel('聊天记录数')
#plt.gcf().autofmt_xdate()
#plt.bar(res_time_day.keys(),res_time_day.values())
#
#show, count = [], 0
#for i in res_time_day.keys():
#    if count%7==0:
#        show.append(i)
#    count=count+1
#ax.set_xticklabels(show)

# 
#plt.xlabel("每小时")
#plt.ylabel('聊天记录数')
#plt.gcf().autofmt_xdate()
#plt.bar(res_time_hour.keys(),res_time_hour.values())

#from wordcloud import WordCloud
#
#strlist=''
#for tag in tags:
#    strlist=strlist+" "+tag
#wordcloud = WordCloud().generate(strlist)
#plt.imshow(wordcloud, interpolation='bilinear')














