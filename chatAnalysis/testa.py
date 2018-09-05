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


#with open('BJ.csv', 'rb') as f:
#    result = chardet.detect(f.read())

df_data = pd.read_csv("BJ.csv", names=['content'])

df_data = df_data.drop([0])
#user_name = df_data.user_name.values
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

#
# 去停用词, 取关键词
jieba.analyse.set_stop_words("stop_word.txt")
tags = jieba.analyse.extract_tags(string, topK=30)

### 文本聚类
#stopkey=[line.strip() for line in open('stop_word.txt',encoding="utf-8").readlines()]
#
#cut = []
#count = 1
#for i in content:
#    if i!=i:
#        continue
#    l=[]
#    l.append(count)
#    count=count+1
#    # l.append(list(set(jieba.cut(i))-set(stopkey)))
#    l.append(jieba.lcut(i))
#    cut.append(l)
#
#
##cluster_docs = "./cluster_result_document.txt"
##cluster_keywords = "./cluster_result_keyword.txt"
##num_clusters = 6
##tfidf_train,word_dict=tfidf_vector(cut)
##cluster_kmeans(tfidf_train,word_dict,cluster_docs,cluster_keywords,num_clusters)
##
#
## plot
##
##fig, ax = plt.subplots()
##plt.xlabel("时间")
##plt.ylabel('聊天记录数')
##plt.gcf().autofmt_xdate()
##plt.bar(res_time_day.keys(),res_time_day.values())
##
##show, count = [], 0
##for i in res_time_day.keys():
##    if count%7==0:
##        show.append(i)
##    count=count+1
##ax.set_xticklabels(show)
#
## 
##plt.xlabel("每小时")
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














