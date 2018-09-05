import jieba
import jieba.analyse
import numpy as np
import pandas as pd
import chardet
from collections import Counter
from sklearn.cluster import KMeans
# from sklearn import feature_extraction
from gensim.models import LdaModel,TfidfModel,LsiModel
from gensim import similarities
from gensim import corpora
from matplotlib import pyplot as plt

def create_data(cut):#构建数据，先后使用doc2bow和tfidf model对文本进行向量表示
    sentences = []
    sentence_dict={}
    count=0
    for line in cut:
        sentences.append(line[1])
        tempStr=''
        for temp in line[1]:
           tempStr = tempStr+temp+' ' 
        sentence_dict[count]=tempStr[:-1]
        count+=1
        
   
    #对文本进行处理，得到文本集合中的词表
    dictionary = corpora.Dictionary(sentences)
    #利用词表，对文本进行cbow表示
    corpus = [dictionary.doc2bow(text) for text in sentences]
    #利用cbow，对文本进行tfidf表示
    tfidf=TfidfModel(corpus)
    corpus_tfidf=tfidf[corpus]
    return sentence_dict,dictionary,corpus,corpus_tfidf
def lda_model(sentence_dict,dictionary,corpus,corpus_tfidf,cluster_keyword_lda):#使用lda模型，获取主题分布   
    lda = LdaModel(corpus=corpus_tfidf, id2word=dictionary, num_topics=5)
    f_keyword = open(cluster_keyword_lda, 'w+')
    for topic in lda.print_topics(5,10):
        print('****'*5)
        words=[]
        for word in topic[1].split('+'):
            word=word.split('*')[1].replace(' ','')
            words.append(word)
        f_keyword.write(str(topic[0])+'\t'+','.join(words)+'\n')
    #利用lsi模型，对文本进行向量表示，这相当于与tfidf文档向量表示进行了降维，维度大小是设定的主题数目  
    corpus_lda = lda[corpus_tfidf]
    for doc in corpus_lda:
        print (len(doc),doc)
    return lda

def lsi_model(sentence_dict,dictionary,corpus,corpus_tfidf,cluster_keyword_lsi):#使用lsi模型，获取主题分布
    lsi = LsiModel(corpus=corpus_tfidf, id2word=dictionary, num_topics=5)
    f_keyword = open(cluster_keyword_lsi, 'w+')
    for topic in lsi.print_topics(5,10):
        print(topic[0])
        words=[]
        for word in topic[1].split('+'):
            word=word.split('*')[1].replace(' ','')
            words.append(word)
        f_keyword.write(str(topic[0])+'\t'+','.join(words)+'\n')
   
    return lsi


#with open('fileNew.csv', 'rb') as f:
#    result = chardet.detect(f.read())

df_data = pd.read_csv("fileNew.csv", names=['group_name','user_name', 'chat_time', 'content'])

df_data = df_data.drop([0])
user_name = df_data.user_name.values
group_name = df_data.group_name.values
time = pd.to_datetime(df_data.chat_time.values, format='%Y%m%d %H:%M')
content = df_data.content.values

#time_day = time.map(lambda x: x.strftime('%Y-%m-%d'))  # 此处大小写有影响
#time_hour = time.map(lambda x: x.strftime('%H'))
#
#res_time_hour = Counter(time_hour)
#res_time_day = Counter(time_day)
#res_user = Counter(user_name).most_common(25)

## 数据拼接
#string = ''
#for i in content:
#    if i!=i:
#        continue
#    # 判断nan值
#    # https: // blog.csdn.net / jpbirdy / article / details / 52333301
#    string = string + i
#
#
# 去停用词, 取关键词
#jieba.analyse.set_stop_words("stop_word.txt")
#tags = jieba.analyse.extract_tags(string, topK=30)

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
    l.append(list(set(jieba.cut(i))-set(stopkey)))
#    l.append(jieba.lcut(i))
    cut.append(l)


cluster_keyword_lda = './cluster_keywords_lda.txt'
cluster_keyword_lsi = './cluster_keywords_lsi.txt'
sentence_dict,dictionary,corpus,corpus_tfidf=create_data(cut)
lsi_model(sentence_dict,dictionary,corpus,corpus_tfidf,cluster_keyword_lsi)
lda_model(sentence_dict, dictionary, corpus, corpus_tfidf,cluster_keyword_lda)


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














