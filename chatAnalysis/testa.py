# -*- coding: utf-8 -*-

import jieba
#
#seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
#print("Full Mode: " + "/ ".join(seg_list))  # 全模式
stopkey=[line.strip() for line in open('stop_word.txt').readlines()] 