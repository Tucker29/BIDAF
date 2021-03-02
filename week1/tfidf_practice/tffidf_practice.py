# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 00:44:10 2020

@author: Nan
"""
import json
import pandas as pd
import re 
from zhon.hanzi import punctuation
from nltk.corpus import stopwords
from nltk import tokenize  
from nltk.tokenize import word_tokenize
import string
import jieba 
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer





Count_vectorizer = CountVectorizer()
transformer = TfidfTransformer() 

class Pro_cn_text:
    def __init__(self):
        self.cn_stopwords = self.read_txt('nlp_source/baidu_stopwords.txt')

    def read_txt(self, f_path):
        with open(f_path, 'r', encoding='utf-8')as f:
            text = f.read()
            text = text.split('\n')
            text = [txt.strip() for txt in text]
        return text 
    
    def cn_cut_sentence(self, text):
        sentence_list = re.split(r'(\.|\!|\?|。|！|？|\.{6})', text)
        return sentence_list
    
    def cn_tokenize(self, sentence):
        seg_list = jieba.cut(sentence)
        return ",".join(seg_list).split(",")
          
    def cn_del_stopwords(self, w_list):  
        res_w = [word for word in w_list if word not in self.cn_stopwords ]
        return res_w 
    
    def remove_numbers(self, w_list): 
        renum_words = [word for word in w_list if not word.isnumeric()]
        new_words = [word for word in renum_words if not re.findall('-\d+',word)]
        return new_words  
    #去除中英文标点符号
    def remove_punctuation(self, w_list):
        new_words = [word for word in w_list if word not in punctuation and word not in string.punctuation]
        return new_words
    
    def cn_corpus(self, text):     
       text_words =""
       sentence_list = self.cn_cut_sentence(text.replace('\n',''))
       for sentence in sentence_list:
           w_list = self.cn_tokenize(sentence)
           seg_words = self.cn_del_stopwords(w_list)  
           ren_words = self.remove_numbers(seg_words)
           rep_words = self.remove_punctuation(ren_words) 
           new_words = " ".join(rep_words)        
           text_words += new_words + " "       
       return text_words
   
    def to_json(self, dic,f_path):
        with open(f_path, 'w') as f:
            json.dump(dic, f, indent=2, ensure_ascii=False)   
    
    
class Pro_en_text:
    def __init__(self):
        self.en_stopwords = stopwords.words('english')
    
    def en_tokenize(self, sentence):
        seg_list = jieba.cut(sentence)
        return ",".join(seg_list).split(",")
    
   
    def en_del_stopwords(self, w_list):  
        res_w = [word for word in w_list if word not in self.en_stopwords]
        return res_w 
    
    #去除文本中的数字 (主要针对分词后的英文，分词后的中文不存在)
    def remove_numbers(self, w_list): 
        renum_words = [word for word in w_list if not word.isnumeric()]
        new_words = [word for word in renum_words if not re.findall('-\d+',word)]
        return new_words
    
    def remove_punctuation(self, w_list):
        new_words = [word for word in w_list if word not in punctuation and word not in string.punctuation]
        return new_words
    
    def en_corpus(self, text):     
       text_words = ""
       sentence_list = tokenize.sent_tokenize(text.replace('\n',''))
       for sent in sentence_list:
           word_list = word_tokenize(sent)
           seg_words = self.en_del_stopwords(word_list)
           ren_words = self.remove_numbers(seg_words)
           rep_words = self.remove_punctuation(ren_words) 
           new_words = " ".join(rep_words)
           text_words += new_words + " "
       return text_words
   
    def to_json(self, dic,f_path):
        with open(f_path, 'w') as f:
            json.dump(dic, f, indent=2)   
   
def calculate_tf(corpus):
    vectorizer = CountVectorizer() 
    X = vectorizer.fit_transform(corpus)  
    words = vectorizer.get_feature_names()   
    df_word_tf = pd.DataFrame(X.toarray(),columns=words) 
    word_tf_sum = df_word_tf.sum().sort_values(ascending=False)
    return  word_tf_sum
    
    
def calculate_tfidf(corpus):    
    vectorizer = CountVectorizer()    
    X = vectorizer.fit_transform(corpus)  
    words = vectorizer.get_feature_names()   
    transformer = TfidfTransformer()  
    tfidf = transformer.fit_transform(X)  
    df_wordFreq = pd.DataFrame(tfidf.toarray(),columns=words) 
    wordFreqSum = df_wordFreq.sum().sort_values(ascending=False) #计算每个特征的总词频并按照降序进行排序
    return wordFreqSum

def calculate_tfidf2(corpus):    
    vectorizer = TfidfVectorizer()   
    X = vectorizer.fit_transform(corpus)  
    words = vectorizer.get_feature_names()    
    df_wordFreq = pd.DataFrame(X.toarray(),columns=words) 
    wordFreqSum = df_wordFreq.sum().sort_values(ascending=False) #计算每个特征的总词频并按照降序进行排序
    return wordFreqSum

 

if __name__ == '__main__':
    print('Processing........')
    
#    with open('datas/input/SQuAD/train-v2.0.json','r') as f:
#        SQuAD_dic = json.load(f)
#    SQuAD_data = SQuAD_dic['data']
#    SQuAD_df = pd.DataFrame(SQuAD_data)
#    SQuAD_df['contexts'] = SQuAD_df['paragraphs'].apply(lambda ps : " ".join([p['context'] for p in ps]))
#    pro_en_text = Pro_en_text()
#    SQuAD_corpus = SQuAD_df['contexts'].apply(pro_en_text.en_corpus)
#    SQuAD_tf = calculate_tf(SQuAD_corpus)
#    pro_en_text.to_json(SQuAD_tf.to_dict(),'datas/output/SQuAD/SQuAD_tf.json')
#    SQuAD_tfidf = calculate_tfidf(SQuAD_corpus)
#    pro_en_text.to_json(SQuAD_tfidf.to_dict(),'datas/output/SQuAD/SQuAD_tfidf.json')

    with open('datas/input/dureader_robust/train.json','r',encoding='UTF-8') as f:
        dr_dic = json.load(f)
    dr_data = dr_dic['data']
    dr_df = pd.DataFrame(dr_data[0]['paragraphs'])
    pro_cn_text = Pro_cn_text()
    dr_corpus = dr_df['context'].apply(pro_cn_text.cn_corpus)
    dr_tf = calculate_tf(dr_corpus)
    pro_cn_text.to_json(dr_tf.to_dict(),'datas/output/dureader_robust/dr_tf.json')
    dr_tfidf = calculate_tfidf(dr_corpus)
    pro_cn_text.to_json(dr_tfidf.to_dict(),'datas/output/dureader_robust/dr_tfidf.json')    
    print('finish...') 
        


