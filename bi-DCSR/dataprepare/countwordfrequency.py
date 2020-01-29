'''
Created on Apr 29, 2019

@author: shuangyinli
'''

from tqdm import tqdm
import math

countdic = {}

idfdic = {}

N= 0

NX = {}



datalist = open("/Users/huihui/experiment/ese/wiki/out/wikiESE.txt", "r", encoding = "utf-8").readlines()

wordset_list = []

for doc in tqdm(datalist):
    N +=1
    sentencelist = doc.split("#")[1:]
    sentences = []
    wordset = {}
    
    for sen in sentencelist:
        keywordlist = sen.lstrip().rstrip().split("@")[0].lstrip().rstrip().split()[1:]
        wordlist = sen.lstrip().rstrip().split("@")[1].lstrip().rstrip().split()
        wordlist_ptr_ = wordlist[1:]
        words_ptr = []
        words_cnt_ptr = []
        for item in wordlist_ptr_:
            key = int(item.split(":")[0])
            value = int(item.split(":")[1])
            words_ptr.append(int(key))
            words_cnt_ptr.append(int(value))
            if key in wordset:
                wordset[key] += value
            else:
                wordset.setdefault(key, value)
            
    for key, value in wordset.items():
        if key in NX:
            NX[key] += 1
        else:
            NX.setdefault(key,1)
            
    wordset_list.append(wordset)
    
    
# compute idf    

print("compute idf ")
for key,value in NX.items():
    NX[key] = math.log((N+1)/(value+1) +1)


#compute the tf-idf

print("compute tf-idf ")
doc_select_words = []

for item in wordset_list:
    
    tfidf_select = []
    for key, value in item.items():
        item[key] = value * NX[key]
    a = sorted(item.items(), key=lambda d: d[1], reverse=True)
    
    a_len = len(a)
    
    count = int(a_len*0.5)
    
    temp = a[:count]
    
    for t in temp:
        tfidf_select.append(t[0])
        
    
    doc_select_words.append(tfidf_select)
    
    
datalist2 = open("/Users/huihui/experiment/ese/wiki/out/wikiESE.txt", "r", encoding = "utf-8").readlines()

countdic = {}

docid = 0
for doc in tqdm(datalist):
    sentencelist = doc.split("#")[1:]
    sentences = []
    for sen in sentencelist:
        select_words = []
        
        wordlist = sen.lstrip().rstrip().split("@")[1].lstrip().rstrip().split()
        wordlist_ptr_ = wordlist[1:]
        words_ptr = []
        for item in wordlist_ptr_:
            key = int(item.split(":")[0])
            words_ptr.append(key)
            
        for key in words_ptr:
            if key in doc_select_words[docid]:
                select_words.append(key)
        
        if len(select_words) in countdic:
            countdic[len(select_words)] +=1
        else:
            countdic.setdefault(len(select_words),1)
    docid += 1
print(countdic)
print(countdic[12] + countdic[13]+countdic[14]+countdic[15]+countdic[16]+countdic[17]+countdic[18]+countdic[19]+countdic[20]+countdic[21]+countdic[22]+countdic[23]+countdic[24])