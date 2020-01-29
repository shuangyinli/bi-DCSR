'''
Created on Apr 2, 2019

@author: shuangyinli
'''
# -*- coding: utf-8 -*-
import re
import operator

import re
import os
import os.path
import sys
import operator
import multiprocessing
from multiprocessing import Process
import nltk
from nltk.corpus import words as ws
from nltk.stem import WordNetLemmatizer
import time 
from progressbar import ProgressBar

# keep the cleaned dictionay from DictionaryOrignal
DICTIONARY=[]
# keep the orignal dictionay
finalorignalDicSet={}
stopwords = {}
MIN_sentence = 0
woDic = set(ws.words())

def readstopwords(stopwordsfile):
    return {}.fromkeys([ line.rstrip() for line in open(stopwordsfile, 'r') ])

def remove(abstract):
    abstract = re.sub("[^A-Za-z]", ' ', abstract)
    ###remove a single word
    abstract=re.sub(' [a-zA-Z] ', ' ', abstract)
    abstract=re.sub('^[ ]*[a-zA-Z] ', ' ', abstract) 
    abstract=re.sub(' [a-zA-Z][ ]*$', ' ', abstract)
    abstract =  ' '.join(abstract.split())
    abstract = abstract.lower()
    return abstract

def split_average_data(corpus, thread_no):
    fn = len(corpus)//thread_no
    rn = len(corpus)%thread_no
    ar = [fn+1]*rn+ [fn]*(thread_no-rn)
    si = [i*(fn+1) if i<rn else (rn*(fn+1)+(i-rn)*fn) for i in range(thread_no)]
    corpusSplitlist = [corpus[si[i]:si[i]+ar[i]] for i in range(thread_no)]
    return corpusSplitlist

def run_multiprocessesDictionary(inputfile, workers_no, outpath):
    workers = []
    inputsourceDocumentslist = []
    for line in inputfile:
        inputsourceDocumentslist.append(line)
    
    corpusSplitlist = split_average_data(inputsourceDocumentslist, workers_no)
    
    for dataSplit in corpusSplitlist:
        worker = Process(target=MakeDictionary, args=(dataSplit,outpath,))
        worker.start()
        workers.append(worker)
    
    for w in workers:
        w.join()
        
## get the dictionany   not use
def OrignalDictionaryFunction(text):
    line = remove(text)
    for word in line.split():
        word = WordNetLemmatizer().lemmatize(word.rstrip().lstrip())
        if word not in ws.words():
            continue
        if word.rstrip() == '':
            continue
        if word in stopwords:
            continue
        if word not in finalorignalDicSet:
            finalorignalDicSet.setdefault(word,1)
        else:
            finalorignalDicSet[word] += 1

def MakeDictionary(inputfilelist, oDir):
    pid = os.getpid()
    partDictionaryfile = open(oDir+"partDictionary"+str(pid),"w", encoding = "utf-8")
    wn = WordNetLemmatizer()
    docno = 0
    partDictionary ={}
    for line in inputfilelist:
        text = line.split("##")[1]
        #OrignalDictionaryFunction(text)
        lines = remove(text)
        for w in lines.split():
            word = wn.lemmatize(w.rstrip().lstrip())
            if word not in woDic:
                continue
            if word.rstrip() == '':
                continue
            if word in stopwords:
                continue
            if word not in partDictionary:
                partDictionary.setdefault(word,1)
            else:
                partDictionary[word] += 1
        docno += 1
        print("doc " + str(docno) + " is done")
        
    print(str(pid)+ " the partDictionary size is " + str(len(partDictionary)))
    print(str(pid)+ " part Dictionary is over. now save the dictionary.")
    
    for key, value in partDictionary.items():
        partDictionaryfile.write(str(key) +":" +str(value)+"\n")
    partDictionaryfile.close()

def cleanDictionary(oDir):
    
    #read all the partDic
    path_list = os.listdir(oDir)
    partDicfile = []
    for path in path_list:
        if "partDictionary" in path:
            partDicfile.append(path)
 
    for dicpath in partDicfile:
        dicfileopen = open(oDir+dicpath,"r", encoding = "utf-8")
        print(oDir+dicpath)
        for line in dicfileopen:
            key = line.split(":")[0]
            value = int(line.split(":")[1])
            if key in finalorignalDicSet:
                finalorignalDicSet[key] += value
            else:
                finalorignalDicSet.setdefault(key, value)
                
    print("Now get the top 10000  frequency words...")
    #a = sorted(finalorignalDicSet.items(), key=lambda d: d[1], reverse=True)[0:10000]
    
    #m = dict(a)
    
    for key, value in finalorignalDicSet.items():
        if value > 15:
            DICTIONARY.append(key)
    print("the DICTIONARY size is " + str(len(DICTIONARY)))
    
def SaveDictionary(dicfile):
    print("now save the dictionary.")
    for word in DICTIONARY:
        dicfile.write(str(DICTIONARY.index(word)) +":" +word)
        dicfile.write("\n")
    dicfile.close()

if __name__ == '__main__':
    
    # /Users/huihui/Data/wiki/ese_out/total-labeled-wikipages5
    # /Users/huihui/Data/wiki/ese_out/stopwords
    workers_no = int(sys.argv[1])
    inputfile = open(sys.argv[2],"r", encoding = "utf-8")
    stopwordsfile = sys.argv[3]
    oDir = sys.argv[4]
    if oDir.endswith("/") is False:
        oDir = oDir+"/"
    
    print("read the stopwords..")
    stopwords = readstopwords(stopwordsfile)
    
    dicfile = open(oDir+"DICTIONARY.txt","w", encoding = "utf-8")
  
    print("run_multiprocessesDictionary...")
    run_multiprocessesDictionary(inputfile,workers_no, oDir)
    cleanDictionary(oDir)
    SaveDictionary(dicfile)
    
    pass