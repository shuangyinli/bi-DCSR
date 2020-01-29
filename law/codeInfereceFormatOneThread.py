'''
Created on Apr 24, 2019

@author: shuangyinli
'''

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

import math
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import re

import nltk
import random

# keep the cleaned dictionay from DictionaryOrignal
DICTIONARY={}
# keep the orignal dictionay
finalorignalDicSet={}
stopwords = {}
MIN_sentence = 0

wn = WordNetLemmatizer()

def readstopwords(stopwordsfile):
    return {}.fromkeys([ line.rstrip() for line in open(stopwordsfile, 'r') ])

def removecontaindot(abstract):
    return abstract

def remove(abstract):
    return abstract

#sentence and doc
def codetext(text):
    wordset ={}
    textwords = remove(text)
    for word in textwords.split():
        word = wn.lemmatize(word.rstrip().lstrip())
        if word in DICTIONARY:
            wordset.setdefault(DICTIONARY[word],0)
            wordset[DICTIONARY[word]] += 1
    return len(wordset), wordset



def onethreadprocess(inputfile, oDir):
    inputsourceDocumentslist = []
    for line in inputfile:
        linetemp = line
        inputsourceDocumentslist.append(linetemp)
        
    code(inputsourceDocumentslist, oDir)
    pass

def code(inputsourceDocuments, oDir):
    pid = os.getpid()
    eseCodedfile = open(oDir+"wiki_test_ESE_Coded"+str(pid),"w", encoding = "utf-8")
    
    documentno = 0
    for line in inputsourceDocuments:
        
        #LDA
        ldadocument = remove(line)
        docwordno, ldawordlist = codetext(ldadocument)
        if docwordno ==0:
            print("ERROR, one test sentence is empty!!!!")
            continue
        
        #7 # 5 123 342 555 110 34 @ 8 11:2 23:3 55:34 1345:1 10:1 44:2 19:9 88:66 # ... #
        #[the number of sentences] # [number of keywords] [keywordsid] ... [keywordsid] @ [number of words] [wordid:wordcount] [wordid:wordcount] [wordid:wordcount] # ... #
        #ese
        #sentences = removecontaindot(line).split(".")
        ##
        
        sentences = []
        originalsentences = removecontaindot(line).split("。")
        for se in originalsentences:
            wordno, wordlist = codetext(se)
            if wordno < 1:
                print("ERROR, one test sentence is empty, too!!!!")
                continue
            _, keywordlist = codetext(se)
            sentences.append((wordno, wordlist,keywordlist))
        
        senno = len(sentences)
        eseCodedfile.write(str(senno)+" ")
        
        for wordno, wordlist, keywordlist in sentences:
            eseCodedfile.write("# ")
            
            # make keywords
            allkeywordlist = list(wordlist)
            eseCodedfile.write(str(len(allkeywordlist)))
            
            for keys in allkeywordlist:
                eseCodedfile.write(" "+str(keys))

                
            eseCodedfile.write(" @ "+str(len(wordlist)) + " ")
           
            
            for key, value in wordlist.items():
                eseCodedfile.write(str(key) + ":" + str(value) + " ")
                
                
        eseCodedfile.write("\n") 
        eseCodedfile.flush()
        
        documentno = documentno +1
        if documentno % 10000 ==1:
            print("now begin to doc :" + str(documentno))
            
    eseCodedfile.close()
    print("now end to doc :" + str(documentno)) 


def split_average_data(corpus, thread_no):
    fn = len(corpus)//thread_no
    rn = len(corpus)%thread_no
    ar = [fn+1]*rn+ [fn]*(thread_no-rn)
    si = [i*(fn+1) if i<rn else (rn*(fn+1)+(i-rn)*fn) for i in range(thread_no)]
    corpusSplitlist = [corpus[si[i]:si[i]+ar[i]] for i in range(thread_no)]
    return corpusSplitlist

def run_multiprocesses(inputfile, workers_no, outpath):
    workers = []
    inputsourceDocumentslist = []
    for line in inputfile:
        linetemp = line
        inputsourceDocumentslist.append(linetemp)
    
    corpusSplitlist = split_average_data(inputsourceDocumentslist, workers_no)
    
    for dataSplit in corpusSplitlist:
        worker = Process(target=code, args=(dataSplit,outpath,))
        worker.start()
        workers.append(worker)
    
    for w in workers:
        w.join()
        
def readDICTIONARY(dicfile):
    for line in dicfile:
        index = int(line.split(":")[0])
        word = line.split(":")[1].rstrip().lstrip()
        DICTIONARY.setdefault(word,index)

def combinefiles(oDir,headname,dicfiled):
    #read all the partDic
    alllines = []
    
    path_list = os.listdir(oDir)
    partDicfile = []
    for path in path_list:
        if headname in path:
            partDicfile.append(path)
    
    for dicpath in partDicfile:
        dicfileopen = open(oDir+dicpath,"r", encoding = "utf-8")
        print(oDir+dicpath)
        for line in dicfileopen:
            key = line.rstrip().lstrip()
            alllines.append(key)
            
    random.shuffle(alllines)
    
    print("\n now save the codes.")
    print("We have "+str(len(alllines))+". \n")
    for sen in alllines:
        dicfiled.write(sen+"\n")
    dicfiled.close()


if __name__ == '__main__':
    
    # /Users/huihui/Data/wiki/ese_out/total-labeled-wikipages5
    # /Users/huihui/Data/wiki/ese_out/stopwords
   
    inputfile = open(sys.argv[1],"r", encoding = "utf-8")
    stopwordsfile = sys.argv[2]
    oDir = sys.argv[3]
    if oDir.endswith("/") is False:
        oDir = oDir+"/"
    
    print("read the stopwords..")
    stopwords = readstopwords(stopwordsfile)

    dicfile = open(oDir+"DICTIONARY.txt","r", encoding = "utf-8")
    readDICTIONARY(dicfile)
    
    print("run_oneprocesses ...")
    #run_multiprocesses(inputfile, workers_no, oDir)
    onethreadprocess(inputfile, oDir)

    #dicfileese = open(oDir+"wiki_test_ESE"+".txt","w", encoding = "utf-8")
    #combinefiles(oDir,"eseCodedfile",dicfileese)
    
    pass