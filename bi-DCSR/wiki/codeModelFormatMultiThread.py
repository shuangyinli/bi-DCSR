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
    abstract = re.sub("[^A-Za-z.]", ' ', abstract)
    ###remove a single word
    abstract=re.sub(' [a-zA-Z] ', ' ', abstract)
    abstract=re.sub('^[ ]*[a-zA-Z] ', ' ', abstract) 
    abstract=re.sub(' [a-zA-Z][ ]*$', ' ', abstract)
    abstract =  ' '.join(abstract.split())
    abstract = abstract.lower()
    return abstract

def remove(abstract):
    abstract = re.sub("[^A-Za-z]", ' ', abstract)
    ###remove a single word
    abstract=re.sub(' [a-zA-Z] ', ' ', abstract)
    abstract=re.sub('^[ ]*[a-zA-Z] ', ' ', abstract) 
    abstract=re.sub(' [a-zA-Z][ ]*$', ' ', abstract)
    abstract =  ' '.join(abstract.split())
    abstract = abstract.lower()
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

#sentence
def codeselectedkeywords(text):
    keywordlist = []

    textwords = remove(text)
    tokens = word_tokenize(textwords)
    cantitateskeywordslist = pos_tag(tokens)
    for keyword in cantitateskeywordslist:
        if keyword[1].startswith('V') or keyword[1].startswith('N') or keyword[1].startswith('J') or keyword[1].startswith('R'):
            word = wn.lemmatize(keyword[0].rstrip().lstrip())
            if word in DICTIONARY:
                keywordlist.append(DICTIONARY[word])
            
    return list(set(keywordlist))

def code(inputsourceDocuments, oDir):
    pid = os.getpid()
    eseCodedfile = open(oDir+"eseCodedfile"+str(pid),"w", encoding = "utf-8")
    eseCodedfile_shortkeys = open(oDir+"shortkeys_eseCoded"+str(pid),"w", encoding = "utf-8")
    LDACodedfile = open(oDir+"LDACodedfile"+str(pid),"w", encoding = "utf-8")
    
    documentno = 0
    for line in inputsourceDocuments:
        
        #LDA
        ldadocument = remove(line)
        docwordno, ldawordlist = codetext(ldadocument)
        if docwordno ==0:
            continue
        if docwordno !=0:
            LDACodedfile.write(str(docwordno)+" ")
            for dw in ldawordlist:
                LDACodedfile.write(str(dw)+":" + str(ldawordlist[dw]) +" ")
            LDACodedfile.write("\n")
        LDACodedfile.flush()
        
        #7 # 5 123 342 555 110 34 @ 8 11:2 23:3 55:34 1345:1 10:1 44:2 19:9 88:66 # ... #
        #[the number of sentences] # [number of keywords] [keywordsid] ... [keywordsid] @ [number of words] [wordid:wordcount] [wordid:wordcount] [wordid:wordcount] # ... #
        #ese
        
        #sentences = removecontaindot(line).split(".")
        ##
        
        sentences = []
        originalsentences = removecontaindot(line).split(".")
        for se in originalsentences:
            wordno, wordlist = codetext(se)
            if wordno < 1:
                continue
            keywordlist = codeselectedkeywords(se)
            sentences.append((wordno, wordlist,keywordlist))
        
        senno = len(sentences)
        eseCodedfile.write(str(senno)+" ")
        eseCodedfile_shortkeys.write(str(senno)+ " ")
        for wordno, wordlist, keywordlist in sentences:
            eseCodedfile.write("# ")
            eseCodedfile_shortkeys.write("# ")
            # make keywords
            allkeywordlist = list(wordlist)
            eseCodedfile.write(str(len(allkeywordlist)))
            eseCodedfile_shortkeys.write(str(len(keywordlist)))
            for keys in allkeywordlist:
                eseCodedfile.write(" "+str(keys))
            for skeys in keywordlist:
                eseCodedfile_shortkeys.write(" "+str(skeys))
                
            eseCodedfile.write(" @ "+str(len(wordlist)) + " ")
            eseCodedfile_shortkeys.write(" @ "+str(len(wordlist)) + " ")
            
            for key, value in wordlist.items():
                eseCodedfile.write(str(key) + ":" + str(value) + " ")
                eseCodedfile_shortkeys.write(str(key) + ":" + str(value)+ " ")
                
        eseCodedfile.write("\n")
        eseCodedfile_shortkeys.write("\n")
        eseCodedfile.flush()
        eseCodedfile_shortkeys.flush()
        
        documentno = documentno +1
        print("now begin to doc :" + str(documentno)) 
     
    LDACodedfile.close()
    eseCodedfile.close()
    eseCodedfile_shortkeys.close()


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
        linetemp = line.split("##")[1]
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
    workers_no = int(sys.argv[1])
    inputfile = open(sys.argv[2],"r", encoding = "utf-8")
    stopwordsfile = sys.argv[3]
    oDir = sys.argv[4]
    if oDir.endswith("/") is False:
        oDir = oDir+"/"
    
    print("read the stopwords..")
    stopwords = readstopwords(stopwordsfile)

    dicfile = open(oDir+"DICTIONARY.txt","r", encoding = "utf-8")
    readDICTIONARY(dicfile)
    
    print("run_multiprocesses ...")
    run_multiprocesses(inputfile, workers_no, oDir)
    
    dicfilelda = open(oDir+"wikiLDA"+".txt","w", encoding = "utf-8")
    combinefiles(oDir,"LDACodedfile",dicfilelda)

    dicfileese = open(oDir+"wikiESE"+".txt","w", encoding = "utf-8")
    combinefiles(oDir,"eseCodedfile",dicfileese)
    
    dicfileldashort = open(oDir+"wikishortESE"+".txt","w", encoding = "utf-8")
    combinefiles(oDir,"shortkeys_eseCoded",dicfileldashort)

    
    pass