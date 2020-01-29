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
from _ast import If

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

def onethreadprocess(inputfile, oDir):
    inputsourceDocumentslist = []
    labelslist = []
    for line in inputfile:
        linetemp = line.split("##")[1].lstrip().rstrip()
        label = line.split("##")[0].lstrip().rstrip()
        inputsourceDocumentslist.append(linetemp.lstrip().rstrip())
        labelslist.append(label.lstrip().rstrip())
    
    code(inputsourceDocumentslist, oDir, labelslist)
    

def code(inputsourceDocuments, oDir, labelslist):
    pid = os.getpid()
    eseCodedfile = open(oDir+"eseCodedfile"+str(pid),"w", encoding = "utf-8")
    eseOriginalfile = open(oDir+"eseOriginalfile"+str(pid),"w", encoding = "utf-8")
    #eseCodedfile_shortkeys = open(oDir+"shortkeys_eseCoded"+str(pid),"w", encoding = "utf-8")
    #LDACodedfile = open(oDir+"LDACodedfile"+str(pid),"w", encoding = "utf-8")
    
    print("num of docs is :" + str(len(inputsourceDocuments)) + " labels : " + str(len(labelslist)))
    
    documentno = 0
    for line in inputsourceDocuments:
        
        sen_doc_label =labelslist[documentno]        
        sentences = []
        
        originalsentences = removecontaindot(line).split(".")
        originalsentenceswithstopwords = line.split(".")
        sennoindex = len(originalsentences)
        originalsentenceswithstopwordsindex = len(originalsentenceswithstopwords)
        
        if sennoindex != originalsentenceswithstopwordsindex:
            print("error 1")
            exit(0)
        
        for index in range(sennoindex):
            wordno, wordlist = codetext(originalsentences[index])
            if wordno < 1:
                continue
            
            sentences.append((wordno, wordlist,originalsentenceswithstopwords[index]))
        
        
        senno = len(sentences)
        eseCodedfile.write(str(senno)+" ")
        eseOriginalfile.write(sen_doc_label.lstrip().rstrip() + "@" )
        
        for wordno, wordlist, orsens in sentences:
            eseCodedfile.write("# ")
            eseOriginalfile.write(orsens.lstrip().rstrip() +"##")
            # make keywords
            allkeywordlist = list(wordlist)
            eseCodedfile.write(str(len(allkeywordlist)))
            
            for keys in allkeywordlist:
                eseCodedfile.write(" "+str(keys)) 
                
            eseCodedfile.write(" @ "+str(len(wordlist)) + " ")
             
            
            for key, value in wordlist.items():
                eseCodedfile.write(str(key) + ":" + str(value) + " ")
                
        eseCodedfile.write("\n")
        eseOriginalfile.write("\n")
        eseCodedfile.flush()
        eseOriginalfile.flush()
         
        
        documentno = documentno +1
        if documentno % 10000 ==1:
            print("now begin to doc :" + str(documentno)) 
     
     
    eseCodedfile.close()
    eseOriginalfile.close()
    print("now end to doc :" + str(documentno)) 

        
def readDICTIONARY(dicfile):
    for line in dicfile:
        index = int(line.split(":")[0])
        word = line.split(":")[1].rstrip().lstrip()
        DICTIONARY.setdefault(word,index)


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
    
    #dicfilelda = open(oDir+"wikiLDA"+".txt","w", encoding = "utf-8")
    #combinefiles(oDir,"LDACodedfile",dicfilelda)

    #dicfileese = open(oDir+"wikiESE"+".txt","w", encoding = "utf-8")
    #combinefiles(oDir,"eseCodedfile",dicfileese)
    
    #dicfileldashort = open(oDir+"wikishortESE"+".txt","w", encoding = "utf-8")
    #combinefiles(oDir,"shortkeys_eseCoded",dicfileldashort)

    
    pass