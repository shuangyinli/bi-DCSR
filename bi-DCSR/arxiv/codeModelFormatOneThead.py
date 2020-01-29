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

def code(inputfile, oDir):
    
    inputsourceDocumentslist = []
    for line in inputfile:
        linetemp = line.split("##")[1].rstrip().lstrip()
        label = line.split("##")[0].rstrip().lstrip()
        inputsourceDocumentslist.append((label, linetemp))
        
    random.shuffle(inputsourceDocumentslist)
    
    pid = os.getpid()
    eseCodedfile = open(oDir+"eseCodedfile"+str(pid),"w", encoding = "utf-8")
    #eseCodedfile_shortkeys = open(oDir+"shortkeys_eseCoded"+str(pid),"w", encoding = "utf-8")
    #LDACodedfile = open(oDir+"LDACodedfile"+str(pid),"w", encoding = "utf-8")
    esecodelabel = open(oDir+"eseCodedfile_label"+str(pid),"w", encoding = "utf-8")
    esesource = open(oDir+"esesource"+str(pid),"w", encoding = "utf-8")
    #keep the middel sentences
    esemiddlesentencesource = open(oDir+"esemiddlesentencesource"+str(pid),"w", encoding = "utf-8")
    
    documentno = 0
    
    for item in inputsourceDocumentslist:
        
        label, line = item[0], item[1]
        # remove empty docs
        ldadocument = remove(line)
        docwordno, ldawordlist = codetext(ldadocument)
        if docwordno ==0:
            continue
        
        #7 # 5 123 342 555 110 34 @ 8 11:2 23:3 55:34 1345:1 10:1 44:2 19:9 88:66 # ... #
        #[the number of sentences] # [number of keywords] [keywordsid] ... [keywordsid] @ [number of words] [wordid:wordcount] [wordid:wordcount] [wordid:wordcount] # ... #
        #ese
        
        sentences = []
        originalsentences = removecontaindot(line).split(".")
        
        if len(originalsentences) < 5:
            continue
        
        for se in originalsentences:
            wordno, wordlist = codetext(se)
            if wordno < 10:
                continue
            #keywordlist = codeselectedkeywords(se)
            sentences.append((wordno, wordlist,se))
        
        senno = len(sentences)
        
        if senno < 6:
            continue
        
        eseCodedfile.write(str(senno)+" ")

        for wordno, wordlist, se in sentences:
            eseCodedfile.write("# ")

            allkeywordlist = list(wordlist)
            eseCodedfile.write(str(len(allkeywordlist)))
            for keys in allkeywordlist:
                eseCodedfile.write(" "+str(keys))
            
            eseCodedfile.write(" @ "+str(len(wordlist)) + " ")
            
            for key, value in wordlist.items():
                eseCodedfile.write(str(key) + ":" + str(value) + " ")
                
        eseCodedfile.write("\n")
        eseCodedfile.flush()
        
        esemiddlesentencesource.write(sentences[3][2].rstrip().lstrip() + "\n")
        esemiddlesentencesource.flush()
        
        esecodelabel.write(label.rstrip().lstrip() + "\n")
        esecodelabel.flush()
        
        esesource.write(line.rstrip().lstrip() + "\n")
        esesource.flush()
        
        documentno = documentno +1
        print("now begin to doc :" + str(documentno))
    
    eseCodedfile.close()
    esecodelabel.close()
    esesource.close()
    esemiddlesentencesource.close()

        
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
    
    #  select out 10000 documents, and each document contains more than 5 sentences
    #  
    # 
    inputfile = open(sys.argv[1],"r", encoding = "utf-8")
    stopwordsfile = sys.argv[2]
    oDir = sys.argv[3]
    if oDir.endswith("/") is False:
        oDir = oDir+"/"
    
    print("read the stopwords..")
    stopwords = readstopwords(stopwordsfile)

    dicfile = open(oDir+"DICTIONARY.txt","r", encoding = "utf-8")
    readDICTIONARY(dicfile)
    
    print("run_multiprocesses ...")
    code(inputfile, oDir)

    pass