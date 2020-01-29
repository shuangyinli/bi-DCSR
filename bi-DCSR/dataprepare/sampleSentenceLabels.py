'''
Created on Apr 15, 2019

@author: shuangyinli
'''
# -*- coding: utf-8 -*-
import re
import math
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

DICTIONARY={}
stopwords = {}
wn = WordNetLemmatizer()

def readDICTIONARY(dicfile):
    for line in dicfile:
        index = int(line.split(":")[0])
        word = line.split(":")[1].rstrip().lstrip()
        DICTIONARY.setdefault(word,index)

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

def removecontaindot(abstract):
    abstract = re.sub("[^A-Za-z.]", ' ', abstract)
    ###remove a single word
    abstract=re.sub(' [a-zA-Z] ', ' ', abstract)
    abstract=re.sub('^[ ]*[a-zA-Z] ', ' ', abstract) 
    abstract=re.sub(' [a-zA-Z][ ]*$', ' ', abstract)
    abstract =  ' '.join(abstract.split())
    abstract = abstract.lower()
    return abstract

def codeword(word):
    tempword = word.rstrip().lstrip()
    if tempword not in stopwords:
        if tempword !="" and tempword != " ":
            tempword = wn.lemmatize(tempword)
    return tempword

def sample(openfile, labelfile, basicfile, forbackfile):
    docno = 0
    for line in openfile:
        print(docno)
        docno = docno +1
        label = line.split("##")[0].rstrip().lstrip()
        labelfile.write(label+"\n")
        sentence = ''
        foresentence = ''
        backsentence = ''
        sentences = line.split("##")[1].rstrip().lstrip().split(".")
        lensentences = len(sentences)
        index = 0
        sizeindex = 0
        for i in range(lensentences):
            if len(sentences[i]) > sizeindex:
                sizeindex = len(sentences[i])
                sentence=sentences[i]
                index = i
        
        if lensentences ==1:
            foresentence = sentences[index]
            backsentence = sentences[index]
        else:    
            if index != 0 and index !=lensentences-1:
                foresentence = sentences[index - 1]
                backsentence = sentences[index + 1]
            if index == 0:
                foresentence = sentences[index]
                backsentence = sentences[index + 1]
            if index == lensentences-1:
                foresentence = sentences[index-1]
                backsentence = sentences[index]
            
        sentence = remove(sentence)
        foresentence = remove(foresentence)
        backsentence = remove(backsentence)
        
        for word in sentence.split():
            tempword = codeword(word)
            if tempword in DICTIONARY:
                basicfile.write(word.lstrip().rstrip() + " ")                
        basicfile.write("\n")
        
        for word in foresentence.split():
            tempword = codeword(word)
            if tempword in DICTIONARY:
                forbackfile.write(word.lstrip().rstrip() + " ")
        forbackfile.write("##")
        
        for word in sentence.split():
            tempword = codeword(word)
            if tempword in DICTIONARY:
                forbackfile.write(word.lstrip().rstrip() + " ")
        
        forbackfile.write("##")
        for word in backsentence.split():
            tempword = codeword(word)
            if tempword in DICTIONARY:
                forbackfile.write(word.lstrip().rstrip() + " ")
        forbackfile.write("\n")
        


if __name__ == '__main__':
    
    #wikifile = open("/Users/huihui/experiment/ese/wiki/wiki/total-labeled-wikipages5","r", encoding = 'utf-8')
    #labelfile = open("/Users/huihui/experiment/ese/classification/wikilabel","w")
    #basicfile = open("/Users/huihui/experiment/ese/classification/wikionesentence","w")
    #forbackfile= open("/Users/huihui/experiment/ese/classification/wikithreesentence","w")
    #dicfile = open("/Users/huihui/experiment/ese/classification/DICTIONARY.txt","r", encoding = "utf-8")
    
    arxivfile = open("/Users/huihui/experiment/ese/arxiv/total.txt","r", encoding = 'utf-8')
    labelfile = open("/Users/huihui/experiment/ese/classification/arxiv/arxivlabel","w")
    basicfile = open("/Users/huihui/experiment/ese/classification/arxiv/arxivonesentence","w")
    forbackfile= open("/Users/huihui/experiment/ese/classification/arxiv/arxivthreesentence","w")
    dicfile = open("/Users/huihui/experiment/ese/classification/arxiv/DICTIONARY.txt","r", encoding = "utf-8")
    
    readDICTIONARY(dicfile)
    
    stopwordsfile = "/Users/huihui/experiment/ese/stopwords"
    stopwords = readstopwords(stopwordsfile)
    
    
    sample(arxivfile, labelfile, basicfile, forbackfile)
    
    arxivfile.close()
    labelfile.close()
    basicfile.close()
    forbackfile.close()
    
    
    pass