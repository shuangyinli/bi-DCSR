'''
Created on May 10, 2016

@author: shuangyinli
'''
import re
import os
import os.path
import sys
import operator
import multiprocessing
from multiprocessing import Process

# keep the cleaned dictionay from DictionaryOrignal
DICTIONARY=[]
# keep the orignal dictionay
DictionaryOrignal={}
stopwords = {}
MIN_sentence = 0

def readstopwords(stopwordsfile):
    return {}.fromkeys([ line.rstrip() for line in open(stopwordsfile, 'r') ])

def removeWithoutDot(abstract):
    abstract = re.sub("[^A-Za-z]", " ", abstract)
    
    ###remove a single word
    abstract=re.sub(' [a-zA-Z] ', ' ', abstract)
    abstract=re.sub('^[ ]*[a-zA-Z] ', ' ', abstract) 
    abstract=re.sub(' [a-zA-Z][ ]*$', ' ', abstract)

    abstract =  ' '.join(abstract.split())
    abstract = abstract.lower()
    return abstract

def remove(abstract):
    abstract = re.sub("[^A-Za-z.]", " ", abstract)
    
    ###remove a single word
    abstract=re.sub(' [a-zA-Z] ', ' ', abstract)
    abstract=re.sub('^[ ]*[a-zA-Z] ', ' ', abstract) 
    abstract=re.sub(' [a-zA-Z][ ]*$', ' ', abstract)

    abstract =  ' '.join(abstract.split())
    abstract = abstract.lower()
    return abstract

## get the dictionany
def OrignalDictionaryFunction(words):
    line = removeWithoutDot(words)
    for word in line.split():
        word = word.rstrip().lstrip()
        if word.rstrip() == '':
            continue
        if word in stopwords:
            continue
        if word not in DictionaryOrignal:
            DictionaryOrignal.setdefault(word,0)
            DictionaryOrignal[word] += 1
        else:
            DictionaryOrignal[word] += 1

def MakeDictionary(inputfile):
    inputsourceDocumentslist =[]
    for line in inputfile:
        text = line.split("##")[-1]
        inputsourceDocumentslist.append(text)
        OrignalDictionaryFunction(text)
    print("dic is over, now clean it.")
    a = sorted(DictionaryOrignal.items(), key=lambda d: d[1], reverse=True)[0:21968]
    for key in a:
        DICTIONARY.append(key[0])
    print("the dic size is " + str(len(DICTIONARY)))
    return inputsourceDocumentslist
            
def codeText(words):
    wordlist ={}
    for word in words.split():
        word = word.rstrip()
        if word in stopwords:
            continue
        if word in DICTIONARY:
            wordlist.setdefault(DICTIONARY.index(word),0)
            wordlist[DICTIONARY.index(word)] += 1
    return wordlist
    
def countSentences(sentences):
    senno = 0
    for sentence in sentences:
        wordlist_ = codeText(sentence)
        wordsno = len(wordlist_)
        if wordsno != 0: 
            senno = senno +1
    return senno

def code(inputsourceDocuments, oDir):
    pid = os.getpid()
    #outputSentencesSourcefile = open(oDir+"outputSourcefile"+str(pid),"w", encoding = "utf-8")
    RATMCodedfile = open(oDir+"RATMCodedfile"+str(pid),"w", encoding = "utf-8")
    LDACodedfile = open(oDir+"LDACodedfile"+str(pid),"w", encoding = "utf-8")
    
    documentno = 0
    for line in inputsourceDocuments:
        sentences = remove(line).split(".")
        # get the sentence's number . if the sentence's No is low than 3, ignore this document
        senno = countSentences(sentences)
        if senno <MIN_sentence:
            continue 
        
        #LDA
        ldadocument = remove(line).replace(".", " ")
        ldawordlist = codeText(ldadocument)
        docwordno = len(ldawordlist)
        if docwordno ==0:
            continue
        if docwordno !=0:
            LDACodedfile.write(str(docwordno)+" ")
            for dw in ldawordlist:
                LDACodedfile.write(str(dw)+":" + str(ldawordlist[dw]) +" ")
            LDACodedfile.write("\n")
        LDACodedfile.flush()        
        
        #biRATM
        RATMCodedfile.write(str(senno)+" ")
        for sentence in sentences:
            wordlist = codeText(sentence)
            wordsno = len(wordlist)
            if wordsno != 0: 
                RATMCodedfile.write("@ "+str(wordsno)+" ")
                for w in wordlist:
                    RATMCodedfile.write(str(w)+":"+str(wordlist[w])+" ")
        RATMCodedfile.write("\n")
        RATMCodedfile.flush()
        
        documentno = documentno +1
        print("now begin to doc :" + str(documentno))       
        
    RATMCodedfile.close()

def SaveDictionary(dicfile):
    print("now save the dictionary.")
    for word in DICTIONARY:
        dicfile.write(str(DICTIONARY.index(word)) +":" +word)
        dicfile.write("\n")
    dicfile.close()
    
    
def split_average_data(corpus, thread_no):
    fn = len(corpus)//thread_no
    rn = len(corpus)%thread_no
    ar = [fn+1]*rn+ [fn]*(thread_no-rn)
    si = [i*(fn+1) if i<rn else (rn*(fn+1)+(i-rn)*fn) for i in range(thread_no)]
    corpusSplitlist = [corpus[si[i]:si[i]+ar[i]] for i in range(thread_no)]
    return corpusSplitlist
    
def run_multiprocesses(inputsourceDocumentslist, workers_no, outpath):
    workers = []
    corpusSplitlist = split_average_data(inputsourceDocumentslist, workers_no)
    
    for dataSplit in corpusSplitlist:
        worker = Process(target=code, args=(dataSplit,outpath,))
        worker.start()
        workers.append(worker)
    
    for w in workers:
        w.join()

if __name__ == '__main__':
    if (len(sys.argv) != 5):
        print("usage: python codeTEXTtoRATMformat.py <processes no> <inputDocumentSourcefile> <stopwords> <output path>\n")
        print("this script is to format document to biRATM per doc.")
        print("The document should contain dots to split the sentences")
        print("The output files include biRATMSentencesCodedfile and LDACodedfile")
        print("")
        sys.exit(1)
        
    MIN_sentence = 0
    
    workers_no = int(sys.argv[1])
    inputfile = open(sys.argv[2], "r", encoding = "utf-8")
    stopwordsfile = sys.argv[3]
    oDir = sys.argv[4]
    if oDir.endswith("/") is False:
        oDir = oDir+"/"
    
    print("read the stopwords..")
    stopwords = readstopwords(stopwordsfile)
    
    dicfile = open(oDir+"DICTIONARY.txt","w", encoding = "utf-8")
    # make the dictionary
    inputsourceDocumentslist = MakeDictionary(inputfile)
    SaveDictionary(dicfile)
    
    run_multiprocesses(inputsourceDocumentslist, workers_no, oDir)
    
    print("end.")