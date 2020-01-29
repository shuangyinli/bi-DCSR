'''
Created on Apr 2, 2019

@author: shuangyinli
'''

# -*- coding: utf-8 -*-
import re
import operator
def remove(abstract):
    abstract = re.sub("[^A-Za-z]", ' ', abstract)
    ###remove a single word
    abstract=re.sub(' [a-zA-Z] ', ' ', abstract)
    abstract=re.sub('^[ ]*[a-zA-Z] ', ' ', abstract) 
    abstract=re.sub(' [a-zA-Z][ ]*$', ' ', abstract)
    abstract =  ' '.join(abstract.split())
    abstract = abstract.lower()
    return abstract

labelset = {}
exitlist = []
remp = ["information retrieval","computer vision","machine learning","image processing","natural language processing","data mining"]

largelist = []
smalllist = []
import random

def reconstruct(wikifile, wikifile2):
    a = 0
    for line in wikifile:
        label = line.split("#")[0].rstrip().lstrip()
        print(a)
        a+=1
        if label in remp:
            wikifile2.write("artificial intelligence##"+line.split("#")[1].rstrip().lstrip())
            wikifile2.write("\n")
        else:
            wikifile2.write(line.rstrip().lstrip())
            wikifile2.write("\n")
            
    wikifile2.close()
    
def reconstruct2(wikifile, wikifile3):
    a = 0
    for key,value in labelset.items():
        if value > 800:
            exitlist.append(key)
    
    for line in wikifile:
        label = line.split("##")[0].rstrip().lstrip()
        content = line.split("##")[1].rstrip().lstrip()
        wordlist = content.split()
        sentencelist = content.split(".")
        print(a)
        a+=1
        if label in exitlist:
            if content != "" and content !="\n" and len(wordlist) >20 and len(sentencelist) >5:
                wikifile3.write(line.rstrip().lstrip())
                wikifile3.write("\n")
    wikifile3.close()

def countLabel(wikifile):
    for line in wikifile:
        label = line.split("##")[0].rstrip().lstrip()
        content = line.split("##")[1].rstrip().lstrip()
        wordlist = content.split()
        sentencelist = content.split(".")
        if content != "" and content !="\n" and len(wordlist) > 20 and len(sentencelist) >5:
            if label not in labelset:
                labelset.setdefault(label,1)
            else:
                labelset[label] +=1

def sampledargedata(wikifile3, wikifile4):
    for line in wikifile3:
        label = line.split("##")[0].rstrip().lstrip()
        
        if labelset[label] < 15000:
            smalllist.append(line)
        else:
            largelist.append(line)
            
    for key, value in labelset.items():
        print(key + " : " + str(value))
        
    random.shuffle(largelist)
    random.shuffle(smalllist)

    totalset = {}
    for item in largelist:
        label = item.split("##")[0].rstrip().lstrip()
        if label not in totalset:
            totalset.setdefault(label,[])
            totalset[label].append(item)
        else:
            totalset[label].append(item)

    print(len(largelist))
    print(len(smalllist))
    print(len(totalset))
    
    selectout = []
    for key, value in totalset.items():
        valuelist = value
        print(str(key) +": "+ str(len(value)))
        random.shuffle(valuelist)
        for i in range(15000):
            temps = valuelist[i]
            selectout.append(temps.lstrip().rstrip())
        
    #selectout.append(smalllist)

    for item in smalllist:
        selectout.append(item)
        
    random.shuffle(selectout)
     
    for item in selectout:
        wikifile4.write(item.lstrip().rstrip())
        wikifile4.write("\n")
        
    wikifile4.close()
    pass

if __name__ == '__main__':
    
#     wikifile2 = open("/Users/huihui/Data/wiki/total-labeled-wikipages2","r")
#     wikifile3 = open("/Users/huihui/Data/wiki/total-labeled-wikipages3","w")
#     countLabel(wikifile2)
#     wikifile2 = open("/Users/huihui/Data/wiki/total-labeled-wikipages2","r")
#     reconstruct2(wikifile2,wikifile3)
    
    #####################
    
# 
    wikifile3 = open("/Users/huihui/Data/wiki/total-labeled-wikipages3","r")
    wikifile4 = open("/Users/huihui/Data/wiki/total-labeled-wikipages4","w")
    countLabel(wikifile3)
    wikifile3 = open("/Users/huihui/Data/wiki/total-labeled-wikipages3","r")
    sampledargedata(wikifile3, wikifile4)
    
#     #reconstruct(wikifile,wikifile2)
#     labelsets = sorted(labelset.items(), key=operator.itemgetter(1), reverse = True)
#     
#     for key,value in labelsets:
#         if value > 999:
#             exitlist.append(key)
# 
#     print(exitlist)
#     print(len(exitlist))
#     reconstruct2(wikifile2,wikifile3)

        
    
    
    pass