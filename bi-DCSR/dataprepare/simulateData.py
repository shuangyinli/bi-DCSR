'''
Created on Mar 23, 2019

@author: shuangyinli
'''
import math
import random

allwords = 30


outfile = open("/Users/huihui/Information/github/ese/testdata/simulate.txt","w");

for n in range(10):
    sentenceno = random.randint(3,20)
    outfile.write(str(sentenceno)+" ")
    for s in range(sentenceno):
        outfile.write("# ")
        keywordno = random.randint(5,10)
        outfile.write(str(keywordno)+" ")
        
        kwset = set()
        while len(kwset) < keywordno:
            kwid = random.randint(1,allwords)
            if kwid not in kwset:
                outfile.write(str(kwid)+" ")
                kwset.add(kwid)
        outfile.write("@ ")
        
        wordno = random.randint(5,20)
        outfile.write(str(wordno)+" ")
        
        wset = set()
        while len(wset) < wordno:
            wid = random.randint(1,allwords)
            if wid not in wset:
                wcnt = random.randint(1,5)
                outfile.write(str(wid)+":"+str(wcnt)+" ")
                wset.add(wid)
            
            pass
        pass
    outfile.write("\n")
    pass
