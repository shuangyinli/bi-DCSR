'''
Created on Apr 2, 2019

@author: shuangyinli
'''
# -*- coding: utf-8 -*-
import re
import operator
def remove(abstract):
    abstract = re.sub("[^A-Za-z.]", ' ', abstract)
    ###remove a single word
    abstract=re.sub(' [a-zA-Z] ', ' ', abstract)
    abstract=re.sub('^[ ]*[a-zA-Z] ', ' ', abstract) 
    abstract=re.sub(' [a-zA-Z][ ]*$', ' ', abstract)
    abstract =  ' '.join(abstract.split())
    abstract = abstract.lower()
    return abstract




if __name__ == '__main__':
    
    wikifile4 = open("/Users/huihui/Data/wiki/total-labeled-wikipages4","r")
    wikifile5 = open("/Users/huihui/Data/wiki/total-labeled-wikipages5","w")
    
    for line in wikifile4:
        linelist = line.split("##")
        label = linelist[0]
        content = linelist[1]
        sentencelist = content.split(".")
        finalsenlist = []
        for sen in sentencelist:
            clearsen = remove(sen).rstrip().lstrip()
            wordlist = clearsen.split()
            if len(wordlist) > 5:
                finalsenlist.append(clearsen)
                
        if len(finalsenlist) > 5:
            wikifile5.write(line.rstrip().lstrip())
            wikifile5.write("\n")
            
    
    
    pass



