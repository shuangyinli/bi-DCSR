'''
Created on Apr 1, 2019

@author: shuangyinli
'''
# -*- coding: utf-8 -*-
import os
import shutil
import re


def remove(abstract):
    abstract = re.sub("[\\w\-\\.]+@([\\w]+\\.)+[a-z]{2,3}", ' ', abstract)
    abstract =  ' '.join(abstract.split())
    abstract = abstract.lower()
    return abstract

HEAD = set()

def extractcontent(newspath):
    #newsfile = open(newspath, encoding="utf-8")
    try:
        with open(newspath, encoding="utf-8") as newsfile:
            list1 = newsfile.readlines()
    except:
        return
    
    for line in list1:
        if ":" in line:
            templist = line.split(":")
            HEAD.add(templist[0].split(" ")[-1])
    
    pass


def processNews(fileslist):
    for line in fileslist:
        extractcontent(line)
        print(line)
        pass
    
    pass


def getfilepath(mainpath):
    fileslist = []
    path_list = os.listdir(mainpath)
    
    news20paths = []
    
    for son_path in path_list:
        son_dir = mainpath + "/"+son_path + "/"
        news20paths.append(son_dir)
        
        sonfilelist =  os.listdir(son_dir)
        
        for sonfile in sonfilelist:
            
            fileslist.append(son_dir+sonfile)
            
    
    return fileslist


if __name__ == '__main__':
    
    newspath = "/Users/huihui/Data/20newsgroups/20_newsgroups"
    
    
    fileslist = getfilepath(newspath)
    
    processNews(fileslist)
    
    print(len(HEAD))
    #for item in HEAD:
        #print(item)
    #print(remove("dadsaf hiu-d-hui@sinan.com.cn dsafd"))
    
    
    
    
    #outputfile = open("/Users/huihui/Data/20newsgroups/output.txt","w")
    
    
    pass