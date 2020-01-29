'''
Created on Jul 19, 2016

@author: shuangyinli
'''

import sys
from scipy import spatial
import operator

dictionary = []
tagsDis = {} #keep tagName & its distribution

def most_similar_tags(tagname):
    distancevec = {}
    #result = [1 - spatial.distance.cosine(tagvec, tagsDis[tag]) for tag in dictionary]
    for tag in dictionary:
        try:
            distancevec[tag] = 1 - spatial.distance.cosine(tagsDis[tagname], tagsDis[tag])
        except TypeError:
            distancevec[tag] = 0
    return sorted(distancevec.items(), key=operator.itemgetter(1),  reverse=True)

def readTagsTopicDistributions(theta_file):
    theta = open(theta_file, "r", encoding = "utf-8")
    tagno = 0
    for ts in theta:
        topicslist = [float(m) for m in ts.split()]
        a = dictionary[tagno]
        tagsDis.setdefault(str(dictionary[tagno]))
        tagsDis[dictionary[tagno]] = topicslist
        tagno = tagno +1
    pass

def getDictionary(vocab_file):
    vocab = open(vocab_file, 'r',encoding = "utf-8").readlines()
    for line in vocab:
        itemlist = line.split(":")
        if len(itemlist) > 1:
            dictionary.append(itemlist[1].strip().rstrip())
        else:
            dictionary.append(line.strip().rstrip())
    pass

if __name__ == '__main__':
    if (len(sys.argv) != 3):
        print("usage: python showSimilarTags.py <tag vocabrory>  <tag-probalility-file> \n")
        sys.exit(1)
    vocab_file = sys.argv[1]
    theta_file = sys.argv[2]
    getDictionary(vocab_file)
    readTagsTopicDistributions(theta_file)
    
    while True:
        tagname = str(input("Enter one tag name:"))
        if tagname is "":
            print("exit")
            exit(0)
        if tagname in dictionary:
            #print(dictionary.index(tagname))
            if tagname in tagsDis.keys():
                topn = int(input("How many tops you want to print?"))
                print(most_similar_tags(tagname)[1:topn])
            else:
                print(tagname + " is not in the tagDis, please try anthor words.")
        else:
            print(tagname + " is not in the tag vocab, please try anthor words.")
            continue
        #topn = input("How many tops you want to print?")
        #most_similar_tags(tagname)[:topn]
        
    pass