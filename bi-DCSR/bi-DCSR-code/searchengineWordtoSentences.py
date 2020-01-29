'''
Created on Apr 27, 2019

@author: shuangyinli
'''
import sys
from scipy import spatial
import operator

dictionary = []

words_phi = {} # phi

sentences_log_topic = {} #sentence log_topic
sentences_context = {} # \sigma_N p(w_n|T) \sigma_Wn p(T | W_n)  without attentio
sentences_all = {} #p(sentence | previous, following, w_n)  \sigma_N p(w_n|T) \sigma_Wn p(T | W_n) 


def most_similar_sentences(tagname, topn):
    distancevec1 = {}
    #result = [1 - spatial.distance.cosine(tagvec, tagsDis[tag]) for tag in dictionary]
    for sen in sentences_log_topic:
        try:
            distancevec1[sen] = 1 - spatial.distance.cosine(words_phi[tagname], sentences_log_topic[sen])
        except TypeError:
            distancevec1[sen] = 0
            
    log_sorted = sorted(distancevec1.items(), key=operator.itemgetter(1),  reverse=True)[:topn]
    
    
    distancevec2 = {}
    for sen in sentences_context:
        try:
            distancevec2[sen] = 1 - spatial.distance.cosine(words_phi[tagname], sentences_context[sen])
        except TypeError:
            distancevec2[sen] = 0
    
    contexts_sorted = sorted(distancevec2.items(), key=operator.itemgetter(1),  reverse=True)[:topn]
    
    distancevec3 = {}
    for sen in sentences_all:
        try:
            distancevec3[sen] = 1 - spatial.distance.cosine(words_phi[tagname], sentences_all[sen])
        except TypeError:
            distancevec3[sen] = 0
    
    all_sorted = sorted(distancevec3.items(), key=operator.itemgetter(1),  reverse=True)[:topn]
    
    
    return log_sorted, contexts_sorted, all_sorted

def loadSentencesembeddings(sentences_probability):
    vectorslines = open(sentences_probability,"r", encoding = "utf-8")
    for line in vectorslines:
        linelist = line.lstrip().rstrip().split("##")
        sentences_log_topic.setdefault(linelist[1].lstrip().rstrip(),linelist[3].lstrip().rstrip().split())
        sentences_context.setdefault(linelist[1].lstrip().rstrip(),linelist[4].lstrip().rstrip().split())
        sentences_all.setdefault(linelist[1].lstrip().rstrip(),linelist[5].lstrip().rstrip().split())


def loadWordembeddings(phifile):
    philines = open(phifile,"r", encoding = "utf-8")
    tagno =0
    for topics_string in philines:
        tagprobabilitylist = topics_string.split()
        word = dictionary[tagno]
        if word not in words_phi:
            words_phi.setdefault(word)
            words_phi[word] = tagprobabilitylist
        tagno = tagno+1


def getDictionary(vocab_file):
    vocab = open(vocab_file, 'r', encoding = "utf-8").readlines()
    for line in vocab:
        dictionary.append(line.strip().rstrip().split(":")[1])
    pass

if __name__ == '__main__':
    
    if (len(sys.argv) != 4):
        print("usage: python showWordsMultiSemantics.py <dictionary file> <keyword_probability_file phi>  <sentences probalility over semantics file> \n")
        #print("<semanic probability over dictionary file(final.beta)>  <word probalility over semantics file(final.theta)> \n")
        sys.exit(1)
        
    
    dictionary_file = sys.argv[1]
    keyword_probability_file_phi = sys.argv[2]
    sentences_probability_file_phi = sys.argv[3]
    
    
    # read 
    print("Begin to read dictionary.")
    getDictionary(dictionary_file)
    
    print("Begin to load word probabilities.")
    loadWordembeddings(keyword_probability_file_phi)
    
    print("Begin to load sentences probabilities.")
    loadSentencesembeddings(sentences_probability_file_phi)
    
    while True:
        tagname = str(input("Please input one word: "))
        if tagname is "":
            print("exit")
            exit(0)
        if tagname in dictionary:
            if tagname in words_phi.keys():
                topn = int(input("How many tops you want to print?"))
                log_sorted, contexts_sorted, all_sorted = most_similar_sentences(tagname,topn)
                
                print("log_topics results: ")
                print(log_sorted)
                
                print(" \n contexts results: ")
                print(contexts_sorted)
                
                print(" \n all results: ")
                print(all_sorted)
                
            else:
                print(tagname + " is not in the dictionary, please try another words.")
        else:
            print(tagname + " is not in the dictionary, please try another words.")
            continue
