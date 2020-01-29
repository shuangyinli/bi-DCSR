#! /usr/bin/python

# usage: python topics.py <beta file> <vocab file> <num words>
#
# <beta file> is output from the lda-c code
# <vocab file> is a list of words, one per line
# <num words> is the number of words to print from each topic

import sys

dictionary = []

def print_topics(beta_file, vocab_file, nwords):
    # for each line in the beta file
    getDictionary(vocab_file)
    beta = open(beta_file, "r")
    topicno = 0
    for topic in beta:
        print("topic %03d :"% topicno)
        dicT = {}
        probabilitylist = topic.split()
        diclen = len(probabilitylist)
        if diclen != len(dictionary):
            print("the size of dictionary doesn't match the probability number.")
            print("the size of dictionary is %d, " % len(dictionary))
            print("and the probability number per line in word-probability-file is %d.\n"% diclen)
            sys.exit(1)
        for word in range(diclen):
            dicT.setdefault(dictionary[word]) # word text
            dicT[dictionary[word]] = float(probabilitylist[word]) # word probability
        
        sortedList = sorted(dicT.items(), key = lambda a:a[1], reverse=True)
        for i in range(nwords):
            print(str(sortedList[i]))
            
        topicno = topicno +1
        
def getDictionary(vocab_file):
    vocab = open(vocab_file, 'r').readlines()
    for line in vocab:
        itemlist = line.split(":")
        if len(itemlist) > 1:
            dictionary.append(itemlist[1].strip().rstrip())
        else:
            dictionary.append(line.strip().rstrip())
    pass

if __name__ == '__main__':

    if (len(sys.argv) != 4):
        print("usage: python showTopicsBywords.py <word-probability-file> <dictionary-file> <top num words>\n")
        sys.exit(1)
    beta_file = sys.argv[1]
    vocab_file = sys.argv[2]
    nwords = int(sys.argv[3])
    print_topics(beta_file, vocab_file, nwords)