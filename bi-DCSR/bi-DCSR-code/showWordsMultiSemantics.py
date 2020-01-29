'''
Created on Jul 18, 2016

@author: shuangyinli
'''
import sys

dictionary = []
tagvocabory = []
topic_words_dic = {}
word_topic_dis = {}

def read_topics(beta_file, nwords):
    # for each line in the beta file
    beta = open(beta_file, "r", encoding = "utf-8")
    topicno = 0
    for topic in beta:
        #print("topic %03d :"% topicno)
        dicT = {}
        probabilitylist = topic.split()
        diclen = len(probabilitylist)
        if diclen != len(dictionary):
            print("the size of dictionary doesn't match the probability number.")
            print("the size of dictionary is %d, "% len(dictionary))
            print("and the probability number per line in word-probability-file is %d.\n"% diclen)
            sys.exit(1)
        for word in range(diclen):
            dicT.setdefault(dictionary[word]) # word text
            dicT[dictionary[word]] = float(probabilitylist[word]) # word probability
        
        sortedtopList = sorted(dicT.items(), key = lambda a:a[1], reverse=True)[:nwords]
            
        topic_words_dic.setdefault(topicno)
        topic_words_dic[topicno] = sortedtopList
        topicno = topicno +1
        


def printTagTopics(thetafile,topn):
    thetalines = open(thetafile,"r", encoding = "utf-8")
    tagno =0
    for topics_string in thetalines:
        print(str(tagvocabory[tagno]) + ": ")
        tagT = {}
        tagprobabilitylist = topics_string.split()
        topic_no = len(tagprobabilitylist)
        for topic in range(topic_no):
            tagT.setdefault(topic) # topic id
            tagT[topic] = float(tagprobabilitylist[topic]) # topic probability
            
        sortedTopList = sorted(tagT.items(), key= lambda a:a[1], reverse = True)[:topn] # top
        
        for i in range(topn):
            topicid = sortedTopList[i][0] # topic id for the tag
            topicpro = sortedTopList[i][1] # topic probability of the tag
            sortedwordList = topic_words_dic[topicid]
            print("    top"+str(i)+" (topic id: "+str(topicid)+"  "+str(topicpro)+") :" +str([word[0].split(":")[1].strip().rstrip() for word in sortedwordList]))
        print("")
        tagno = tagno+1
    pass

def loadWordembeddings(thetafile):
    thetalines = open(thetafile,"r", encoding = "utf-8")
    tagno =0
    for topics_string in thetalines:
        tagprobabilitylist = topics_string.split()
        word = tagvocabory[tagno]
        if word not in word_topic_dis:
            word_topic_dis.setdefault(word)
            word_topic_dis[word] = tagprobabilitylist
        tagno = tagno+1
    pass

def printonewordTopics(wordstring,topn):
    print(wordstring + ": ")
    tagT = {}
    tagprobabilitylist = word_topic_dis[wordstring]
    topic_no = len(tagprobabilitylist)
    for topic in range(topic_no):
        tagT.setdefault(topic) # topic id
        tagT[topic] = float(tagprobabilitylist[topic]) # topic probability
    sortedTopList = sorted(tagT.items(), key= lambda a:a[1], reverse = True)[:topn] # top
    
    for i in range(topn):
        topicid = sortedTopList[i][0] # topic id for the tag
        topicpro = sortedTopList[i][1] # topic probability of the tag
        sortedwordList = topic_words_dic[topicid]
        print("    top"+str(i)+" (topic id: "+str(topicid)+"  "+str(topicpro)+") :" +str([word[0].strip().rstrip() for word in sortedwordList]))
    print("")
    pass

def getDictionary(vocab_file):
    vocab = open(vocab_file, 'r', encoding = "utf-8").readlines()
    for line in vocab:
        dictionary.append(line.strip().rstrip().split(":")[1])
    pass

def getTagVocab(tagfile):
    tagvocab = open(tagfile, 'r', encoding = "utf-8").readlines()
    for line in tagvocab:
        tagvocabory.append(line.strip().rstrip().split(":")[1])
    pass

if __name__ == '__main__':
    if (len(sys.argv) != 6):
        print("usage: python showWordsMultiSemantics.py <dictionary file> <word_probability_file beta> <top num words in each semantic>  <word probalility over semantics file(phi)> <top num topic> \n")
        #print("<semanic probability over dictionary file(final.beta)>  <word probalility over semantics file(final.theta)> \n")
        sys.exit(1)
        
    dictionary_file = sys.argv[1]
    word_probability_file = sys.argv[2]
    topnwords = int(sys.argv[3])
    tag_probability_file = sys.argv[4]
    tag_vocabrory_file = sys.argv[1]
    top_num_topics = int(sys.argv[5])
    
    # read 
    print("Begin to read dictionary.")
    getDictionary(dictionary_file)
    getTagVocab(tag_vocabrory_file)
    #
    print("Begin to load Semantics.")
    read_topics(word_probability_file, topnwords)
    #
    print("Begin to load word probabilities.")
    loadWordembeddings(tag_probability_file)
    
    while True:
        tagname = str(input("Please input one word: "))
        if tagname is "":
            print("exit")
            exit(0)
        if tagname in dictionary:
            if tagname in word_topic_dis.keys():
                printonewordTopics(tagname,top_num_topics)
            else:
                print(tagname + " is not in the dictionary, please try another words.")
        else:
            print(tagname + " is not in the dictionary, please try another words.")
            continue
    pass