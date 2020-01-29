'''
Created on Apr 18, 2019

@author: shuangyinli
'''
from numpy import *
import numpy as np
import random
import copy
import sys
import re
import time
import os
from multiprocessing import Process, Manager
from copy import deepcopy
from collections import Counter

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import cross_validate

from sklearn import metrics

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.datasets import load_svmlight_file

from sklearn.ensemble import RandomForestClassifier

from nltk.stem import WordNetLemmatizer

wn = WordNetLemmatizer()

stopwords = {}

def readstopwords(stopwordsfile):
    return {}.fromkeys([ line.rstrip() for line in open(stopwordsfile, 'r') ])

def remove(abstract):
    abstract = re.sub("[^A-Za-z.]", " ", abstract)
    ###remove a single word
    abstract = re.sub(' [a-zA-Z] ', ' ', abstract)
    abstract = re.sub('^[ ]*[a-zA-Z] ', ' ', abstract)
    abstract = re.sub(' [a-zA-Z][ ]*$', ' ', abstract)

    abstract = ' '.join(abstract.split())
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

def codetext(text, model):
    wordset ={}
    textwords = remove(text)
    for word in textwords.split():
        word = wn.lemmatize(word.rstrip().lstrip())
        if word in model.dictionary:
            wordset.setdefault(model.DICTIONARY[word],0)
            wordset[model.DICTIONARY[word]] += 1
    return len(wordset), wordset

class Sentence():
    def __init__(self, senid_, num_keywords_, num_words_, num_topics_, lik_,
                keywords_ptr_, words_ptr_, words_cnt_ptr_, neighbor_):
        '''
        Constructor
        '''
        # ensure the args are numpy array
        
        self.sentenceid = senid_
        
        self.num_keyword = num_keywords_
        self.num_words = num_words_
        self.num_topics = num_topics_
        self.lik = lik_
        self.keywords_ptr = keywords_ptr_
        self.words_ptr = words_ptr_
        self.words_cnt_ptr = words_cnt_ptr_
        self.neighbor = neighbor_

        self.xi = np.zeros(self.num_keyword + 2*neighbor_, dtype=np.float64) #[0 for i in range(self.num_tags)]
        self.log_gamma = np.zeros(shape = (self.num_words, self.num_topics), dtype=np.float64)
        self.sentenceattention = {}
        
        self.log_topic = np.zeros(self.num_topics, dtype=np.float64)
        self.log_neighbortopics = np.zeros(shape=(2*neighbor_, self.num_topics),dtype=np.float64)
        
        self.sentence_init()

    def sentence_init(self):
        for i in range(self.num_keyword + 2*self.neighbor):
            self.xi[i] = random.random()+0.5
        
        for i in range(2*self.neighbor):
            for k in range(self.num_topics):
                self.log_neighbortopics[i][k] = log(1.0 / self.num_topics)
            
        for i in range(self.num_words):
            for k in range(self.num_topics):
                self.log_gamma[i][k] = log(1.0 / self.num_topics)
                

class Model():
    def __init__(self, num_topics_, neighbor_, dictionary_, DICTIONARY_, pi_,  log_phi_, log_beta_):
        ''' 
        Constructor
        '''
        self.neighbor = neighbor_
        self.num_topics = num_topics_
        self.dictionary = dictionary_
        
        self.pi = pi_
        self.log_phi = log_phi_
        self.log_beta = log_beta_
        
        self.num_words = len(dictionary_)
        self.num_all_keywords = len(dictionary_) 
        self.dictionary = dictionary
        self.DICTIONARY = DICTIONARY_
          
class Configuration():
    def __init__(self, settingsfile):
        '''
        Constructor
        '''
        self.pi_learn_rate = 0.00001
        self.max_pi_iter=100
        self.pi_min_eps=1e-5
        self.max_xi_iter=100
        self.xi_min_eps=1e-5
        self.xi_learn_rate = 10
        
        self.max_em_iter=30
        self.num_threads=1
        self.sen_var_converence = 1e-4
        self.sen_max_var_iter = 5
        self.doc_var_converence =0.01
        self.doc_max_var_iter = 2
        
        self.num_topics=10
        self.neighbor = 2
        
        self.em_converence = 1e-4
        self.read_settingfile(settingsfile)

    def read_settingfile(self,settingsfile):
        settingslist = open(settingsfile, "r", encoding = 'utf-8')
        for line in settingslist:
            confname = line.split()[0]
            confvalue = line.split()[1]
            if confname == "pi_learn_rate":
                self.pi_learn_rate = float(confvalue)
            if confname == "max_pi_iter":
                self.max_pi_iter = int(confvalue)
            if confname == "pi_min_eps":
                self.pi_min_eps = float(confvalue)
            if confname == "max_xi_iter":
                self.max_xi_iter = int(confvalue)
            if confname == "xi_learn_rate":
                self.xi_learn_rate = float(confvalue)
            if confname == "xi_min_eps":
                self.xi_min_eps = float(confvalue)
                
            if confname == "max_em_iter":
                self.max_em_iter = int(confvalue)
            if confname == "num_threads":
                self.num_threads = int(confvalue)
            if confname == "sen_var_converence":
                self.sen_var_converence = float(confvalue)
            if confname == "sen_max_var_iter":
                self.sen_max_var_iter = int(confvalue)
                
            if confname == "doc_var_converence":
                self.doc_var_converence = float(confvalue)
                
            if confname == "doc_max_var_iter":
                self.doc_max_var_iter = int(confvalue)
                
            if confname == "num_topics":
                self.num_topics = int(confvalue)
                
            if confname == "neighbor":
                self.neighbor = int(confvalue)
                
            if confname == "em_converence":
                self.em_converence = float(confvalue)

def split_average_data(thread_no, text_list):
    fn = len(text_list)//thread_no
    rn = len(text_list)%thread_no
    ar = [fn+1]*rn+ [fn]*(thread_no-rn)
    si = [i*(fn+1) if i<rn else (rn*(fn+1)+(i-rn)*fn) for i in range(thread_no)]
    corpusSplitlist = [text_list[si[i]:si[i]+ar[i]] for i in range(thread_no)]
    return corpusSplitlist


class ESE():
    def __init__(self, datafilename, sentencetopicfile, sentencexifile, configuration, model):
        '''
        Constructor
        '''
        self.corpus = []
        self.num_docs=0

        self.num_words = len(model.dictionary)
        self.configuration = configuration
        self.model = model
        self.num_total_words = 0
        
        self.num_topics = self.configuration.num_topics
        self.neighbor = self.configuration.neighbor
        
        if datafilename != "":
            self.read_sentencesforesecode(datafilename)
        
        if sentencetopicfile != "":
            self.read_sentencetopics(sentencetopicfile)
        
        if sentencexifile!= "":
            self.read_sentenceattention(sentencexifile)
        
    
    def read_sentencetopics(self, sentencetopicfile):
        datalist = open(sentencetopicfile, "r", encoding = "utf-8").readlines()
        lineno = 0
        for line in datalist:
            linelist = line.split("#")
            #self.sentencetopics = linelist[0].rstrip().lstrip().split()
            temp2 =  linelist[0].rstrip().lstrip().split()
            temp = [float(n) for n in temp2]
            self.corpus[lineno].log_topic = array(temp).copy()
            lineno +=1
            
            
    def read_sentenceattention(self, sentencexifile):
        datalist = open(sentencexifile, "r", encoding = "utf-8").readlines()
        lineno = 0
        for line in datalist:
            linelist = line.split("#")
            for item in linelist[0].rstrip().lstrip().split():
                itemlist = item.split(":")
                
                wordidstring = itemlist[0].rstrip().lstrip()
                if wordidstring == " " or wordidstring == "" or wordidstring.rstrip().lstrip() == "":
                    pass
                if "n" not in wordidstring:
                    wordid = int(itemlist[0].rstrip().lstrip())
                else:
                    wordid = wordidstring
                attenvalue = float(itemlist[1].rstrip().lstrip())
                self.corpus[lineno].sentenceattention.setdefault(wordid, 0)
                self.corpus[lineno].sentenceattention[wordid] = attenvalue
            lineno +=1
    
    def read_sentencesforesecode(self,filename):
        senid =0
        datalist = open(filename, "r", encoding = "utf-8").readlines()
        for sentence in datalist:
            allwordno = 0
            sentencelist = sentence.split("#")
            keywordlist = sentencelist[1].lstrip().rstrip().split("@")[0].lstrip().rstrip().split()
            wordlist = sentencelist[1].lstrip().rstrip().split("@")[1].lstrip().rstrip().split()
            
            keyword_ptr = [int(n) for n in keywordlist[1:]] 
            keywordno = int(keywordlist[0])
            wordno = int(wordlist[0])
            wordlist_ptr_ = wordlist[1:]
            words_ptr = []
            words_cnt_ptr = []
            
            for item in wordlist_ptr_:
                key = item.split(":")[0]
                value = item.split(":")[1]
                words_ptr.append(int(key))
                words_cnt_ptr.append(int(value))
                allwordno += int(value)
                
            sentence = Sentence(senid, keywordno, wordno, self.num_topics, 100, keyword_ptr, words_ptr, words_cnt_ptr, self.neighbor)
            self.corpus.append(sentence)
            self.num_total_words += allwordno
            self.num_docs += 1
            senid+=1
            #print("num_sentences: "+str(self.num_docs))
    
    def read_sentences(self, filename):
        senid =0
        datalist = open(filename, "r", encoding = "utf-8").readlines()
        for sentence in datalist:
            allwordno = 0
            wordno, wordlist = codetext(sentence,self.model)
            keyword_ptr = list(wordlist)
            keywordno = wordno
            words_ptr = []
            words_cnt_ptr = []
            for key, value in wordlist.items():
                words_ptr.append(key)
                words_cnt_ptr.append(value)
                allwordno += value
            sentence = Sentence(senid, keywordno, wordno, self.num_topics, 100, keyword_ptr, words_ptr, words_cnt_ptr, self.neighbor)
            
            self.corpus.append(sentence)
            self.num_total_words += allwordno
            self.num_docs += 1
            senid+=1
            #print("num_sentences: "+str(self.num_docs))
            
    def inference(self):
        cur_round_begin_time = time.time()
        managerDoclist = []
        for i in range(len(self.corpus)):
            
            sentence = self.corpus[i]
            
            all_context_topics, all_context_topics_xi = self.do_inferencebywords(sentence) #not log
            vector1 = all_context_topics # \sigma_N p(w_n|T) \sigma_Wn p(T | W_n)  without attentions
            vector2 = all_context_topics_xi # \sigma_N xi_n * p(w_n|T) \sigma_Wn p(T | W_n)  with attentions            
            temp = np.array([exp(sentence.log_topic[n]) for n in range(self.num_topics)])
            vector3 = np.add(temp,all_context_topics) # p(sentence | previous, following, w_n)  \sigma_N p(w_n|T) \sigma_Wn p(T | W_n) 
            vector4 = sentence.log_topic
            
            
            vector5 = np.concatenate((vector1,vector2))
            vector6 = np.concatenate((vector1,vector3))
            vector7 = np.concatenate((vector2,vector3))
            vector8 = np.concatenate((vector1,vector2,vector3))
            vector9 = np.concatenate((vector1,vector4))
            
            ##more
            managerDoclist.append((sentence.sentenceid,vector1,vector2, vector3, vector4, vector5, vector6, vector7, vector8,vector9))
            
        cur_round_cost_time = time.time() - cur_round_begin_time
        pid = os.getpid()
        print("Process "+str(pid)+", doc "+ str(i)+" cost_time= "+str(cur_round_cost_time))
        all_sentence_sorted = sorted(managerDoclist, key=lambda item : item[0])
        return all_sentence_sorted     
    
    def multiprocesses_inferece(self):
        workers = []
        workers_no = self.configuration.num_threads
        
        corpusSplitlist = split_average_data(workers_no, self.corpus)
        manager = Manager()
        ManagerReturn_corpusSplitlist = []
        for dataSplit in corpusSplitlist:
            return_dataSplit = manager.list()
            
            worker = Process(target=self.inferenceDatasplit, args=(dataSplit, return_dataSplit))
            
            worker.start()
            workers.append(worker)
            ManagerReturn_corpusSplitlist.append(return_dataSplit)
        for w in workers:
            w.join()
            pass
        
        corpus = []
        for split_ in ManagerReturn_corpusSplitlist:
            for sen in split_:
                corpus.append(sen)
        
        print(len(corpus))
        all_sentence_sorted = sorted(corpus, key=lambda item : item[0])
        return all_sentence_sorted
    
    def inferenceDatasplit(self, datasplit, managerDoclist):
        datasize = len(datasplit)
        cur_round_begin_time = time.time()
        for i in range(datasize):
            
            sentence = datasplit[i]
            
            all_context_topics, all_context_topics_xi = self.do_inferencebywords(sentence) #not log
            vector1 = all_context_topics # \sigma_N p(w_n|T) \sigma_Wn p(T | W_n)  without attentions
            vector2 = all_context_topics_xi # \sigma_N xi_n * p(w_n|T) \sigma_Wn p(T | W_n)  with attentions            
            temp = np.array([exp(sentence.log_topic[n]) for n in range(self.num_topics)])
            vector3 = np.add(temp,all_context_topics) # p(sentence | previous, following, w_n)  \sigma_N p(w_n|T) \sigma_Wn p(T | W_n) 
            vector4 = sentence.log_topic
            
            
            vector5 = np.concatenate((vector1,vector2))
            vector6 = np.concatenate((vector1,vector3))
            vector7 = np.concatenate((vector2,vector3))
            vector8 = np.concatenate((vector1,vector2,vector3))
            vector9 = np.concatenate((vector1,vector4))
            
            ##more
            managerDoclist.append((sentence.sentenceid,vector1,vector2, vector3, vector4, vector5, vector6, vector7, vector8,vector9))
        cur_round_cost_time = time.time() - cur_round_begin_time
        pid = os.getpid()
        print("Process "+str(pid)+", doc "+ str(i)+" cost_time= "+str(cur_round_cost_time))
        pass

    def do_inferencebywords(self,sentence):
        
        all_context_topics = np.array([0.0 for n in range(self.num_topics)])
        sigma_all_words_topic_beta = np.array([0.0 for n in range(self.num_topics)]) # not log

        all_context_topics_xi = np.array([0.0 for n in range(self.num_topics)]) # not log
        
        wordno = len(sentence.words_ptr)
        for i in range(self.num_topics):
            for wno in range(wordno):
                sigma_all_words_topic_beta[i] += exp(self.model.log_beta[i][sentence.words_ptr[wno]]) * sentence.words_cnt_ptr[wno]
            
        for wno in range(wordno):
            #word is an id
            wid = sentence.words_ptr[wno]
            widxi = sentence.sentenceattention[wid]
            log_topics_word = self.model.log_phi[wid][:]
            for i in range(self.num_topics):
                all_context_topics[i] += exp(log_topics_word[i]) * sigma_all_words_topic_beta[i]
                all_context_topics_xi[i] += widxi * exp(log_topics_word[i]) * sigma_all_words_topic_beta[i]
                
        return all_context_topics, all_context_topics_xi
                    
                

def reconstruct_beta_phi_pi(betafile, phifile, pifile,vocabularyfile, num_topics):
    dictionary = []
    dicsDis ={}
    DICTIONARY = {}
        
    phi = open(phifile, "r", encoding = "utf-8")
    pif = open(pifile, "r", encoding = "utf-8")
    vocabularyf = open(vocabularyfile, "r", encoding = "utf-8")
        
    for vs in vocabularyf:
        index = int(vs.split(":")[0])
        word = vs.split(":")[1].rstrip().lstrip()
        DICTIONARY.setdefault(word,index)
        dictionary.append(vs.split(":")[1].strip().lstrip().rstrip())
    
    num_all_keywords = len(dictionary)
    
    pi = []
    for p in pif:
        pi.append(float(p.lstrip().rstrip()))
  
    log_phi = np.zeros((num_all_keywords, num_topics), dtype = np.float64)
    
    wordno = 0
    for ts in phi:
        topicslist = np.array([float(m) for m in ts.split()], dtype=np.float64)
        dicsDis.setdefault(dictionary[wordno])
        dicsDis[dictionary[wordno]] = topicslist
        log_phi[wordno] = topicslist[:]
        wordno += 1
        
    log_beta = np.loadtxt(betafile)
    return dictionary, DICTIONARY, dicsDis, pi, log_phi, log_beta

def convertToSVMinput(inputsourceVectors, inputsourceLabels,svmfile):
    vectors = []
    classlabel = []
    classlabeltypes = []
    vectorDimension = 0
    for line in inputsourceVectors:
        #vector = line.strip().rstrip().lstrip().split(" ")
        vectorDimension = len(line)
        vectors.append(line)
    inputsourceLabels.seek(0)
    for lab in inputsourceLabels:
        lab = lab.strip().rstrip().lstrip()
        if lab not in classlabeltypes:
            classlabeltypes.append(lab)
    inputsourceLabels.seek(0)
    for linel in inputsourceLabels:
        label = linel.strip().rstrip().lstrip()
        classlabel.append(classlabeltypes.index(label))

    for i in range(len(vectors)):
        svmfile.write(str(classlabel[i]) + " ")
        temp = vectors[i]
        for j in range(vectorDimension):
            svmfile.write(str(j))
            svmfile.write(":")
            svmfile.write(str(temp[j]))
            svmfile.write(" ")
        svmfile.write('\n')
    svmfile.flush()
    svmfile.close()

def svmClassifer5Fold(svmfile):
    Data_train, label_train = load_svmlight_file(svmfile)
    
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(Data_train.shape[0]))
    
    x_shuffled = Data_train[shuffle_indices]
    y_shuffled = label_train[shuffle_indices]
    
    #vectorNumber = x_shuffled.shape[0]
        
    #kf = cross_validation.KFold(vectorNumber, n_folds=5, shuffle=True, random_state=3)
    kf = KFold(n_splits=5)
    
    scoreNum = []
    f1Num = []
    for train_index, test_index in kf.split(x_shuffled):
        X_train, X_test = x_shuffled[train_index], x_shuffled[test_index]
        y_train, y_test = y_shuffled[train_index], y_shuffled[test_index]

        f1value, score = computeAcc(X_train, y_train,X_test, y_test)
        
        scoreNum.append(score)
        f1Num.append(f1value)
    return scoreNum,f1Num
    
def computeACCwithrandomforest(X_train, y_train,X_test, y_test):
    clf = RandomForestClassifier(min_samples_leaf=20, n_jobs = 30)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    f1value = f1_score(y_test, y_pred,average="micro")
    acc = clf.score(X_test, y_test)
    return f1value, acc
    pass

def computeAcc(X_train, y_train,X_test, y_test):
    clf = SVC(C=2.0, cache_size=200, class_weight=None, coef0=0.0, degree=5, kernel='rbf', max_iter=-1, probability=True,
    random_state=None, shrinking=True, tol=0.1, verbose=False)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    f1value = f1_score(y_test, y_pred,average="micro")
    acc = clf.score(X_test, y_test)
    return f1value, acc

def crossvalidationfunction(oDir, num,):
    print("\n start classify: "+str(num))
    start1 = time.time()
    
    senvector_svmfile = open(oDir +"senvectorsvmFormatVectors_"+str(num), "rb")
    senvectorscoreNum,f1Num = svmClassifer5Fold(senvector_svmfile)
    print("The accuracy of sentence vectors is : "+str(num))
    print(str(np.sum(senvectorscoreNum) / 5.0) + " acc  std : " + str(np.std(senvectorscoreNum)))
    print(str(num)+"------------")
    print("The f1 of sentence vectors is : "+str(num))
    print(str(np.sum(f1Num) / 5.0) + " std: " + str(np.std(f1Num)))
    print(str(num)+"*****************")
    
    end1 = time.time()
    print("end classify "+str(num)+", and the time is " + str(end1-start1)+"\n\n")

if __name__ == '__main__':
    if (len(sys.argv) != 12):
        print(" ddd")
        print(" ")
        exit(0)
    
    inputtestfile = sys.argv[1]
    inputsourceLabels = open(sys.argv[2],"r", encoding = "utf-8")
    settingsfile = sys.argv[3]
    pifile = sys.argv[4]
    betafile = sys.argv[5]
    phifile = sys.argv[6]
    vocabularyfile = sys.argv[7]
    stopwords = readstopwords(sys.argv[8])
    sentencetopicfile = sys.argv[9]
    sentencexifile = sys.argv[10]
    
    model_root = sys.argv[11]
    
    if model_root.endswith("/") is False:
        model_root = model_root+"/"
    
    configuration = Configuration(settingsfile)
    
    dictionary,DICTIONARY, dicsDis, pi, log_phi, log_beta = reconstruct_beta_phi_pi(betafile, phifile, pifile,vocabularyfile, configuration.num_topics)
    
    #reconstruct model
    print("Build the model...")
    model = Model(configuration.num_topics, configuration.neighbor, dictionary, DICTIONARY, pi, log_phi, log_beta)

    ese = ESE(inputtestfile, sentencetopicfile, sentencexifile, configuration, model)
    print("begin multiprocesses_inferece")
    #all_sentence_sorted = ese.multiprocesses_inferece()
    all_sentence_sorted = ese.inference()
    
    print("begin build the vectors")
    all_senvector1 = []; all_senvector2 = []; all_senvector3 = [];all_senvector4 = []; all_senvector5 = []; all_senvector6 = [];all_senvector7 = [];all_senvector8 = [];all_senvector9 = [];
    
    for sen in all_sentence_sorted:
        all_senvector1.append(sen[1])
        all_senvector2.append(sen[2])
        all_senvector3.append(sen[3])
        all_senvector4.append(sen[4])
        all_senvector5.append(sen[5])
        all_senvector6.append(sen[6])
        all_senvector7.append(sen[7])
        all_senvector8.append(sen[8])
        all_senvector9.append(sen[9])
        
    print("convert vectors to svm format...")
    
    senvector1svmfile = open(model_root +"senvectorsvmFormatVectors_1", "w", encoding = "utf-8")
    convertToSVMinput(all_senvector1, inputsourceLabels,senvector1svmfile)
    
    senvector2svmfile = open(model_root +"senvectorsvmFormatVectors_2", "w", encoding = "utf-8")
    convertToSVMinput(all_senvector2, inputsourceLabels,senvector2svmfile)
    
    senvector3svmfile = open(model_root +"senvectorsvmFormatVectors_3", "w", encoding = "utf-8")
    convertToSVMinput(all_senvector3, inputsourceLabels,senvector3svmfile)
    
    senvector4svmfile = open(model_root +"senvectorsvmFormatVectors_4", "w", encoding = "utf-8")
    convertToSVMinput(all_senvector4, inputsourceLabels,senvector4svmfile)
    
    senvector5svmfile = open(model_root +"senvectorsvmFormatVectors_5", "w", encoding = "utf-8")
    convertToSVMinput(all_senvector5, inputsourceLabels,senvector5svmfile)
    
    senvector6svmfile = open(model_root +"senvectorsvmFormatVectors_6", "w", encoding = "utf-8")
    convertToSVMinput(all_senvector6, inputsourceLabels,senvector6svmfile)
    
    senvector7svmfile = open(model_root +"senvectorsvmFormatVectors_7", "w", encoding = "utf-8")
    convertToSVMinput(all_senvector7, inputsourceLabels,senvector7svmfile)
    
    senvector8svmfile = open(model_root +"senvectorsvmFormatVectors_8", "w", encoding = "utf-8")
    convertToSVMinput(all_senvector8, inputsourceLabels,senvector8svmfile)
    
    senvector9svmfile = open(model_root +"senvectorsvmFormatVectors_9", "w", encoding = "utf-8")
    convertToSVMinput(all_senvector9, inputsourceLabels,senvector9svmfile)
    
    
    
    print("begin to multiProcesses classify...\n")
    
    works = []
    worker1 = Process(target=crossvalidationfunction, args=(model_root, 1,))
    worker1.start()
    works.append(worker1)
    
    worker2 = Process(target=crossvalidationfunction, args=(model_root, 2,))
    worker2.start()
    works.append(worker2)
    
    worker3 = Process(target=crossvalidationfunction, args=(model_root, 3,))
    worker3.start()
    works.append(worker3)
    
    worker4 = Process(target=crossvalidationfunction, args=(model_root, 4,))
    worker4.start()
    works.append(worker4)
    
    worker5 = Process(target=crossvalidationfunction, args=(model_root, 5,))
    worker5.start()
    works.append(worker5)
    
    worker6 = Process(target=crossvalidationfunction, args=(model_root, 6,))
    worker6.start()
    works.append(worker6)
    
    worker7 = Process(target=crossvalidationfunction, args=(model_root, 7,))
    worker7.start()
    works.append(worker7)
    
    worker8 = Process(target=crossvalidationfunction, args=(model_root, 8,))
    worker8.start()
    works.append(worker8)
    
    worker9 = Process(target=crossvalidationfunction, args=(model_root, 9,))
    worker9.start()
    works.append(worker9)
    
    
    for work in works:
        work.join()
    
    
#     
#     print("start classify 1 .\n")
#     start1 = time.time()
#     crossvalidationfunction(model_root, 1)
#     end1 = time.time()
#     print("end classify 1, and the time is " + str(end1-start1))
#     
#     print("start classify 2 .\n")
#     start2 = time.time()
#     crossvalidationfunction(model_root, 2)
#     end2 = time.time()
#     print("end classify 2, and the time is " + str(end2-start2))
#     
#     print("start classify 3 .\n")
#     start3 = time.time()
#     crossvalidationfunction(model_root, 3)
#     end3 = time.time()
#     print("end classify 0, and the time is " + str(end3-start3))
    
    
    
#     works = []
#     worker0 = Process(target=crossvalidationfunction, args=(model_root, 0,))
#     worker0.start()
#     works.append(worker0)
#     
#     
#     worker1 = Process(target=crossvalidationfunction, args=(model_root,1,))
#     worker1.start()
#     works.append(worker1)
#     
#     worker2 = Process(target=crossvalidationfunction, args=(model_root, 2,))
#     worker2.start()
#     works.append(worker2)
#     
#     for work in works:
#         work.join()
        
    pass
        