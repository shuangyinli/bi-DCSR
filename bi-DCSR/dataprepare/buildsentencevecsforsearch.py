'''
Created on Apr 27, 2019

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
from tqdm import tqdm



class Sentence():
    def __init__(self, num_keywords_, num_words_, num_topics_, lik_,
                keywords_ptr_, words_ptr_, words_cnt_ptr_, neighbor_):
        '''
        Constructor
        '''
        # ensure the args are numpy array
    
        self.num_keyword = num_keywords_
        self.num_words = num_words_
        self.num_topics = num_topics_
        self.lik = lik_
        self.keywords_ptr = keywords_ptr_
        self.words_ptr = words_ptr_
        self.words_cnt_ptr = words_cnt_ptr_
        self.neighbor = neighbor_

        self.log_topic = np.zeros(self.num_topics, dtype=np.float64)
                

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


class ESE():
    def __init__(self, originalsentencesfile, codedatafilename, sentencetopicfile, configuration, model):
        '''
        Constructor
        '''
        self.corpus = []
        self.num_docs=0
        
        self.N= 0

        self.NX = {} # idf
        
        self.wordset_list = []  # keep word_id and the frequency for each doc
        
        self.doc_select_words = [] # keep the selected keyword list for each doc
        
        self.original_corpus =[]
        self.labellist =[]
        
        self.corpus_coded = []

        self.num_words = len(model.dictionary)
        self.configuration = configuration
        self.model = model
        self.num_total_words = 0
        
        self.num_topics = self.configuration.num_topics
        self.neighbor = self.configuration.neighbor
        
        if codedatafilename != "":
            self.read_sentencesforesecode(codedatafilename)
        
        self.select_keyword_bytfidf()
        
        if sentencetopicfile != "":
            self.read_sentencetopics(sentencetopicfile)
        
        print("read originalsentences...")
        
        if originalsentencesfile != "":
            self.read_originaldocandsentences(originalsentencesfile)
        
        
    
    def read_originaldocandsentences(self,originalsentencesfile):
        originalsentences = open(originalsentencesfile, "r", encoding = "utf-8").readlines()
        for doc in tqdm(originalsentences):
            label = doc.lstrip().rstrip().split("@")[0].lstrip().rstrip()
            sentences = doc.lstrip().rstrip().split("@")[1].lstrip().rstrip().split("##")
            self.labellist.append(label)
            sentencelist = []
            for sen in sentences:
                if sen.rstrip().lstrip() != "" and sen.rstrip().lstrip() != " ":
                    sentencelist.append(sen.rstrip().lstrip())
            self.original_corpus.append(sentencelist)
            
        pass
      
    def select_keyword_bytfidf(self):
        for key,value in self.NX.items():
            self.NX[key] = math.log((self.N+1)/(value+1) +1)
        
        for item in self.wordset_list:
            
            tfidf_select = []
            for key, value in item.items():
                item[key] = value * self.NX[key]
                
            a = sorted(item.items(), key=lambda d: d[1], reverse=True)
            count = int(len(a)*0.5)
            
            temp = a[:count]
            for t in temp:
                tfidf_select.append(t[0])
            
            self.doc_select_words.append(tfidf_select)
        
    def read_sentencesforesecode(self,filename):
        datalist = open(filename, "r", encoding = "utf-8").readlines()
        
        print("now read_sentencesforesecode in ESE")

        for doc in tqdm(datalist):
            self.N +=1
            wordset = {}
            
            sentencelist = doc.split("#")[1:]
            sentences = []
            self.corpus_coded.append(sentencelist)
            for sen in sentencelist:
                keywordlist = sen.lstrip().rstrip().split("@")[0].lstrip().rstrip().split()
                wordlist = sen.lstrip().rstrip().split("@")[1].lstrip().rstrip().split()
                
                keyword_ptr = [int(n) for n in keywordlist[1:]]
                keywordno = int(keywordlist[0])
                wordno = int(wordlist[0])
                wordlist_ptr_ = wordlist[1:]
                words_ptr = []
                words_cnt_ptr = []
                for item in wordlist_ptr_:
                    key = int(item.split(":")[0])
                    value = int(item.split(":")[1])
                    words_ptr.append(key)
                    words_cnt_ptr.append(value)
                    if key in wordset:
                        wordset[key] += value
                    else:
                        wordset.setdefault(key, value)                
                    
                sentence = Sentence(keywordno, wordno, self.num_topics, 100, keyword_ptr, words_ptr, words_cnt_ptr, self.neighbor)
                sentences.append(sentence)
            
            for key, value in wordset.items():
                if key in self.NX:
                    self.NX[key] += 1
                else:
                    self.NX.setdefault(key,1)
            self.wordset_list.append(wordset)
            
            self.corpus.append(sentences)

            self.num_docs += 1
        
    
    def read_sentencetopics(self, sentencetopicfile):
        datalist = open(sentencetopicfile, "r", encoding = "utf-8").readlines()
        lineno = 0
        print("now read_sentencetopics in ESE")
        
        for line in tqdm(datalist):
            linelist = line.lstrip().rstrip().split("#")
            #self.sentencetopics = linelist[0].rstrip().lstrip().split()
            sentences_log = []
            for sen_log in linelist:
                sen_log = sen_log.lstrip().rstrip()
                if sen_log != "" and sen_log != " ":
                    temp = [float(n) for n in sen_log.rstrip().lstrip().split()]
                    sentences_log.append(temp)
            
            sen_num = len(sentences_log)
            
            for i in range(sen_num):
                self.corpus[lineno][i].log_topic = array(sentences_log[i]).copy()
            lineno +=1

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
        all_doc_sorted = sorted(corpus, key=lambda item : item[0])
        return all_doc_sorted
    
    def inferenceDatasplit(self,datasplit, managerDoclist):
        #managerDoclist = []
        #print("now inference for the corpus in ESE")
        for i in tqdm(range(len(datasplit))):

            sentences = self.corpus[i][1]
            docid = self.corpus[i][0]
            vectorsentences = []
       
            for sen in sentences:
                all_context_topics = self.do_inferencebywords(sen) #not log
                vector1 = all_context_topics # \sigma_N p(w_n|T) \sigma_Wn p(T | W_n)  without attentions
                temp = np.array([exp(sen.log_topic[n]) for n in range(self.num_topics)])
                vector2 = np.add(temp,all_context_topics) # p(sentence | previous, following, w_n)  \sigma_N p(w_n|T) \sigma_Wn p(T | W_n) 
                vector3 = sen.log_topic
                vector4 = np.concatenate((vector1,vector2))
                vector5 = np.concatenate((vector1,vector3))
                vector6 = np.concatenate((vector2,vector3))
                vector7 = np.concatenate((vector1,vector2,vector3))
                
                vectorsentences.append((vector1,vector2, vector3, vector4, vector5, vector6, vector7))

            managerDoclist.append((docid,vectorsentences))
        return managerDoclist  
    
    def inference(self):
        managerDoclist = []
        
        print("now inference for the corpus in ESE")
        
        ERROR = 0
        for i in tqdm(range(len(self.corpus))):

            sentences = self.corpus[i]
            #vectorsentences = []
            set_select_keywords_doc = set(self.doc_select_words[i])
            num_sen = len(sentences)
            
            if len(self.original_corpus[i]) != num_sen or len(self.corpus_coded[i]) != num_sen or len(self.original_corpus[i]) != len(self.corpus_coded[i]):
                print("ERROR: " + str(ERROR))
                print("len(self.original_corpus[i]) : " + str(len(self.original_corpus[i])))
                print("len(self.corpus_coded[i]) : "+str(len(self.corpus_coded[i])))
                print("len(sentences): " + str(len(sentences)))
                ERROR +=1
                continue
            
            for s in range(num_sen):
                sen = sentences[s]
                select_words = set(sen.words_ptr).intersection(set_select_keywords_doc)
                if len(select_words) < 12:
                    continue
                
                all_context_topics = self.do_inferencebywords(sen,select_words) #not log
                vector1 = all_context_topics # \sigma_N p(w_n|T) \sigma_Wn p(T | W_n)  without attentions
                vector2 = np.array([0.0 for n in range(self.num_topics)])
                temp = np.array([exp(sen.log_topic[n]) for n in range(self.num_topics)])
                for k in range(self.num_topics):
                    vector2[k] = all_context_topics[k] * temp[k] # p(sentence | previous, following, w_n)  \sigma_N p(w_n|T) \sigma_Wn p(T | W_n) 
                
                #vector4 = np.concatenate((vector1,vector3))
                label = self.labellist[i]
                
                orginal_sentencesstring = self.original_corpus[i][s]
                log_topic_vector = sen.log_topic
                coded_file = self.corpus_coded[i][s]
                #vectorsentences.append((vector1,vector2, vector3))

                managerDoclist.append([label, orginal_sentencesstring, coded_file, log_topic_vector, vector1, vector2])

        #all_sentence_sorted = sorted(managerDoclist, key=lambda item : item[0])
        return managerDoclist     
    

    def do_inferencebywords(self,sentence,select_words):
        
        all_context_topics = np.array([0.0 for n in range(self.num_topics)])
        sigma_all_words_topic_beta = np.array([0.0 for n in range(self.num_topics)]) # not log
        
        for i in range(self.num_topics):
            for wno in select_words:
                sigma_all_words_topic_beta[i] += exp(self.model.log_beta[i][wno]) * sentence.words_cnt_ptr[sentence.words_ptr.index(wno)]
            
        for wno in select_words:
            #word is an id
            #wid = sentence.words_ptr[wno]
            # wno is the word id in dic
            log_topics_word = self.model.log_phi[wno][:]
            for i in range(self.num_topics):
                all_context_topics[i] += exp(log_topics_word[i]) * sigma_all_words_topic_beta[i]
            
        return all_context_topics

def split_average_data(thread_no, text_list):
    fn = len(text_list)//thread_no
    rn = len(text_list)%thread_no
    ar = [fn+1]*rn+ [fn]*(thread_no-rn)
    si = [i*(fn+1) if i<rn else (rn*(fn+1)+(i-rn)*fn) for i in range(thread_no)]
    corpusSplitlist = [text_list[si[i]:si[i]+ar[i]] for i in range(thread_no)]
    return corpusSplitlist
               
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



if __name__ == '__main__':
    
    if (len(sys.argv) != 10):
        print(" python build <originalsentences> <codefile> <sentence_log_topic_file> <pi_file> <betafile> <phi_file> <vocabularyfile> <settingsfile> <model_root>")
        print(" ")
        exit(0)
    
    #originalsentences = open("/Users/huihui/experiment/ese/search/wiki500_n2_init_beta_update_beta_phi/eseOriginalfile","r")  # not total-labeled-wikipages5. I have processed one orginal file
    originalsentences = sys.argv[1]
    #codefile = open("/Users/huihui/experiment/ese/search/wiki500_n2_init_beta_update_beta_phi/esewiki_test","r")
    codefile = sys.argv[2]
    
    sentence_log_topic_file = sys.argv[3]
    
    #labelfile = open("","r")
    pifile = sys.argv[4]
    betafile = sys.argv[5]
    phifile = sys.argv[6]
    vocabularyfile = sys.argv[7]
    settingsfile = sys.argv[8]
    model_root = sys.argv[9]
    
    if model_root.endswith("/") is False:
        model_root = model_root+"/"
    
    configuration = Configuration(settingsfile)
    
    dictionary,DICTIONARY, dicsDis, pi, log_phi, log_beta = reconstruct_beta_phi_pi(betafile, phifile, pifile,vocabularyfile, configuration.num_topics)
    
    print("Build the model...")
    model = Model(configuration.num_topics, configuration.neighbor, dictionary, DICTIONARY, pi, log_phi, log_beta)

    print("Build the ESE to get the vectors...")
    ese = ESE(originalsentences, codefile, sentence_log_topic_file, configuration, model)
    
    #all_doc_sorted = ese.multiprocesses_inferece()
    
    all_doc_sorted = ese.inference()
        
    print("begin to verify and write.. : " + str(len(all_doc_sorted)))
    
    wiki_sentencelevel_vectorsfile = open(model_root +"wiki_sentencelevel_vectors", "w", encoding = "utf-8")
    num_select = len(all_doc_sorted)
    
    for i in range(num_select):
        wiki_sentencelevel_vectorsfile.write(str(all_doc_sorted[i][0].lstrip().rstrip()).lstrip().rstrip()+"##")
        wiki_sentencelevel_vectorsfile.write(str(all_doc_sorted[i][1].lstrip().rstrip()).lstrip().rstrip()+"##")
        wiki_sentencelevel_vectorsfile.write(str(all_doc_sorted[i][2].lstrip().rstrip()).lstrip().rstrip()+"##")
        for item in all_doc_sorted[i][3]:
            wiki_sentencelevel_vectorsfile.write(str(item).lstrip().rstrip()+" ")
        wiki_sentencelevel_vectorsfile.write("##")
        
        for item in all_doc_sorted[i][4]:
            wiki_sentencelevel_vectorsfile.write(str(item).lstrip().rstrip()+" ")
        wiki_sentencelevel_vectorsfile.write("##")
        
        for item in all_doc_sorted[i][5]:
            wiki_sentencelevel_vectorsfile.write(str(item).lstrip().rstrip()+" ")
        wiki_sentencelevel_vectorsfile.write("\n")
        wiki_sentencelevel_vectorsfile.flush()
        
    wiki_sentencelevel_vectorsfile.close()
    
        
    