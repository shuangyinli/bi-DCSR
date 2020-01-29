'''
Created on Apr 29, 2019

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


def convertToSVMinput(inputsourceVectors, inputsourceLabels,svmfile):
    vectors = []
    classlabel = []
    classlabeltypes = []
    vectorDimension = 0
    for line in inputsourceVectors:
        #vector = line.strip().rstrip().lstrip().split(" ")
        vectorDimension = len(line)
        vectors.append(line)
    #inputsourceLabels.seek(0)
    for lab in inputsourceLabels:
        lab = lab.strip().rstrip().lstrip()
        if lab not in classlabeltypes:
            classlabeltypes.append(lab)
    #inputsourceLabels.seek(0)
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
    
    if (len(sys.argv) != 3):
        print(" python build <dropoutfile>  <model_root>")
        print(" ")
        exit(0)
    
    openfile = open(sys.argv[1], "r", encoding = "utf-8").readlines()
    model_root = sys.argv[2]
    
    all_senvector1 = []; all_senvector2 = []; all_senvector3 = [];all_senvector4 = [];all_senvector5 = [];all_senvector6 = [];all_senvector7 = [];
    
    Labels = []
    
    for line in openfile:
        linelist = line.lstrip().rstrip().split("##")
        Labels.append(linelist[0])
        all_senvector1.append(linelist[3].lstrip().rstrip().split()) # log p(sentence | previous, following, w_n)
        all_senvector2.append(linelist[4].lstrip().rstrip().split()) # \sigma_N p(w_n|T) \sigma_Wn p(T | W_n)  without attentions
        all_senvector3.append(linelist[5].lstrip().rstrip().split()) # p(sentence | previous, following, w_n)  \sigma_N p(w_n|T) \sigma_Wn p(T | W_n) 
        all_senvector4.append(linelist[3].lstrip().rstrip().split() + linelist[4].lstrip().rstrip().split())
        all_senvector5.append(linelist[3].lstrip().rstrip().split() + linelist[5].lstrip().rstrip().split())
        all_senvector6.append(linelist[4].lstrip().rstrip().split() + linelist[5].lstrip().rstrip().split())
        all_senvector7.append(linelist[3].lstrip().rstrip().split() + linelist[4].lstrip().rstrip().split() + linelist[5].lstrip().rstrip().split())
        
    
    print("convert vectors to svm format...")
    
    senvector1svmfile = open(model_root +"senvectorsvmFormatVectors_1", "w", encoding = "utf-8")
    convertToSVMinput(all_senvector1, Labels,senvector1svmfile)
    
    senvector2svmfile = open(model_root +"senvectorsvmFormatVectors_2", "w", encoding = "utf-8")
    convertToSVMinput(all_senvector2, Labels,senvector2svmfile)
    
    senvector3svmfile = open(model_root +"senvectorsvmFormatVectors_3", "w", encoding = "utf-8")
    convertToSVMinput(all_senvector3, Labels,senvector3svmfile)
    
    senvector4svmfile = open(model_root +"senvectorsvmFormatVectors_4", "w", encoding = "utf-8")
    convertToSVMinput(all_senvector4, Labels,senvector4svmfile)
    
    senvector5svmfile = open(model_root +"senvectorsvmFormatVectors_5", "w", encoding = "utf-8")
    convertToSVMinput(all_senvector5, Labels,senvector5svmfile)
    
    senvector6svmfile = open(model_root +"senvectorsvmFormatVectors_6", "w", encoding = "utf-8")
    convertToSVMinput(all_senvector6, Labels,senvector6svmfile)
    
    senvector7svmfile = open(model_root +"senvectorsvmFormatVectors_7", "w", encoding = "utf-8")
    convertToSVMinput(all_senvector7, Labels,senvector7svmfile)
    
    
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
        
    for work in works:
        work.join()
        
        