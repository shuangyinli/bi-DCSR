import os
import os.path
import sys
import random

alllines = []
    
def combinefiles(oDir,headname):
    #read all the partDic
    path_list = os.listdir(oDir)
    partDicfile = []
    for path in path_list:
        if headname in path:
            partDicfile.append(path)
    
    for dicpath in partDicfile:
        dicfileopen = open(oDir+dicpath,"r", encoding = "utf-8")
        print(oDir+dicpath)
        for line in dicfileopen:
            key = line.rstrip().lstrip()
            alllines.append(key)
            
    random.shuffle(alllines)
    
def SaveDictionary(dicfile):
    print("\n now save the codes.")
    print("We have "+str(len(alllines))+". \n")
    for sen in alllines:
        dicfile.write(sen+"\n")
    dicfile.close()

if __name__ == '__main__':
    
    oDir = sys.argv[1]
    if oDir.endswith("/") is False:
        oDir = oDir+"/"
    headname = sys.argv[2]
    endname = sys.argv[3]
    
    combinefiles(oDir,headname)
    dicfile = open(oDir+endname+".txt","w", encoding = "utf-8")
    SaveDictionary(dicfile)
    
    
    
    