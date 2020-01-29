'''
Created on Apr 4, 2019

@author: shuangyinli
'''
import ast
import random

def samplephysics(inputfile, outputfile,category):
    lines = []
    for line in inputfile:
        tempdic = ast.literal_eval(line)
        stringtemp = category + "##" + tempdic['abstract'] + "\n"
        lines.append(stringtemp)
        
    random.shuffle(lines)
    
    for i in range(50000):
        outputfile.write(lines[i])
        pass
    outputfile.close()
    pass

def parsearxivpaper(inputfile, outputfile,category):
    for line in inputfile:
        #print(line)
        tempdic = ast.literal_eval(line)
        #tempdic = dict(line)
        print(tempdic['abstract'])
        outputfile.write(category + "##" + tempdic['abstract'] + "\n")  
        #print(line)
    outputfile.close()
    pass

if __name__ == '__main__':
    
    inputfile = open("/Users/huihui/Data/wiki/ese_out/arxiv/physics","r",encoding='utf-8')
    outputfile = open("/Users/huihui/Data/wiki/ese_out/arxiv/total.txt","a")
    category = "physics"
    
    #parsearxivpaper(inputfile, outputfile, category)
    samplephysics(inputfile, outputfile,category)
    
    
    pass
