'''
Created on May 13, 2019

@author: shuangyinli
'''
# -*- coding: utf-8 -*-



if __name__ == '__main__':
    
    openfile = open("","r")
    
    total = 0
    
    for line in openfile:
        wordlist = line.strip().split()[1:]
        for item in wordlist:
            word = item.strip().split(":")[0]
            count = int(item.strip().split(":")[1])
            total += count
    
    pass