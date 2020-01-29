'''
Created on May 5, 2019

@author: shuangyinli
'''
inputfile = open("","r")

labels = {}

for line in inputfile:
    label = line.rstrip().lstrip().split("##")[0].rstrip().lstrip()
    
    if label not in labels:
        labels.setdefault(label,1)
    else:
        labels[label] +=1
        

a = sorted(labels.items(), key=lambda d: d[1], reverse=True)

print(a)
print(len(a))