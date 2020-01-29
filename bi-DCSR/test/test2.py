'''
Created on Apr 12, 2019

@author: shuangyinli
'''

import numpy as np
import math
from tqdm import tqdm
from progressbar import *


all_context_topics = np.array([0.0 for n in range(5)])

for i in range(5):
    all_context_topics[i] = i
    
print(all_context_topics)


all_context_topic = np.array([0.0 for n in range(4)])

for i in range(4):
    all_context_topic[i] = 8
    
print(all_context_topic)

vector = np.concatenate((all_context_topic, all_context_topics))

print(vector)


all_context_topicsd = np.array([math.exp(all_context_topics[n]) for n in range(5)])


print(all_context_topicsd)


a = ((1,3),(3,4))

print(list(a))
print(set(a))


wrfile = open("/Users/huihui/Downloads/test","w")

a = np.array([n for n in range(10)])

print(str(a[1]) + " dd")
print(list(a))

wrfile.write(str(list(a)))



a = [1,2,3]
b=[3,4,5]

print(a+b)



    
    
    










