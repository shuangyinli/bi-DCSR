'''
Created on Apr 2, 2019

@author: shuangyinli
'''


a = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

b = [',','.','/','\\','[',']','{','}',':',';','\'','\"','|','~','!','@','#','$','%','^','&','*','(',')','_','-','+','=','1','2','3','4','5','6','7','8','9','0','<','>']

aa =[]
bb=[]
ab=[]
ba=[]

stopfile = open("/Users/huihui/Data/wiki/ese_out/newstops.txt","w")

for i in a:
    for j in a:
        aa.append(i+j)
        stopfile.write(i+j)
        stopfile.write("\n")
        
for i in b:
    for j in b:
        bb.append(i+j)
        stopfile.write(i+j)
        stopfile.write("\n")

for i in a:
    for j in b:
        ab.append(i+j)
        stopfile.write(i+j)
        stopfile.write("\n")

for i in b:
    for j in a:
        ba.append(i+j)
        stopfile.write(i+j)
        stopfile.write("\n")
        
stopfile.close()

