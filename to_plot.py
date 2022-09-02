import matplotlib.pyplot as plt
import os
import numpy as np
resample=open("v3_result/reample_average_performance.txt",'r')
no_resample=open("v3_result/no_reample_average_performance.txt",'r')
r=resample.read()
nr=no_resample.read()
resample.close()
no_resample.close()

r_s=r.split(": ")
nr_s=nr.split(": ")
r_s=r_s[1:]
nr_s=nr_s[1:]

resample_data=[]
no_resample_data=[]

for i in range(len(r_s)):
    r=r_s[i].split("\n")
    n=nr_s[i].split('\n')
    resample_data.append(float(r[0][0:5]))
    no_resample_data.append(float(n[0][0:5]))


score=["ROC","P-R","accuracy","sensitivity","specificity","mAP","dice"]
x=np.arange(len(resample_data))
width=0.3
plt.figure(figsize=(10,8))

plt.bar(x,no_resample_data,width,color="blue",label="no_resample_data")
plt.bar(x+width,resample_data,width,color="red",label="resample_data")

for a,b in enumerate(no_resample_data):plt.text(a,b,'%s'%b,ha='right',color="blue") 
for a,b in enumerate(resample_data):plt.text(a,b,'%s'%b,ha='left',color="red") 
plt.legend(bbox_to_anchor=(1,1))
plt.xticks(x+width/2,score)
plt.ylabel('Score')
plt.ylim(0,1.05)
# plt.show()
plt.savefig("v3_result/difference_plot")
  
