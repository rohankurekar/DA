import math
import numpy as np
import pandas as pd

df=pd.read_csv('diabetes.csv', sep=',',header=None)
dataset=df.values

#Shuffle the dataset
np.random.shuffle(dataset)

#Select Training dataset size and Testing dataset size
trainsize=int(len(dataset)*0.67)
testsize=len(dataset)-trainsize

#separate training and testing data
dataset=dataset.tolist()
trainSet=dataset[0:trainsize+1]
testSet=dataset[trainsize+2:len(dataset)]

trainSet=pd.DataFrame(trainSet)

#Separate data based on label
mask = trainSet[8] == 0
df0 = trainSet[mask]
df1 = trainSet[~mask]

#Calculate mean and standard deviation
list0_ms=[df0.mean().values.tolist(),df0.std().values.tolist()]
list1_ms=[df1.mean().values.tolist(),df1.std().values.tolist()]

#Predict
probability=[]
for data in testSet:
    j=0
    prob0=1
    prob1=1
    while(j<8): 
        mean=list0_ms[0][j]
        std=list0_ms[1][j]
        prob0*=(1 / (math.sqrt(2*math.pi) * std)) * math.exp(-(math.pow(data[j]-mean,2)/(2*math.pow(std,2))))
        mean=list1_ms[0][j]
        std=list1_ms[1][j]
        prob1*=(1 / (math.sqrt(2*math.pi) * std)) * math.exp(-(math.pow(data[j]-mean,2)/(2*math.pow(std,2))))
        j+=1
    if(prob1>prob0):
        probability.append(1)
    else:
        probability.append(0)

#Calculate accuracy
i=0
accu=0
while(i<len(testSet)):
    if(testSet[i][-1]==probability[i]):
        accu+=1
    i+=1
print("Accuracy : "+str((accu/len(testSet))*100))
