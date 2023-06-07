# -*- coding: utf-8 -*-
"""
Created on Sun Dec 4 15:15:01 2022

@author: hbonen
"""

import re
import codecs  
import random
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn import metrics

import collections
from collections import Counter

def tokenize(str):
    return tokenize2(str, r'[A-Za-z]+')

def tokenize2(str, wordPattern):
    return re.findall(wordPattern,str)

print('Please write your complaints: ')
txt = input()
# txt = 'My neck hurts after I fell down'
# txt = 'I have acidity, vomiting, chest pain and cough'
txt = txt.lower()
print('Please write true diagnose:')
diag = input()
# diag = 'GERD'

tokens = tokenize(txt)

df = pd.read_csv("datasetUnk.csv")

# print(len(df["Symptom_1"].values))

symptoms = set()

for s in range(1, df.shape[1]-1):
    symps = df["Symptom_" + str(s)].values
    for c in symps:
     symptoms.add(c)

symptoms = list(symptoms)

diseases = set()
dis = df["Disease"].values
for d in dis:
    diseases.add(d)

diseases = list(diseases)    

nSymptoms = len(symptoms)
nDisease = len(diseases)

sympMap = {}
disMap = {}
sympSplit = []
disSplit = []

for s in range(nSymptoms):
    sympMap.update({symptoms[s] : s+1})

for d in range(nDisease):
    disMap.update({diseases[d] : d+1})

for sym in range(nSymptoms):
    symp = symptoms[sym].split('_')
    symp[0] = symp[0].strip()
    sympSplit.append(symp)

for dis in range(nDisease):
    dises = diseases[dis].split('_')
    dises[0] = dises[0].strip()
    disSplit.append(dises)

# print(sympSplit)
# print(disSplit)    

for d in range(nDisease):
    df.Disease[df.Disease == diseases[d]] = d+1

for s in range(nSymptoms):
    if symptoms[s] != '0':
        df.Symptom_1[df.Symptom_1 == symptoms[s]] = s+1
    else:
        df.Symptom_1[df.Symptom_1 == symptoms[s]] = 0     
    
for s in range(nSymptoms):
    if symptoms[s] != '0':
        df.Symptom_2[df.Symptom_2 == symptoms[s]] = s+1
    else:
        df.Symptom_2[df.Symptom_2 == symptoms[s]] = 0   
    
for s in range(nSymptoms):
    if symptoms[s] != '0':
        df.Symptom_3[df.Symptom_3 == symptoms[s]] = s+1
    else:
        df.Symptom_3[df.Symptom_3 == symptoms[s]] = 0   

for s in range(nSymptoms):
    if symptoms[s] != '0':
        df.Symptom_4[df.Symptom_4 == symptoms[s]] = s+1
    else:
        df.Symptom_4[df.Symptom_4 == symptoms[s]] = 0   

for s in range(nSymptoms):
    if symptoms[s] != '0':
        df.Symptom_5[df.Symptom_5 == symptoms[s]] = s+1
    else:
        df.Symptom_5[df.Symptom_5 == symptoms[s]] = 0   

for s in range(nSymptoms):
    if symptoms[s] != '0':
        df.Symptom_6[df.Symptom_6 == symptoms[s]] = s+1
    else:
        df.Symptom_6[df.Symptom_6 == symptoms[s]] = 0   

for s in range(nSymptoms):
    if symptoms[s] != '0':
        df.Symptom_7[df.Symptom_7 == symptoms[s]] = s+1
    else:
        df.Symptom_7[df.Symptom_7 == symptoms[s]] = 0   

for s in range(nSymptoms):
    if symptoms[s] != '0':
        df.Symptom_8[df.Symptom_8 == symptoms[s]] = s+1
    else:
        df.Symptom_8[df.Symptom_8 == symptoms[s]] = 0   

for s in range(nSymptoms):
    if symptoms[s] != '0':
        df.Symptom_9[df.Symptom_9 == symptoms[s]] = s+1
    else:
        df.Symptom_9[df.Symptom_9 == symptoms[s]] = 0    

for s in range(nSymptoms):
    if symptoms[s] != '0':
        df.Symptom_10[df.Symptom_10 == symptoms[s]] = s+1
    else:
        df.Symptom_10[df.Symptom_10 == symptoms[s]] = 0   

for s in range(nSymptoms):
    if symptoms[s] != '0':
        df.Symptom_11[df.Symptom_11 == symptoms[s]] = s+1
    else:
        df.Symptom_11[df.Symptom_11 == symptoms[s]] = 0

for s in range(nSymptoms):
    if symptoms[s] != '0':
        df.Symptom_12[df.Symptom_12 == symptoms[s]] = s+1
    else:
        df.Symptom_12[df.Symptom_12 == symptoms[s]] = 0  

for s in range(nSymptoms):
    if symptoms[s] != '0':
        df.Symptom_13[df.Symptom_13 == symptoms[s]] = s+1
    else:
        df.Symptom_13[df.Symptom_13 == symptoms[s]] = 0  

for s in range(nSymptoms):
    if symptoms[s] != '0':
        df.Symptom_14[df.Symptom_14 == symptoms[s]] = s+1
    else:
        df.Symptom_14[df.Symptom_14 == symptoms[s]] = 0  

for s in range(nSymptoms):
    if symptoms[s] != '0':
        df.Symptom_15[df.Symptom_15 == symptoms[s]] = s+1
    else:
        df.Symptom_15[df.Symptom_15 == symptoms[s]] = 0  

for s in range(nSymptoms):
    if symptoms[s] != '0':
        df.Symptom_16[df.Symptom_16 == symptoms[s]] = s+1
    else:
        df.Symptom_16[df.Symptom_16 == symptoms[s]] = 0  

for s in range(nSymptoms):
    if symptoms[s] != '0':
        df.Symptom_17[df.Symptom_17 == symptoms[s]] = s+1
    else:
        df.Symptom_17[df.Symptom_17 == symptoms[s]] = 0  

Y = df["Disease"].values 
Y=Y.astype(int)

X = df.drop(labels = ["Disease"], axis=1)  
listX = X.values.tolist()
shuffListX = []
for u in range(len(X)):
    arrX = np.array(listX[u])
    random.shuffle(arrX)
    shuffListX.append(arrX)
   
# shuffArrX = np.array([shuffListX])

X = pd.DataFrame(shuffListX, columns = ['Symptom_1','Symptom_2','Symptom_3','Symptom_4',
                                            'Symptom_5','Symptom_6','Symptom_7','Symptom_8',
                                            'Symptom_9','Symptom_10','Symptom_11','Symptom_12',
                                            'Symptom_13','Symptom_14','Symptom_15','Symptom_16','Symptom_17'])
# X = shuffX.astype(int)
# X =X.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=20)

model = LogisticRegression(max_iter = 1000) 

model.fit(X_train, y_train) 


prediction_test = model.predict(X_test)

print ("Accuracy = ", metrics.accuracy_score(y_test, prediction_test))

for d in range(nDisease):
    weights = pd.Series(model.coef_[d], index=X.columns.values)
    
    print("Weights for each variables is a follows...")
    print(weights)

sympTakenInput = []
indexOf = set()
for t in tokens:
    for sympt in range(nSymptoms):
        for splitted in sympSplit[sympt]:
            found = re.findall(splitted, t)
            if found != [] and found != ['in'] and found != ['and'] and found != ['on'] and found != ['the'] and found != ['pain']:
                indexOf.add(sympt)
                sympTakenInput.append(found)
            # elif found = ['pain']:
                

sympTakenInputSet = set()
for h in range(len(sympTakenInput)-1):
    sympTakenInputSet.add(sympTakenInput[h][0])

# sympTakenInput = list(sympTakenInputSet)


# newlist = [] 
# duplist = [] 
# for dup in indexOf:
#     if dup not in newlist:
#         newlist.append(dup)
#     else:
#         duplist.append(dup)

# indexOf = duplist

inByUser = np.array([[diag,'0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0']],dtype='<U1000')


sympUser = []
for i in indexOf:
    print(symptoms[i])
    sympUser.append(symptoms[i])
    
for k in range(1,len(indexOf)+1):
    inByUser[0][17-k] = sympUser[k-1]


# inByUser = np.array([['GERD', ' stomach_pain',' vomiting',' cough',' chest_pain','0', '0', '0', '0', '0','0','0','0','0','0','0','0','0']])

# inByUser = np.array([['GERD', ' stomach_pain',' vomiting',' cough',' chest_pain','0', '0', '0', '0', '0','0','0','0','0','0','0','0','0']])

dfInput = pd.DataFrame(inByUser, columns = ['Disease', 'Symptom_1','Symptom_2','Symptom_3','Symptom_4',
                                            'Symptom_5','Symptom_6','Symptom_7','Symptom_8',
                                            'Symptom_9','Symptom_10','Symptom_11','Symptom_12',
                                            'Symptom_13','Symptom_14','Symptom_15','Symptom_16','Symptom_17'])

dfInput.iat[0,0] = disMap[dfInput.iat[0,0]] 

for k in range(1,18):
    dfInput.iat[0,k] = sympMap[dfInput.iat[0,k]]

dfInput = dfInput.drop(labels = ["Disease"], axis=1)    
prediction = model.predict(dfInput)
print(prediction)
keys = [key for key, v in disMap.items() if v == prediction]
print(keys)
