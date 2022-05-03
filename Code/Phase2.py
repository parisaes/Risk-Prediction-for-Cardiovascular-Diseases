#!/usr/bin/env python
# coding: utf-8

# In[114]:


import warnings
warnings.filterwarnings('ignore')


# In[115]:


import random
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


# # Importing Dataset

# In[3]:


data=pd.read_csv("D:\B.Sc. Project\CardioVascular Disease (Cleaned).csv", sep='\t')
df=pd.DataFrame(data)
df.drop(['Unnamed: 0', 'male'], axis=1, inplace=True)
df.head()


# In[4]:


df.describe().loc[['min', 'max']]


# # Building the Model

# In[82]:


rand=30
x = df.drop('cardio', axis=1, inplace=False).values
y = list(df['cardio'][:])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=rand)
sc = preprocessing.StandardScaler()
x_train = sc.fit_transform(x_train)
classifier = MLPClassifier(random_state=rand)
classifier.fit(x_train,y_train)


# In[42]:


def predict(sc, classifier, test):
    test = sc.transform(test)
    pred = classifier.predict(test)
    prob = classifier.predict_proba(test)[:,1]
    return prob


# In[43]:


y_prob = predict(sc, classifier, x_test)


# # Plot

# In[54]:


#age (real, predicted)
age = x_test[:,0] / 365
age = pd.cut(age, bins=np.arange(29, 70, 10), labels=[3,4,5,6])
real = np.array([])
predicted = np.array([])
for i in np.arange(3,7,1):
    tmp = np.array(y_test)[np.where(age==i)]
    real = np.append(real, tmp.mean())
    tmp = y_prob[np.where(age==i)]
    predicted = np.append(predicted, tmp.mean())
#plot-------
# plt.figure(figsize=(9,6), dpi=100)
barwidth = 0.25
p1 = np.arange(len(real))
p2 = [x + barwidth for x in p1]
plt.bar(p1, real, barwidth, label='Real')
plt.bar(p2, predicted, barwidth, label='Predicted')
plt.xlabel('Age')
plt.ylabel('Probability')
plt.xticks([r + barwidth/2 for r in np.arange(len(real))], ['30s', '40s', '50s', '60s'])
plt.legend()
plt.show()


# In[9]:


#height, weight
bmi = x_test[:,2] / ((x_test[:,1]/100)**2)
bmi = pd.cut(bmi, bins=[0, 18.5, 24.9, 29.9, 100], labels=np.arange(4))
real = np.array([])
predicted = np.array([])
for i in np.arange(4):
    tmp = np.array(y_test)[np.where(bmi==i)]
    real = np.append(real, tmp.mean())
    tmp = y_prob[np.where(bmi==i)]
    predicted = np.append(predicted, tmp.mean())
#----
# plt.figure(figsize=(9,6), dpi=100)
barwidth = 0.25
p1 = np.arange(len(real))
p2 = [x + barwidth for x in p1]
plt.bar(p1, real, barwidth, label='Real')
plt.bar(p2, predicted, barwidth, label='Predicted')
plt.xlabel('BMI')
plt.ylabel('Probability')
plt.xticks([r + barwidth/2 for r in np.arange(len(real))], ['Underweight', 'Normal weight', 'Overweight', 'Obese'])
plt.legend()
plt.show()


# In[100]:


# #systolic, diastolic (real)
# sys = x_test[:,3]
# dias = x_test[:,4]
# plt.scatter(sys, dias, c=y_test)
# plt.xlabel('Systolic BP (mm Hg)')
# plt.ylabel ('Diastolic BP (mm Hg)')
# plt.show()
#systolic, diastolic (real)
sys = x_test[:,3]
dias = x_test[:,4]
color = ['tab:green', 'tab:red']
label = ['No', 'Yes']
plt.figure(figsize=(9,6), dpi=100)
for i in [0,1]:
    dots = np.where(np.array(y_test)==i)
    plt.scatter(sys[dots], dias[dots], c=color[i], label=label[i])
plt.xlabel('Systolic BP (mm Hg)')
plt.ylabel ('Diastolic BP (mm Hg)')
plt.legend()
plt.show()


# In[9]:


#systolic, diastolic (predicted)
sys = x_test[:,3]
dias = x_test[:,4]
plt.scatter(sys, dias, c=np.round(y_prob, 0))
plt.xlabel('Systolic BP (mm Hg)')
plt.ylabel ('Diastolic BP (mm Hg)')
plt.show()


# In[97]:


#systolic, diastolic (predicted)
sys = x_test[:,3]
dias = x_test[:,4]
p = pd.cut(y_prob, bins=np.arange(0,1.1, 0.25), labels=np.arange(4))
color = ['yellowgreen', 'indigo', 'gold', 'crimson']
label = ['Prob: [0,0.25]', 'Prob: (0.25,0.5]', 'Prob: (0.5,0.75]', 'Prob: (0.75,1]']
plt.figure(figsize=(9,6), dpi=100)
for i in range(4):
    dots = np.where(p==i)
    plt.scatter(sys[dots], dias[dots], c=color[i], label=label[i])
plt.xlabel('Systolic BP (mm Hg)')
plt.ylabel ('Diastolic BP (mm Hg)')
plt.legend()
plt.show()


# In[10]:


# #systolic bp min=90, max=240
# bp = x_test[:,3]
# real = np.array([])
# predicted = np.array([])
# for i in np.arange(90,231,20):
#     tmp = np.array(y_test)[np.intersect1d(np.where(bp>=i), np.where(bp<=i+10))]
#     real = np.append(real, tmp.mean())
#     tmp = y_prob[np.intersect1d(np.where(bp>=i), np.where(bp<=i+10))]
#     predicted = np.append(predicted, tmp.mean())
# #----
# barwidth = 0.25
# p1 = np.arange(len(real))
# p2 = [x + barwidth for x in p1]
# plt.bar(p1, real, barwidth, label='Real')
# plt.bar(p2, predicted, barwidth, label='Predicted')
# plt.xlabel('Systolic BP (cm Hg)')
# plt.xticks([r + barwidth/2 for r in np.arange(len(real))], ['9-10', '11-12','13-14','15-16','17-18', '19-20','21-22','23-24'])
# plt.legend()
# plt.show()


# In[11]:


# #diastolic bp min=40, max=160
# bp = x_test[:,4]
# real = np.array([])
# predicted = np.array([])
# tmp = np.array(y_test)[np.intersect1d(np.where(bp>=40), np.where(bp<=60))]
# real = np.append(real, tmp.mean())
# tmp = y_prob[np.intersect1d(np.where(bp>40), np.where(bp<=60))]
# predicted = np.append(predicted, tmp.mean())
# for i in np.arange(70,151,20):
#     tmp = np.array(y_test)[np.intersect1d(np.where(bp>=i), np.where(bp<=i+10))]
#     real = np.append(real, tmp.mean())
#     tmp = y_prob[np.intersect1d(np.where(bp>=i), np.where(bp<=i+10))]
#     predicted = np.append(predicted, tmp.mean())
# #----
# barwidth = 0.25
# p1 = np.arange(len(real))
# p2 = [x + barwidth for x in p1]
# plt.bar(p1, real, barwidth, label='Real')
# plt.bar(p2, predicted, barwidth, label='Predicted')
# plt.xlabel('Diastolic BP (cm Hg)')
# plt.xticks([r + barwidth/2 for r in np.arange(len(real))], ['4-6', '7-8','9-10','11-12','13-14', '15-16'])
# plt.legend()
# plt.show()


# In[12]:


# #pulse pressure min=10, max=140
# pp = x_test[:,3] - x_test[:,4]
# real = np.array([])
# predicted = np.array([])
# for i in np.arange(10,131,20):
#     tmp = np.array(y_test)[np.intersect1d(np.where(pp>=i), np.where(pp<=i+10))]
#     real = np.append(real, tmp.mean())
#     tmp = y_prob[np.intersect1d(np.where(pp>=i), np.where(pp<=i+10))]
#     predicted = np.append(predicted, tmp.mean())
# #----
# barwidth = 0.25
# p1 = np.arange(len(real))
# p2 = [x + barwidth for x in p1]
# plt.bar(p1, real, barwidth, label='Real')
# plt.bar(p2, predicted, barwidth, label='Predicted')
# plt.xlabel('Pulse Pressure (cm Hg)')
# plt.xticks([r + barwidth/2 for r in np.arange(len(real))], ['1-2', '3-4','5-6','7-8', '9-10','11-12','13-14'])
# plt.legend()
# plt.show()


# In[13]:


#cholesterol
chol = x_test[:,5]+2*x_test[:,6]
chol[chol==0] = 3
real = np.array([])
predicted = np.array([])
for i in np.arange(1,4,1):
    tmp = np.array(y_test)[np.where(chol==i)]
    real = np.append(real, tmp.mean())
    tmp = y_prob[np.where(chol==i)]
    predicted = np.append(predicted, tmp.mean())
#----
# plt.figure(figsize=(9,6), dpi=100)
barwidth = 0.25
p1 = np.arange(len(real))
p2 = [x + barwidth for x in p1]
plt.bar(p1, real, barwidth, label='Real')
plt.bar(p2, predicted, barwidth, label='Predicted')
plt.xlabel('Cholesterol')
plt.ylabel('Probability')
plt.xticks([r + barwidth/2 for r in np.arange(len(real))], ['Normal', 'Above Normal', 'Well Above Normal'])
plt.legend()
plt.show()


# In[14]:


#glucose
gluc = x_test[:,7]+2*x_test[:,8]
gluc[gluc==0] = 3
real = np.array([])
predicted = np.array([])
for i in np.arange(1,4,1):
    tmp = np.array(y_test)[np.where(gluc==i)]
    real = np.append(real, tmp.mean())
    tmp = y_prob[np.where(gluc==i)]
    predicted = np.append(predicted, tmp.mean())
#----
# plt.figure(figsize=(9,6), dpi=100)
barwidth = 0.25
p1 = np.arange(len(real))
p2 = [x + barwidth for x in p1]
plt.bar(p1, real, barwidth, label='Real')
plt.bar(p2, predicted, barwidth, label='Predicted')
plt.xlabel('Glucose')
plt.ylabel('Probability')
plt.xticks([r + barwidth/2 for r in np.arange(len(real))], ['Normal', 'Above Normal', 'Well Above Normal'])
plt.legend()
plt.show()


# In[15]:


# #smoke
# real = np.array([])
# predicted = np.array([])
# for i in np.arange(0,2):
#     tmp = np.array(y_test)[np.where(x_test[:,9]==i)]
#     real = np.append(real, tmp.mean())
#     tmp = y_prob[np.where(x_test[:,9]==i)]
#     predicted = np.append(predicted, tmp.mean())
# #----
# barwidth = 0.25
# p1 = np.arange(len(real))
# p2 = [x + barwidth for x in p1]
# plt.bar(p1, real, barwidth, label='Real')
# plt.bar(p2, predicted, barwidth, label='Predicted')
# plt.xlabel('Smoking')
# plt.xticks([r + barwidth/2 for r in np.arange(len(real))], ['NO', 'YES'])
# plt.legend()
# plt.show()


# In[16]:


# #alcohol
# real = np.array([])
# predicted = np.array([])
# for i in np.arange(0,2):
#     tmp = np.array(y_test)[np.where(x_test[:,10]==i)]
#     real = np.append(real, tmp.mean())
#     tmp = y_prob[np.where(x_test[:,10]==i)]
#     predicted = np.append(predicted, tmp.mean())
# #----
# barwidth = 0.25
# p1 = np.arange(len(real))
# p2 = [x + barwidth for x in p1]
# plt.bar(p1, real, barwidth, label='Real')
# plt.bar(p2, predicted, barwidth, label='Predicted')
# plt.xlabel('Alcohol')
# plt.xticks([r + barwidth/2 for r in np.arange(len(real))], ['NO', 'YES'])
# plt.legend()
# plt.show()


# In[17]:


# #active
# real = np.array([])
# predicted = np.array([])
# for i in np.arange(0,2):
#     tmp = np.array(y_test)[np.where(x_test[:,11]==i)]
#     real = np.append(real, tmp.mean())
#     tmp = y_prob[np.where(x_test[:,11]==i)]
#     predicted = np.append(predicted, tmp.mean())
# #----
# barwidth = 0.25
# p1 = np.arange(len(real))
# p2 = [x + barwidth for x in p1]
# plt.bar(p1, real, barwidth, label='Real')
# plt.bar(p2, predicted, barwidth, label='Predicted')
# plt.xlabel('Activity')
# plt.xticks([r + barwidth/2 for r in np.arange(len(real))], ['NO', 'YES'])
# plt.legend()
# plt.show()


# # Export

# In[99]:


def export(df, loc):
    df.to_csv(loc+'.csv')
    df.to_excel(loc+'.xlsx')


# # Critical Factor

# In[59]:


def random_record():    
#     random.seed(rand)
    age = random.randint(10798, 23713)
#     male = random.randint(0, 1)
    height = random.randint(143, 186)
    weight = random.randint(40, 107)
    sys = random.randint(9, 24)*10
    dias = random.randint(4, 16)*10
    chol_n=0
    chol_nplus=0
    chol = random.randint(1, 3)
    if chol==1:
        chol_n=1
    elif chol==2:
        chol_nplus=1
    gluc_n=0
    gluc_nplus=0
    gluc = random.randint(1, 3)
    if gluc==1:
        gluc_n=1
    elif gluc==2:
        gluc_nplus=1
    smoke = random.randint(0, 1)

    alcohol = random.randint(0, 1)
    active = random.randint(0, 1)
    test = np.array([])
    test = np.append(test, [age, height, weight, sys, dias, chol_n, chol_nplus, gluc_n, gluc_nplus, smoke, alcohol, active])
    test = np.reshape(test, (-1,12))
    return test


# In[60]:


def append_change(records, tmp, prob, code):
    records = np.append(records, tmp)
    records = np.append(records, prob)
    records = np.append(records, code)
    return records


# In[61]:


def change_min(test):
    records = np.copy(test)
    records = np.append(records, predict(sc, classifier, test))
    records = np.append(records, 10000)
    #weight
    tmp = np.copy(test)
    for i in np.arange(1): #5kg
        if tmp[0,2]-5 >= 40:
            tmp[0,2] -=5
            records = append_change(records, tmp, predict(sc, classifier, tmp), -3001-i)
    #systolic
    tmp = np.copy(test)
    pp1 = tmp[0,3] - tmp[0,4]
    for i in np.arange(1): #10
        if tmp[0,3]+10 <= 240:
            tmp[0,3] +=10
            pp2 = tmp[0,3] - tmp[0,4]
            if pp2 not in range(40,61): 
                if abs(pp2-50) > abs(pp1-50):
                    continue
            records = append_change(records, tmp, predict(sc, classifier, tmp), 4001+i)
    tmp = np.copy(test)
    pp1 = tmp[0,3] - tmp[0,4]
    for i in np.arange(1): #10
        if (tmp[0,3]-10 >= 90) and (tmp[0,3]-10 >= tmp[0,4]):
            tmp[0,3] -=10
            pp2 = tmp[0,3] - tmp[0,4]
            if pp2 not in range(40,61): 
                if abs(pp2-50) > abs(pp1-50):
                    continue
            records = append_change(records, tmp, predict(sc, classifier, tmp), -4001-i)
    #diastolic
    tmp = np.copy(test)
    pp1 = tmp[0,3] - tmp[0,4]
    for i in np.arange(1): #10
        if (tmp[0,4]+10 <= 160) and (tmp[0,4]+10 <= tmp[0,3]):
            tmp[0,4] +=10
            pp2 = tmp[0,3] - tmp[0,4]
            if pp2 not in range(40,61): 
                if abs(pp2-50) > abs(pp1-50):
                    continue
            records = append_change(records, tmp, predict(sc, classifier, tmp), 5001+i)
    tmp = np.copy(test)
    pp1 = tmp[0,3] - tmp[0,4]
    for i in np.arange(1): #10
        if tmp[0,4]-10 >= 40:
            tmp[0,4] -=10
            pp2 = tmp[0,3] - tmp[0,4]
            if pp2 not in range(40,61): 
                if abs(pp2-50) > abs(pp1-50):
                    continue
            records = append_change(records, tmp, predict(sc, classifier, tmp), -5001-i)
    #cholesterol, glucose
    for i in np.arange(5,8,2):
        tmp = np.copy(test)
        code = -(i+1)*1000-(i+2)*100
        if tmp[0,i]+tmp[0,i+1]==0: #1 level
            tmp[0,i+1] = 1
            records = append_change(records, tmp, predict(sc, classifier, tmp), code-3)
#             tmp[0,i], tmp[0,i+1] = tmp[0,i+1], tmp[0,i]
#             records = append_change(records, tmp, predict(sc, classifier, tmp), code-2)
        elif tmp[0,i+1]==1:
            tmp[0,i], tmp[0,i+1] = tmp[0,i+1], tmp[0,i]
            records = append_change(records, tmp, predict(sc, classifier, tmp), code-1)
    #smoke, alcohol
    for i in np.arange(9,11):
        tmp = np.copy(test)
        if tmp[0,i]==1:
            tmp[0,i] = 0
            records = append_change(records, tmp, predict(sc, classifier, tmp), -(i+1)*100)
    #active
    tmp = np.copy(test)
    if tmp[0,11]==0:
        tmp[0,11] = 1
        records = append_change(records, tmp, predict(sc, classifier, tmp), (11+1)*100)
    #DataFrame
    records = np.reshape(records, (-1,14))
    columns = np.array(df.columns)
    columns = np.append(columns, 'change')
    records = pd.DataFrame(records, columns=columns)
    records.rename(columns={'cardio':'prob'}, inplace=True)
    return records


# In[62]:


def change_max(test):
    records = np.copy(test)
    records = np.append(records, predict(sc, classifier, test))
    records = np.append(records, 10000)
    #age
    tmp = np.copy(test)
    for i in np.arange(1): #5 years
        if tmp[0,0]+1825 <= 23713:
            tmp[0,0] +=1825
            records = append_change(records, tmp, predict(sc, classifier, tmp), 1001+i)
    #weight
    tmp = np.copy(test)
    for i in np.arange(1): #5kg
        if tmp[0,2]+5 <= 107:
            tmp[0,2] +=5
            records = append_change(records, tmp, predict(sc, classifier, tmp), 3001+i)
    #systolic
    tmp = np.copy(test)
    pp1 = tmp[0,3] - tmp[0,4]
    for i in np.arange(1): #10
        if tmp[0,3]+10 <= 240:
            tmp[0,3] +=10
            pp2 = tmp[0,3] - tmp[0,4]
            if pp2 not in range(40,61): 
                if abs(pp2-50) > abs(pp1-50):
                    continue
            records = append_change(records, tmp, predict(sc, classifier, tmp), 4001+i)
    tmp = np.copy(test)
    pp1 = tmp[0,3] - tmp[0,4]
    for i in np.arange(1): #10
        if (tmp[0,3]-10 >= 90) and (tmp[0,3]-10 >= tmp[0,4]):
            tmp[0,3] -=10
            pp2 = tmp[0,3] - tmp[0,4]
            if pp2 not in range(40,61): 
                if abs(pp2-50) > abs(pp1-50):
                    continue
            records = append_change(records, tmp, predict(sc, classifier, tmp), -4001-i)
    #diastolic
    tmp = np.copy(test)
    pp1 = tmp[0,3] - tmp[0,4]
    for i in np.arange(1): #10
        if (tmp[0,4]+10 <= 160) and (tmp[0,4]+10 <= tmp[0,3]):
            tmp[0,4] +=10
            pp2 = tmp[0,3] - tmp[0,4]
            if pp2 not in range(40,61): 
                if abs(pp2-50) > abs(pp1-50):
                    continue
            records = append_change(records, tmp, predict(sc, classifier, tmp), 5001+i)
    tmp = np.copy(test)
    pp1 = tmp[0,3] - tmp[0,4]
    for i in np.arange(1): #10
        if tmp[0,4]-10 >= 40:
            tmp[0,4] -=10
            pp2 = tmp[0,3] - tmp[0,4]
            if pp2 not in range(40,61): 
                if abs(pp2-50) > abs(pp1-50):
                    continue
            records = append_change(records, tmp, predict(sc, classifier, tmp), -5001-i)
    #cholesterol, glucose
    for i in np.arange(5,8,2):
        tmp = np.copy(test)
        code = +(i+1)*1000+(i+2)*100
        if tmp[0,i+1]==1:
            tmp[0,i], tmp[0,i+1] = 0, 0
            records = append_change(records, tmp, predict(sc, classifier, tmp), code+3)
        if tmp[0,i]==1: #1 level
            tmp[0,i], tmp[0,i+1] = tmp[0,i+1], tmp[0,i]
            records = append_change(records, tmp, predict(sc, classifier, tmp), code+1)
#             tmp[0,i], tmp[0,i+1] = 0, 0
#             records = append_change(records, tmp, predict(sc, classifier, tmp), code+2)
    #smoke, alcohol
    for i in np.arange(9,11):
        tmp = np.copy(test)
        if tmp[0,i]==0:
            tmp[0,i] = 1
            records = append_change(records, tmp, predict(sc, classifier, tmp), (i+1)*100)
    #active
    tmp = np.copy(test)
    if tmp[0,11]==1:
        tmp[0,11] = 0
        records = append_change(records, tmp, predict(sc, classifier, tmp), -(11+1)*100)
    #DataFrame
    records = np.reshape(records, (-1,14))
    columns = np.array(df.columns)
    columns = np.append(columns, 'change')
    records = pd.DataFrame(records, columns=columns)
    records.rename(columns={'cardio':'prob'}, inplace=True)
    return records


# In[63]:


def minimize_prob(test):
    result = np.array([])
    result = np.append(result, test)
    result = np.append(result, predict(sc, classifier, test))
    result = np.append(result, 0)
    for i in np.arange(2):
        records = change_min(test)
        minimum = np.array(records[records['prob']==records['prob'].min()])
        result = np.append(result, minimum)
        test = minimum[:, :12]
    result = np.reshape(result, (-1,14))
    columns = np.array(df.columns)
    columns = np.append(columns, 'change')
    result = pd.DataFrame(result, columns=columns)
    result.rename(columns={'cardio':'prob'}, inplace=True)
    return result


# In[64]:


def maximize_prob(test):
    result = np.array([])
    result = np.append(result, test)
    result = np.append(result, predict(sc, classifier, test))
    result = np.append(result, 0)
    for i in np.arange(2):
        records = change_max(test)
        maximum = np.array(records[records['prob']==records['prob'].max()])
        result = np.append(result, maximum)
        test = maximum[:, :12]
    result = np.reshape(result, (-1,14))
    columns = np.array(df.columns)
    columns = np.append(columns, 'change')
    result = pd.DataFrame(result, columns=columns)
    result.rename(columns={'cardio':'prob'}, inplace=True)
    return result


# In[65]:


def clean_df(df):
    df['age'] = round(df['age']/365, 0)
    chol = df['chol normal']+2*df['chol normal+']
    chol[chol==0] = 3
    gluc = df['gluc normal']+2*df['gluc normal+']
    gluc[gluc==0] = 3
    df.insert(5, 'cholesterol', chol)
    df.insert(6, 'glucose', gluc)
    df.drop(columns=['chol normal', 'chol normal+', 'gluc normal', 'gluc normal+'], axis=1, inplace=True)
    df.insert(5, 'pulse pressure', df['systolic bp']-df['diastolic bp'])
    return df


# In[66]:


def show_change(df):
    prob1 = df['prob'][1]
    prob2 = df['prob'][2]
    change1 = df['change'][1]
    change2 = df['change'][2]
    data = df.drop([1,2])
    data.insert(12, 'prob1', prob1)
    data.insert(13, 'prob2', prob2)
    data.insert(14, 'change1', change1)
    data.insert(15, 'change2', change2)
    data.drop(['change'], axis=1, inplace=True)
    return data


# In[67]:


def min_tests():
    result=pd.DataFrame()
    for i in range(len(x_test)):
        x = np.array(x_test[i,:]).reshape(1,-1)
        tmp = clean_df(minimize_prob(x)).round(2)
        frames = [result, show_change(tmp)]
        result = pd.concat(frames)
    return result


# In[68]:


def max_tests():
    result=pd.DataFrame()
    for i in range(len(x_test)):
        x = np.array(x_test[i,:]).reshape(1,-1)
        tmp = clean_df(maximize_prob(x)).round(2)
        frames = [result, show_change(tmp)]
        result = pd.concat(frames)
    return result


# In[127]:


decode = {1001: 'age+5', 1002: 'age+10',
       3001: 'weight+5', 3002: 'weight+10', -3001: 'weight-5', -3002: 'weight-10',
       4001: 'SBP+10', 4002: 'SBP+20', -4001: 'SBP-10', -4002: 'SBP-20', 
       5001: 'DBP+10', 5002: 'DBP+20', -5001: 'DBP-10', -5002: 'DBP-20',
       6701: 'chol 1->2', 6702: 'chol 1->3', 6703: 'chol 2->3', -6701: 'chol 2->1', -6702: 'chol 3->1', -6703: 'chol 3->2',
       8901: 'gluc 1->2', 8902: 'gluc 1->3', 8903: 'gluc 2->3', -8901: 'gluc 2->1', -8902: 'gluc 3->1', -8903: 'gluc 3->2',
       1000: 'smoke N->Y', -1000: 'smoke Y->N',
       1100: 'alcohol N->Y', -1100: 'alcohol Y->N',
       1200: 'active N->Y', -1200: 'active Y->N',
       10000: '-'}
encode = {v: k for k, v in decode.items()}


# In[ ]:


result_min = min_tests()
result_min = result_min.reset_index()
result_min.drop('index', axis=1, inplace=True)
result_min = result_min.replace(decode)
export(result_min, 'D:\B.Sc. Project\Result\Minimizing Probs')


# In[102]:


result_min = pd.read_csv("D:\B.Sc. Project\Result\Minimizing Probs.csv")
result_min.drop('Unnamed: 0', axis=1, inplace=True)


# In[47]:


result_max = max_tests()
result_max = result_max.reset_index()
result_max.drop('index', axis=1, inplace=True)
result_max = result_max.replace(decode)
export(result_max, 'D:\B.Sc. Project\Result\Maximizing Probs')


# In[116]:


result_max = pd.read_csv("D:\B.Sc. Project\Result\Maximizing Probs.csv")
result_max.drop('Unnamed: 0', axis=1, inplace=True)


# In[5]:


result_min.head()


# In[6]:


result_max.head()


# In[117]:


def categorize(item):
    item = ((item.value_counts(normalize=True))*100).round(2)
    all_item = decode.values()
    for i in all_item:
        if i not in item:
            item = item.append(pd.Series(0, index=[i]))
    dic = {}
    dic.update({'NC' : item['-']})
    dic.update({'Age' : item['age+5']})
    dic.update({'wgt.' : item['weight-5']+item['weight+5']})    
    dic.update({'BP' : item['SBP+10']+item['SBP-10']+item['DBP+10']+item['DBP-10']})
    dic.update({'CHOL' : item['chol 2->1']+item['chol 3->2']+item['chol 1->2']+item['chol 2->3']})
    dic.update({'GLC' : item['gluc 2->1']+item['gluc 3->2']+item['gluc 1->2']+item['gluc 2->3']})
    dic.update({'SMOK' : item['smoke Y->N']+item['smoke N->Y']})
    dic.update({'ALC' : item['alcohol Y->N']+item['alcohol N->Y']})
    dic.update({'Activity' : item['active Y->N']+item['active N->Y']})
    return dic


# In[118]:


def get_value(key, dic):
    for k, v in dic.items():
        if k==key:
            return v


# In[119]:


def get_key(value, dic):
    for k, v in dic.items():
        if v==value:
            return k


# In[120]:


def bar_plot1(data): 
    p = np.arange(8) 
    data = categorize(data)
    sort = sorted(data.values(), reverse=True)
    labels = np.array([])
    for v in sort:
        labels = np.append(labels, get_key(v, data))
    plt.bar(p, sort)
    plt.xticks(p, labels)
    plt.show()


# In[121]:


# bar_plot1(result_min['change1'])


# In[122]:


def bar_plot2(data1, data2, bar_label, xlabel, color): 
    p = np.arange(8) 
    data1 = categorize(data1)
    data2 = categorize(data2)
    sort1 = sorted(data1.values(), reverse=True)
    labels = np.array([])
    for v in sort1:
        labels = np.append(labels, get_key(v, data1))
    sort2 = np.array([])
    for l in labels:
        sort2 = np.append(sort2, get_value(l, data2))
    #plot
    barwidth = 0.25
    p1 = np.arange(len(data1))
    p2 = [x + barwidth for x in p1]
    plt.bar(p1, sort1, barwidth, label=bar_label[0], color=color[0])
    plt.bar(p2, sort2, barwidth, label=bar_label[1], color=color[1])
    plt.xlabel(xlabel)
    plt.ylabel('Percentage of Records')
    plt.xticks([r + barwidth/2 for r in np.arange(len(data1))], labels)
    plt.legend()
    plt.show()


# In[123]:


def bar_plot3(data1, data2, data3, bar_label, xlabel, color): 
    p = np.arange(8) 
    data1 = categorize(data1)
    data2 = categorize(data2)
    data3 = categorize(data3)
    sort1 = sorted(data1.values(), reverse=True)
    labels = np.array([])
    for v in sort1:
        labels = np.append(labels, get_key(v, data1))
    sort2 = np.array([])
    sort3 = np.array([])
    for l in labels:
        sort2 = np.append(sort2, get_value(l, data2))
        sort3 = np.append(sort3, get_value(l, data3))
    #plot
    barwidth = 0.25
    p1 = np.arange(len(data1))
    p2 = [x + barwidth for x in p1]
    p3 = [x + barwidth for x in p2]
    plt.bar(p1, sort1, barwidth, label=bar_label[0], color=color[0])
    plt.bar(p2, sort2, barwidth, label=bar_label[1], color=color[1])
    plt.bar(p3, sort3, barwidth, label=bar_label[2], color=color[2])
    plt.xlabel(xlabel)
    plt.ylabel('Percentage of Records')
    plt.xticks([r + barwidth for r in p1], labels)
    plt.legend()
    plt.show()


# In[124]:


def bar_plot4(data1, data2, data3, data4, bar_label, xlabel): 
    p = np.arange(8) 
    data1 = categorize(data1)
    data2 = categorize(data2)
    data3 = categorize(data3)
    data4 = categorize(data4)
    sort1 = sorted(data1.values(), reverse=True)
    labels = np.array([])
    for v in sort1:
        labels = np.append(labels, get_key(v, data1))
    sort2 = np.array([])
    sort3 = np.array([])
    sort4 = np.array([])
    for l in labels:
        sort2 = np.append(sort2, get_value(l, data2))
        sort3 = np.append(sort3, get_value(l, data3))
        sort4 = np.append(sort4, get_value(l, data4))    
    #plot
    barwidth = 0.25
    p1 = np.arange(len(data1))*1.3
    p2 = [x + barwidth for x in p1]
    p3 = [x + barwidth for x in p2]
    p4 = [x + barwidth for x in p3]
    plt.bar(p1, sort1, barwidth, label=bar_label[0])
    plt.bar(p2, sort2, barwidth, label=bar_label[1])
    plt.bar(p3, sort3, barwidth, label=bar_label[2])
    plt.bar(p4, sort4, barwidth, label=bar_label[3])
    plt.xlabel(xlabel)
    plt.ylabel('Percentage of Records')
    plt.xticks([r + barwidth*1.5 for r in p1], labels)
    plt.legend()
    plt.show()


# In[125]:


def bar_plot5(data1, data2, data3, data4, data5, bar_label, xlabel): 
    p = np.arange(8) 
    data1 = categorize(data1)
    data2 = categorize(data2)
    data3 = categorize(data3)
    data4 = categorize(data4)
    data5 = categorize(data5)
    sort1 = sorted(data1.values(), reverse=True)
    labels = np.array([])
    data = data1.copy()
    for v in sort1:
        labels = np.append(labels, get_key(v, data))
        del data[get_key(v, data)]
    sort2 = np.array([])
    sort3 = np.array([])
    sort4 = np.array([])
    sort5 = np.array([])
    for l in labels:
        sort2 = np.append(sort2, get_value(l, data2))
        sort3 = np.append(sort3, get_value(l, data3))
        sort4 = np.append(sort4, get_value(l, data4))
        sort5 = np.append(sort5, get_value(l, data5)) 
    #plot
    barwidth = 0.25
    p1 = np.arange(len(data1))*1.7
    p2 = [x + barwidth for x in p1]
    p3 = [x + barwidth for x in p2]
    p4 = [x + barwidth for x in p3]
    p5 = [x + barwidth for x in p4]
    plt.bar(p1, sort1, barwidth, label=bar_label[0])
    plt.bar(p2, sort2, barwidth, label=bar_label[1])
    plt.bar(p3, sort3, barwidth, label=bar_label[2])
    plt.bar(p4, sort4, barwidth, label=bar_label[3])
    plt.bar(p5, sort5, barwidth, label=bar_label[4])
    plt.xlabel(xlabel)
    plt.ylabel('Percentage of Records')
    plt.xticks([r + barwidth*2 for r in p1], labels)
    plt.legend()
    plt.show()


# In[19]:


# plt.figure(figsize=(9,6), dpi=100)
bar_plot2(result_min['change1'], result_min['change2'], ['Stage 1', 'Stage 2'], 'The best Factor for reducing the Probability', ['palevioletred', 'steelblue'])


# In[129]:


# plt.figure(figsize=(9,6), dpi=100)
bar_plot2(result_max['change1'], result_max['change2'], ['Stage 1', 'Stage 2'], 'The best Factor for increasing the Probability', ['palevioletred', 'steelblue'])


# In[57]:


# age = x_test[:,0] / 365
# age = pd.cut(age, bins=np.arange(29, 70, 10), labels=np.arange(4))
# tmp = result_max.copy(deep=True)
# tmp.insert(0,'age_c', age)
# # plt.figure(figsize=(9,6), dpi=100)
# bar_plot4(tmp[tmp['age_c']==3]['change1'], tmp[tmp['age_c']==2]['change1'], tmp[tmp['age_c']==1]['change1'], tmp[tmp['age_c']==0]['change1'], ['60s', '50s', '40s', '30s'], 'Stage 1')


# In[160]:


# bmi = result_min['weight'] / ((result_min['height']/100)**2)
# bmi = pd.cut(bmi, bins=[0, 18.5, 24.9, 29.9, 100], labels=np.arange(4))
# tmp = result_min.copy(deep=True)
# tmp.insert(0,'bmi', bmi)
# # plt.figure(figsize=(9,6), dpi=100)
# bar_plot4(tmp[tmp['bmi']==3]['change1'], tmp[tmp['bmi']==2]['change1'], tmp[tmp['bmi']==1]['change1'], tmp[tmp['bmi']==0]['change1'], ['Obese', 'Overweight', 'Normal weight', 'Underweight'], 'Stage 1')


# In[194]:


# bmi = result_max['weight'] / ((result_max['height']/100)**2)
# bmi = pd.cut(bmi, bins=[0, 18.5, 24.9, 29.9, 100], labels=np.arange(4))
# tmp = result_max.copy(deep=True)
# tmp.insert(0,'bmi', bmi)
# # plt.figure(figsize=(9,6), dpi=100)
# bar_plot4(tmp[tmp['bmi']==3]['change1'], tmp[tmp['bmi']==2]['change1'], tmp[tmp['bmi']==1]['change1'], tmp[tmp['bmi']==0]['change1'], ['Obese', 'Overweight', 'Normal weight', 'Underweight'], 'Stage 1')


# In[197]:


# sys = pd.cut(result_min['systolic bp'], bins=[0, 119, 129, 139, 180, 1000], labels=np.arange(5))
# dias = pd.cut(result_min['diastolic bp'], bins=[0, 79, 89, 120, 1000], labels=[0,2,3,4])
# sys = sys.tolist()
# dias = dias.tolist()
# hbp = np.maximum(sys, dias)
# tmp = result_min.copy(deep=True)
# tmp.insert(0,'hbp', hbp)
# # plt.figure(figsize=(9,6), dpi=100)
# bar_plot5(tmp[tmp['hbp']==4]['change1'], tmp[tmp['hbp']==3]['change1'], tmp[tmp['hbp']==2]['change1'], tmp[tmp['hbp']==1]['change1'], tmp[tmp['hbp']==0]['change1'], ['Hypertensive Crisis', 'Hypertension Stage 2', 'Hypertension Stage 1', 'Elevated', 'Normal'], 'Stage 1')


# In[199]:


# sys = pd.cut(result_max['systolic bp'], bins=[0, 119, 129, 139, 180, 1000], labels=np.arange(5))
# dias = pd.cut(result_max['diastolic bp'], bins=[0, 79, 89, 120, 1000], labels=[0,2,3,4])
# sys = sys.tolist()
# dias = dias.tolist()
# hbp = np.maximum(sys, dias)
# tmp = result_max.copy(deep=True)
# tmp.insert(0,'hbp', hbp)
# # plt.figure(figsize=(9,6), dpi=100)
# bar_plot5(tmp[tmp['hbp']==4]['change1'], tmp[tmp['hbp']==3]['change1'], tmp[tmp['hbp']==2]['change1'], tmp[tmp['hbp']==1]['change1'], tmp[tmp['hbp']==0]['change1'], ['Hypertensive Crisis', 'Hypertension Stage 2', 'Hypertension Stage 1', 'Elevated', 'Normal'], 'Stage 1')


# In[22]:


# plt.figure(figsize=(9,6), dpi=100)
bar_plot3(result_min[result_min['cholesterol']==3]['change1'], result_min[result_min['cholesterol']==2]['change1'], result_min[result_min['cholesterol']==1]['change1'],
          ['Cholesterol: Well Above Normal', 'Cholesterol: Above Normal', 'Cholesterol: Normal'], 'The best Factor for reducing the Probability (Stage 1)', ['tab:red', 'orange', 'yellowgreen'])


# In[131]:


# plt.figure(figsize=(9,6), dpi=100)
bar_plot3(result_max[result_max['cholesterol']==3]['change1'], result_max[result_max['cholesterol']==2]['change1'], result_max[result_max['cholesterol']==1]['change1'],
          ['Cholesterol: Well Above Normal', 'Cholesterol: Above Normal', 'Cholesterol: Normal'], 'The best Factor for increasing the Probability (Stage 1)', ['tab:red', 'orange', 'yellowgreen'])


# In[232]:


# plt.figure(figsize=(9,6), dpi=100)
# bar_plot3(result_max[result_max['cholesterol']==3]['change2'], result_max[result_max['cholesterol']==2]['change2'], result_max[result_max['cholesterol']==1]['change2'],
#           ['Cholesterol: Well Above Normal', 'Cholesterol: Above Normal', 'Cholesterol: Normal'], 'The best Factor for increasing the Probability (Stage 2)', ['tab:red', 'orange', 'yellowgreen'])


# In[205]:


# plt.figure(figsize=(9,6), dpi=100)
# bar_plot3(result_min[result_min['cholesterol']==3]['change2'], result_min[result_min['cholesterol']==2]['change2'], result_min[result_min['cholesterol']==1]['change2'],
#           ['Cholesterol: Well Above Normal', 'Cholesterol: Above Normal', 'Cholesterol: Normal'], 'The best Factor for reducing the Probability (Stage 2)', ['tab:red', 'orange', 'yellowgreen'])


# In[26]:


# plt.figure(figsize=(9,6), dpi=100)
bar_plot3(result_min[result_min['glucose']==3]['change1'], result_min[result_min['glucose']==2]['change1'], result_min[result_min['glucose']==1]['change1'],
          ['Glucose: Well Above Normal', 'Glucose: Above Normal', 'Glucose: Normal'], 'The best Factor for reducing the Probability (Stage 1)', ['tab:red', 'orange', 'yellowgreen'])


# In[233]:


# plt.figure(figsize=(9,6), dpi=100)
# bar_plot3(result_max[result_max['glucose']==3]['change1'], result_max[result_max['glucose']==2]['change1'], result_max[result_max['glucose']==1]['change1'],
#           ['Glucose: Well Above Normal', 'Glucose: Above Normal', 'Glucose: Normal'], 'The best Factor for increasing the Probability (Stage 1)', ['tab:red', 'orange', 'yellowgreen'])


# In[207]:


# plt.figure(figsize=(9,6), dpi=100)
# bar_plot3(result_min[result_min['cholesterol']==3]['change2'], result_min[result_min['cholesterol']==2]['change2'], result_min[result_min['cholesterol']==1]['change2'],
#           ['Cholesterol: Well Above Normal', 'Cholesterol: Above Normal', 'Cholesterol: Normal'], 'The best Factor for reducing the Probability (Stage 2)', ['tab:red', 'orange', 'yellowgreen'])


# In[217]:


# plt.figure(figsize=(9,6), dpi=100)
# bar_plot3(result_max[result_max['glucose']==3]['change2'], result_max[result_max['glucose']==2]['change2'], result_max[result_max['glucose']==1]['change2'],
#           ['Glucose: Well Above Normal', 'Glucose: Above Normal', 'Glucose: Normal'], 'The best Factor for increasing the Probability (Stage 2)', ['tab:red', 'orange', 'yellowgreen'])


# In[28]:


# plt.figure(figsize=(9,6), dpi=100)
bar_plot2(result_min[result_min['smoke']==1]['change1'], result_min[result_min['smoke']==0]['change1'], ['SMOK YES', 'SMOK NO'],
          'The best Factor for reducing the Probability (Stage 1)', ['crimson', 'mediumseagreen'])


# In[133]:


# plt.figure(figsize=(9,6), dpi=100)
bar_plot2(result_max[result_max['smoke']==1]['change1'], result_max[result_max['smoke']==0]['change1'], ['SMOK YES', 'SMOK NO'],
          'The best Factor for increasing the Probability (Stage 1)', ['crimson', 'mediumseagreen'])


# In[218]:


# plt.figure(figsize=(9,6), dpi=100)
# bar_plot2(result_min[result_min['smoke']==1]['change2'], result_min[result_min['smoke']==0]['change2'], ['SMOK YES', 'SMOK NO'],
#           'The best Factor for reducing the Probability (Stage 2)', ['crimson', 'mediumseagreen'])


# In[32]:


# plt.figure(figsize=(9,6), dpi=100)
bar_plot2(result_min[result_min['alcohol']==1]['change1'], result_min[result_min['alcohol']==0]['change1'], ['ALC YES', 'ALC NO'],
          'The best Factor for reducing the Probability (Stage 1)', ['crimson', 'mediumseagreen'])


# In[135]:


# plt.figure(figsize=(9,6), dpi=100)
bar_plot2(result_max[result_max['alcohol']==1]['change1'], result_max[result_max['alcohol']==0]['change1'], ['ALC YES', 'ALC NO'],
          'The best Factor for increasing the Probability (Stage 1)', ['crimson', 'mediumseagreen'])


# In[170]:


# plt.figure(figsize=(9,6), dpi=100)
# bar_plot2(result_min[result_min['alcohol']==1]['change2'], result_min[result_min['alcohol']==0]['change2'], ['ALC YES', 'ALC NO'],
#           'The best Factor for reducing the Probability (Stage 2)', ['crimson', 'mediumseagreen'])


# In[222]:


# plt.figure(figsize=(9,6), dpi=100)
# bar_plot2(result_max[result_max['alcohol']==1]['change2'], result_max[result_min['alcohol']==0]['change2'], ['ALC YES', 'ALC NO'],
#           'The best Factor for increasing the Probability (Stage 2)', ['crimson', 'mediumseagreen'])


# In[36]:


# plt.figure(figsize=(9,6), dpi=100)
bar_plot2(result_min[result_min['active']==1]['change1'], result_min[result_min['active']==0]['change1'], ['Activity YES', 'Activity NO'],
          'The best Factor for reducing the Probability (Stage 1)', ['mediumseagreen', 'crimson'])


# In[228]:


# # plt.figure(figsize=(9,6), dpi=100)
# bar_plot2(result_max[result_max['active']==1]['change1'], result_max[result_max['active']==0]['change1'], ['Activity YES', 'Activity NO'],
#           'The best Factor for increasing the Probability (Stage 1)', ['mediumseagreen', 'crimson'])


# In[172]:


# plt.figure(figsize=(9,6), dpi=100)
# bar_plot2(result_min[result_min['alcohol']==1]['change2'], result_min[result_min['alcohol']==0]['change2'], ['ALC YES', 'ALC NO'],
#           'The best Factor for reducing the Probability (Stage 2)', ['crimson', 'mediumseagreen'])


# In[230]:


# plt.figure(figsize=(9,6), dpi=100)
# bar_plot2(result_max[result_max['alcohol']==1]['change2'], result_max[result_max['alcohol']==0]['change2'], ['ALC YES', 'ALC NO'],
#           'The best Factor for increasing the Probability (Stage 2)', ['crimson', 'mediumseagreen'])


# In[111]:


# (result_min.groupby('change1')['change2'].value_counts(normalize=True)*100).round(2)


# In[113]:


# result_min[result_min['prob']-result_min['prob2']>=0.1]
# plt.hist(result_min['prob']-result_min['prob2'], cumulative=True)


# In[95]:


x = np.array(x_test[48,:]).reshape(1,-1)
min_exm = clean_df(minimize_prob(x)).round(2)
min_exm= min_exm.replace(decode)
min_exm


# In[96]:


x = np.array(x_test[48,:]).reshape(1,-1)
max_exm = clean_df(maximize_prob(x)).round(2)
max_exm = max_exm.replace(decode)
max_exm


# In[100]:


frames = [min_exm, max_exm]
exm = pd.concat(frames)
export(exm, 'D:\B.Sc. Project\Result\Example')


# In[ ]:




