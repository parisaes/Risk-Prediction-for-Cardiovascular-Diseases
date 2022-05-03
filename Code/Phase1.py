#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


import time
import math
import random
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
import sklearn.metrics
from sklearn.calibration import calibration_curve


# # Importing DataSet

# In[3]:


data=pd.read_csv("D:\B.Sc. Project\CardioVascular Disease.csv", sep=';')
df=pd.DataFrame(data)
df.head()


# In[4]:


df.drop("id", axis=1, inplace=True)
df.rename(columns={'ap_hi':'systolic bp', 'ap_lo':'diastolic bp', 'gluc':'glucose', 'alco':'alcohol'}, inplace=True)
df.head()


# # What is it?

# In[5]:


df.isnull().values.sum()


# In[6]:


df.describe()


# # Cleaning

# In[7]:


print(df[df['diastolic bp']>df['systolic bp']].count())
df.drop(df[df['diastolic bp']>df['systolic bp']].index, inplace=True)


# In[8]:


print(df[df['systolic bp']<90].count())
df.drop(df[df['systolic bp']<90].index, inplace=True)


# In[9]:


print(df[df['systolic bp']>250].count())
df.drop(df[df['systolic bp']>250].index, inplace=True)


# In[10]:


print(df[df['diastolic bp']<40].count())
df.drop(df[df['diastolic bp']<40].index, inplace=True)


# In[11]:


print(df[df['diastolic bp']>160].count())
df.drop(df[df['diastolic bp']>160].index, inplace=True)


# In[12]:


plt.scatter(df['height'], df['weight'])
plt.show()


# In[13]:


sb.boxplot(df['weight'])


# In[14]:


q1_weight =  df['weight'].quantile(0.25)
q3_weight = df['weight'].quantile(0.75)
iqr_weight = q3_weight - q1_weight


# In[15]:


print(df[df['weight']>q3_weight+1.5*iqr_weight].count())
df.drop(df[df['weight']>q3_weight+1.5*iqr_weight].index, inplace=True)


# In[16]:


print(df[df['weight']<q1_weight-1.5*iqr_weight].count())
df.drop(df[df['weight']<q1_weight-1.5*iqr_weight].index, inplace=True)


# In[17]:


sb.boxplot(df['weight'])


# In[18]:


sb.boxplot(df['height'])


# In[19]:


q1_height =  df['height'].quantile(0.25)
q3_height = df['height'].quantile(0.75)
iqr_height = q3_height - q1_height


# In[20]:


print(df[df['height']>q3_height+1.5*iqr_height].count())
df.drop(df[df['height']>q3_height+1.5*iqr_height].index, inplace=True)


# In[21]:


print(df[df['height']<q1_height-1.5*iqr_height].count())
df.drop(df[df['height']<q1_height-1.5*iqr_height].index, inplace=True)


# In[22]:


plt.scatter(df['height'], df['weight'])
plt.show()


# In[23]:


df.shape[0]


# In[24]:


# df_hw=pd.DataFrame(df['height'])
# df_hw.insert(1,'weight',df['weight'])
# kmeans=KMeans(n_clusters=3, random_state=23).fit(df_hw)
# plt.scatter(df['height'], df['weight'], c=kmeans.labels_)


# In[25]:


pd.DataFrame.drop_duplicates(df, inplace=True)
df.reset_index(drop=True, inplace=True)
df.tail()


# In[26]:


df.describe()


# # Creating Indicator (Cholestrol, Glucose)

# In[27]:


lb=preprocessing.LabelBinarizer()
lb_results=lb.fit_transform(df['cholesterol'])
# lb.classes_
# lb_results
tmp = pd.DataFrame(lb_results, columns=lb.classes_)
df.insert(6,'chol normal', tmp [1])
df.insert(7,'chol normal+', tmp [2])
df.drop("cholesterol", axis=1, inplace=True)
df.head()


# In[28]:


lb_results=lb.fit_transform(df['glucose'])
# lb.classes_
# lb_results
tmp = pd.DataFrame(lb_results, columns=lb.classes_)
df.insert(8,'gluc normal', tmp [1])
df.insert(9,'gluc normal+', tmp [2])
df.drop("glucose", axis=1, inplace=True)
df.head()


# In[29]:


df.insert(1,'male', df["gender"]-1)
df.drop("gender", axis=1, inplace=True)
df.head()


# # How does it look like?

# In[30]:


df['cardio'].value_counts()


# In[31]:


sb.countplot(df['cardio'], palette='Blues')


# In[32]:


df.groupby('cardio').mean()


# In[33]:


df.var()


# In[34]:


plt.figure(figsize=(12,10))
cor = df.corr()
sb.heatmap(cor, annot=True, cmap=plt.cm.YlOrRd)
plt.show()


# In[35]:


df.to_csv('D:\B.Sc. Project\CardioVascular Disease (Cleaned).csv', sep='\t')


# # Defining Metrics

# In[36]:


def TPR(c_matrix):
    tp=c_matrix[1][1]
    fn=c_matrix[1][0]
    return tp/(tp+fn)


# In[37]:


def TNR(c_matrix):
    tn=c_matrix[0][0]
    fp=c_matrix[0][1]
    return tn/(tn+fp)


# In[38]:


def RMSE(prob):
    return math.sqrt(sum(2*(prob**2))/len(prob))


# In[39]:


prob_true=df[df['cardio']==1].shape[0]/df['cardio'].shape[0]
E=-(prob_true*np.log2(prob_true)+(1-prob_true)*np.log2(1-prob_true))
def KBIS(prob):
    i=0
    for m in np.arange(len(prob)):
        if prob[m]>=prob_true:
            i+=-np.log2(prob_true)+np.log2(prob[m])
        else:
            i+=-np.log2(1-prob_true)+np.log2(1-prob[m])
    return i/len(prob)/E*100


# In[40]:


def brier(prob):
    return sum((1-prob)**2)/len(prob)


# In[41]:


def cal(x, y):
    return math.sqrt(sum((x-y)**2)/len(x))


# In[42]:


metric_num=7
rand=30
bin_num=50


# # Building a Model

# In[43]:


def model(data, classifier):
    x = data.values
    y = list(df['cardio'][:])
    result = np.array([])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=rand)
    sc = preprocessing.StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    classifier.fit(x_train,y_train)
    y_pred = classifier.predict(x_test)
    prob = classifier.predict_proba(x_test)[:,1]
    cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
    cal_y, cal_x = calibration_curve(y_test, prob, n_bins=bin_num)
    result = np.append(result, sklearn.metrics.accuracy_score(y_test, y_pred)*100)
    result = np.append(result, TPR(cm))
    result = np.append(result, TNR(cm))
    result = np.append(result, RMSE(1-prob))
    result = np.append(result, KBIS(prob))
    result = np.append(result, brier(prob))
    result = np.append(result, cal(cal_x, cal_y))
    result = np.reshape(result, (-1,metric_num))
    result = pd.DataFrame(result, columns=['Accuracy', 'TPR', 'TNR', 'RMSE', 'KBIS', 'brier', 'Cal'])
    # plot
    fig, ax = plt.subplots()
    plt.plot(cal_x, cal_y, marker='o', linewidth=1)
    line = mlines.Line2D([0, 1], [0, 1], color='black')
    ax.add_line(line)
    fig.suptitle('Calibration Plot')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('True Probability in each Bin')
#     plt.legend()
    plt.show()
    plt.hist(prob, bins=bin_num)
    plt.show()
    return classifier, prob, result


# In[44]:


def model_repeat(data, classifier):
    x = data.values
    y = list(df['cardio'][:])
    result = np.array([])
    for i in range(10):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=rand+i)
        sc = preprocessing.StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)
        classifier.fit(x_train,y_train)
        y_pred = classifier.predict(x_test)
        prob = classifier.predict_proba(x_test)[:,1]
        cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
        cal_y, cal_x = calibration_curve(y_test, prob, n_bins=bin_num)
        result = np.append(result, sklearn.metrics.accuracy_score(y_test, y_pred)*100)
        result = np.append(result, TPR(cm))
        result = np.append(result, TNR(cm))
        result = np.append(result, RMSE(1-prob))
        result = np.append(result, KBIS(prob))
        result = np.append(result, brier(prob))
        result = np.append(result, cal(cal_x, cal_y))
    result = np.reshape(result, (-1,metric_num))
    result = np.array(result.mean(axis=0))
    result = np.reshape(result, (-1,metric_num))
    result = pd.DataFrame(result, columns=['Accuracy', 'TPR', 'TNR', 'RMSE', 'KBIS', 'brier', 'Cal'])
    return result


# In[45]:


def model_add_prob(data, classifier1, classifier2):
    x = data.values
    y = list(df['cardio'][:])
    result = np.array([])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=rand)
    sc = preprocessing.StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    #first classifier
    classifier1.fit(x_train,y_train)
#     y_pred = classifier.predict(x_test)
    #tarin data
    prob_train = classifier1.predict_proba(x_train)[:,1]
    x_train = pd.DataFrame(x_train)
    x_train.insert(x.shape[1], 'prob', prob_train)
    x_train = x_train.values
    #test data
    prob_test = classifier1.predict_proba(x_test)[:,1]
    x_test = pd.DataFrame(x_test)
    x_test.insert(x.shape[1], 'prob', prob_test)
    x_test = x_test.values
    #second classifier
    classifier2.fit(x_train,y_train)
    y_pred = classifier2.predict(x_test)
    prob = classifier2.predict_proba(x_test)[:,1]
    cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
    cal_y, cal_x = calibration_curve(y_test, prob, n_bins=bin_num)
    result = np.append(result, sklearn.metrics.accuracy_score(y_test, y_pred)*100)
    result = np.append(result, TPR(cm))
    result = np.append(result, TNR(cm))
    result = np.append(result, RMSE(1-prob))
    result = np.append(result, KBIS(prob))
    result = np.append(result, brier(prob))
    result = np.append(result, cal(cal_x, cal_y))
    result = np.reshape(result, (-1,metric_num))
    result = pd.DataFrame(result, columns=['Accuracy', 'TPR', 'TNR', 'RMSE', 'KBIS', 'brier', 'Cal'])
    # plot
    fig, ax = plt.subplots()
    plt.plot(cal_x, cal_y, marker='o', linewidth=1)
    line = mlines.Line2D([0, 1], [0, 1], color='black')
    ax.add_line(line)
    fig.suptitle('Calibration Plot')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('True Probability in each Bin')
    plt.show()
    plt.hist(prob, bins=bin_num)
    plt.show()
    return classifier2, prob, result


# In[46]:


def feature_selection(data, classifier):
    x = data.values
    y = list(df['cardio'][:])
    features = np.array([])
    result = np.array([])
    #split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=rand)
    sc = preprocessing.StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    #building the model
    for i in range(x.shape[1]):
        selector = RFE(classifier, n_features_to_select=i+1, step=1)
        selector.fit(x_train,y_train)
        y_pred = selector.predict(x_test)
        prob = selector.predict_proba(x_test)[:,1]
        cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
        cal_y, cal_x = calibration_curve(y_test, prob, n_bins=bin_num)
        #fearures
        features = np.append(features, selector.support_)
        #metrics
        result = np.append(result, sklearn.metrics.accuracy_score(y_test, y_pred)*100)
        result = np.append(result, TPR(cm))
        result = np.append(result, TNR(cm))
        result = np.append(result, RMSE(1-prob))
        result = np.append(result, KBIS(prob))
        result = np.append(result, brier(prob))
        result = np.append(result, cal(cal_x, cal_y))
    features = np.reshape(features, (-1,x.shape[1]))
    features = pd.DataFrame(features, columns=data.columns)
    features.replace({0: False, 1: True}, inplace=True)
    features.insert(0,'Number of Features', np.arange(x.shape[1])+1)
    result = np.reshape(result, (-1,metric_num))
    result = pd.DataFrame(result, columns=['Accuracy', 'TPR', 'TNR', 'RMSE', 'KBIS', 'Brier', 'Cal'])
    result.insert(0,'Number of Features', np.arange(x.shape[1])+1)
    return features, result


# In[47]:


def lr_feature_selection (data, classifier):
    x=data
    result = pd.DataFrame()
    for i in range(x.shape[1]):
        for j in df.columns:
            if str(j)!='cardio':
                if lr_rfe_features[str(j)][i]==False:
                    x=pd.DataFrame(x.drop(str(j), axis=1, inplace=False))
        tmp = model_repeat(x, classifier)
        frames=[result, tmp]
        result = pd.concat(frames)
    #     print(x.columns)
        x=data
    result.insert(0, "Number of features", np.arange(1,x.shape[1]+1,1))
    df.reset_index(drop=True, inplace=True)
    return result


# In[48]:


def rf_feature_selection (data, classifier):
    x=data
    result = pd.DataFrame()
    for i in range(x.shape[1]):
        for j in df.columns:
            if str(j)!='cardio':
                if rf_tuned_rfe_features[str(j)][i]==False:
                    x=pd.DataFrame(x.drop(str(j), axis=1, inplace=False))
        tmp = model_repeat(x, classifier)
        frames=[result, tmp]
        result = pd.concat(frames)
    #     print(x.columns)
        x=data
    result.insert(0, "Number of features", np.arange(1,x.shape[1]+1,1))
    df.reset_index(drop=True, inplace=True)
    return result


# In[49]:


def add_bmi(data, classifier):
    bmi = data['weight'] / ((data["height"]/100)**2)
    data_bminum = data.copy(deep=True)
    data_bminum.insert(2,'bmi', bmi)
    data_bminum.drop(columns=['height', 'weight'], axis=1, inplace=True)
    for i in range(len(bmi)):
        if bmi[i]<18.5:
            bmi[i]=-1
        elif bmi[i]>25:
            bmi[i]=1
        else:
            bmi[i]=0
    # bmi.value_counts()
    lb_results=lb.fit_transform(bmi)
    # lb.classes_
    # lb_results
    data_bmi = data.copy(deep=True)
    tmp = pd.DataFrame(lb_results, columns=lb.classes_)
    data_bmi.insert(2,'BMI normal', tmp [0])
    data_bmi.insert(3,'BMI normal+', tmp [1])
    data_bmi.drop(columns=['height', 'weight'], axis=1, inplace=True)
    #building model
    hw = model_repeat(pd.DataFrame(data.drop('cardio', axis=1, inplace=False)), classifier)
    bminum = model_repeat(pd.DataFrame(data_bminum.drop('cardio', axis=1, inplace=False)), classifier)
    bmi = model_repeat(pd.DataFrame(data_bmi.drop('cardio', axis=1, inplace=False)), classifier)
    bmi_normal = model_repeat(pd.DataFrame(data_bmi.drop(columns=['cardio', 'BMI normal+'], axis=1, inplace=False)), classifier)
    bmi_normalplus = model_repeat(pd.DataFrame(data_bmi.drop(columns=['cardio', 'BMI normal'], axis=1, inplace=False)), classifier)
    frames = [hw, bminum, bmi, bmi_normal, bmi_normalplus]
    result = pd.concat(frames)
    result.insert(0,'Description', ['Height & Weight', 'BMI Num.', 'BMI Catagories', 'BMI Nomal', 'BMI Normal+'])
    return result


# In[50]:


sys=(df['systolic bp']>=130)+0
dias=(df['diastolic bp']>=80)+0
hbp=np.maximum(sys,dias)
def add_hbp(data, classifier):
    data_hbp = data.copy(deep=True)
    data_hbp.insert(4,'HBP', hbp)
    data_hbp.drop(columns=['systolic bp', 'diastolic bp'], axis=1, inplace=True)
    #building model
    sd = model_repeat(pd.DataFrame(data.drop(columns=['cardio'], axis=1, inplace=False)), classifier)
    s = model_repeat(pd.DataFrame(data.drop(columns=['cardio', 'diastolic bp'], axis=1, inplace=False)), classifier)
    d = model_repeat(pd.DataFrame(data.drop(columns=['cardio', 'systolic bp'], axis=1, inplace=False)), classifier)
    hbd = model_repeat(pd.DataFrame(data_hbp.drop(columns=['cardio'], axis=1, inplace=False)), classifier)
    frames = [sd, s, d, hbd]
    result = pd.concat(frames)
    result.insert(0,'Description', ['S&D', 'Systolic BP', 'Diastolic BP', 'HBP'])
    return result


# In[51]:


def drop_chol(data, classifier):
    #building model
    chol = model_repeat(pd.DataFrame(data.drop(columns=['cardio'], axis=1, inplace=False)), classifier)
    cholnormal = model_repeat(pd.DataFrame(data.drop(columns=['cardio', 'chol normal+'], axis=1, inplace=False)), classifier)
    cholplus = model_repeat(pd.DataFrame(data.drop(columns=['cardio', 'chol normal'], axis=1, inplace=False)), classifier)
    frames = [chol, cholnormal, cholplus]
    result = pd.concat(frames)
    result.insert(0,'Description', ['Cholesterol', 'Cholesterol Normal', 'Cholesterol Normal+'])
    return result


# In[52]:


def drop_gluc(data, classifier):
    #building model
    gluc = model_repeat(pd.DataFrame(data.drop(columns=['cardio'], axis=1, inplace=False)), classifier)
    glucnormal = model_repeat(pd.DataFrame(data.drop(columns=['cardio', 'gluc normal+'], axis=1, inplace=False)), classifier)
    glucplus = model_repeat(pd.DataFrame(data.drop(columns=['cardio', 'gluc normal'], axis=1, inplace=False)), classifier)
    frames = [gluc, glucnormal, glucplus]
    result = pd.concat(frames)
    result.insert(0,'Description', ['Glucose', 'Glucose Normal', 'Glucose Normal+'])
    return result


# # Export

# In[53]:


def export(df, loc):
    df.to_csv(loc+'.csv')
    df.to_excel(loc+'.xlsx')


# # Logistic Regression

# In[113]:


lr, lr_prob , lr_result = model(pd.DataFrame(df.drop('cardio', axis=1, inplace=False)), LogisticRegression(random_state=rand))


# In[85]:


lr_result.round(2)


# In[78]:


lr.coef_


# In[79]:


feature_importances = abs(lr.coef_.flatten())
plt.title('Feature Importances')
plt.barh(range(len(feature_importances)), feature_importances)
plt.yticks(range(len(feature_importances)), df.columns)
plt.show()


# In[63]:


lr_rfe_features, lr_rfe_result = feature_selection(pd.DataFrame(df.drop('cardio', axis=1, inplace=False)), LogisticRegression(random_state=rand))


# In[62]:


lr_rfe_features


# In[63]:


lr_rfe_result.round(2)


# In[113]:


lr_result = lr_feature_selection(pd.DataFrame(df.drop('cardio', axis=1, inplace=False)), LogisticRegression(random_state=rand))


# In[114]:


lr_result.round(2)


# In[132]:


export(lr_rfe_features, 'D:\B.Sc. Project\Result\Logistic Regression Features (RFE)')
export(lr_rfe_result, 'D:\B.Sc. Project\Result\Logistic Regression Result (RFE)')


# In[115]:


export(lr_result, 'D:\B.Sc. Project\Result\Logistic Regression Result (RFE-10Run)')


# In[69]:


# lr_rfe_result.drop(columns=['Number of Features', 'RMSE'], inplace=False).plot()


# # Random Forest

# In[113]:


rf, rf_prob , rf_result = model(pd.DataFrame(df.drop('cardio', axis=1, inplace=False)), classifier=RandomForestClassifier(random_state=rand))


# In[114]:


rf_result.round(2)


# In[133]:


export(rf_result, 'D:\B.Sc. Project\Result\Random Forest Result')


# In[110]:


feature_importances = rf.feature_importances_
plt.title('Feature Importances')
plt.barh(range(len(feature_importances)), feature_importances)
plt.yticks(range(len(feature_importances)), df.columns)
plt.show()


# In[111]:


# rf_rfe_features, rf_rfe_result = feature_selection(pd.DataFrame(df.drop('cardio', axis=1, inplace=False)), RandomForestClassifier(random_state=rand))


# In[112]:


# rf_rfe_features


# In[113]:


# rf_rfe_result.round(2)


# In[69]:


x = pd.DataFrame(df.drop('cardio', axis=1, inplace=False)).values
y = list(df['cardio'][:])
RandomForestClassifier(random_state=rand)
model_params = {
    'n_estimators': np.arange(100, 501, 100),
    'max_depth': np.arange(1, 11, 1),
    'min_samples_split': np.arange(2, 51, 2),
    'max_features': ['log2', None] #log2=sqrt=auto
}
rf_tuning = RandomForestClassifier(random_state=rand)
rf_randomsearch = RandomizedSearchCV(rf_tuning, model_params, n_iter=20, random_state=rand)
rf_tuned = rf_randomsearch.fit(x, y)
print(rf_tuned.best_estimator_.get_params())


# In[71]:


rf_tuned, rf_tuned_prob , rf_tuned_result = model(pd.DataFrame(df.drop('cardio', axis=1, inplace=False)), RandomForestClassifier(random_state=rand, n_estimators=100, max_depth=10, min_samples_split=16, max_features='log2'))


# In[72]:


rf_tuned_result.round(2)


# In[73]:


# export(rf_tuned_result, 'D:\B.Sc. Project\Result\Random Forest Result (Tuned)')


# In[74]:


feature_importances = rf_tuned.feature_importances_
plt.title('Feature Importances')
plt.barh(range(len(feature_importances)), feature_importances)
plt.yticks(range(len(feature_importances)), df.columns)
plt.show()


# In[117]:


rf_tuned_rfe_features, rf_tuned_rfe_result = feature_selection(pd.DataFrame(df.drop('cardio', axis=1, inplace=False)), RandomForestClassifier(random_state=rand, n_estimators=100, max_depth=10, min_samples_split=16, max_features='log2', warm_start=False))


# In[77]:


rf_tuned_rfe_features


# In[78]:


rf_tuned_rfe_result.round(2)


# In[118]:


rf_result = rf_feature_selection(pd.DataFrame(df.drop('cardio', axis=1, inplace=False)), RandomForestClassifier(random_state=rand, n_estimators=100, max_depth=10, min_samples_split=16, max_features='log2', warm_start=False))


# In[122]:


rf_result.round(2)


# In[79]:


export(rf_tuned_rfe_features, 'D:\B.Sc. Project\Result\Random Forest Features (RFE)')
export(rf_tuned_rfe_result, 'D:\B.Sc. Project\Result\Random Forest Result (RFE)')


# In[120]:


export(rf_result, 'D:\B.Sc. Project\Result\Random Forest Result (RFE-10Run)')


# # Neural Network

# In[93]:


nn, nn_prob , nn_result = model(pd.DataFrame(df.drop('cardio', axis=1, inplace=False)), MLPClassifier(random_state=rand))


# In[94]:


nn_result.round(2)


# In[95]:


x = pd.DataFrame(df.drop('cardio', axis=1, inplace=False)).values
y = list(df['cardio'][:])
model_params = {
    'hidden_layer_sizes': [(random.randrange(20, 200, step=20),), (random.randrange(20, 200, step=20),random.randrange(20, 200, step=20),), (random.randrange(20, 200, step=20),random.randrange(20, 200, step=20),random.randrange(20, 200, step=20),)],
    'activation':['identity', 'logistic', 'tanh', 'relu'],
    'learning_rate':['constant', 'invscaling', 'adaptive'],
    'batch_size':np.arange(200, 1001, 200)
}
nn_tuning = MLPClassifier(random_state=rand, early_stopping=True)
nn_randomsearch = RandomizedSearchCV(nn_tuning, model_params, n_iter=20, random_state=rand+1)
nn_tuned = nn_randomsearch.fit(x, y)
print(nn_tuned.best_estimator_.get_params())


# In[96]:


nn_tuned, nn_tuned_prob , nn_tuned_result = model(pd.DataFrame(df.drop('cardio', axis=1, inplace=False)), MLPClassifier(random_state=rand, hidden_layer_sizes=(180,140,140), activation='relu', batch_size=600))


# In[97]:


nn_tuned_result.round(2)


# In[135]:


export(nn_tuned_result,'D:\B.Sc. Project\Result\\Neural Network Result (Tuned)')


# In[123]:


nn_result = rf_feature_selection(pd.DataFrame(df.drop('cardio', axis=1, inplace=False)), MLPClassifier(random_state=rand))


# In[112]:


nn_rfe_result.round(2)


# In[127]:


nn_result.round(2)


# In[138]:


export(nn_rfe_result, 'D:\B.Sc. Project\Result\\Neural Network Result (RFE)')


# In[126]:


export(nn_result, 'D:\B.Sc. Project\Result\\Neural Network Result (RFE-10Run)')


# # K Neighbors

# In[135]:


knn, knn_prob , knn_result = model(pd.DataFrame(df.drop('cardio', axis=1, inplace=False)), KNeighborsClassifier())


# In[136]:


knn_result.round(2)


# In[58]:


knn, knn_prob , knn_result = model(pd.DataFrame(df.drop('cardio', axis=1, inplace=False)), KNeighborsClassifier(n_neighbors=220))


# In[56]:


knn_result.round(2)


# In[125]:


knn, knn_prob , knn_result = model(pd.DataFrame(df.drop('cardio', axis=1, inplace=False)), KNeighborsClassifier(n_neighbors=220, p=1))


# In[126]:


knn_result.round(2)


# In[139]:


export(knn_result,'D:\B.Sc. Project\Result\KNN Result (Manhattan)')


# In[128]:


knn_result = rf_feature_selection(pd.DataFrame(df.drop('cardio', axis=1, inplace=False)), KNeighborsClassifier(n_neighbors=220))


# In[142]:


knn_rfe_result.round(2)


# In[129]:


knn_result.round(2)


# In[143]:


export(knn_rfe_result, 'D:\B.Sc. Project\Result\KNN Result (RFE)')


# In[130]:


export(knn_result, 'D:\B.Sc. Project\Result\KNN Result (RFE-10Run)')


# # Improving Logistic Regression

# In[135]:


lr_df = df.copy(deep=True)
lr_df.drop(columns=['male', 'height', 'gluc normal', 'gluc normal+', 'smoke', 'alcohol'], inplace=True)
lr_df.head()


# In[169]:


# lr_bmi = add_bmi(lr_df, LogisticRegression(random_state=rand))


# In[170]:


# export(lr_bmi, 'D:\B.Sc. Project\Result\Logistic Regression (BMI)')
# lr_bmi.round(2)


# In[136]:


lr_hbp = add_hbp(lr_df, LogisticRegression(random_state=rand))


# In[59]:


export(lr_hbp, 'D:\B.Sc. Project\Result\Logistic Regression (HBP)')
lr_hbp.round(2)


# In[137]:


export(lr_hbp, 'D:\B.Sc. Project\Result\Logistic Regression (HBP-10Run)')
lr_hbp.round(2)


# In[138]:


lr_chol = drop_chol(lr_df, LogisticRegression(random_state=rand))


# In[179]:


export(lr_chol, 'D:\B.Sc. Project\Result\Logistic Regression (Cholesterol)')
lr_chol.round(2)


# In[139]:


export(lr_chol, 'D:\B.Sc. Project\Result\Logistic Regression (Cholesterol-10Run)')
lr_chol.round(2)


# In[180]:


# lr_gluc = drop_gluc(lr_df, LogisticRegression(random_state=rand))


# In[181]:


# export(lr_gluc, 'D:\B.Sc. Project\Result\Logistic Regression (Glucose)')
# lr_gluc.round(2)


# # Improving Random Forest

# In[140]:


rf_df = df.copy(deep=True)
# rf_df.drop(columns=['male', 'gluc normal', 'gluc normal+', 'smoke', 'alcohol'], inplace=True)
rf_df.drop(columns=['male', 'gluc normal+', 'alcohol'], inplace=True)
rf_df.head()


# In[141]:


rf_bmi = add_bmi(rf_df, RandomForestClassifier(random_state=rand, n_estimators=100, max_depth=10, min_samples_split=16, max_features='log2'))


# In[65]:


export(rf_bmi, 'D:\B.Sc. Project\Result\Random Forest (BMI)')
rf_bmi.round(2)


# In[142]:


export(rf_bmi, 'D:\B.Sc. Project\Result\Random Forest (BMI-10Run)')
rf_bmi.round(2)


# In[143]:


rf_hbp = add_hbp(rf_df, RandomForestClassifier(random_state=rand, n_estimators=100, max_depth=10, min_samples_split=16, max_features='log2'))


# In[62]:


export(rf_hbp, 'D:\B.Sc. Project\Result\Random Forest (HBP)')
rf_hbp.round(2)


# In[144]:


export(rf_hbp, 'D:\B.Sc. Project\Result\Random Forest (HBP-10Run)')
rf_hbp.round(2)


# In[145]:


rf_chol = drop_chol(rf_df, RandomForestClassifier(random_state=rand, n_estimators=100, max_depth=10, min_samples_split=16, max_features='log2'))


# In[69]:


export(rf_chol, 'D:\B.Sc. Project\Result\Random Forest (Cholesterol)')
rf_chol.round(2)


# In[146]:


export(rf_chol, 'D:\B.Sc. Project\Result\Random Forest (Cholesterol-10Run)')
rf_chol.round(2)


# In[61]:


# rf_gluc = drop_gluc(rf_df, RandomForestClassifier(random_state=rand, n_estimators=100, max_depth=10, min_samples_split=16, max_features='log2'))


# In[62]:


# export(rf_gluc, 'D:\B.Sc. Project\Result\Random Forest (Glucose)')
# rf_gluc.round(2)


# # Improving Neural Network

# In[147]:


nn_df = df.copy(deep=True)
nn_df.drop(columns=['male'], inplace=True)
nn_df.head()


# In[148]:


nn_bmi = add_bmi(nn_df, MLPClassifier(random_state=rand))


# In[196]:


export(nn_bmi, 'D:\B.Sc. Project\Result\\Neural Network (BMI)')
nn_bmi.round(2)


# In[149]:


export(nn_bmi, 'D:\B.Sc. Project\Result\\Neural Network (BMI-10Run)')
nn_bmi.round(2)


# In[150]:


nn_hbp = add_hbp(nn_df, MLPClassifier(random_state=rand))


# In[68]:


export(nn_hbp, 'D:\B.Sc. Project\Result\\Neural Network (HBP)')
nn_hbp.round(2)


# In[151]:


export(nn_hbp, 'D:\B.Sc. Project\Result\\Neural Network (HBP-10Run)')
nn_hbp.round(2)


# In[152]:


nn_chol = drop_chol(rf_df, MLPClassifier(random_state=rand))


# In[200]:


export(nn_chol, 'D:\B.Sc. Project\Result\\Neural Network (Cholesterol)')
nn_chol.round(2)


# In[153]:


export(nn_chol, 'D:\B.Sc. Project\Result\\Neural Network (Cholesterol-10Run)')
nn_chol.round(2)


# In[154]:


nn_gluc = drop_gluc(nn_df, MLPClassifier(random_state=rand))


# In[202]:


export(nn_gluc, 'D:\B.Sc. Project\Result\\Neural Network (Glucose)')
nn_gluc.round(2)


# In[155]:


export(nn_gluc, 'D:\B.Sc. Project\Result\\Neural Network (Glucose-10Run)')
nn_gluc.round(2)


# # Improving KNN

# In[156]:


knn_df = df.copy(deep=True)
knn_df.drop(columns=['male', 'gluc normal', 'gluc normal+','smoke', 'alcohol', 'active'], inplace=True)
knn_df.head()


# In[157]:


knn_bmi = add_bmi(knn_df, KNeighborsClassifier(n_neighbors=220))


# In[72]:


export(knn_bmi, 'D:\B.Sc. Project\Result\KNN (BMI)')
knn_bmi.round(2)


# In[158]:


export(knn_bmi, 'D:\B.Sc. Project\Result\KNN (BMI-10Run)')
knn_bmi.round(2)


# In[159]:


knn_hbp = add_hbp(knn_df, KNeighborsClassifier(n_neighbors=220))


# In[65]:


export(knn_hbp, 'D:\B.Sc. Project\Result\KNN (HBP)')
knn_hbp.round(2)


# In[160]:


export(knn_hbp, 'D:\B.Sc. Project\Result\KNN (HBP-10Run)')
knn_hbp.round(2)


# In[161]:


knn_chol = drop_chol(knn_df, KNeighborsClassifier(n_neighbors=220))


# In[76]:


export(knn_chol, 'D:\B.Sc. Project\Result\KNN (Cholesterol)')
knn_chol.round(2)


# In[162]:


export(knn_chol, 'D:\B.Sc. Project\Result\KNN (Cholesterol10-Run)')
knn_chol.round(2)


# In[77]:


# knn_gluc = drop_gluc(knn_df, KNeighborsClassifier(n_neighbors=220))


# In[78]:


# export(knn_gluc, 'D:\B.Sc. Project\Result\KNN (Glucose)')
# knn_gluc.round(2)


# In[ ]:


#this is not updated!!!!!


# In[58]:


bmi = data['weight'] / ((data["height"]/100)**2)
knn_df.insert(2,'BMI', bmi)
knn_df.drop(columns=['height', 'weight'], axis=1, inplace=True)
knn_df.head()


# In[95]:


knn_hbp = add_hbp(knn_df, KNeighborsClassifier(n_neighbors=220))


# In[96]:


export(knn_hbp, 'D:\B.Sc. Project\Result\KNN Improved (HBP)')
knn_hbp.round(2)


# # 4 Best Models

# In[59]:


duration = np.array([])
#Logistic Regression
start = time.time()
lr, lr_prob , lr_result = model(lr_df.drop('cardio', axis=1, inplace=False), LogisticRegression(random_state=rand))
duration = np.append(duration, time.time()-start)
#Random Forest
start = time.time()
rf, rf_prob , rf_result = model(rf_df.drop('cardio', axis=1, inplace=False), RandomForestClassifier(random_state=rand, n_estimators=100, max_depth=10, min_samples_split=16, max_features='log2'))
duration = np.append(duration, time.time()-start)
#Neural Network
start = time.time()
nn, nn_prob , nn_result = model(nn_df.drop('cardio', axis=1, inplace=False), MLPClassifier(random_state=rand))
duration = np.append(duration, time.time()-start)
#KNN
start = time.time()
knn, knn_prob , knn_result = model(knn_df.drop('cardio', axis=1, inplace=False), KNeighborsClassifier(n_neighbors=220))
duration = np.append(duration, time.time()-start)
#result
frames=[lr_result, rf_result, nn_result, knn_result]
selection_result = pd.concat(frames)
selection_result.insert(0, 'Model', ['Logistic Regression', 'Random Forest', 'Neural Network', 'KNN'])
selection_result.insert(1, 'Duration', duration)
export(selection_result, 'D:\B.Sc. Project\Result\Four Best Models')


# In[ ]:


#this is not updated!!!!!


# In[60]:


selection_result.round(2)


# In[163]:


nn_rf, nn_rf_prob , nn_rf_result = model_add_prob(nn_df.drop('cardio', axis=1, inplace=False), RandomForestClassifier(random_state=rand, n_estimators=100, max_depth=10, min_samples_split=16, max_features='log2'), MLPClassifier(random_state=rand))


# In[164]:


export(nn_rf_result, 'D:\B.Sc. Project\Result\\Neural Network on Random Forest Result')
nn_rf_result.round(2)


# In[165]:


rf_nn, rf_nn_prob , rf_nn_result = model_add_prob(nn_df.drop('cardio', axis=1, inplace=False), MLPClassifier(random_state=rand), RandomForestClassifier(random_state=rand, n_estimators=100, max_depth=10, min_samples_split=16, max_features='log2'))


# In[166]:


export(rf_nn_result, 'D:\B.Sc. Project\Result\\Random Forest on Neural Network Result')
rf_nn_result.round(2)


# In[ ]:


lr, lr_prob , lr_result = model(pd.DataFrame(df.drop(['cardio', 'male', 'age', 'height', 'weight', 'diastolic bp', 'chol normal', 'chol normal+', 'gluc normal', 'gluc normal+', 'smoke', 'alcohol', 'active'], axis=1, inplace=False)), RandomForestClassifier(random_state=rand, n_estimators=100, max_depth=10, min_samples_split=16, max_features='log2'))


# In[105]:


x1 = df.drop('cardio', axis=1, inplace=False).values
x2 = df.drop(columns=['cardio', 'male', 'age', 'height', 'weight', 'diastolic bp', 'chol normal', 'chol normal+', 'gluc normal', 'gluc normal+', 'smoke', 'alcohol', 'active'],
             axis=1, inplace=False).values
y = list(df['cardio'][:])
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y, test_size=0.3, random_state=rand)
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y, test_size=0.3, random_state=rand)
sc1 = preprocessing.StandardScaler()
x1_train = sc1.fit_transform(x1_train)
x1_test = sc1.transform(x1_test)
sc2 = preprocessing.StandardScaler()
x2_train = sc2.fit_transform(x2_train)
x2_test = sc2.transform(x2_test)  
classifier1 = RandomForestClassifier(random_state=rand, n_estimators=100, max_depth=10, min_samples_split=16, max_features='log2')
classifier1.fit(x1_train,y1_train)
prob1 = classifier1.predict_proba(x1_test)[:,1]
cal_y1, cal_x1 = calibration_curve(y1_test, prob1, n_bins=bin_num)
classifier2 = RandomForestClassifier(random_state=rand, n_estimators=100, max_depth=10, min_samples_split=16, max_features='log2')
classifier2.fit(x2_train,y2_train)
prob2 = classifier2.predict_proba(x2_test)[:,1]
cal_y2, cal_x2 = calibration_curve(y2_test, prob2, n_bins=bin_num)
# plot
fig, ax = plt.subplots()
plt.plot(cal_x1, cal_y1, marker='o', linewidth=1, label='13 Features')
plt.plot(cal_x2, cal_y2, marker='o', linewidth=1, label='1 Feature')
line = mlines.Line2D([0, 1], [0, 1], color='black')
ax.add_line(line)
fig.suptitle('Calibration Plot')
ax.set_xlabel('Mean Predicted Probability')
ax.set_ylabel('True Probability in each Bin')
plt.legend()
plt.show()


# In[ ]:




