import re
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import warnings
from sklearn.model_selection import train_test_split
from keras.layers import Dense,Input
from keras.models import Model
from keras.models import Sequential
from sklearn.externals import joblib
warnings.filterwarnings('ignore')
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')
import keras
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
train_data.info()
print("-" * 40)
test_data.info()
# train_data['Survived'].value_counts().plot.pie(autopct = '%1.2f%%')
# # # plt.show()
train_data.Embarked[train_data.Embarked.isnull()] = train_data.Embarked.dropna().mode().values
train_data['Cabin'] = train_data.Cabin.fillna('U0')
age_df = train_data[['Age','Survived','Fare', 'Parch', 'SibSp', 'Pclass']]
age_df_notnull = age_df.loc[(train_data['Age'].notnull())]
age_df_isnull = age_df.loc[(train_data['Age'].isnull())]
X = age_df_notnull.values[:,1:]

Y = age_df_notnull.values[:,0]
Y[Y<=10]=1
Y[(Y<=20)&(Y>10)]=2
Y[(Y<=30)&(Y>20)]=3
Y[(Y<=40)&(Y>30)]=4
Y[(Y<=50)&(Y>40)]=5
Y[(Y<=60)&(Y>50)]=6
Y[(Y<=70)&(Y>60)]=7
Y[(Y<=80)&(Y>70)]=8
Y[(Y<=90)&(Y>90)]=9




Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y,test_size=0.1)
Ytrain = keras.utils.to_categorical(Ytrain, 9)
Ytest = keras.utils.to_categorical(Ytest, 9)
# # use RandomForestRegression to train data
# RFR = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
# RFC = RandomForestClassifier( n_jobs=-1)
# rfc = RFC.fit(Xtrain,Ytrain)
# Ypre = rfc.predict(Xtest)
# score_r1 = rfc.score(Xtest,Ytest)
# score_r2= rfc.score(X,Y)
# score_r3= rfc.score(Xtrain,Ytrain)
# # score_r = r2_score(Ytest,Ypre)
# # predictAges = RFR.predict(age_df_isnull.values[:,1:])
# # train_data.loc[train_data['Age'].isnull(), ['Age']]= predictAges
# superpa = []
# def baseline_model():
#     model=Sequential()
#     model.add(Dense(8,input_dim=5,activation='relu'))
#     model.add(Dense(16, activation='relu'))
#     model.add(Dense(9, activation='softmax'))
#     model.compile(optimizer='adam',loss='categorical_crossentropy',  metrics=['mae', 'acc','accuracy'])

input=Input(shape=(5,))
x=Dense(8,activation='relu')(input)
x=Dense(16,activation='relu')(x)
x=Dense(9,activation='softmax')(x)
model=Model(input,x)
model.compile(optimizer='adam',loss='categorical_crossentropy',  metrics=['mae', 'acc','accuracy'])
model.fit(Xtrain, Ytrain,epochs=1000,validation_split=0.2)
eval=model.evaluate(Xtest,Ytest)