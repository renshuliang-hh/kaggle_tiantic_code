import re
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import warnings
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import keras
from sklearn.externals import joblib
from sklearn.ensemble import AdaBoostClassifier

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

warnings.filterwarnings('ignore')
train_data = pd.read_csv('data/tran_test_all.csv')
test_data = pd.read_csv('data/test.csv')

from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# ['Age', 'Embarked', 'Sex', 'Title', 'Name_length', 'Family_Size', 'Family_Size_Category','Fare', 'Fare_bin_id', 'Pclass']
age_df = train_data[
    ['Age', 'Embarked', 'Sex', 'Title', 'Name_length', 'Family_Size', 'Family_Size_Category', 'Fare', 'Pclass']]
age_df_notnull = age_df.loc[(train_data['Age'].notnull())]
age_df_isnull = age_df.loc[(train_data['Age'].isnull())]
X = age_df_notnull.values[:, 1:]

Y = age_df_notnull.values[:, 0]
pre = age_df_isnull.values[:, 1:]

# Y[Y<=10]=1
# Y[(Y<=20)&(Y>10)]=2
# Y[(Y<=30)&(Y>20)]=3
# Y[(Y<=40)&(Y>30)]=4
# Y[(Y<=50)&(Y>40)]=5
# Y[(Y<=60)&(Y>50)]=6
# Y[(Y<=70)&(Y>60)]=7
# Y[(Y<=80)&(Y>70)]=8
# Y[(Y<=90)&(Y>90)]=9
Y[Y <= 10] = 0
Y[(Y <= 20) & (Y > 10)] = 1
Y[(Y <= 40) & (Y > 20)] = 2
Y[(Y <= 60) & (Y > 40)] = 3
Y[(Y <= 80) & (Y > 60)] = 4
Y[(Y <= 100) & (Y > 80)] = 5

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2)
# # use RandomForestRegression to train data
# RFR = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
# RFC = RandomForestClassifier(n_estimators=370, n_jobs=-1)
# RFC = LogisticRegression(penalty='l2')
from xgboost import XGBClassifier
# RFC = KNeighborsClassifier()
# rfc = RFC.fit(Xtrain, Ytrain)
# Ypre = rfc.predict(Xtest)
# score_r1 = rfc.score(Xtest, Ytest)
# score_r2 = rfc.score(X, Y)
# score_r3 = rfc.score(Xtrain, Ytrain)
# Age_pre = rfc.predict(pre)

xgc = XGBClassifier()
RFC = RandomForestClassifier( n_estimators=370,n_jobs=-1)
xgc_param = {
    'n_estimators': range(30, 50, 2),
    'max_depth': range(2, 7, 1)
}



# predictAges = RFR.predict(age_df_isnull.values[:,1:])
# train_data.loc[train_data['Age'].isnull(), ['Age']]= predictAges
# superpa = []
# joblib.dump(rfc, 'modle/age_6_'+str(score_r1)+'_ABC.model')
# # joblib.dump(rfc, 'modle/age_5_0.6112.model')
# superpa = []
# for i in range(100):
#     i=(i+1)*10
#     rfc = RandomForestClassifier(n_estimators=i,n_jobs=-1)
#     rfc_s = cross_val_score(rfc,X,Y,cv=4).mean()
#     superpa.append(rfc_s)
#     print(max(superpa),superpa.index(max(superpa)))
# plt.figure(figsize=[20,5])
# plt.plot(range(1,101),superpa)
# plt.show()
