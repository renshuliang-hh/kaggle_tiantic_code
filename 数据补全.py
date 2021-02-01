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
import keras
from sklearn.externals import joblib
warnings.filterwarnings('ignore')
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

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
RF=joblib.load('modle/age_0.41666.model')
predictAges = RF.predict(age_df_isnull.values[:,1:])

train_data.loc[train_data['Age'].isnull(), ['Age']]= (predictAges-1)*10+5
# train_data['Age'][train_data['Age']<=10]=1
# train_data['Age'][(train_data['Age']<=20)&(train_data['Age']>10)]=2
# train_data['Age'][(train_data['Age']<=30)&(train_data['Age']>20)]=3
# train_data['Age'][(train_data['Age']<=40)&(train_data['Age']>30)]=4
# train_data['Age'][(train_data['Age']<=50)&(train_data['Age']>40)]=5
# train_data['Age'][(train_data['Age']<=60)&(train_data['Age']>50)]=6
# train_data['Age'][(train_data['Age']<=70)&(train_data['Age']>60)]=7
# train_data['Age'][(train_data['Age']<=80)&(train_data['Age']>70)]=8
# train_data['Age'][(train_data['Age']<=90)&(train_data['Age']>80)]=9
print("-" * 40)
train_data.info()
train_data.to_csv('data/train_age_0-100_10.csv')