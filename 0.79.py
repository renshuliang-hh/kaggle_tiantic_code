import pandas as pd
import numpy as np
import random as rnd
# 可视化
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression    #逻辑回归
from sklearn.svm import SVC, LinearSVC                 #支持向量机
from sklearn.ensemble import RandomForestClassifier    #随机森林
from sklearn.neighbors import KNeighborsClassifier     #K最邻近算法
from sklearn.naive_bayes import GaussianNB             #朴素贝叶斯
from sklearn.linear_model import Perceptron            #感知机算法
from sklearn.linear_model import SGDClassifier         #梯度下降分类
from sklearn.tree import DecisionTreeClassifier        #决策树算法
from sklearn.model_selection import StratifiedKFold    #K折交叉切分
from sklearn.model_selection import GridSearchCV       #网格搜索
import warnings

warnings.filterwarnings("ignore")
# 获取数据
train_df = pd.read_csv('data\\train.csv')
test_df = pd.read_csv('data\\test.csv')
# 合并数据集
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]
for dataset in combine:
    # A-Z匹配任何大写字母，a-z匹配任何小写字母，'+'1次或多次匹配，\.就是匹配符号'.' (点)
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)# expand=False表示返回DataFrame

#crosstab交叉列表取值， 第一个参数是列, 第二个参数是行
pd.crosstab(train_df['Title'], train_df['Sex'])
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',
    'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

#进一步的，我们可以将这些离散型的Title转换为有序的数值型
title_mapping = {"Mr":1, "Miss":2, "Mrs":3, "Master":4, "Rare":5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
# test中的PassengerId等会提交结果要用，不能删
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
train_df[['Title', 'Survived','Age']].groupby(['Title'], as_index=False).mean()
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean()
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]
guess_ages = np.zeros((2, 3))
#迭代sex（0或1）和pclass（1，2，3）来计算六个组合的年龄估计值。
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) &  (dataset['Pclass'] == j+1)]['Age'].dropna()
            age_guess = guess_df.median()
            #将随机年龄浮动转换为最接近的 0.5 岁
            guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5
    for i in range(0, 2):
            for j in range(0, 3):
                dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),
                            'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'],
as_index=False).mean()
for dataset in combine:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
freq_port = train_df.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
# 展示Embarked与Survived的关系图表
train_df[['Embarked', 'Survived']].groupby(['Embarked'],
as_index=False).mean()
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0 , 'C': 1 , 'Q': 2}).astype(int)
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
#我们也可以将Fare的数值做四舍五入，将它精确到2位
for dataset in combine:
    dataset['Fare']=round(dataset['Fare'],2)
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby(['FareBand'],as_index=False).mean()
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]
for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass
train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)
train_df.to_csv("new_data/0.79_train.csv")
test_df.to_csv("new_data/0.79_test.csv")

X_train = train_df.drop('Survived', axis=1)
Y_train = train_df['Survived']

X_test = test_df.drop('PassengerId', axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape
forestModel = RandomForestClassifier(n_estimators=100)
# 训练模型
forestModel.fit(X_train, Y_train)
# 对模型进行评分
acc_random_forest = round(forestModel.score(X_train, Y_train) * 100, 2)
forestModel = RandomForestClassifier(n_estimators=250,max_depth=5,criterion='gini')
# 训练模型
forestModel.fit(X_train, Y_train)
# 对模型进行评分
acc_random_forest = round(forestModel.score(X_train, Y_train) * 100, 2)
acc_random_forest
Y_pred = forestModel.predict(X_test)
passenger_id = test_df['PassengerId']
submitDf = pd.DataFrame( {
    'PassengerId': passenger_id ,
    'Survived': Y_pred } )
submitDf.to_csv('new_data/0.79.csv')