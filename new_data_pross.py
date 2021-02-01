import re
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import warnings
from sklearn.model_selection import train_test_split
import keras
from sklearn import preprocessing
from sklearn.externals import joblib
# warnings.filterwarnings('ignore')
from sklearn.model_selection import GridSearchCV
from sklearn import  metrics

train_df_org = pd.read_csv('data/train.csv')
test_df_org = pd.read_csv('data/test.csv')
test_df_org['Survived'] = 0
combined_train_test = train_df_org.append(test_df_org)
PassengerId = test_df_org['PassengerId']

emb_map={
    'C':2,
    'Q':1,
    'S':0
}
combined_train_test['Embarked'].fillna(combined_train_test['Embarked'].mode().iloc[0], inplace=True)
combined_train_test['Embarked'] = combined_train_test['Embarked'].map(emb_map)
emb_dummies_df = pd.get_dummies(combined_train_test['Embarked'], prefix=combined_train_test[['Embarked']].columns[0])
combined_train_test = pd.concat([combined_train_test, emb_dummies_df], axis=1)



sex_map={
    'male':0,
    'female':1
}
combined_train_test['Sex'] = combined_train_test['Sex'].map(sex_map)
emb_dummies_df = pd.get_dummies(combined_train_test['Sex'], prefix=combined_train_test[['Sex']].columns[0])
combined_train_test = pd.concat([combined_train_test, emb_dummies_df], axis=1)




pclass_map={
    3:0,
    2:1,
    1:2
}
combined_train_test['Pclass'] = combined_train_test['Pclass'].map(pclass_map)
emb_dummies_df = pd.get_dummies(combined_train_test['Pclass'], prefix=combined_train_test[['Pclass']].columns[0])
combined_train_test = pd.concat([combined_train_test, emb_dummies_df], axis=1)

title_map={
    'Royalty':5,
    'Officer':1,
    'Mr':0,
'Miss':3,
'Master':2,
'Mrs':4,

}
combined_train_test['Title'] = combined_train_test['Name'].map(lambda x: re.compile(", (.*?)\.").findall(x)[0])
title_Dict = {}
title_Dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
title_Dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
title_Dict.update(dict.fromkeys(['Mr'], 'Mr'))
title_Dict.update(dict.fromkeys(['Master','Jonkheer'], 'Master'))

combined_train_test['Title'] = combined_train_test['Title'].map(title_Dict).map(title_map)
emb_dummies_df = pd.get_dummies(combined_train_test['Title'], prefix=combined_train_test[['Title']].columns[0])
combined_train_test = pd.concat([combined_train_test, emb_dummies_df], axis=1)


combined_train_test['Name_length'] = combined_train_test['Name'].apply(len)
combined_train_test['Fare'] = combined_train_test[['Fare']].fillna(combined_train_test.groupby('Pclass').transform(np.mean))
combined_train_test['Group_Ticket'] = combined_train_test['Fare'].groupby(by=combined_train_test['Ticket']).transform('count')
combined_train_test['self_Fare'] = combined_train_test['Fare'] / combined_train_test['Group_Ticket']
combined_train_test['Fare_bin'] = pd.qcut(combined_train_test['Fare'], 5)
fare_map = {
0:0,
1:4,
2:1,
3:2,
4:3

}

combined_train_test['Fare_bin'] = pd.factorize(combined_train_test['Fare_bin'])[0]
combined_train_test['Fare_bin']=combined_train_test['Fare_bin'].map(fare_map)
emb_dummies_df = pd.get_dummies(combined_train_test['Fare_bin'], prefix=combined_train_test[['Fare_bin']].columns[0])
combined_train_test = pd.concat([combined_train_test, emb_dummies_df], axis=1)
family_map={
'Small_Family':2,
'Single':0,
'Large_Family':1

}


def family_size_category(family_size):
    if family_size <= 1:
        return 'Single'
    elif family_size <= 4:
        return 'Small_Family'
    else:
        return 'Large_Family'
combined_train_test['Family_Size'] = combined_train_test['Parch'] + combined_train_test['SibSp'] + 1
combined_train_test['Family_Size_Category'] = combined_train_test['Family_Size'].map(family_size_category).map(family_map)
emb_dummies_df = pd.get_dummies(combined_train_test['Family_Size_Category'], prefix=combined_train_test[['Family_Size_Category']].columns[0])
combined_train_test = pd.concat([combined_train_test, emb_dummies_df], axis=1)

age_yuce=pd.read_csv('new_data/trmp.csv')['Age']
combined_train_test['Age']=age_yuce
combined_train_test['Age_c']=age_yuce

combined_train_test['Age_c'][combined_train_test['Age_c']<=16]=1
combined_train_test['Age_c'][combined_train_test['Age_c']>16]=0
emb_dummies_df = pd.get_dummies(combined_train_test['Age_c'], prefix=combined_train_test[['Age_c']].columns[0])
combined_train_test = pd.concat([combined_train_test, emb_dummies_df], axis=1)

combined_train_test.loc[combined_train_test.Cabin.isnull(), 'Cabin'] = 'U0'
combined_train_test['Cabin'] = combined_train_test['Cabin'].apply(lambda x: 0 if x == 'U0' else 1)
emb_dummies_df = pd.get_dummies(combined_train_test['Cabin'], prefix=combined_train_test[['Cabin']].columns[0])
combined_train_test = pd.concat([combined_train_test, emb_dummies_df], axis=1)
combined_train_test.to_csv('new_data/all_train_test_data.csv')






