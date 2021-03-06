import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC



from sklearn.feature_selection import SelectKBest

from sklearn.model_selection import GridSearchCV, train_test_split,cross_validate
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('data/train.csv',dtype={"Age": np.float64})
test = pd.read_csv('data/test.csv',dtype={"Age": np.float64})
PassengerId=test['PassengerId']
all_data = pd.concat([train, test])
facet = sns.FacetGrid(train, hue="Survived",aspect=2)
facet.map(sns.kdeplot,'Age',shade= True)

facet = sns.FacetGrid(train, hue="Survived",aspect=2)
facet.map(sns.kdeplot,'Fare',shade= True)

train.loc[train.Cabin.isnull(), 'Cabin'] = 'U0'
train['Has_Cabin'] = train['Cabin'].apply(lambda x: 0 if x == 'U0' else 1)
all_data['Cabin'] = all_data['Cabin'].fillna('Unknown')
all_data['Deck']=all_data['Cabin'].str.get(0)
all_data['Title'] = all_data['Name'].apply(lambda x:x.split(',')[1].split('.')[0].strip())
Title_Dict = {}
Title_Dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
Title_Dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
Title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
Title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
Title_Dict.update(dict.fromkeys(['Mr'], 'Mr'))
Title_Dict.update(dict.fromkeys(['Master','Jonkheer'], 'Master'))
all_data['Title'] = all_data['Title'].map(Title_Dict)
all_data['FamilySize']=all_data['SibSp']+all_data['Parch']+1
def Fam_label(s):
    if (s >= 2) & (s <= 4):
        return 2
    elif ((s > 4) & (s <= 7)) | (s == 1):
        return 1
    elif (s > 7):
        return 0
all_data['FamilyLabel']=all_data['FamilySize'].apply(Fam_label)
Ticket_Count = dict(all_data['Ticket'].value_counts())
all_data['TicketGroup'] = all_data['Ticket'].apply(lambda x:Ticket_Count[x])
def Ticket_Label(s):
    if (s >= 2) & (s <= 4):
        return 2
    elif ((s > 4) & (s <= 8)) | (s == 1):
        return 1
    elif (s > 8):
        return 0

all_data['TicketGroup'] = all_data['TicketGroup'].apply(Ticket_Label)

train = all_data[all_data['Survived'].notnull()]
test  = all_data[all_data['Survived'].isnull()]
# 分割数据，按照 训练数据:cv数据 = 1:1的比例
train_split_1, train_split_2 = train_test_split(train, test_size=0.5, random_state=0)


def predict_age_use_cross_validationg(df1, df2, dfTest):
    age_df1 = df1[['Age', 'Pclass', 'Sex', 'Title']]
    age_df1 = pd.get_dummies(age_df1)
    age_df2 = df2[['Age', 'Pclass', 'Sex', 'Title']]
    age_df2 = pd.get_dummies(age_df2)

    known_age = age_df1[age_df1.Age.notnull()].as_matrix()
    unknow_age_df1 = age_df1[age_df1.Age.isnull()].as_matrix()
    unknown_age = age_df2[age_df2.Age.isnull()].as_matrix()

    print(unknown_age.shape)

    y = known_age[:, 0]
    X = known_age[:, 1:]

    rfr = RandomForestRegressor(random_state=0, n_estimators=100, n_jobs=-1)
    rfr.fit(X, y)
    predictedAges = rfr.predict(unknown_age[:, 1::])
    df2.loc[(df2.Age.isnull()), 'Age'] = predictedAges
    predictedAges = rfr.predict(unknow_age_df1[:, 1::])
    df1.loc[(df1.Age.isnull()), 'Age'] = predictedAges

    age_Test = dfTest[['Age', 'Pclass', 'Sex', 'Title']]
    age_Test = pd.get_dummies(age_Test)
    age_Tmp = df2[['Age', 'Pclass', 'Sex', 'Title']]
    age_Tmp = pd.get_dummies(age_Tmp)

    age_Tmp = pd.concat([age_Test[age_Test.Age.notnull()], age_Tmp])

    known_age1 = age_Tmp.as_matrix()
    unknown_age1 = age_Test[age_Test.Age.isnull()].as_matrix()
    y = known_age1[:, 0]
    x = known_age1[:, 1:]

    rfr.fit(x, y)
    predictedAges = rfr.predict(unknown_age1[:, 1:])
    dfTest.loc[(dfTest.Age.isnull()), 'Age'] = predictedAges

    return dfTest


t1 = train_split_1.copy()
t2 = train_split_2.copy()
tmp1 = test.copy()
t5 = predict_age_use_cross_validationg(t1, t2, tmp1)
t1 = pd.concat([t1, t2])

t3 = train_split_1.copy()
t4 = train_split_2.copy()
tmp2 = test.copy()
t6 = predict_age_use_cross_validationg(t4, t3, tmp2)
t3 = pd.concat([t3, t4])

train['Age'] = (t1['Age'] + t3['Age']) / 2

test['Age'] = (t5['Age'] + t6['Age']) / 2

print(train.describe())
print(test.describe())

all_data = pd.concat([train, test])
all_data[all_data['Embarked'].isnull()]
all_data['Embarked'] = all_data['Embarked'].fillna('C')

fare=all_data[(all_data['Embarked'] == "S") & (all_data['Pclass'] == 3)].Fare.median()
all_data['Fare']=all_data['Fare'].fillna(fare)

all_data['Surname']=all_data['Name'].apply(lambda x:x.split(',')[0].strip())
Surname_Count = dict(all_data['Surname'].value_counts())
all_data['FamilyGroup'] = all_data['Surname'].apply(lambda x:Surname_Count[x])
Female_Child_Group=all_data.loc[(all_data['FamilyGroup']>=2) & ((all_data['Age']<=12) | (all_data['Sex']=='female'))]
Male_Adult_Group=all_data.loc[(all_data['FamilyGroup']>=2) & (all_data['Age']>12) & (all_data['Sex']=='male')]
Female_Child=pd.DataFrame(Female_Child_Group.groupby('Surname')['Survived'].mean().value_counts())
Female_Child.columns=['GroupCount']

Male_Adult=pd.DataFrame(Male_Adult_Group.groupby('Surname')['Survived'].mean().value_counts())
Male_Adult.columns=['GroupCount']
Female_Child_Group=Female_Child_Group.groupby('Surname')['Survived'].mean()
Dead_List=set(Female_Child_Group[Female_Child_Group.apply(lambda x:x==0)].index)
print(Dead_List)
Male_Adult_List=Male_Adult_Group.groupby('Surname')['Survived'].mean()
Survived_List=set(Male_Adult_List[Male_Adult_List.apply(lambda x:x==1)].index)
print(Survived_List)

train=all_data.loc[all_data['Survived'].notnull()]
test=all_data.loc[all_data['Survived'].isnull()]
test.loc[(test['Surname'].apply(lambda x:x in Dead_List)),'Sex'] = 'male'
test.loc[(test['Surname'].apply(lambda x:x in Dead_List)),'Age'] = 60
test.loc[(test['Surname'].apply(lambda x:x in Dead_List)),'Title'] = 'Mr'
test.loc[(test['Surname'].apply(lambda x:x in Survived_List)),'Sex'] = 'female'
test.loc[(test['Surname'].apply(lambda x:x in Survived_List)),'Age'] = 5
test.loc[(test['Surname'].apply(lambda x:x in Survived_List)),'Title'] = 'Miss'

all_data=pd.concat([train, test])
all_data=all_data[['Survived','Pclass','Sex','Age','Fare','Embarked','Title','FamilyLabel','Deck','TicketGroup']]
all_data=pd.get_dummies(all_data)
# all_data.to_csv('cug_result/all_data.csv')
# test_df_org = pd.read_csv('data/test.csv')
# PassengerId = test_df_org['PassengerId']
# data_all = pd.read_csv('new_data/train_test_data.csv')
# combined_train_test = data_all
# combined_train_test.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
# all_data=combined_train_test
# train_data = combined_train_test[:891]
# test_data = combined_train_test[891:]
# scale_age_fare = preprocessing.StandardScaler().fit(combined_train_test[['Age', 'Fare', 'Name_length']])
# combined_train_test[['Age', 'Fare','self_Fare' github'Name_length']] = scale_age_fare.transform(
#     combined_train_test[['Age', 'Fare', 'Name_length']])

# train_data_X = train_data.drop(['Survived'], axis=1)
# train_data_Y = train_data['Survived']
# test_data_X = test_data.drop(['Survived'], axis=1)


train=all_data[all_data['Survived'].notnull()]
test=all_data[all_data['Survived'].isnull()].drop('Survived',axis=1)
X = train.as_matrix()[:,1:]
y = train.as_matrix()[:,0]

piperf=Pipeline([('select',SelectKBest(k=20)),
               ('classify', RandomForestClassifier(random_state = 10,max_features='sqrt'))])
param_rf = {'classify__n_estimators':[24],
'classify__max_depth':[6],
            'classify__min_samples_split':[2],






              }
gsearchrf = GridSearchCV(estimator = piperf, param_grid = param_rf, scoring='f1', cv=10)
gsearchrf.fit(X,y)
print(gsearchrf.best_params_, gsearchrf.best_score_)
# pipeet=Pipeline([('select',SelectKBest(k=20)),
#                ('classify', ExtraTreesClassifier(random_state = 10, max_features = 'sqrt'))])
# param_et = {'classify__n_estimators':range(20,50,2),
#               'classify__max_depth': range(3, 60, 3)
#               }
# gsearchet = GridSearchCV(estimator = pipeet, param_grid = param_et, scoring='roc_auc', cv=10)
# gsearchet.fit(X,y)
# print(gsearchet.best_params_, gsearchet.best_score_)
# pipegb=Pipeline([('select',SelectKBest(k=20)),
#                ('classify', GradientBoostingClassifier(random_state = 10, max_features = 'sqrt'))])
# param_gb = {'classify__n_estimators':[42,48],
#               'classify__max_depth': [3],
# 'classify__max_depth': [3],
# 'classify__learning_rate':[0.21947368421052632],
# 'classify__subsample':[0.878],
# 'classify__min_child_weight': range(1, 9, 1)
#
#
#
#               }
# gsearchgb = GridSearchCV(estimator = pipegb, param_grid = param_gb, scoring='roc_auc', cv=10)
# gsearchgb.fit(X,y)
# print(gsearchgb.best_params_, gsearchgb.best_score_)
# pipexg=Pipeline([('select',SelectKBest(k=20)),
#                ('classify', XGBClassifier(random_state = 10))])
# param_xg = {'classify__n_estimators':[20],
#               'classify__max_depth':[9],
#             'classify__learning_rate':[0.01],
# 'classify__subsample':[0.876],
# 'classify__min_child_weight': range(1, 9, 1)
#
#               }
# gsearchxg = GridSearchCV(estimator = pipexg, param_grid = param_xg, scoring='roc_auc', cv=10)
# gsearchxg.fit(X,y)
# print(gsearchxg.best_params_, gsearchxg.best_score_)

# clf=XGBClassifier(random_state = 10, max_features = 'sqrt',
# n_estimators=22,
#               max_depth=9,
#               learning_rate=0.428,
#               subsample=0.742105,
#
# colsample_bytree=0.98,
# min_child_weight=5,
# gamma=0.001


                  # )
select = SelectKBest(k = 25)
clf = RandomForestClassifier(random_state = 10, warm_start = True,
                                  n_estimators = 30,
                                  max_depth = 6,
                                  max_features = 15,

                             )
pipeline = make_pipeline(select, clf)
pipeline.fit(X, y)

predictions = pipeline.predict(test)
submission = pd.DataFrame({"PassengerId": PassengerId, "Survived": predictions.astype(np.int32)})
submission.to_csv("cug_result/submission_5.csv", index=False)