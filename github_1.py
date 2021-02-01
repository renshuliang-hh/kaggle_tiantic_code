import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
rf = RandomForestClassifier(n_estimators=370, warm_start=True, max_features='sqrt', max_depth=6,
                            min_samples_split=3, min_samples_leaf=2, n_jobs=-1, verbose=0)

ada = AdaBoostClassifier(n_estimators=370, learning_rate=0.1)

et = ExtraTreesClassifier(n_estimators=370, n_jobs=-1, max_depth=8, min_samples_leaf=2, verbose=0)

gb = GradientBoostingClassifier(n_estimators=370, learning_rate=0.008, min_samples_split=3, min_samples_leaf=2,
                                max_depth=5, verbose=0)

dt = DecisionTreeClassifier(max_depth=8)

knn = KNeighborsClassifier(n_neighbors=5)

svm = SVC(kernel='linear', C=0.025)

# 特征预处理map函数
def change_age(x):
    if x < 16:
        return 1
    else:
        return 0


def change_family(x):
    if x == 1 or x == 2 or x == 3:
        return 1
    else:
        return 0


def change_embarked(x):
    if x == 'S':
        return 0
    elif x == 'C':
        return 1
    elif x == 'Q':
        return 2


def pro_train_data(datas):
    datas['family'] = datas.SibSp + datas.Parch
    datas.loc[[65, 159, 176, 709], 'Age'] = 15
    age = datas.Age.map(change_age)
    datas['Age'] = age
    datas['family'] = datas.family.map(change_family)
    cabin_datas = datas.Cabin.replace(np.nan, 0)
    datas['Cabin'] = np.where(cabin_datas == 0, 0, 1)
    datas.Embarked.fillna('C', inplace=True)
    datas['Embarked'] = datas.Embarked.map(change_embarked)
    datas['Sex'] = np.where(datas['Sex'] == 'female', 1, 0)
    to_drop = ['PassengerId', 'Survived', 'Name', 'Ticket', 'SibSp',
               'Parch']
    target = datas.Survived
    datas_handle = datas.drop(to_drop, axis=1)
    return datas_handle, target


def pro_test_data(test_datas):
    test_datas.loc[(test_datas.Fare.isnull()), 'Fare'] = test_datas[
        (test_datas.Embarked == 'S') & (test_datas.Pclass == 3)].Fare.median()
    test_datas['family'] = test_datas.SibSp + test_datas.Parch
    test_datas.loc[[244, 344, 417], 'Age'] = 15
    age = test_datas.Age.map(change_age)
    test_datas['Age'] = age
    test_datas['family'] = test_datas.family.map(change_family)
    cabin_datas = test_datas.Cabin.replace(np.nan, 0)
    test_datas['Cabin'] = np.where(cabin_datas == 0, 0, 1)
    test_datas.Embarked.fillna('S', inplace=True)
    test_datas['Embarked'] = test_datas.Embarked.map(change_embarked)
    test_datas['Sex'] = np.where(test_datas['Sex'] == 'female', 1, 0)
    to_drop = ['PassengerId', 'Name', 'Ticket', 'SibSp',
               'Parch']
    new_test_data_handle = test_datas.drop(to_drop, axis=1)

    return new_test_data_handle


def clf(datas, label, pre_datas):

    x_train, x_test, y_train, y_test = train_test_split(datas, label)
    ntrain = x_train.shape[0]
    ntest = x_test.shape[0]
    npre=pre_datas.shape[0]
    rf_oof_train, rf_oof_test, rf_oof_pre, = get_out_fold(rf, x_train, y_train, x_test,pre_datas,ntrain,ntest,npre)  # Random Forest
    ada_oof_train, ada_oof_test,ada_oof_pre, = get_out_fold(ada, x_train, y_train, x_test,pre_datas,ntrain,ntest,npre)  # AdaBoost
    et_oof_train, et_oof_test, et_oof_pre, = get_out_fold(et, x_train, y_train, x_test,pre_datas,ntrain,ntest,npre)  # Extra Trees
    gb_oof_train, gb_oof_test, gb_oof_pre, = get_out_fold(gb, x_train, y_train, x_test,pre_datas,ntrain,ntest,npre)  # Gradient Boost
    dt_oof_train, dt_oof_test, dt_oof_pre, = get_out_fold(dt, x_train, y_train, x_test,pre_datas,ntrain,ntest,npre)  # Decision Tree
    # knn_oof_train, knn_oof_test,knn_oof_pre, = get_out_fold(knn, x_train, y_train, x_test,pre_datas,ntrain,ntest,npre)  # KNeighbors
    svm_oof_train, svm_oof_test,svm_oof_pre, = get_out_fold(svm, x_train, y_train, x_test,pre_datas,ntrain,ntest,npre)  # Support Vector
    x_train = np.concatenate((rf_oof_train, ada_oof_train, et_oof_train, gb_oof_train, dt_oof_train, svm_oof_train),
                             axis=1)
    x_test = np.concatenate((rf_oof_test, ada_oof_test, et_oof_test, gb_oof_test, dt_oof_test, svm_oof_test), axis=1)
    x_pre = np.concatenate((rf_oof_pre, ada_oof_pre, et_oof_pre, gb_oof_pre, dt_oof_pre,svm_oof_pre), axis=1)

    xgc = XGBClassifier()
    xgc_param = {
        'n_estimators': range(30, 70, 2),
        'max_depth': range(2, 7, 1),
    #     'learning_rate' :[0.0001, 0.001, 0.01, 0.1, 0.2, 0.3], # 学习率
    # 'gamma': [1, 0.1, 0.01, 0.001],
    #     'subsample': np.linspace(0.7, 0.9, 20),
    #     'colsample_bytree': np.linspace(0.5, 0.98, 10),
    #     'min_child_weight': range(1, 9, 1)

    }
    gc = GridSearchCV(xgc, param_grid=xgc_param, cv=5)
    gc.fit(x_train, y_train)
    print("训练集样本为：", x_train.shape[0])
    print("测试集样本为：", x_test.shape[0])
    print("预测率为：", gc.score(x_test, y_test))
    print("交叉验证最好结果：", gc.best_score_)
    print("交叉验证最好参数模型：", gc.best_estimator_)

    return gc.predict(x_pre)

SEED = 0 # for reproducibility
NFOLDS = 7 # set folds for out-of-fold prediction
kf = KFold(n_splits = NFOLDS, random_state=SEED, shuffle=False)
def get_out_fold(clf, x_train, y_train, x_test,pre_data,ntrain,ntest,npre):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_pre=np.zeros((npre,))
    oof_test_skf = np.empty((NFOLDS, ntest))
    oof_pre_skf = np.empty((NFOLDS, npre))

    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        x_tr = x_train.values[train_index]
        y_tr = y_train.values[train_index]
        x_te = x_train.values[test_index]


        clf.fit(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)
        oof_pre_skf[i, :] = clf.predict(pre_data)

    oof_test[:] = oof_test_skf.mean(axis=0)
    oof_pre[:]=oof_pre_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1),oof_pre.reshape(-1, 1)
def generate_result(test_data, result):
    result = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': result.astype(np.int32)})
    result.to_csv("new_result/stakde.csv", index=False)



if __name__ == '__main__':
    # train_data = pd.read_csv('data/train.csv')
    test_df_org = pd.read_csv('data/test.csv')
    PassengerId = test_df_org['PassengerId']
    data_all = pd.read_csv('new_data/all_train_test_data.csv')
    combined_train_test = data_all
    combined_train_test.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)
    train_data = combined_train_test[:891]
    test_data = combined_train_test[891:]
    train_data_X = train_data.drop(['Survived'], axis=1)
    train_data_Y = train_data['Survived']
    test_data_X = test_data.drop(['Survived'], axis=1)

    # pro_datas, target = pro_train_data(train_data)
    # pre_datas = pro_test_data(test_data)



    pre_y = clf(train_data_X, train_data_Y , test_data_X)
    StackingSubmission = pd.DataFrame({'PassengerId': PassengerId, 'Survived': pre_y})
    StackingSubmission.to_csv('new_result/stakdall.csv', index=False, sep=',')
    # generate_result(test_data, pre_y)

