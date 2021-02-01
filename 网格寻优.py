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
from sklearn.metrics import accuracy_score,roc_auc_score,recall_score
from sklearn.model_selection import KFold
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
rf = RandomForestClassifier(n_estimators=370, warm_start=True, max_features='sqrt', max_depth=6,
                            min_samples_split=3, min_samples_leaf=2, n_jobs=-1, verbose=0)

ada = AdaBoostClassifier(n_estimators=370, learning_rate=0.1)

et = ExtraTreesClassifier(n_estimators=370, n_jobs=-1, max_depth=8, min_samples_leaf=2, verbose=0)

gb = GradientBoostingClassifier(n_estimators=370, learning_rate=0.008, min_samples_split=3, min_samples_leaf=2,
                                max_depth=5, verbose=0)

dt = DecisionTreeClassifier(max_depth=8)

knn = KNeighborsClassifier(n_neighbors=2)

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

def get_top_n_features(titanic_train_data_X, titanic_train_data_Y, top_n_features):
    # random forest
    rf_est = RandomForestClassifier(random_state=0)
    rf_param_grid = {'n_estimators': [100], 'min_samples_split': [2, 3], 'max_depth': [20]}
    rf_grid = model_selection.GridSearchCV(rf_est, rf_param_grid, n_jobs=25, cv=10, verbose=1)
    rf_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    print('Top N Features Best RF Params:' + str(rf_grid.best_params_))
    print('Top N Features Best RF Score:' + str(rf_grid.best_score_))
    print('Top N Features RF Train Score:' + str(rf_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_rf = pd.DataFrame({'feature': list(titanic_train_data_X),
                                          'importance': rf_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_rf = feature_imp_sorted_rf.head(top_n_features)['feature'][:top_n_features]
    print('Sample 10 Features from RF Classifier')
    print(str(features_top_n_rf[:top_n_features]))

    # AdaBoost
    ada_est = AdaBoostClassifier(random_state=0)
    ada_param_grid = {'n_estimators': [100], 'learning_rate': [0.01, 0.1]}
    ada_grid = model_selection.GridSearchCV(ada_est, ada_param_grid, n_jobs=25, cv=10, verbose=1)
    ada_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    print('Top N Features Best Ada Params:' + str(ada_grid.best_params_))
    print('Top N Features Best Ada Score:' + str(ada_grid.best_score_))
    print('Top N Features Ada Train Score:' + str(ada_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_ada = pd.DataFrame({'feature': list(titanic_train_data_X),
                                           'importance': ada_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_ada = feature_imp_sorted_ada.head(top_n_features)['feature'][:top_n_features]
    print('Sample 10 Feature from Ada Classifier:')
    print(str(features_top_n_ada[:top_n_features]))

    # ExtraTree
    et_est = ExtraTreesClassifier(random_state=0)
    et_param_grid = {'n_estimators': [100], 'min_samples_split': [3, 4], 'max_depth': [20]}
    et_grid = model_selection.GridSearchCV(et_est, et_param_grid, n_jobs=25, cv=10, verbose=1)
    et_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    print('Top N Features Best ET Params:' + str(et_grid.best_params_))
    print('Top N Features Best ET Score:' + str(et_grid.best_score_))
    print('Top N Features ET Train Score:' + str(et_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_et = pd.DataFrame({'feature': list(titanic_train_data_X),
                                          'importance': et_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_et = feature_imp_sorted_et.head(top_n_features)['feature'][:top_n_features]
    print('Sample 10 Features from ET Classifier:')
    print(str(features_top_n_et[:top_n_features]))

    # GradientBoosting
    gb_est = GradientBoostingClassifier(random_state=0)
    gb_param_grid = {'n_estimators': [100], 'learning_rate': [0.01, 0.1], 'max_depth': [20]}
    gb_grid = model_selection.GridSearchCV(gb_est, gb_param_grid, n_jobs=25, cv=10, verbose=1)
    gb_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    print('Top N Features Best GB Params:' + str(gb_grid.best_params_))
    print('Top N Features Best GB Score:' + str(gb_grid.best_score_))
    print('Top N Features GB Train Score:' + str(gb_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_gb = pd.DataFrame({'feature': list(titanic_train_data_X),
                                          'importance': gb_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_gb = feature_imp_sorted_gb.head(top_n_features)['feature'][:top_n_features]
    print('Sample 10 Feature from GB Classifier:')
    print(str(features_top_n_gb[:top_n_features]))

    # DecisionTree
    dt_est = DecisionTreeClassifier(random_state=0)
    dt_param_grid = {'min_samples_split': [2, 4], 'max_depth': [20]}
    dt_grid = model_selection.GridSearchCV(dt_est, dt_param_grid, n_jobs=25, cv=10, verbose=1)
    dt_grid.fit(titanic_train_data_X, titanic_train_data_Y)
    print('Top N Features Best DT Params:' + str(dt_grid.best_params_))
    print('Top N Features Best DT Score:' + str(dt_grid.best_score_))
    print('Top N Features DT Train Score:' + str(dt_grid.score(titanic_train_data_X, titanic_train_data_Y)))
    feature_imp_sorted_dt = pd.DataFrame({'feature': list(titanic_train_data_X),
                                          'importance': dt_grid.best_estimator_.feature_importances_}).sort_values(
        'importance', ascending=False)
    features_top_n_dt = feature_imp_sorted_dt.head(top_n_features)['feature'][:top_n_features]
    print('Sample 10 Features from DT Classifier:')
    print(str(features_top_n_dt[:top_n_features]))

    # merge the three models
    features_top_n = pd.concat(
        [features_top_n_rf, features_top_n_ada, features_top_n_et, features_top_n_gb, features_top_n_dt],
        ignore_index=True).drop_duplicates()

    features_importance = pd.concat([feature_imp_sorted_rf, feature_imp_sorted_ada, feature_imp_sorted_et,
                                     feature_imp_sorted_gb, feature_imp_sorted_dt], ignore_index=True)

    return features_top_n, features_importance





from sklearn.model_selection import KFold

# Some useful parameters which will come in handy later on


def clf(datas, label, pre_datas):

    x_train, x_test, y_train, y_test = train_test_split(datas, label)
    ntrain = x_train.shape[0]
    ntest = x_test.shape[0]
    npre=pre_datas.shape[0]
    rfc = RandomForestClassifier(random_state=0,n_estimators=100,max_depth=4,max_features=14)
    ada = AdaBoostClassifier()
    et = ExtraTreesClassifier()
    # 'max_depth': range(2, 15, 1),'learning_rate':[0.05]


    xgc = XGBClassifier()
    param_grid = {'n_estimators': range(20,1000,20)}


    gc = GridSearchCV(ada , param_grid=param_grid, cv=5,scoring='roc_auc')
    gc.fit(x_train, y_train)
    print("训练集样本为：", x_train.shape[0])
    print("测试集样本为：", x_test.shape[0])
    print("预测率为：", gc.score(x_test, y_test))
    print("交叉验证最好结果：", gc.best_score_)
    yt_pre=gc.predict(x_test)
    accuracy_score, roc_auc_score
    print("精度：", accuracy_score(y_test,yt_pre))
    print("召回率：", recall_score(y_test,yt_pre))
    print("AUC", roc_auc_score(y_test,yt_pre))
    print(gc.best_estimator_)




    return gc.predict(pre_datas)

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
    data_all = pd.read_csv('new_data/train_test_data.csv')
    combined_train_test = data_all
    combined_train_test.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)


    train_data = combined_train_test[:891]
    test_data = combined_train_test[891:]
    # scale_age_fare = preprocessing.StandardScaler().fit(combined_train_test[['Age', 'Fare', 'Name_length']])
    # combined_train_test[['Age', 'Fare','self_Fare' 'Name_length']] = scale_age_fare.transform(
    #     combined_train_test[['Age', 'Fare', 'Name_length']])

    train_data_X = train_data.drop(['Survived'], axis=1)
    train_data_Y = train_data['Survived']
    test_data_X = test_data.drop(['Survived'], axis=1)
    # feature_to_pick = 10
    # feature_top_n, feature_importance = get_top_n_features(train_data_X, train_data_Y, feature_to_pick)
    # train_data_X = pd.DataFrame(train_data_X[feature_top_n])
    # test_data_X = pd.DataFrame(test_data_X[feature_top_n])

    # pro_datas, target = pro_train_data(train_data)
    # pre_datas = pro_test_data(test_data)
    pre_y = clf(train_data_X, train_data_Y , test_data_X)

    # StackingSubmission = pd.DataFrame({'PassengerId': PassengerId, 'Survived': pre_y})
    # StackingSubmission.to_csv('new_result/rf_3.csv', index=False, sep=',')



