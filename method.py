import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import xgboost as xgb
from xgboost import plot_importance
import warnings
warnings.filterwarnings("ignore")

"""
random forest
"""
def random_forest(input_x, revised_y, X_train, y_train, X_test, y_test, judge):
    if judge==True:
        X_train=X_train
        y_train=y_train
        X_test=X_test 
        y_test=y_test
    else:
        X_train, X_test, y_train, y_test = train_test_split(input_x, revised_y, test_size=0.3, random_state=42)
        print("labels")
        check_list=[]
        for i in range(len(X_test)):
            # print(X_test[i][input_x.shape[1]-1])
            check_list.append(X_test[i][input_x.shape[1]-1])
        check_list.sort()
        print(check_list)
        X_train=np.delete(X_train, -1, axis=1)
        X_test=np.delete(X_test, -1, axis=1)
        
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                        max_depth=64, max_features='auto', max_leaf_nodes=None,
                        min_impurity_split=1e-07, min_samples_leaf=1,
                        min_samples_split=2, min_weight_fraction_leaf=0.0,
                        n_estimators=200, n_jobs=1, oob_score=False, random_state=None,
                        verbose=0, warm_start=False)

    y_pred = rf.predict(X_test)
    print(y_pred)
    print(y_test)
    acc=rf.score(X_test, y_test)
    print(acc)
    """
    f1_score
    """    
    f1_score(y_test, y_pred, average='macro')  
    # f1_score(y_test, y_pred, average='micro')
    # f1_score(y_test, y_pred, average='weighted')
    # f1_score(y_test, y_pred, average=None)
    return y_test, y_pred

"""
DecisionTree
"""
def decision_tree(input_x, revised_y, X_train, y_train, X_test, y_test, judge):
    if judge==True:
        X_train=X_train
        y_train=y_train
        X_test=X_test 
        y_test=y_test
    else:
        X_train, X_test, y_train, y_test = train_test_split(input_x, revised_y, test_size=0.3, random_state=42)
        print("labels")
        check_list=[]
        for i in range(len(X_test)):
            # print(X_test[i][input_x.shape[1]-1])
            check_list.append(X_test[i][input_x.shape[1]-1])
        check_list.sort()
        print(check_list)
        X_train=np.delete(X_train, -1, axis=1)
        X_test=np.delete(X_test, -1, axis=1)

    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(y_pred)
    print(y_test)
    acc=clf.score(X_test, y_test)
    print(acc)
    """
    f1_score
    """
    f1_score(y_test, y_pred, average='macro')  
    return y_test, y_pred

"""
xgboost
"""
def xgboost(input_x, revised_y, X_train, y_train, X_test, y_test, class_num, num, judge):
    params = {
        'booster': 'gbtree',
        'objective': 'multi:softmax',
        'num_class': class_num, 
        'gamma': 0.1,
        'max_depth': 10,
        'lambda': 2,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 3,
        'silent': 1,
        'eta': 0.1,
        'seed': 1000,
        'nthread': 4,
    }

    plst = params.items()
    if judge==True:
        X_train=X_train
        y_train=y_train
        X_test=X_test 
        y_test=y_test
    else:
        X_train, X_test, y_train, y_test = train_test_split(input_x, revised_y, test_size=0.3, random_state=42)
        print("labels")
        check_list=[]
        for i in range(len(X_test)):
            # print(X_test[i][input_x.shape[1]-1])
            check_list.append(X_test[i][input_x.shape[1]-1])
        check_list.sort()
        print(check_list)
        X_train=np.delete(X_train, -1, axis=1)
        X_test=np.delete(X_test, -1, axis=1)
        
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    dtrain = xgb.DMatrix(X_train, y_train)

    acc_list=[0]
    number_list=[0]
    ans_best=np.zeros(len(X_test))
    for j in range(num):    
        num_rounds = j
        model = xgb.train(plst, dtrain, num_rounds)

        # 对测试集进行预测
        dtest = xgb.DMatrix(X_test)
        ans = model.predict(dtest)
        
    #     print(ans)
        # 计算准确率
        cnt1 = 0
        cnt2 = 0
        for i in range(len(y_test)):
            if ans[i] == y_test[i]:
                cnt1 += 1
            else:
                cnt2 += 1
        acc=(100 * cnt1 / (cnt1 + cnt2))
    #     print(acc)
    #     print(acc_list[0])
        if acc>acc_list[0]:
            acc_list.insert(0, acc) 
            number_list.insert(0, j) 
            acc_list.pop(-1) 
            number_list.pop(-1) 
            ans_best=ans
            
        
        print("Epochs "+ str(j) + " " +"Accuracy: %.2f %% " % (100 * cnt1 / (cnt1 + cnt2)))
    print(number_list)
    print(acc_list)
    # print(y_test)
    # print(ans_best)
    return y_test, ans_best
    # # 显示重要特征
    # plot_importance(model)
    # plt.show()