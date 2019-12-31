import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from class_method import pie
from method import random_forest, decision_tree, xgboost
from confusion import plot_confusion_matrix
from nn import nn

def data_preprocess(source_data, nan=True):
    # print(len(source_data))
    box=source_data["累計銷售金額"]
    watch=source_data["預告觀看次數"]
    like=source_data["喜歡"]
    dislike=source_data["不喜歡"]
    variable=source_data["新類別"]
    month=pd.to_datetime(source_data["上映日期"]).dt.month

    revised_y=np.zeros(len(box))
    for i in range(len(source_data)):
        # print("money:")
        # print(source_data["累計銷售金額"].describe()['50%'])
        # if source_data["累計銷售金額"][i] < 684206:        
        #     revised_y[i]=0
        # else:        
        #     revised_y[i]=1
        if source_data["累計銷售金額"][i] < source_data["累計銷售金額"].describe()['25%']:        
            revised_y[i]=0
        elif source_data["累計銷售金額"].describe()['25%'] < source_data["累計銷售金額"][i] < source_data["累計銷售金額"].describe()['50%']:        
            revised_y[i]=1
        elif source_data["累計銷售金額"].describe()['50%'] < source_data["累計銷售金額"][i] < source_data["累計銷售金額"].describe()['75%']:
            revised_y[i]=2
        elif source_data["累計銷售金額"][i] > source_data["累計銷售金額"].describe()['75%']:
            revised_y[i]=3
    
    """
    watch preprocess
    """
    watch_data=np.zeros(len(watch))
    for i in range(len(watch_data)):
        watch_data[i]=watch[i]
    watch_data=preprocessing.scale(watch_data) 
    """
    like preprocess
    """
    like_ratio=np.zeros(len(like))
    for i in range(len(like)):
        tmp=like[i]/(like[i]+dislike[i])
    #     if pd.isna(tmp):
        if np.isnan(tmp)==True :
            if nan==True:
                tmp=np.nan
            else:
                tmp=0            
        like_ratio[i]=tmp
    like_ratio=preprocessing.scale(like_ratio)
    """
    dislike preprocess
    """
    dislike_ratio=np.zeros(len(like))
    for i in range(len(like)):
        tmp=dislike[i]/(like[i]+dislike[i])
    #     if pd.isna(tmp):
        if np.isnan(tmp)==True :
            if nan==True:
                tmp=np.nan
            else:
                tmp=0     
        dislike_ratio[i]=tmp
    dislike_ratio=preprocessing.scale(dislike_ratio)
    """
    like_audience preprocess
    """
    like_audience_ratio=np.zeros(len(like))
    for i in range(len(like)):
        tmp=like[i]/(watch_data[i])
        if np.isnan(tmp)==True :
            if nan==True:
                tmp=np.nan
            else:
                tmp=0            
        like_audience_ratio[i]=tmp
    like_audience_ratio=preprocessing.scale(like_audience_ratio)
    """
    dislike_audience preprocess
    """
    dislike_audience_ratio=np.zeros(len(like))
    for i in range(len(like)):
        tmp=dislike[i]/(watch_data[i])
        if np.isnan(tmp)==True :
            if nan==True:
                tmp=np.nan
            else:
                tmp=0     
        dislike_audience_ratio[i]=tmp
    dislike_audience_ratio=preprocessing.scale(dislike_audience_ratio)
    """
    like inverse
    """
    like_inverse=np.zeros(len(like))
    for i in range(len(like)):
        if like[i]==0:
            if nan==True:
                tmp=np.nan
            else:
                tmp=0     
        else:
            tmp=1/(like[i])
        like_inverse[i]=tmp
    # print(like_inverse)
    like_inverse=preprocessing.scale(like_inverse)
    """
    dislike inverse
    """
    dislike_inverse=np.zeros(len(like))
    for i in range(len(like)):
        if dislike[i]==0:
            if nan==True:
                tmp=np.nan
            else:
                tmp=0
        else:
            tmp=1/(dislike[i])
        dislike_inverse[i]=tmp
    dislike_inverse=preprocessing.scale(dislike_inverse)      
    input_x=np.stack((watch_data, 
                      like_ratio, dislike_ratio,
                      like_audience_ratio, dislike_audience_ratio, 
                      like_inverse, dislike_inverse,
                      variable, month),axis=1)
    # print(input_x.shape)
    return input_x, revised_y

def main(class_num, epochs, batch_size, optimizer, loss, method, kind, judge, nan):
    if kind=="seven":
        train_source="./new/new_seven_train_data.csv"
        test_source="./new/new_seven_test_data.csv"
    elif kind=="eight":
        train_source="./new/new_eight_train_data.csv"
        test_source="./new/new_eight_test_data.csv"
    else:
        train_source="./new/new_nine_train_data.csv"
        test_source="./new/new_nine_test_data.csv"


    train_data=pd.read_csv(train_source)
    train_data.reset_index(drop=True)
    train_final_data=pie(num=11, source_data=train_data, all_option=True)        
    test_data=pd.read_csv(test_source)
    test_data.reset_index(drop=True)
    test_final_data=pie(num=11, source_data=test_data, all_option=True)

    X_train, y_train=data_preprocess(train_final_data, nan=nan)
    X_test, y_test=data_preprocess(test_final_data, nan=nan)

    input_x=0
    revised_y=0
    if method=="random_forest":
        y_test, ans_best=random_forest(input_x, revised_y, X_train, y_train, X_test, y_test, judge=judge)
        plot(class_num, y_test, ans_best)
    elif method=="decision_tree":
        y_test, ans_best=decision_tree(input_x, revised_y, X_train, y_train, X_test, y_test, judge=judge)
        plot(class_num, y_test, ans_best)
    elif method=="xgboost":                            
        y_test, ans_best=xgboost(input_x, revised_y, X_train, y_train, X_test, y_test, class_num=class_num,
        num=100, judge=judge)
        print(y_test)
        plot(class_num, y_test, ans_best)
    else:
        ohe = OneHotEncoder()
        y_train = ohe.fit_transform(y_train.reshape(-1, 1)).toarray()
        y_test = ohe.fit_transform(y_test.reshape(-1, 1)).toarray()    
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        nn(X_train, X_test, y_train, y_test, class_num=class_num, input_dim=X_train.shape[1], 
        epochs=epochs, batch_size=batch_size, optimizer=optimizer, loss=loss)

def plot(class_num, y_test, ans_best):
    if class_num==2:
        classes = ['0', '1']
    elif class_num==3:
        classes = ['0', '1', '2']
    elif class_num==4:
        classes = ['0', '1', '2', '3']

    np.set_printoptions(precision=2)
    plot_confusion_matrix(y_test, ans_best, classes=classes,
                                        normalize=False,
                                        title=None,
                                        cmap=plt.cm.Blues)
    plt.show()

if __name__ == "__main__":

    main(class_num=4, epochs=200, batch_size=32, optimizer="adam", loss="binary", method="xgboost", kind="eight", judge=True, nan=False)
