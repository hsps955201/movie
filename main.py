from method import random_forest, decision_tree, xgboost
from data_preprocessing import data_preprocess
from confusion import plot_confusion_matrix
from class_method import pie
from combine_data import combine
from nn import nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


def ml(class_num, epochs, method, source_data, twitter_source, google_source, ig_source, judge=True, nan=True):
    if judge==True:
        tmp_data=pie(num=11, source_data=source_data, all_option=True)
        final_data=combine(tmp_data, twitter_source, google_source, ig_source)
        input_x, revised_y=data_preprocess(final_data, nan=nan)
        a=pd.to_datetime(final_data['上映日期'])
        cut=a.dt.weekofyear
        test_list=[]
        train_list=[]
        test=0
        train=0
        # print(cut)
        for i in range(len(final_data)):    
            if cut[i]%4==0:  
                test+=1
                test_list.append(i)
            else:
                train+=1
                train_list.append(i)
        # print(train_list)
        # print(test_list)
        print(train)
        print(test)
        # print("final",final_data.shape)
        train_final_data=final_data
        test_final_data=final_data
        for i in test_list:
            train_final_data=train_final_data.drop(final_data.index[i])
        # print(len(train_final_data))
        train_final_data=train_final_data.reset_index(drop=True)
        # print(train_new_youtube_file_v3_data)
        for i in train_list:
            test_final_data=test_final_data.drop(final_data.index[i])
        # print(len(test_new_youtube_file_v3_data))
        test_final_data=test_final_data.reset_index(drop=True)
        # print(test_new_youtube_file_v3_data)

        X_train, y_train=data_preprocess(train_final_data, nan=nan)
        X_test, y_test=data_preprocess(test_final_data, nan=nan)
      
        if method=="random_forest":
            y_test, ans_best=random_forest(input_x, revised_y, X_train, y_train, X_test, y_test, judge=judge)
        elif method=="decision_tree":
            y_test, ans_best=decision_tree(input_x, revised_y, X_train, y_train, X_test, y_test, judge=judge)
        else:                            
            y_test, ans_best=xgboost(input_x, revised_y, X_train, y_train, X_test, y_test, class_num=class_num,
            num=epochs, judge=judge)
        
        if class_num==2:
            classes = ['0', '1']
        elif class_num==4:
            classes = ['0', '1', '2', '3']

        np.set_printoptions(precision=2)
        plot_confusion_matrix(y_test, ans_best, classes=classes,
                                normalize=False,
                                title=None,
                                cmap=plt.cm.Blues)
        plt.show()

    else:
        tmp_data=pie(num=11, source_data=source_data, all_option=True)
        final_data=combine(tmp_data, twitter_source, google_source, ig_source)
        input_x, revised_y=data_preprocess(final_data, nan=nan) 
        if method=="random_forest":
            y_test, ans_best=random_forest(input_x, revised_y, X_train=0, y_train=0, X_test=0, y_test=0, judge=judge)
        elif method=="decision_tree":
            y_test, ans_best=decision_tree(input_x, revised_y, X_train=0, y_train=0, X_test=0, y_test=0, judge=judge) 
        else:
            y_test, ans_best=xgboost(input_x, revised_y, X_train=0, y_train=0, X_test=0, y_test=0, class_num=class_num,
            num=epochs, judge=judge)
        
        if class_num==2:
            classes = ['0', '1']
        elif class_num==4:
            classes = ['0', '1', '2', '3']

        np.set_printoptions(precision=2)
        plot_confusion_matrix(y_test, ans_best, classes=classes,
                                normalize=False,
                                title=None,
                                cmap=plt.cm.Blues)
        plt.show()

def dl(source_data, twitter_source, google_source, ig_source, class_num, epochs, batch_size, optimizer, loss,
       judge=True, nan=True):
    if judge == True:
        tmp_data=pie(num=11, source_data=source_data, all_option=True)
        final_data=combine(tmp_data, twitter_source, google_source, ig_source)
        a=pd.to_datetime(final_data['上映日期'])
        cut=a.dt.weekofyear
        test_list=[]
        train_list=[]
        test=0
        train=0
        # print(cut)
        for i in range(len(final_data)):    
            if cut[i]%4==0:  
                test+=1
                test_list.append(i)

            else:
                train+=1
                train_list.append(i)       
                
        # print(train_list)
        # print(test_list)
        print(train)
        print(test)

        train_final_data=final_data
        test_final_data=final_data
        for i in test_list:
            train_final_data=train_final_data.drop(final_data.index[i])
        # print(len(train_final_data))
        train_final_data=train_final_data.reset_index(drop=True)
        # print(train_final_data)

        for i in train_list:
            test_final_data=test_final_data.drop(final_data.index[i])
        # print(len(test_final_data))
        test_final_data=test_final_data.reset_index(drop=True)
        # print(test_new_final_data)

        X_train, y_train=data_preprocess(train_final_data, nan=nan)
        X_test, y_test=data_preprocess(test_final_data, nan=nan) 
        # print(X_train.shape)       
        ohe = OneHotEncoder()
        y_train = ohe.fit_transform(y_train.reshape(-1, 1)).toarray()
        y_test = ohe.fit_transform(y_test.reshape(-1, 1)).toarray()    
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        nn(X_train, X_test, y_train, y_test, class_num=class_num, input_dim=X_train.shape[1], 
        epochs=epochs, batch_size=batch_size, optimizer=optimizer, loss=loss)
    else:
        tmp_data=pie(num=11, source_data=source_data, all_option=True)
        final_data=combine(tmp_data, twitter_source, google_source, ig_source)
        input_x, revised_y=data_preprocess(final_data, nan=nan)
        ohe = OneHotEncoder()
        revised_y = ohe.fit_transform(revised_y.reshape(-1, 1)).toarray()
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
        
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        nn(X_train, X_test, y_train, y_test, class_num=class_num, input_dim=input_x.shape[1], 
        epochs=epochs, batch_size=batch_size, optimizer=optimizer, loss=loss)

    

if __name__ == "__main__":
    source="./original/main.csv"
    source_data=pd.read_csv(source)
    source_data.reset_index(drop=True)
    source_data.pop("Unnamed: 13")
    twitter_source="./original/twitter.csv"
    google_source="./original/google_search.csv"
    ig_source="./original/ig.csv"
    
    
    """
    judge=True:表示用每個月分的前三or四週來預測下一週的收入區段
    """

    """
    nan==True:將空值視為nan
    """

    """        
    ml:"random_forest", "decision_tree", "xgboost"
    """
    ml(class_num=2, epochs=200, method="xgboost", source_data=source_data, twitter_source=twitter_source, 
       google_source=google_source, ig_source=ig_source, judge=False, nan=False)

    """
    dl:NN
    optimizer: adam, sgd
    loss: binary, categorical
    """
    # dl(source_data=source_data, twitter_source=twitter_source, google_source=google_source, 
    # ig_source=ig_source, class_num=2, epochs=500, batch_size=16, optimizer="adam", loss="binary",
    # judge=False, nan=False)