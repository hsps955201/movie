import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")

def data_preprocess(source_data, nan=True):
    # print(len(source_data))
    # name=source_data["中文片名"].index.values
    # print(name)
    name_index=source_data.index.values
    # print("index",name_index)
    box=source_data["累計銷售金額"]
    watch=source_data["預告觀看次數"]
    like=source_data["喜歡"]
    dislike=source_data["不喜歡"]
    google_trend=source_data["google trend"]
    variable=source_data["新類別"]
    month=pd.to_datetime(source_data["上映日期"]).dt.month

    ig_director=source_data["導演1分數"]
    ig_actor_one=source_data["演員1分數"]
    ig_actor_two=source_data["演員2分數"]
    ig_writer=source_data["編劇1分數"]

    twitter_director=source_data["導演1_fans"]
    twitter_actor_one=source_data["演員1_fans"]
    twitter_actor_two=source_data["演員2_fans"]
    twitter_writer=source_data["編劇1_fans"]

    google_director=source_data["導演1搜尋"]
    google_actor_one=source_data["演員1搜尋"]
    google_actor_two=source_data["演員2搜尋"]
    google_writer=source_data["編劇1搜尋"]



    revised_y=np.zeros(len(box))
    for i in range(len(source_data)):
        if source_data["累計銷售金額"][i] < source_data["累計銷售金額"].describe()['25%']:        
            revised_y[i]=0
        elif source_data["累計銷售金額"].describe()['25%'] < source_data["累計銷售金額"][i] < source_data["累計銷售金額"].describe()['50%']:        
            revised_y[i]=0
        elif source_data["累計銷售金額"].describe()['50%'] < source_data["累計銷售金額"][i] < source_data["累計銷售金額"].describe()['75%']:
            revised_y[i]=1
        elif source_data["累計銷售金額"][i] > source_data["累計銷售金額"].describe()['75%']:
            revised_y[i]=1
    # print(revised_y)

    """
    google search

    """
    google_director_data=np.zeros(len(google_director))
    google_actor_one_data=np.zeros(len(google_actor_one))
    google_actor_two_data=np.zeros(len(google_actor_two))
    google_writer_data=np.zeros(len(google_writer))

    for i in range(len(google_director_data)):
        if np.isnan(google_director[i])==True :
            if nan==True:
                google_director_data[i]=np.nan
            else:
                google_director_data[i]=0
        else:
            google_director_data[i]=google_director[i]


    for i in range(len(google_director_data)):
        if np.isnan(google_actor_one[i])==True :
            if nan==True:
                google_actor_one_data[i]=np.nan
            else:
                google_actor_one_data[i]=0
        else:
            google_actor_one_data[i]=google_actor_one[i]

    for i in range(len(google_director_data)):
        if np.isnan(google_actor_two[i])==True :
            if nan==True:
                google_actor_two_data[i]=np.nan
            else:
                google_actor_two_data[i]=0
        else:
            google_actor_two_data[i]=google_actor_two[i]

    for i in range(len(google_director_data)):
        if np.isnan(google_writer[i])==True :
            if nan==True:
                google_writer_data[i]=np.nan
            else:
                google_writer_data[i]=0
        else:
            google_writer_data[i]=google_writer[i]


    google_director_data=preprocessing.scale(google_director_data)
    google_actor_one_data=preprocessing.scale(google_actor_one_data)
    google_actor_two_data=preprocessing.scale(google_actor_two_data)
    google_writer_data=preprocessing.scale(google_writer_data)

    """
    twitter

    """
    twitter_director_data=np.zeros(len(twitter_director))
    twitter_actor_one_data=np.zeros(len(twitter_actor_one))
    twitter_actor_two_data=np.zeros(len(twitter_actor_two))
    twitter_writer_data=np.zeros(len(twitter_writer))

    for i in range(len(twitter_director_data)):
        if np.isnan(twitter_director[i])==True :
            if nan==True:
                twitter_director_data[i]=np.nan
            else:
                twitter_director_data[i]=0
        else:
            twitter_director_data[i]=twitter_director[i]


    for i in range(len(twitter_director_data)):
        if np.isnan(twitter_actor_one[i])==True :
            if nan==True:
                twitter_actor_one_data[i]=np.nan
            else:
                twitter_actor_one_data[i]=0
        else:
            twitter_actor_one_data[i]=twitter_actor_one[i]

    for i in range(len(twitter_director_data)):
        if np.isnan(twitter_actor_two[i])==True :
            if nan==True:
                twitter_actor_two_data[i]=np.nan
            else:
                twitter_actor_two_data[i]=0
        else:
            twitter_actor_two_data[i]=twitter_actor_two[i]

    for i in range(len(twitter_director_data)):
        if np.isnan(twitter_writer[i])==True :
            if nan==True:
                twitter_writer_data[i]=np.nan
            else:
                twitter_writer_data[i]=0
        else:
            twitter_writer_data[i]=twitter_writer[i]


    twitter_director_data=preprocessing.scale(twitter_director_data)
    twitter_actor_one_data=preprocessing.scale(twitter_actor_one_data)
    twitter_actor_two_data=preprocessing.scale(twitter_actor_two_data)
    twitter_writer_data=preprocessing.scale(twitter_writer_data)  


    """
    ig
    """
    ig_director_data=np.zeros(len(ig_director))
    for i in range(len(ig_director_data)):
        if ig_director[i]==-1 :
            if nan==True:
                ig_director_data[i]=np.nan
            else:
                ig_director_data[i]=0
        else:        
            ig_director_data[i]=ig_director[i]
    ig_director_data=preprocessing.scale(ig_director_data) 

  
    ig_actor_one_data=np.zeros(len(ig_actor_one))
    ig_actor_two_data=np.zeros(len(ig_actor_two))
    for i in range(len(ig_actor_one_data)):
        if ig_actor_one[i]==-1 :
            if nan==True:
                ig_actor_one_data[i]=np.nan
            else:
                ig_actor_one_data[i]=0
        if ig_actor_two[i]==-1 :
            if nan==True:
                ig_actor_two_data[i]=np.nan
            else:
                ig_actor_two_data[i]=0
        else:
            ig_actor_one_data[i]=ig_actor_one[i]
            ig_actor_two_data[i]=ig_actor_two[i]
    ig_actor_one_data=preprocessing.scale(ig_actor_one_data)
    ig_actor_two_data=preprocessing.scale(ig_actor_two_data)       
    
    ig_writer_data=np.zeros(len(ig_writer))
    for i in range(len(ig_writer_data)):
        if ig_writer[i]==-1 :
            if nan==True:
                ig_writer_data[i]=np.nan
            else:
                ig_writer_data[i]=0
        else:    
            ig_writer_data[i]=ig_writer[i]
    ig_writer_data=preprocessing.scale(ig_writer_data)        
    
    """
    watch preprocess
    """
    watch_data=np.zeros(len(watch))
    for i in range(len(watch_data)):
        watch_data[i]=watch[i]
    watch_data=preprocessing.scale(watch_data)   
    # watch_max=np.nanmax(watch_data)
    # watch_min=np.nanmin(watch_data)
    # for i in range(len(watch_data)):
    #     if np.isnan(watch_data[i])==True :
    #         watch_data[i]=np.nan
    #     else:
    #         watch_data[i]=(watch_data[i]-watch_min)/(watch_max-watch_min)

    # print(watch_data)


    """
    google trend preprocess
    """
    google_data=np.zeros(len(google_trend))
    for i in range(len(google_data)):
        google_data[i]=google_trend[i]
    # print(google_data)
    google_data=preprocessing.scale(google_data)


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
    # like_ratio_max=np.nanmax(like_ratio)
    # like_ratio_min=np.nanmin(like_ratio)
    # for i in range(len(like_ratio)):
    #     if np.isnan(like_ratio[i])==True :
    #         like_ratio[i]=np.nan
    #     else:
    #         like_ratio[i]=(like_ratio[i]-like_ratio_min)/(like_ratio_max-like_ratio_min)
    # print(like_ratio)

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
    # like_audience_ratio_max=np.nanmax(like_audience_ratio)
    # like_audience_ratio_min=np.nanmin(like_audience_ratio)
    # for i in range(len(like_audience_ratio)):
    #     if np.isnan(like_audience_ratio[i])==True :
    #         like_audience_ratio[i]=np.nan
    #     else:
    #         like_audience_ratio[i]=(like_audience_ratio[i]-like_audience_ratio_min)/(like_audience_ratio_max-like_audience_ratio_min)
    # print(like_audience_ratio)

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
    # dislike_ratio_max=np.nanmax(dislike_ratio)
    # dislike_ratio_min=np.nanmin(dislike_ratio)
    # for i in range(len(dislike_ratio)):
    #     if np.isnan(dislike_ratio[i])==True :
    #         dislike_ratio[i]=np.nan
    #     else:
    #         dislike_ratio[i]=(dislike_ratio[i]-dislike_ratio_min)/(dislike_ratio_max-dislike_ratio_min)
    # print(dislike_ratio)


    """
    y:box
    x:watch, like_ratio, google_trend
    """

    # from sklearn.preprocessing import PolynomialFeatures
    # poly = PolynomialFeatures(3)
    # # poly.fit_transform(X)

    # # poly = PolynomialFeatures(interaction_only=True)
    # new_data=poly.fit_transform(writer_data, director_data, actor_one_data, actor_two_data)
    # print(new_data)

    # input_x=np.stack((watch_data, like_ratio, dislike_ratio, like_audience_ratio, dislike_audience_ratio,
    #                   variable, like_inverse, dislike_inverse, month,
    #                   google_director_data, google_actor_one_data, google_actor_two_data, google_writer_data                                   

    #                  ),axis=1)
    
    print(watch_data.shape)
    print(name_index.shape)
    input_x=np.stack((
                      watch_data, 
                      like, dislike,
                      variable,
                      google_data,
                      like_inverse, dislike_inverse, 
                      like_audience_ratio, dislike_audience_ratio,
                      ig_actor_one_data,
                      twitter_actor_one_data,
                      google_actor_one_data,
                      name_index          # for check   

                     ),axis=1)
    # print(name_index)
    
    # print(input_x)
    # print(input_x.shape)
    return input_x, revised_y

