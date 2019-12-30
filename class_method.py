import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
font = font_manager.FontProperties(fname="C:\\Windows\\Fonts\\AdobeFanHeitiStd-Bold.otf",size='20')
from kmodes.kmodes import KModes
import datetime
import warnings
warnings.filterwarnings("ignore")

def main (num, source_data):
    count=0
    all_count=[]
    month_count=[]
    all_data=[]
    data_list=[]
    data_num=[]
    # print(len(source_data["類型"]))
    for i in range(len(source_data["類型"])):
        tmp=datetime.datetime.strptime(source_data["上映日期"][i],"%Y/%m/%d")
        all_data.append(source_data["類型"][i].split('、'))  
        all_count.append(i)
        if(tmp.month==num):
            data_list.append(source_data["類型"][i].split('、')) 
            count+=1  
            month_count.append(i)
    for i in range(len(data_list)):
        for j in (data_list[i]):
            data_num.append(j)   
#     from collections import Counter
#     print(Counter(data_num))
    labels=list(set(data_num)) 
    data_count=np.zeros(len(labels))
    for i in (data_num):
        for j in range(len(data_count)):
            if i in list(set(data_num))[j]:
                data_count[j]+=1
    return data_count, labels, all_data, count, data_list, month_count, all_count

# data_count, labels, all_data=main(10)

def kmode(source_data, all_data, data_list, month_count, all_count, clusters_num, init, n_init, verbose, all_option=False):
    center_list=[]
    if all_option==False:
        revenue = source_data.iloc[month_count[0]:month_count[-1]]        
        revenue.pop("上映日期")
        revenue.pop("出品")
        revenue.pop("google trend")
        revenue.pop("導演")
        revenue.pop("演員1")
        revenue.pop("演員2")
        revenue.pop("演員3")
        revenue.pop("演員4")
        revenue.pop("編劇")
        revenue.pop("喜歡")
        revenue.pop("不喜歡")
        revenue.pop("製片預算")
        revenue=revenue.nlargest(7, ["累計銷售金額"])    
        print(revenue.to_string(index=False))
        for i in (data_list):
            if len(i)!=5:
                a=5-len(i)        
                for j in range(a):            
                    i.append("no")  
        data_array=np.array(data_list)
    else:
        revenue = source_data.iloc[all_count[0]:all_count[-1]]
        revenue.pop("上映日期")
        revenue.pop("出品")
        revenue.pop("google trend")
        revenue.pop("導演")
        revenue.pop("演員1")
        revenue.pop("演員2")
        revenue.pop("演員3")
        revenue.pop("演員4")
        revenue.pop("編劇")
        revenue.pop("喜歡")
        revenue.pop("不喜歡")
        revenue.pop("製片預算")
        revenue=revenue.nlargest(7, ["累計銷售金額"])    
        print(revenue.to_string(index=False))
        for i in (all_data):
            if len(i)!=5:
                a=5-len(i)        
                for j in range(a):            
                    i.append("no")
        data_array=np.array(all_data)  

    km = KModes(n_clusters=clusters_num, init=init, n_init=5, verbose=1)
    clusters = km.fit_predict(data_array)
   
    """
     Print the cluster centroids
    """   
#     print(km.cluster_centroids_)
    for i in range(clusters_num):
        if km.cluster_centroids_[i][1]=="no":
            center_list.append(km.cluster_centroids_[i][0])
        else:
            center_list.append(km.cluster_centroids_[i][0]+km.cluster_centroids_[i][1])

    return center_list, clusters


# center_list, clusters=kmode(all_data, clusters_num=7, init='Huang', n_init=5, verbose=1)   


def pie(num, source_data, all_option=True):
    if all_option==False:
        for ii in range(num):
            if ii==0:
                pass
            else:            
                data_count, labels, all_data, count, data_list, month_count, all_count=main(ii,source_data=source_data)
                print("month_"+str(ii)+":")
                print("movie number:",count)
                print("class number:",data_count)
                center_list, clusters=kmode(source_data,all_data, data_list, month_count, all_count, clusters_num=7, init='Huang', n_init=5, verbose=1, all_option=False)   
                print("cluster center:",center_list)
                print("class:",clusters)
                check_labels=np.zeros(len(center_list))
                for i in range(len(center_list)):
                    check_labels[i]=i
                print(check_labels)  

    #             from collections import Counter
    #             print(Counter(clusters))

                plot_data=np.zeros(len(center_list))
                for i in clusters:
                    for j in range(len(check_labels)):
    #                     print(check_labels[j])
                        if i == check_labels[j]:
                            plot_data[i]+=1           
                print(plot_data) 

                plt.figure(figsize=(12,18))    # 顯示圖框架大小
#                 labels = check_labels    # 製作圓餅圖的類別標籤
                labels = center_list
                # separeted = (0, 0, 0.3, 0, 0.3)                  # 依據類別數量，分別設定要突出的區塊
                # size = accident["count"]                         # 製作圓餅圖的數值來源

                patches,l_text,p_text =plt.pie(plot_data,                           # 數值
                        labels = labels,                # 標籤
                        autopct = "%1.1f%%",            # 將數值百分比並留到小數點一位
                #         explode = separeted,            # 設定分隔的區塊位置
                        pctdistance = 0.6,              # 數字距圓心的距離
                        textprops = {"fontsize" : 24},  # 文字大小
                        shadow=True)                    # 設定陰影   
                
                
                for t in l_text: 
                    t.set_fontproperties(font)
                month=["0","January","February","March","April","May","June","July","August","September","October"]
                plt.axis('equal')                                          # 使圓餅圖比例相等
                plt.title("Film Genre Statistics of "+ month[ii], {"fontsize" : 24})  # 設定標題及其文字大小
                plt.legend(loc = "best", prop=font)                                   # 設定圖例及其位置為最佳
#                 plt.savefig("Film Genre Statistics of "+ month[ii]+".jpg",   # 儲存圖檔
#                             bbox_inches='tight',               # 去除座標軸占用的空間
#                             pad_inches=0.0)                    # 去除所有白邊
#                 plt.close()      # 關閉圖表
    else:
        data_count, labels, all_data, count, data_list, month_count, all_count=main(0,source_data=source_data)
        print("All:")
#         print("movie number:",count)
#         print("class number:",data_count)        

        center_list, clusters=kmode(source_data,all_data, data_list, month_count, all_count, clusters_num=7, init='Huang', n_init=5, verbose=1, all_option=True)   
        print("cluster center:",center_list)
        print("class:",clusters)
        print(len(clusters))
        check_labels=np.zeros(len(center_list))
        for i in range(len(center_list)):
            check_labels[i]=i
        print(check_labels) 
        dataframe=pd.DataFrame(clusters, columns=['新類別'])
#         print(dataframe)
        class_data=pd.concat([source_data,dataframe],axis=1, ignore_index=False)
#         print(class_data)
        plot_data=np.zeros(len(center_list))
        for i in clusters:
            for j in range(len(check_labels)):
    #               print(check_labels[j])
                if i == check_labels[j]:
                    plot_data[i]+=1           
#         print(plot_data) 

        plt.figure(figsize=(12,18))    # 顯示圖框架大小
#         labels = check_labels    # 製作圓餅圖的類別標籤
        labels = center_list
                # separeted = (0, 0, 0.3, 0, 0.3)                  # 依據類別數量，分別設定要突出的區塊
                # size = accident["count"]                         # 製作圓餅圖的數值來源

        patches,l_text,p_text =plt.pie(plot_data,                           # 數值
                labels = labels,                # 標籤
                autopct = "%1.1f%%",            # 將數值百分比並留到小數點一位
                #         explode = separeted,            # 設定分隔的區塊位置
                pctdistance = 0.6,              # 數字距圓心的距離
                textprops = {"fontsize" : 24},  # 文字大小
                shadow=True
               )                    # 設定陰影

        for t in l_text: 
            t.set_fontproperties(font)
        plt.axis('equal')                                          # 使圓餅圖比例相等
        plt.title("Annual Film Genre Statistics", {"fontsize" : 24})  # 設定標題及其文字大小        
        plt.legend(loc = "best",prop=font)                                   # 設定圖例及其位置為最佳
        plt.show()
#         plt.savefig("Annual Film Genre Statistics.jpg",   # 儲存圖檔
#                             bbox_inches='tight',               # 去除座標軸占用的空間
#                             pad_inches=0.0)                    # 去除所有白邊
#         plt.close()      # 關閉圖表
        return class_data

# class_data=pie(num=11, all_option=True)