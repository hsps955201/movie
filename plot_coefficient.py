import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd

def plot(final_data, method):    
    corr_matrix = final_data.corr(method) 
    corr_matrix["revised_y"].sort_values(ascending=True)
    values_list=[]
    index_list=[]
    values_list=corr_matrix["revised_y"].sort_values(ascending=True).values
    index_list=corr_matrix["revised_y"].sort_values(ascending=True).index
    print(corr_matrix["revised_y"].sort_values(ascending=True))
    # print(values_list)
    font = font_manager.FontProperties(fname="C:\\Windows\\Fonts\\AdobeFanHeitiStd-Bold.otf",size='20')
    plt.figure(figsize=(15,9))
    plt.xticks(fontsize=20,rotation=45)
    plt.yticks(fontsize=20)
    plt.ylim(-1,1)

    plt.xlabel("Different parameter",fontsize=20, fontproperties=font)
    plt.ylabel("Coefficient",fontsize=20)
    plt.title(method,fontsize=20)

    for i in range(len(index_list)):    
        plt.bar(index_list[i], values_list[i],color='black')
    plt.show()
    # plt.bar(x2,c_y2,color='r')
    # plt.bar(x3,c_y3,color='orange')
    # plt.bar(x4,c_y4,color='limegreen')
    # plt.bar(x5,c_y5,color='dodgerblue')

if __name__ == "__main__":
    final_source="./analysis.csv"
    final_data=pd.read_csv(final_source)
    final_data.reset_index(drop=True)
    method="kendall"
    """
    'pearson', 'kendall', 'spearman'
    """
    plot(final_data, method)