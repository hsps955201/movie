import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def combine(source_data, twitter_source, google_source, ig_source):
    twitter=twitter_source  #"C:\\Users\\hsps9\\Desktop\\twitter.csv"
    twitter_data=pd.read_csv(twitter)
    twitter_data.reset_index(drop=True)

    new_google=google_source #"C:\\Users\\hsps9\\Desktop\\final_google_using.csv"
    new_google_data=pd.read_csv(new_google)
    new_google_data.reset_index(drop=True)

    new_fans=ig_source #"C:\\Users\\hsps9\\Desktop\\new_fans.csv"
    new_fans_data=pd.read_csv(new_fans)
    new_fans_data.reset_index(drop=True)

    final_data=pd.concat([source_data, new_fans_data, twitter_data, new_google_data],axis=1, ignore_index=False)


    trans_list=["預告觀看次數","導演1搜尋","導演2搜尋","導演3搜尋","演員1搜尋","演員2搜尋","演員3搜尋","演員4搜尋",
           "編劇1搜尋","編劇2搜尋","編劇3搜尋","編劇4搜尋"]
    for i in trans_list:
        final_data[i]=final_data[i].astype(str).str.replace(",","")
        final_data[i]=pd.to_numeric(final_data[i], errors='coerce')
    final_data.reset_index(drop=True)

    return final_data