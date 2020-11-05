
# coding: utf-8

# In[1]:

import pandas as pd
import matplotlib as plt
import numpy as np
import os,sys
#import sox
import sys,time


# In[2]:

dp=[f for f in os.listdir('data/e/') if f.startswith('cv-') and f.endswith('csv')]
fn='data/e/'
data=[pd.read_csv(fn+i) for i in dp]
data[0].head(1)


# In[3]:

df = data[0].iloc[0:0]
for i in data:
    df=pd.concat([df,i])
df=df.reset_index(drop=True)


# In[4]:

def printnnl(string):
    import sys
    sys.stdout.write('\r%s'%(string))
    sys.stdout.flush()


# In[ ]:

total=len(df)
start_time = time.time()
t=1

import os,datetime
for k in range(len(df)):
    end_time = time.time()
    uptime =end_time - start_time
    el=(datetime.timedelta(seconds=int(uptime)))
    pc=(total/(t+1)*uptime)-uptime
    eta=(datetime.timedelta(seconds=int(pc)))
    if t%2000==0 or t==10 or t==100 or t==750 or t==250 or t==total:
        printnnl("{:%d, %b %Y %H:%M:%S}>> ".format(datetime.datetime.today())+'processing {} of {} files .. {:.2f}% complete. time elapsed {} eta {}'.format(t,total,t/total*100,str(el),str(eta)))
        #df.to_csv('data/e/cvd.csv',encoding='latin_1')
    t+=1
    wpath=df['wav_filename'][k]#.replace("/wave","").replace(".wav",".mp3")
    mpath=df['scatterc'][k]#.replace("/wave","").replace(".wav",".mp3")
    if df['wav_exists'][k]==1 or df['wav_exists'][k]==0:
        continue
    if os.path.exists(wpath):
        df['wav_exists'][k]=1
    else:
        print("\n{:%d, %b %Y %H:%M:%S}>> ".format(datetime.datetime.today())+'could not find',wpath)
        df['wav_exists'][k]=0
    if os.path.exists(mpath):
        df['mfc_exists'][k]=1
        df['mfc_size'][k]=os.path.getsize(mpath)
    else:
        df['mfc_exists'][k]=0
        print("\n{:%d, %b %Y %H:%M:%S}>> ".format(datetime.datetime.today())+'could not find',mpath)
df.head(1)
df.to_csv('data/e/cvd.csv',encoding='latin_1')


# In[4]:

os.path.exists("P:/Student_Save/Backup/s/cv_corpus_v1/cv-valid-test/wave/sample-000000.wav")


# In[ ]:



