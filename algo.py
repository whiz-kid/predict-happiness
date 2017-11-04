import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

train.loc[train["Browser_Used"]=="Google Chrome","Browser_Used"]=0
train.loc[train["Browser_Used"]=="Chrome","Browser_Used"]=0
train.loc[train["Browser_Used"]=="InternetExplorer","Browser_Used"]=1
train.loc[train["Browser_Used"]=="IE","Browser_Used"]=1
train.loc[train["Browser_Used"]=="Internet Explorer","Browser_Used"]=1
train.loc[train["Browser_Used"]=="Mozilla Firefox","Browser_Used"]=2
train.loc[train["Browser_Used"]=="Firefox","Browser_Used"]=2
train.loc[train["Browser_Used"]=="Mozilla","Browser_Used"]=2
train.loc[train["Browser_Used"]=="Edge","Browser_Used"]=3
train.loc[train["Browser_Used"]=="Safari","Browser_Used"]=4
train.loc[train["Browser_Used"]=="Opera","Browser_Used"]=5

train["Device_Used"]=train["Device_Used"].astype("category")
train["Device_Used"].cat.categories=[0,1,2]

train["Is_Response"]=train["Is_Response"].astype("category")
train["Is_Response"].cat.categories=[0,1]


description=train.loc[:,"Description"]
vec=TfidfVectorizer(stop_words='english')
print("Conveting document into tf-idf vector")
a=vec.fit_transform(description)

# Only description as feature
print("Saving the first array")
des=a.toarray()
np.save("train1.npy",des)

# Description, browser_used and deviced used as feature
print("Stacking cloumns")
brow=np.array(train.loc[:,"Browser_Used"])
dev=np.array(train.loc[:,"Device_Used"])
ans=np.column_stack((des,brow,dev))
print("saving second model")
np.save("train2.npy",ans)

# print(train.head(10))
print(ans.shape)

