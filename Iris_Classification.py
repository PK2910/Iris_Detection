#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("iris.csv")
df


# In[2]:


df.info()


# In[3]:


df.describe()


# In[6]:


df.groupby('Species').count()


# In[7]:


sns.pairplot(df)


# In[8]:


sns.heatmap(df.corr())


# In[34]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
x=df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species']]
for col in x.columns:
    df[col]=le.fit_transform(df[col])
df


# In[53]:


x=df.drop('Species',axis=1)
y=df['Species']


# In[54]:


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
x=ss.fit_transform(x)


# In[55]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=42)
x_train.shape


# In[44]:


from sklearn.linear_model import LogisticRegression
le=LogisticRegression(solver='liblinear')
model=le.fit(x_train,y_train)
print(model.coef_)
print(model.intercept_)


# In[57]:


y_pred=model.predict(x_test)
print(y_pred)


# In[58]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report


# In[72]:


print(accuracy_score(y_test,y_pred))


# In[80]:


f1_score(y_test, y_pred, average='micro')


# In[82]:


precision_score(y_test,y_pred,average='micro')


# In[81]:


recall_score(y_test,y_pred,average='micro')


# In[76]:


print(classification_report(y_pred,y_test))


# In[ ]:




