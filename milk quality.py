#!/usr/bin/env python
# coding: utf-8

# # Import Libraries , Function and Data #

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


# In[2]:


milk = pd.read_csv("milknewdata.csv")


# In[3]:


milk.head()


# In[ ]:





# In[4]:


milk.tail()


# In[5]:


milk.shape


# # Data Cleaning and Visulization

# In[6]:


milk.info()


# In[7]:


milk.describe()


# In[8]:


milk.isnull().sum()


# In[9]:


sns.heatmap(milk.isnull())


# In[10]:


milk.corr()


# In[11]:


sns.heatmap(milk.corr(),annot=True,cmap="Greens")


# In[14]:


sns.displot(milk['Taste'])


# In[15]:


sns.distplot(milk['Odor'])


# In[16]:


sns.distplot(milk['Turbidity'])


# In[17]:


sns.distplot(milk['Colour'])


# In[18]:


milk_corr = milk.corr()
milk_corr.plot(figsize=(20,10))


# In[19]:


milk['Taste'].value_counts().plot(kind='pie',autopct='%.2f')


# In[20]:


milk['Grade'].value_counts().plot(kind='pie',autopct='%.2f')


# In[21]:


sns.pairplot(milk)


# In[22]:


milk


# # Preprocessing.LabelEncoder Convert String to Number #

# In[23]:


label_encoder = preprocessing.LabelEncoder()
milk['Grade']= label_encoder.fit_transform(milk['Grade'])
milk['Grade'].unique()


# In[24]:


milk


# In[25]:


milk__feature = ['pH','Temprature','Taste','Odor','Turbidity','Colour']

x = milk[milk__feature]

y = milk['Grade']


# In[26]:


x.head()


# In[27]:


y.head()


# In[28]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
x_train.head()


# In[29]:


y_train.head()


# In[30]:


x_test.head()


# In[31]:


y_test.head()


# # RandomForest Model #

# In[32]:


from sklearn.ensemble import RandomForestClassifier

model =RandomForestClassifier()


# In[33]:


model.fit(x_train,y_train)


# In[34]:


model.score(x_train,y_train)


# In[35]:


model.score(x_test,y_test)*100


# # KNeighbors Model #

# In[36]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()


# In[37]:


model.fit(x_train,y_train)


# In[38]:


model.score(x_train,y_train)*100


# In[39]:


model.score(x_test,y_test)*100


# # Thank You #

# In[ ]:




