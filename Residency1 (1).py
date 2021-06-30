#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


import pickle


# In[4]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


# In[5]:


Home = pd.read_csv('ResidencyRenew.csv')
Home.head()


# In[6]:


Home.drop('Area live-in Satisfaction', axis=1, inplace=True)
Home.drop('Stress Management', axis=1, inplace=True)


# In[11]:


area = Home.Residence.unique()


# In[14]:


np.count_nonzero(area)


# In[ ]:


# In[57]:


#Home['Nearby Places'].value_counts()


# In[ ]:


# In[58]:


# Home['Residence'].value_counts()


# In[15]:


Home['Residence'].value_counts()


# In[16]:


Home.Residence.hist(bins=30, alpha=0.5)
plt.show()


# In[17]:


Home.isna().sum()


# In[18]:


Home.info()


# In[19]:


# Home.head()


# In[20]:


# Home.head()


# # Categorical to Numeric

# In[21]:


labelencoder = LabelEncoder()
Home['Nearby Places'] = labelencoder.fit_transform(Home['Nearby Places'])
Home['Area related Info'] = labelencoder.fit_transform(
    Home['Area related Info'])
Home['Nature'] = labelencoder.fit_transform(Home['Nature'])
Home['MentalPeace'] = labelencoder.fit_transform(Home['MentalPeace'])
Home['Reaction on lack of something'] = labelencoder.fit_transform(
    Home['Reaction on lack of something'])
Home['Free time activities'] = labelencoder.fit_transform(
    Home['Free time activities'])
Home['GoOut'] = labelencoder.fit_transform(Home['GoOut'])
#Home['Stress Management'] = labelencoder.fit_transform(Home['Stress Management'])
Home['Descrimination'] = labelencoder.fit_transform(Home['Descrimination'])
Home['Outing Preference'] = labelencoder.fit_transform(
    Home['Outing Preference'])
Home['Residence'] = labelencoder.fit_transform(Home['Residence'])


# In[22]:


Home['Residence'].value_counts()


# In[23]:


corrmat = Home.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20, 20))
g = sns.heatmap(Home[top_corr_features].corr(), annot=True, cmap='RdYlGn')


# In[24]:


X = Home.drop('Residence', axis=1).values
y = Home['Residence'].values


# In[25]:


# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)


# In[26]:


X_train[0]


# In[27]:


# try some regression
random_forest = RandomForestClassifier(n_estimators=10)
rf = random_forest.fit(X_train, y_train)

Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, y_train)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)


# In[28]:


acc_random_forest


# In[29]:


Y_prediction


# In[30]:


X_test[0]


# In[31]:


Y_prediction[0]


# In[32]:


pickle.dump(rf, open('iri.pkl', 'wb'))


# In[33]:


Y_prediction = random_forest.predict([[5, 1, 2, 2, 1, 0, 2, 1, 1, 0]])


# In[34]:


Y_prediction[0]


# %%

# %%
