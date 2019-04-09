#!/usr/bin/env python
# coding: utf-8

# In[1]:


# check Python version
get_ipython().system('python -V')


# In[2]:


import pandas as pd # import library to read data into dataframe
pd.set_option('display.max_columns', None)
import numpy as np # import numpy library
import re # import library for regular expression


# In[3]:


recipes = pd.read_csv("https://ibm.box.com/shared/static/5wah9atr5o1akuuavl2z9tkjzdinr1lv.csv")

print("Data read into dataframe!") # takes about 30 seconds


# In[4]:


recipes.head()


# In[5]:


recipes.shape


# In[ ]:




