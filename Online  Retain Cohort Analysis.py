#!/usr/bin/env python
# coding: utf-8

# # Cohort Analysis With Python
# In this article, I am going to talk about Cohort Analysis and how to analyze it with Python. It widely used for mobile applications/games. Let's say we created a mobile game and published it. How do we know the game will be popular or die. It depends on the relationship with the users.  If the entrepreneurs analyze relations with the users, fix bugs and errors and take the necessary steps to organize their relations, they will at least increase their chance of survival.
# 
# But, how to understand these relations?
# 
# The answer is Cohort Analysis.
# 
# # What is Cohort Analysis?
# Literally, a cohort is a group who shared similar behaviours within a specified period. A group of people born in Turkey in 2022 is an example for cohort related to the number of births in a country. In terms of bussiness problems, cohort represents a group of customers or users. And a cohort analysis is when you try to derive insights from the behaviour of this group. Cohort analysis makes it easy to analyze the user behaviour and trends without having to look at the behaviour of each user individually. It can also be used to estimate customer lifetime value using this output
# 
# # Why Cohort Analysis?
# The most valuable feature of cohort analysis is that it helps companies answer some of the targeted questions by examining the relevant data. Cohort Analysis helps to understand how the behaviour of users can affect the business in terms of acquisition and retention and to analyze the customer churn rate.

# In[5]:


# import libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime as dt

# data visualization libraries
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from operator import attrgetter

# machine learning libraries
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

online_retail_df = pd.read_csv(r"C:\Users\jki\Downloads\Year 2009-2010.csv", encoding='latin1')
online_retail_df.head(5)


#  will directly delete the missing values and duplicates as they are not in the scope of our topic.

# In[8]:


online_retail_df.dropna(inplace=True)


# In[10]:


online_retail_df.describe()


# There some negative values in the Quantity and Price features. It can't be possible. So I will filter the data greater than zero.

# In[11]:


online_retail_df = online_retail_df[(online_retail_df['Quantity'] > 0) & (online_retail_df['Price'] > 0)]


# # Data Preparation
# For cohort analysis, we need three labels. These are payment period, cohort group and cohort period/index
# 

# To work with the time series, we need to convert the type of related feature. The format shuld be as in the dataset.

# In[12]:


online_retail_df['InvoiceDate'] = pd.to_datetime(online_retail_df['InvoiceDate'], format='%m/%d/%Y %H:%M')


# Now, we need to create the cohort and order_month variables. The first one indicates the monthly cohort based on the first purchase date and the second one is the truncated month of the purchase date.

# In[13]:


online_retail_df['order_month'] = online_retail_df['InvoiceDate'].dt.to_period('M')


# In[14]:


online_retail_df['cohort'] = online_retail_df.groupby('Customer ID')['InvoiceDate'].transform('min').dt.to_period('M')


# Then, we aggregate the data per cohort and order_month and count the number of unique customers in each group.

# In[16]:


online_retail_df_cohort = online_retail_df.groupby(['cohort', 'order_month']).agg(n_customers=('Customer ID', 'nunique')).reset_index(drop=False)


# In[17]:


online_retail_df_cohort['period_number'] = (online_retail_df_cohort.order_month - online_retail_df_cohort.cohort).apply(attrgetter('n'))


# In[18]:


online_retail_df_cohort.head()


# Then, we aggregate the data per cohort and order_month and count the number of unique customers in each group.

# In[19]:


cohort_pivot = online_retail_df_cohort.pivot_table(index='cohort', columns='period_number', values='n_customers')


# In[20]:


cohort_pivot


# Actually, cohort_pivot shows us what we want to see. But we need to convert the table to see more clearly.

# In[21]:


cohort_size = cohort_pivot.iloc[:, 0]


# In[22]:


retention_matrix = cohort_pivot.divide(cohort_size, axis=0)


# Lastly, we plot the retention matrix as a heatmap. Additionally, we wanted to include extra information regarding the cohort size. That is why we in fact created two heatmaps, where the one indicating the cohort size is using a white only colormap — no coloring at all.

# In[24]:


with sns.axes_style("white"):
    fig, ax = plt.subplots(1, 2, figsize=(12, 8), sharey=True, gridspec_kw={'width_ratios': [1, 11]})
    
    # retention matrix
    sns.heatmap(retention_matrix, 
                mask=retention_matrix.isnull(), 
                annot=True, 
                fmt='.0%', 
                cmap='RdYlGn', 
                ax=ax[1])
    ax[1].set_title('Monthly Cohorts: User Retention', fontsize=16)
    ax[1].set(xlabel='# of periods',
              ylabel='')

    # cohort size
    cohort_size_df = pd.DataFrame(cohort_size).rename(columns={0: 'cohort_size'})
    white_cmap = mcolors.ListedColormap(['white'])
    sns.heatmap(cohort_size_df, 
                annot=True, 
                cbar=False, 
                fmt='g', 
                cmap=white_cmap, 
                ax=ax[0])

    fig.tight_layout()


# In[ ]:




