#!/usr/bin/env python
# coding: utf-8

# In[ ]:





#                                                 **RISK OF LOAN DEFAULT**
# 
#                                                      Aman Panwar

# **Introduction**
# 
# Loan delinquency and loan default are becoming a growing problem for the banking and other financial institutions. Failure to recover loans on time puts these loan lending institutions at financial loss. This eventually makes it difficult for people who genuinely need loan but never had any banking history. While it is very difficult to accurately claim who will pay back the loan, it is possible to estimate the probability of person paying back the loan. To help answer this question, we obtained data posted by Home Credit on Kaggle. Home credit, a non-bank financial institution, focuses primarily on lending loans to people with little or no credit history.
# 
# *The main objective of this project is to prepare a dataset that has necessary variables to predict how likely a person would repay a loan. 
# 
# *The results from this study can also be used to identify the factors that are common to people who faced a difficulty in paying back the loan. 
# 
# *Correlation studies was also performed between some of the variables in final dataset to check if there existed any significant relation between those variables as this may help with selection of variables for machine learning model.
# 

# In[55]:


#![flowchart](https://github.com/panwaraman3031/Loan-Default/blob/master/flowchart.png)
#![](http://localhost:8888/tree/Desktop/GitHub/BI/flowchart.jpg)
#![](C:/Users/panwaraman/Desktop/GitHub/BI/flowchart.jpg)

from PIL import Image
flowchart = Image.open("flowchart.jpg");
flowchart.show();


# This flow chart explains the life cycle of our project from gathering data to visualizing the data. 

# In[53]:


#import libraries
import pandas as pd
import numpy as np
import os

#working directory
path= "C:/Users/panwaraman/Desktop/GitHub/BI"
#path= "https://drive.google.com/drive/folders/1f2m1fB5Y8bC2ddfIvjyXHWBdMs8KII5R"
os.chdir(path)


# **Short description of datasets used in this project:**
# 
# **application.csv**
# This is the main table that contains data for all loan applications.
# One row represents one loan in our data sample.
# 
# **bureau.csv**
# This contains all client's previous credits that were provided by other financial institutions to Credit Bureau. The information is for those clients who have a loan application in our sample.
# For every loan in our sample, there are as many rows as number of credits the client had in Credit Bureau before the application date.
# 
# **credit_card_balance.csv**
# Monthly balance snapshots of previous credit cards that the applicant has with Home Credit.
# 
# **previous_application.csv**
# All previous applications for Home Credit loans of clients who have loans in our sample.
# There is one row for each previous application related to loans in our data sample.

# In[10]:


#read datasets
df_application=pd.read_csv("application_train.csv")
df_bureau=pd.read_csv("bureau.csv")
df_credit_balance=pd.read_csv("credit_card_balance.csv")
df_previous=pd.read_csv("previous_application.csv")

#choose the columns that we have to work on
df= df_application[['SK_ID_CURR', 'TARGET', 'NAME_CONTRACT_TYPE','CODE_GENDER',
                    'FLAG_OWN_CAR', 'FLAG_OWN_REALTY','AMT_INCOME_TOTAL',
                    'AMT_CREDIT','AMT_GOODS_PRICE','NAME_INCOME_TYPE',
                    'NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS',
                    'REGION_POPULATION_RELATIVE','DAYS_BIRTH','DAYS_EMPLOYED',
                    'FLAG_CONT_MOBILE','OCCUPATION_TYPE','CNT_FAM_MEMBERS',
                    'REGION_RATING_CLIENT','ORGANIZATION_TYPE',
                    'AMT_REQ_CREDIT_BUREAU_YEAR']]
df.head()


# In[4]:


df1= df_bureau[['SK_ID_CURR','CREDIT_ACTIVE']]
df1.head()


# **Multilevel data**
# 
# There were some multilevel data in our datasets where the “SK_ID_CURR” had multiple records for each customer so we grouped all those values by “SK_ID_CURR” and summed them together. Categorical data were converted to numeric columns by creating a new column for each unique category.

# In[5]:


#divide'CREDIT_ACTIVE'categorical variable column into numeric columns
df1= pd.get_dummies(df1, columns=['CREDIT_ACTIVE'])
#rename the newly created numeric columns
df1= df1.rename(columns={x: x.split('_')[0]+'_1_'+
    x.split('_')[2] for x in df1.columns[1:]})
#groups the data by SK_ID_CURR
df1= df1.groupby('SK_ID_CURR',as_index = False).sum()

df1.head()


# In[6]:


#subsets the data to remove unecessary columns and groups the data by'SK_ID_CURR
df2= df_credit_balance[['SK_ID_CURR','AMT_BALANCE','AMT_CREDIT_LIMIT_ACTUAL']]
df2=df2.groupby(['SK_ID_CURR'],as_index = False)['AMT_BALANCE','AMT_CREDIT_LIMIT_ACTUAL'].sum()
df2.head()


# In[7]:


#subsets the data to remove unecessary columns
df3= df_previous[['SK_ID_CURR','NAME_CONTRACT_STATUS']]
#divide'NAME_CONTRACT_STATUS'categorical variable column into numeric columns
df3= pd.get_dummies(df3, columns=['NAME_CONTRACT_STATUS'])
#rename the newly created numeric columns
df3= df3.rename(columns={x: x.split('_')[3]+'_1_'+x.split('_')[1]+'_'+x.split('_')[2] for x in df3.columns[1:]})
#groups the data by SK_ID_CURR
df3= df3.groupby('SK_ID_CURR',as_index = False).sum()
df3.head()


# **Data Merging**
# 
# Data merging helps to combine all the distributed information through datasets to fulfill the requirements of the project.
# 
# Firstly, we merged all the datasets using a common column named “SK_ID_CURR” which is unique id for each loan application.

# In[9]:


#merges 'application' and 'bureau' data using SK_ID_CURR,a unique record id
#Left join is used because 'application' is our primary dataset and we would like to keep all the data of application dataset.
dft= df.merge(df1,on='SK_ID_CURR', how='left', indicator=True)
dft.head()


# **Data Cleaning**

# In[21]:


#check the missing values and computes all the missing values
nan_cols = [i for i in dft.columns if dft[i].isnull().any()]
dft.isnull().sum()


# In[11]:


#replace missing values with 'unknown'
dft['OCCUPATION_TYPE'] = dft['OCCUPATION_TYPE'].fillna('Unknown')
#replace missing values with mean
mean_value=dft['AMT_GOODS_PRICE'].mean()
dft['AMT_GOODS_PRICE']=dft['AMT_GOODS_PRICE'].fillna(mean_value)
mean_value=dft['CNT_FAM_MEMBERS'].mean()
dft['CNT_FAM_MEMBERS']=dft['CNT_FAM_MEMBERS'].fillna(mean_value)
#replace missing year with 0
dft['AMT_REQ_CREDIT_BUREAU_YEAR']=dft['AMT_REQ_CREDIT_BUREAU_YEAR'].fillna
dft.head()


# In[13]:


#removes all missing values that resulted from left merge and cannot be replaced
dft_70 = dft[(dft != 'left_only').all(axis=1)]
dft_70.head()


# **Data Visualization & Analysis**
# 
# Tableau was used to visualize categorical and continuous variables in our dataset.
# 

# In[ ]:


#![occupation](https://github.com/panwaraman3031/Loan-Default/blob/master/occupation.png)


# **Interpretation of the plot:**
# 
# “% of total target” represents proportion of people who had difficulty paying back loan. Proportion of total target by occupation type shows that laborers and “unknown” occupation type had most difficulties paying back loan. 
# “Unknown” occupation type refers to all the people who had missing values in their occupation type column. We believe that laborers are unskilled people who do manual work for wages and therefore have difficultes paying back the loan.

# In[ ]:


#![organization](https://github.com/panwaraman3031/Loan-Default/blob/master/organization.png)


# **Interpretation of the plot:**
# 
# Among all the people who had difficulty paying back the loan, people belonging to business organization type comprised a major portion because there is relatively more uncertainty involved with the outcome of a business and thus this category of people may find it hard to pay loan on time.

# In[ ]:


#![education](https://github.com/panwaraman3031/Loan-Default/blob/master/education.png)


# **Interpretation of the plot:**
# Proportion of people by education type shows that people with secondary education had most difficulty paying back the loan.

# In[ ]:


#![income](https://github.com/panwaraman3031/Loan-Default/blob/master/income.png)


# **Interpretation of the plot:**
# People with lower income were more likely to default the payment of loan as compared to higher income group.

# In[47]:


#computes correlation matrix
c = dft_70.corr().abs()
#converts to series
s = c.unstack()
#sorts correlation matrix descending order
so = s.sort_values(kind="quicksort",ascending=False)
#prints unique correlation values
so[15:25:2]


# In[ ]:


#![correlation](https://github.com/panwaraman3031/Loan-Default/blob/master/correlation.png)


# *Based on the correlation matrix, “Amt Credit” and “Amt Goods Price” are strongly correlated and one of these variable can be removed to make our dataset more compact.*

# **Conclusion:**
# 
# Based on our results, we would remove one variable out of pair of highly correlated variables to make our dataset more simpler. Additional screening can be performed for the loan applications with high risk factors. For example, people with laborers as education type and secondary school as education type may need an additional verification before approving the loan.
