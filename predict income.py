
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


train_df = pd.read_csv(r"C:\Users\Roshan Doss\Desktop\data sci\Sample Data.csv")

# In[3]:


print(train_df.head())

# In[4]:


print(train_df.shape)   

# In[5]:


train_df.info()


# In[6]:


train_df.isnull().sum()


# In[7]:


train_df.select_dtypes(include=["object"]).head()


# In[44]:


for value in['Region','City']:train_df[value].fillna(train_df[value].mode()[0],inplace = True)

# In[45]:


train_df.isnull().sum()

daily_sales_vol = train_df["Sale Date"].value_counts()
daily_sales_vol.plot()
plt.title('Sales in Aug 2018')
plt.ylabel('Number of sales')
plt.xlabel('Date')
plt.rcParams['figure.figsize'] = (20.0, 10.0)


# In[11]:


for label in list(train_df):
    if len(train_df[value].unique()) < 5:
        print(train_df[label].value_counts())
        print("\n")


# In[12]:


plt.figure(figsize = (40,20))
sns.set_context("paper", font_scale=1.0)

sns.heatmap(train_df.corr(), vmax=.8, square=True, annot=True, fmt='.2f')


# In[24]:


train_df = train_df.drop('ID',axis = 1.00)


# In[50]:


colname = ['Workclass','Education','Marital.Status','Occupation','Relationship','Race','Sex','Native.Country','Income.Group']
colname1 = ['Workclass','Education','Marital.Status','Occupation','Relationship','Race','Sex','Native.Country']


# In[47]:


from sklearn import preprocessing

le = {}
for x in colname:
    le[x]=preprocessing.LabelEncoder()
for x in colname:
    train_df[x]=le[x].fit_transform(train_df[x])


# In[51]:


from sklearn import preprocessing

le = {}
for x in colname1:
    le[x]=preprocessing.LabelEncoder()
for x in colname1:
    test_df[x]=le[x].fit_transform(test_df[x])


# In[52]:


test_df.isnull().sum()


# In[53]:


x = train_df.values[:,:-1]
y = train_df.values[:,-1]


# In[54]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3,random_state = 40)


# In[55]:


from sklearn.linear_model import LogisticRegression

classifier =(LogisticRegression())


# In[56]:



classifier=(LogisticRegression(random_state=0))

model =classifier.fit(x_train, y_train)

#Test the model
y_pred=classifier.predict(x_test)
print(list(zip(y_test, y_pred)))


# In[57]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cfm = confusion_matrix(y_test, y_pred)
print(cfm)

print("Classification Report:")
print(classification_report(y_test, y_pred))

acc=accuracy_score(y_test, y_pred)
print("Accuracy of the model:", acc)


# In[58]:


#predicting using the Random forest Classifier
from sklearn.ensemble import RandomForestClassifier

model_RandomForest=(RandomForestClassifier(100))
#fit the model on the data and predict the values
model_RandomForest=model_RandomForest.fit(x_train, y_train)

y_pred=model_RandomForest.predict(x_test)


# In[59]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cfm = confusion_matrix(y_test, y_pred)
print(cfm)

print("Classification Report:")
print(classification_report(y_test, y_pred))

acc=accuracy_score(y_test, y_pred)
print("Accuracy of the model:", acc)


# In[67]:


submission = pd.DataFrame()
submission['ID'] = test_df.ID


# In[68]:


feats = test_df.drop(['ID'], axis=1).interpolate()


# In[69]:


pred = classifier.predict(feats)


# In[70]:


submission['Income.Group'] = pred
#submission.head()
print(submission.head())


# In[72]:


submission["Income.Group"] = submission["Income.Group"].map({0: "<=50k", 1: ">50k"})


# In[74]:


submission.head()

