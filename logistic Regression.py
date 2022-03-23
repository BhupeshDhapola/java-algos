#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics 
import seaborn as sn 
import matplotlib.pyplot as plt 


# In[6]:


c = {'A': [780,750,690,710,680,730,690,720,740,690,610,690,710,680,770,610,580,650,540,590,
           620,600,550,550,570,670,660,580,650,660,640,620,660,660,680,650,670,580,590,690],
    'B': [4,3.9,3.3,3.7,3.9,3.7,2.3,3.3,3.3,1.7,2.7,3.7,3.7,3.3,3.3,3,2.7,3.7,2.7,2.3,3.3,2,2.3,
          2.7,3,3.3,3.7,2.3,3.7,3.3,3,2.7,4,3.3,3.3,2.3,2.7,3.3,1.7,3.7],
    'C': [3,4,3,5,4,6,1,4,5,1,3,5,6,4,3,1,4,6,2,3,2,1,4,1,2,6,4,2,6,5,1,2,4,6,5,1,2,1,4,5],
    'D': [1,1,0,1,0,1,0,1,1,0,0,1,1,0,1,0,0,1,0,0,1,0,0,0,0,1,1,0,1,1,0,0,1,1,1,0,0,0,0,1]
              }
df = pd.DataFrame(c,columns= ['A', 'B', 'C','D'])


# In[13]:


#PRINT (df)
X=df[['A','B','C']]
y=df['D']


# In[14]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)


# In[15]:


logistic_regression=LogisticRegression()
logistic_regression.fit(X_train,y_train)
y_pred=logistic_regression.predict(X_test)


# In[16]:


confusion_matrix=pd.crosstab(y_test,y_pred,rownames=['Actual'],colnames=['Predict'])
sn.heatmap(confusion_matrix,annot=True)


# In[17]:


print('Accuracy:',metrics.accuracy_score(y_test,y_pred))
plt.show()


# In[ ]:




