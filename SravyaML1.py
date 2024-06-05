#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().system('pip install numpy pandas scikit-learn matplotlib seaborn')



# In[5]:


import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
columns = ['ID', 'Diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
data = pd.read_csv(url, header=None, names=columns)
data.head()


# In[6]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

X = data.drop(['ID', 'Diagnosis'], axis=1)
y = data['Diagnosis']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[7]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")


# In[ ]:




