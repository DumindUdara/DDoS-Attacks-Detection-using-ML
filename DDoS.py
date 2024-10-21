#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
# import csv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc, confusion_matrix

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle


# ### Data Loading

# In[3]:


df = pd.read_csv("C:\\Users\\Udara\\Downloads\\Data-DDoS\\New folder\\DDoS.csv")


# ### Data Preprocessing

# In[4]:


df.head(5)


# In[5]:


# Remove the space before the column names
df.columns = df.columns.str.strip()


# In[6]:


# Unique values in the lable targe column
df.loc[:,'Label'].unique()


# In[7]:


# Checking the null vaues in  the dataset
plt.figure(1,figsize=( 10,4))
plt.hist(df.isna().sum())

# Set the title and axis lable 
plt.xticks([0, 1], labels=['Not Null=0', 'Null=1'])

plt.title('Columns with Null Values')
plt.xlabel('Feature')
plt.ylabel('The number of feature')
plt.show()


# In[8]:


def plotMissingValues(dataframe):
    missing_values = dataframe.isnull().sum() # Counting null values for each column
    fig = plt.figure(figsize=(18, 5))
    missing_values.plot(kind='bar')
    plt.xlabel('Features')
    plt.ylabel('Missing values')
    plt.title('Total number of Missing Values in each feature')
    plt.show()
    
plotMissingValues(df)


# In[9]:


# Removeing the null values
data_f = df.dropna()


# In[10]:


# checking the null values in the dataset
plt.figure(1, figsize=(10,4))
plt.hist(data_f.isna().sum())

plt.title('Data after removing the Null Values')
plt.xlabel('The number of the null values')
plt.ylabel('Number of columns')
plt.show()


# In[11]:


pd.set_option('use_inf_as_na', True) #Treat inf as NaN
null_values = data_f.isnull().sum() # Check the NaN values


# In[12]:


# To know the data type of the columns
(data_f.dtypes=='object')


# In[13]:


# check label values
data_f.head(5)


# In[14]:


data_f.loc[:,'Label'].unique()


# ### Create Dummpy Varibale 

# In[15]:


# Convert the lables is the DataFrame to numerical values
data_f['Label'] = data_f['Label'].map({'BENIGN':0, 'DDoS':1})


# In[16]:


# Check dummpy variabels 
data_f.head(5)


# In[17]:


# Print DataFrame
plt.hist(data_f['Label'], bins=[0,0.3,0.7,1], edgecolor='black') # Specify bins as [0,1]
plt.xticks([0,1], label=['BENIGN=0', 'DDoS=1'])
plt.xlabel('Classes')
plt.ylabel('Count')
plt.show()


# ### Data Exploring

# In[18]:


df.describe()


# In[19]:


data_f.describe()


# ### Plot the distibution of the features

# In[20]:


# Create a histrogram for each feature
plt.figure(5)
for col in data_f.columns:
    plt.hist(data_f[col])
    plt.title(col)
    plt.show()


# ### Data splitting into train and test

# In[21]:


# Split data into featuers and targert variable
X = data_f.drop('Label', axis=1)
y = data_f['Label']

# Split the data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

print('The train dataset size : ',X_train.shape)
print('The test dataset size : ',X_test.shape)


# ## Training the models

# #### Randome Forest

# In[22]:


# Randome forest
rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)


# In[23]:


# Getting feaure imporances from the trained model
importance = rf_model.feature_importances_

# Getting the incluices of the features sorted by importance 
indices = sorted(range(len(importance)), key=lambda i: importance[i], reverse=False)
feature_names = [f"Features {i}" for i in indices] # Replace with your column names

# Ploting feature imporances horizantally
plt.figure(figsize=(8, 16))
plt.barh(range(X_train.shape[1]), importance[indices], align='center')
plt.yticks(range(X_train.shape[1]), feature_names)
plt.xlabel('Importance')
plt.ylabel('Feature Importances')
plt.show()


# In[24]:


from sklearn.tree import plot_tree

estimator = rf_model.estimators_[0] # Selecting the first estimator from the random forest model

plt.figure(figsize=(20,10))
plot_tree(estimator, filled=True, rounded=True)
plt.show()


# #### Model Evaluation

# In[25]:


# Funtion to generate and display a detailed confussion marix
def plot_confusion_matrix(y_train, y_pred, classes, title):
    cm = confusion_matrix(y_train, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


# In[26]:


# Evaluate Randome Forest 
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)
rf_precission = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)

print('\nRandom Forest Metrics:')
print(f'Accurancy : {rf_accuracy:.4f}')
print(f'F1 Score : {rf_f1:.4f}')
print(f'Precision : {rf_precission:.4f}')
print(f'Recall : {rf_recall:.4f}')


# In[27]:


# Confusion Matrix for Random Forest
plot_confusion_matrix(y_test, rf_pred, ['Benign', 'DDoS'], 'Random Forest Confusion Matrix')


# ### Logistic Regression

# In[28]:


lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)


# #### Evaluate Logistic Regression 

# In[29]:


lr_accuracy = accuracy_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred)
lr_precission = precision_score(y_test, lr_pred)
lr_recall = recall_score(y_test, lr_pred)

print('\nLogistic Regression:')
print(f'Accurancy : {lr_accuracy:.4f}')
print(f'F1 Score : {lr_f1:.4f}')
print(f'Precision : {lr_precission:.4f}')
print(f'Recall : {lr_recall:.4f}')


# In[32]:


# Confusion matrix for Logistic Regression
plot_confusion_matrix(y_test, lr_pred, ['Benign', 'DDoS'], 'Logistic Regression Confusion Matrix')


# ### Nural Network

# In[33]:


nn_model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=10, random_state=42)
nn_model.fit(X_train, y_train)
nn_pred = nn_model.predict(X_test)


# #### Evaluate Nural Network

# In[34]:


nn_accuracy = accuracy_score(y_test, nn_pred)
nn_f1 = f1_score(y_test, nn_pred)
nn_precission = precision_score(y_test, nn_pred)
nn_recall = recall_score(y_test, nn_pred)

print('\nNural Networ:')
print(f'Accurancy : {nn_accuracy:.4f}')
print(f'F1 Score : {nn_f1:.4f}')
print(f'Precision : {nn_precission:.4f}')
print(f'Recall : {nn_recall:.4f}')


# In[36]:


# Confusion Matrix for Nural Network
plot_confusion_matrix(y_test, nn_pred, ['Benign', 'DDoS'], 'Nural Network Confusion Matrix')


# ### Model Comparison

# In[38]:


# Random Forest
rf_proba = rf_model.predict_proba(X_test)

# Logistic Regression
lr_proba = lr_model.predict_proba(X_test)

# Neural Network
nn_proba = nn_model.predict_proba(X_test)


# In[39]:


# Combine predictions for ROC Curve

# Calculate ROC curve for Random Forest
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_proba[:, 1])
rf_auc = auc(rf_fpr, rf_tpr)

# Calculate ROC curve for Logistic Regression
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_proba[:, 1])
lr_auc = auc(lr_fpr, lr_tpr)

# Callculate ROC curve for Neural Netork
nn_fpr, nn_tpr, _ = roc_curve(y_test, nn_proba[:, 1])
nn_auc = auc(nn_fpr, nn_tpr)


# In[40]:


# Plot Roc Cureve for all Models
plt.figure(figsize=(10, 6))
plt.plot(rf_fpr, rf_tpr, label=f'Randome Forest (AUC = {rf_auc:.2f})')
plt.plot(lr_fpr, lr_tpr, label=f'Logistic Regression (AUC = {lr_auc:.2f})')
plt.plot(nn_fpr, nn_tpr, label=f'Neural Network (AUC = {nn_auc:.2f})')

# Plot ROC curve for random classifier (50% area)
plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='Random Classifier (ACU = 0.50)')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.grid()
plt.show()


# In[ ]:




