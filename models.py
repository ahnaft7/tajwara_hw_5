"""
Ahnaf Tajwar
Class: CS 677
Date: 4/14/24
Homework Problem # 1
Description of Problem (just a 1-2 line summary!): This problem is to
"""

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

# Reading in excel file
df = pd.read_excel('CTG.xls', sheet_name='Raw Data')
# Dropping null rows
df = df.drop(0)
df = df[:-3]
# Replacing NSP values 2 and 3 with 0
df.loc[df['NSP'].isin([2, 3]), 'NSP'] = 0

X = df[['LB', 'ALTV', 'Min', 'Mean']]
Y = df[['NSP']]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=1)
Y_train = np.ravel(Y_train)
Y_test = np.ravel(Y_test)
# train_df = pd.concat([X_train, Y_train], axis=1)
# test_df = pd.concat([X_test, Y_test], axis=1)
# print(train_df)
# print(test_df)

#---------------------Naive-Bayes-------------------

print("\n-------------Naive-Bayes--------------")
# Initialize and train the Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, Y_train)

# Predict class labels in X_test
Y_pred = nb_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(Y_test, Y_pred)
print("\nAccuracy:", accuracy)


# Compute confusion matrix
conf_matrix = confusion_matrix(Y_test, Y_pred)

print("Confusion Matrix: (0 is Positive (Abnormal), 1 is Negative (Normal))")
print("\t\tPredicted labels")
print("\t\t 0    1")
print("Actual labels 0", conf_matrix[0])
print("              1", conf_matrix[1])

tp = conf_matrix[0][0]
fn = conf_matrix[0][1]
fp = conf_matrix[1][0]
tn = conf_matrix[1][1]
print("TP: ", tp)
print("FP: ", fp)
print("TN: ", tn)
print("FN: ", fn)

#---------------------Decision Tree-------------------

print("\n-------------Decision Tree--------------")
# Initialize and train the Decision Tree classifier
dt_classifier = DecisionTreeClassifier(random_state=1)
dt_classifier.fit(X_train, Y_train)

# Predict class labels in X_test
Y_pred = dt_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(Y_test, Y_pred)
print("\nAccuracy:", accuracy)


# Compute confusion matrix
conf_matrix = confusion_matrix(Y_test, Y_pred)

print("Confusion Matrix: (0 is Positive (Abnormal), 1 is Negative (Normal))")
print("\t\tPredicted labels")
print("\t\t 0    1")
print("Actual labels 0", conf_matrix[0])
print("              1", conf_matrix[1])

tp = conf_matrix[0][0]
fn = conf_matrix[0][1]
fp = conf_matrix[1][0]
tn = conf_matrix[1][1]
print("TP: ", tp)
print("FP: ", fp)
print("TN: ", tn)
print("FN: ", fn)

