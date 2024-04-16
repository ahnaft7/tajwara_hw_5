"""
Ahnaf Tajwar
Class: CS 677
Date: 4/14/24
Homework Problems # 1-5
Description of Problem (just a 1-2 line summary!): These problems are to compare the accuracy for each classifier model and compute the confusion matrices.
    For the Random Forest classifier, the best Subtree count and depth combination was also computed and the error rates were plotted.
"""

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

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

# Dictionary for metrics
model_metrics = {}

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

# Calculate True Positive Rate
tpr = tp / (tp + fn)

# Calculate True Negative Rate
tnr = tn / (tn + fp)

print("True Positive Rate:", tpr)
print("True Negative Rate:", tnr)

model_metrics['NB'] = {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn, 'accuracy': accuracy, 'TPR': tpr, 'TNR': tnr}

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

# Calculate True Positive Rate
tpr = tp / (tp + fn)

# Calculate True Negative Rate
tnr = tn / (tn + fp)

print("True Positive Rate:", tpr)
print("True Negative Rate:", tnr)

model_metrics['DT'] = {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn, 'accuracy': accuracy, 'TPR': tpr, 'TNR': tnr}

#---------------------Random Forest-------------------

print("\n-------------Random Forest--------------")

# Initialize lists to store error rates and hyperparameters
error_rates = []
N_values = range(1, 11)
d_values = range(1, 6)

# Loop through each combination of N and d
for N in N_values:
    for d in d_values:
        
        # Initialize and train the Random Forest classifier
        rf_classifier = RandomForestClassifier(n_estimators=N, max_depth=d, criterion='entropy', random_state=1)
        rf_classifier.fit(X_train, Y_train)
        
        # Predict class labels for the testing set
        Y_pred = rf_classifier.predict(X_test)
        
        # Compute the error rate
        error_rate = 1 - accuracy_score(Y_test, Y_pred)
        
        # Store the error rate and hyperparameters
        error_rates.append((N, d, error_rate))

# Convert the error rates to a NumPy array
error_rates = np.array(error_rates)

# Plot the error rates
plt.figure(figsize=(10, 6))
for d in d_values:
    plt.plot(N_values, error_rates[error_rates[:, 1] == d][:, 2], label=f'd={d}')
plt.title('Error Rates for Different Values of N and d')
plt.xlabel('Number of (Sub)Trees (N)')
plt.ylabel('Error Rate')
plt.xticks(N_values)
plt.legend(title='Max Depth (d)')
plt.grid(True)
plt.savefig('Random_Forest_Combinations.png')

# Best N and d combination (N = 10, d = 5)
N = 10
d = 5
print(f"\nBest N and d combination: N = {N} d = {d}")

# Initialize and train the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=N, max_depth=d, criterion='entropy', random_state=1)
rf_classifier.fit(X_train, Y_train)

# Predict class labels for the testing set
Y_pred = rf_classifier.predict(X_test)

accuracy = accuracy_score(Y_test, Y_pred)
print("\nAccuracy: ", accuracy)

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

# Calculate True Positive Rate
tpr = tp / (tp + fn)

# Calculate True Negative Rate
tnr = tn / (tn + fp)

print("True Positive Rate:", tpr)
print("True Negative Rate:", tnr)

model_metrics['RF'] = {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn, 'accuracy': accuracy, 'TPR': tpr, 'TNR': tnr}

# Create table of metrics
metrics_df = pd.DataFrame(model_metrics).T

print("\n", metrics_df)
