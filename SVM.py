# =============================================================================
#
# Machine Learning for Acoustic Anomaly Detection
# Support Machine Vector (SVM) program (version 1.6)
# Alex Renaudin 31 October 2023
#
# This program builds a Support Vector Classifier (SVC) and records its
# performance by averaging scores across 1000 runs. 
#
# Tuneable program parameters:
#    - Testing split proportion 
#    - Number of runs
#    - C value
#    - Kernel type
# 
# Based off YouTube playlist 'Machine Learning with Python' by Sentdex
# https://www.youtube.com/playlist?list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v
#
# =============================================================================

import numpy as np
import sklearn.metrics
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

# Load in dataset
df = pd.read_csv('dataset.csv') 

# Assign class features to X and y arrays
X = np.array(df.drop(labels=['class'], axis=1))
y = np.array(df['class'])

# Program parameters:
SPLIT = 0.20        # Proportion of testing set (e.g. 0.200 = 20% testing)
NUM_RUNS = 1000     # Number of iterations of model 
C = 100000          # 'Regularisation parameter' for SVC
KERNEL = 'rbf'      # Kernel type for SVC

# Initialise performance metric variables
accuracy = []                           # Total accuracy score
precision_S = []; precision_A = []      # Precision
recall_S = []; recall_A = []            # Recall
F1_S = []; F1_A = []                    # F1 score
support_S = []; support_A = []          # Support
cfm = []                                # Confusion matrix

# Print parameters
print(f'\n\nRunning {NUM_RUNS} SVM models with a {SPLIT:.0%} split...\n')
    
# Iterate NUM_RUNS times, creating and testing new SVM model each time
for _ in range(NUM_RUNS):

    # Randomly seperate data into training and testing clumps
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=SPLIT)
    model = svm.SVC(kernel = KERNEL, C = C) # Creates the model classifier
    model.fit(X_train, y_train) # Fits the training sets to the classifier
    
    # Compute predicted answers based off testing dataset
    y_pred = model.predict(X_test)
    
    # Compute performance metrics for this run
    report = precision_recall_fscore_support(y_test, y_pred)
    
    # Append performance metrics values
    accuracy.append(model.score(X_test, y_test))
    precision_S.append(report[0][0])
    recall_S.append(report[0][1])
    precision_A.append(report[1][0])
    recall_A.append(report[1][1])
    F1_S.append(report[2][0])
    F1_A.append(report[2][1])
    support_S.append(report[3][0])
    support_A.append(report[3][1])
    cfm.append(sklearn.metrics.confusion_matrix(y_test, y_pred))

# Print results
print('Complete! Mean model performance metrics:\n')

print(f'\tSize of dataset          = {len(df)} samples')
print(f'\tOverall accuracy         = {100*np.mean(accuracy):.2f}%')
print(f'\tStandard deviation       = {100*np.std(accuracy):.2f}%\n')

print(f'\tSatisfactory:  Precision = {100*np.mean(precision_S):.2f}%')
print(f'\t               Recall    = {100*np.mean(recall_S):.2f}%')
print(f'\t               F1 score  = {100*np.mean(F1_S):.2f}%')
print(f'\t               Support   = {round(np.mean(support_S))} samples\n')


print(f'\tAnomalous:     Precision = {100*np.mean(precision_A):.2f}%')
print(f'\t               Recall    = {100*np.mean(recall_A):.2f}%')
print(f'\t               F1 score  = {100*np.mean(F1_A):.2f}%')
print(f'\t               Support   = {round(np.mean(support_A))} samples\n')

# Function to generate confusion matrix plot (CMP)
def cmp(cfm, display_labels, title='KNN Confusion Matrix (Averaged)'):

    # Plot confusion matrix using matplotlib
    d=sklearn.metrics.ConfusionMatrixDisplay(cfm,display_labels=display_labels)
    d.plot(colorbar=False)
    plt.title(title)
    plt.show()
    
# Compute confusion matrix averaged & rounded across all runs
cfm_averaged = np.round(np.mean(cfm, axis=0)).astype(int)
    
# Call confusion matrix function to display results for last run
cmp(cfm_averaged, ['Satisfactory', 'Anomalous'])
