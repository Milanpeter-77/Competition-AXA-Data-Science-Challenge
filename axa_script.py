# -*- coding: utf-8 -*-
"""
Created on 14/10/23

@author: Juline, Milán, Hans

"""
#%%
"""
PART 0 - Dependencies and Initial Settings
"""
# import libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV


# setting working directory
os.chdir("/Users/petermilan/Documents/University/University of Antwerp/Subjects/Machine Learning/AXA")

#%%
"""
PART I - Data Understanding
"""
# Reading Train and Test sets
Data = pd.read_csv('conversion_DB_Train.csv', sep = ',', index_col = 0, na_values = np.nan)
Test = pd.read_csv('conversion_DB_Test.csv', sep = ',', index_col = 0, na_values = np.nan)

# Infos
Test.info()
Data.info()

# Year_Month conversion into Year and Month variable
Data[['Year', 'Month']] = Data['Year_Month'].str.split('-', expand = True).astype(float)
Data = Data.drop(columns = ['Year_Month'])

Test[['Year', 'Month']] = Test['Year_Month'].str.split('-', expand = True).astype(float)
Test = Test.drop(columns = ['Year_Month'])

# Determine the types of variables
all_variables = Data.columns.tolist()
nominal_variables = ['profession', 'fuelType', 'make', 'model', 'postal_code_XX']
ordinal_variables = ['dero', 'animation', 'animation_2', 'customerScore', 'Year', 'Month']
discrete_variables = ['nbYearsAttest', 'birthday_5y', 'license_year_5y', 'vhConstructionYear', 'availableActions', 'purchaseTva', 'nbbackoffice']
continuous_variables = ['powerKW', 'catalogValue']
binary_variables = ['isSecondHandVehicle', 'mainDriverNotDesignated', 'premiumCustomer']


#%% Nominal Variables
for variable in nominal_variables:
    print('\n-----\nDescription and Plot(s) of variable ', variable)
    print('Number of unique values: ', len(Data[variable].unique()))
    print('Proportion of missing values: %1.2f%%' % (Data[variable].isna().sum() / (Data[variable].count() + Data[variable].isna().sum()) * 100))
    Data[variable].value_counts(dropna = False).plot(kind = 'bar')
    plt.show()


#%% Ordinal Variables
for variable in ordinal_variables:
    print('\n-----\nDescription and Plot(s) of variable ', variable)
    print('Number of unique values: ', len(Data[variable].unique()))
    print('Proportion of missing values: %1.2f%%' % (Data[variable].isna().sum() / (Data[variable].count() + Data[variable].isna().sum()) * 100))
    Data[variable].value_counts(dropna = False).sort_index(ascending = True).plot(kind = 'bar')
    plt.show()


#%% Discrete Variables
for variable in discrete_variables:
    print('\n-----\nDescription and Plot(s) of variable ', variable)
    print('Number of unique values: ', len(Data[variable].unique()))
    print('Proportion of missing values: %1.2f%%' % (Data[variable].isna().sum() / (Data[variable].count() + Data[variable].isna().sum()) * 100))
    Data[variable].value_counts(dropna = False).sort_index(ascending = True).plot(kind = 'bar')
    plt.show()
    print('\nStatistics:\n',Data[variable].describe())


#%% Countinuous Variables
for variable in continuous_variables:
    print('\n-----\nDescription and Plot(s) of variable ', variable)
    print('Number of unique values: ', len(Data[variable].unique()))
    print('Proportion of missing values: %1.2f%%' % (Data[variable].isna().sum() / (Data[variable].count() + Data[variable].isna().sum()) * 100))
    # Bosxplot
    Data.boxplot(column=[variable])
    plt.show()
    # Histogram
    Data[variable].hist(bins = 30)
    plt.show()
    print('\nStatistics:\n', Data[variable].describe())


#%% Binary Variables
for variable in binary_variables:
    print('\n-----\nDescription and Plot(s) of variable ', variable)
    print('Proportion of missing values: %1.2f%%' % (Data[variable].isna().sum() / (Data[variable].count() + Data[variable].isna().sum()) * 100))
    Data[variable].value_counts(dropna = False).plot(kind = 'pie', autopct = '%1.2f%%')
    plt.show()


#%%
"""
PART II - Data Preparation
"""
#%% Nominal Variables' Preparation
# Delete varibles
Data = Data.drop(columns = ['make', 'model', 'profession'])

# Replace values
# profession - nan to 0
# Data['profession'].fillna(0, inplace = True)

# fuelType - nan to 5
Data['fuelType'].fillna(5, inplace = True)

# postal_code_XX - 33-34 to 33; 64-65 to 64
Data['postal_code_XX'].replace('33-34', '33', inplace = True)
Data['postal_code_XX'].replace('64-65', '64', inplace = True)


# Converting varible data types
for variable in nominal_variables:
    if variable in Data.columns:
        Data[variable] = Data[variable].astype(int)
    else:
        continue

# Variable encoding - dummy encoding
for variable in nominal_variables:
    if variable in Data.columns:
        Data = pd.get_dummies(Data, columns = [variable], prefix = variable, drop_first = True, dtype = int)
    else:
        continue


#%% Ordinal Variables' Preparation
# Delete varibles
Data = Data.drop(columns = ['Year'])

# Converting varible data types
for variable in ordinal_variables:
    if variable in Data.columns:
        Data[variable] = Data[variable].astype(int)
    else:
        continue

# Variable encoding - thermometer encoding
for variable in ordinal_variables:
    if variable in Data.columns:
        for i in range(1, Data[variable].max() + 1):
            Data[f'{variable}_{i}'] = (Data[variable] >= i).astype(int)
            if i == (Data[variable].max()):
                Data.drop(columns = [variable], inplace = True)
    else:
        continue


#%% Discrete Variables' Preparation
# Delete varibles (due to data leakage)
Data = Data.drop(columns = ['availableActions', 'license_year_5y'])

# Replace values
# nbYearsAttest - nan to 6<
Data['nbYearsAttest'].fillna(7, inplace = True)

# birthday_5y - nan to mode
Data['birthday_5y'].fillna(Data.birthday_5y.median(), inplace = True)

# vhConstructionYear - nan to mode
Data['vhConstructionYear'].fillna(Data.vhConstructionYear.median(), inplace = True)

# purchaseTva - nan to 4<
Data['purchaseTva'].fillna(4, inplace = True)

# license_year_5y - nan to 0 - only for testing
# Data['license_year_5y'].fillna(2025, inplace = True)


# Converting varible data types
for variable in discrete_variables:
    if variable in Data.columns:
        Data[variable] = Data[variable].astype(int)
    else:
        continue


#%% Countinuous Variables' Preparation
# Replace values
# catalogValue - nan to mean
Data['catalogValue'].fillna(Data.catalogValue.mean(), inplace = True)

# Standardise variables
for variable in continuous_variables:
    Data[variable] = (Data[variable] - Data[variable].mean()) / Data[variable].std()

# Dealing with outliers
for variable in continuous_variables:
    Data[variable] = Data[variable].clip(lower = -3, upper = 3)


#%% Binary Variables' Preparation
# Delete varibles (due to data leakage)
Data = Data.drop(columns = ['isSecondHandVehicle'])

# Replace values
# mainDriverNotDesignated - nan to True; False to 0
Data['mainDriverNotDesignated'].fillna('1', inplace = True)
Data['mainDriverNotDesignated'].replace('False', '0', inplace = True)

# Converting varible data types
for variable in binary_variables:
    if variable in Data.columns:
        Data[variable] = Data[variable].astype(bool)
    else:
        continue

#%% Outlier observations
Data = Data[Data['birthday_5y'] != 1695]
Data = Data[Data['vhConstructionYear'] != 1014]


#%% Export data frame

Data.to_csv("Processed_Data.csv", index = False)
# Test.to_csv("Processed_Test_Data.csv", index = False)

#%%
"""
PART III - Data Modelling
"""
#%% Load the preprocessed data set
Data = pd.read_csv('Processed_Data.csv', sep = ',')

# Decrease the number of observation (so my computer does not die fitting and predicting)
# Data = Data.drop(index = Data.sample(frac = 0.75, random_state = 42).index)

# Divide the target variable and the 
X = Data.drop(columns = ['converted'])
y = Data['converted']

# Split the data into training and validation sets (80:20)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Load the preprocessed test set
# X_test = pd.read_csv('Processed_Test.csv', sep = ',')


#%% K Nearest Neighbors

# Create classifier
clf_KNN = KNeighborsClassifier()
# Fit the model on training data
clf_KNN.fit(X_train, y_train)
# Make predictions on validation set
y_val_labels = clf_KNN.predict(X_val)
y_val_scores = clf_KNN.predict_proba(X_val)[:,1]
# Calculate the accuracy on validation data
accuracy_val = accuracy_score(y_val, y_val_labels)
# AUC value on validation data
AUC_val = roc_auc_score(y_val, y_val_scores)
print("AUC Score with initial kNN: ", AUC_val)

# Grid search
param_grid = {
    'n_neighbors': list(range(1, 101)),
    'weights': ['uniform', 'distance'],
}
grid_search_KNN = GridSearchCV(clf_KNN, param_grid, scoring = 'roc_auc', cv = 5, verbose = 3)
# Fit the model on training data
grid_search_KNN.fit(X_train, y_train) #TAKES A LONG TIME TO RUN
# Make predictions on validation set
y_val_labels = grid_search_KNN.predict(X_val)
y_val_scores = grid_search_KNN.predict_proba(X_val)[:,1]
# Calculate the accuracy on validation data
accuracy_val = accuracy_score(y_val, y_val_labels)
# AUC value on validation data
AUC_val = roc_auc_score(y_val, y_val_scores)
print("AUC Score with grid search kNN: ", AUC_val)



# Draw ROC curve
fpr, tpr, threshold = metrics.roc_curve(y_val, y_val_scores)
roc_auc = metrics.auc(fpr, tpr)

plt.title('Receiver Operating Characteristic on validation data') #title
plt.plot(fpr, tpr, 'b', label = 'AUC KNN val = %0.4f' %roc_auc) #label shown in the legend
plt.legend(loc = 'lower right') #plot a legend in the lower right
plt.plot([0, 1], [0, 1],'r--') #plot diagonal indicating a random model
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate') #ylabel
plt.xlabel('False Positive Rate') #xlabel
plt.show()


#%% Decision Tree

# Create classifier
clf_DT = DecisionTreeClassifier(criterion = 'entropy')
# Fit the model on training data
clf_DT.fit(X_train, y_train)
# Make predictions on validation set
y_val_labels = clf_DT.predict(X_val)
y_val_scores = clf_DT.predict_proba(X_val)[:,1]
# Calculate the accuracy on validation data
accuracy_val = accuracy_score(y_val, y_val_labels)
# AUC value on validation data
AUC_val = roc_auc_score(y_val, y_val_scores)
print("AUC Score with initial DT: ", AUC_val)

# Grid search
param_grid = {
    'min_samples_leaf': list(range(1, 501))
}
grid_search_DT = GridSearchCV(clf_DT, param_grid, scoring = 'roc_auc', cv = 5, verbose = 3)
# Fit the model on training data
grid_search_DT.fit(X_train, y_train) #TAKES A LONG TIME TO RUN
# Make predictions on validation set
y_val_labels = grid_search_DT.predict(X_val)
y_val_scores = grid_search_DT.predict_proba(X_val)[:,1]
# Calculate the accuracy on validation data
accuracy_val = accuracy_score(y_val, y_val_labels)
# AUC value on validation data
AUC_val = roc_auc_score(y_val, y_val_scores)
print("AUC Score with grid search DT: ", AUC_val)


#%% Logistic Regression

# Create classifier
clf_LR = LogisticRegression()
# Fit the model on training data
clf_LR.fit(X_train, y_train)
# Make predictions on validation set
y_val_labels = clf_LR.predict(X_val)
y_val_scores = clf_LR.predict_proba(X_val)[:,1]
# Calculate the accuracy on validation data
accuracy_val = accuracy_score(y_val, y_val_labels)
# AUC value on validation data
AUC_val = roc_auc_score(y_val, y_val_scores)
print("AUC Score with initial LR: ", AUC_val)

# Grid search
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
}
grid_search_LR = GridSearchCV(clf_LR, param_grid, scoring = 'roc_auc', cv = 5, verbose = 3) #TAKES A LONG TIME TO RUN
# Fit the model on training data
grid_search_LR.fit(X_train, y_train)
# Make predictions on validation set
y_val_labels = grid_search_LR.predict(X_val)
y_val_scores = grid_search_LR.predict_proba(X_val)[:,1]
# Calculate the accuracy on validation data
accuracy_val = accuracy_score(y_val, y_val_labels)
# AUC value on validation data
AUC_val = roc_auc_score(y_val, y_val_scores)
print("AUC Score with grid search LR: ", AUC_val)



#%% Support Vector Classification

# Create classifier
clf_SVM = SVC(probability = True)
# Fit the model on training data
clf_SVM.fit(X_train, y_train)
# Make predictions on validation set
y_val_labels = clf_SVM.predict(X_val)
y_val_scores = clf_SVM.predict_proba(X_val)[:,1]
# Calculate the accuracy on validation data
accuracy_val = accuracy_score(y_val, y_val_labels)
# AUC value on validation data
AUC_val = roc_auc_score(y_val, y_val_scores)
print("AUC Score with initial SVM: ", AUC_val)



#%% Random Forest

# Create classifier
clf_RF = RandomForestClassifier(random_state = 42)
# Fit the model on training data
clf_RF.fit(X_train, y_train)
# Make predictions on validation set
y_val_labels = clf_RF.predict(X_val)
y_val_scores = clf_RF.predict_proba(X_val)[:,1]
# Calculate the accuracy on validation data
accuracy_val = accuracy_score(y_val, y_val_labels)
# AUC value on validation data
AUC_val = roc_auc_score(y_val, y_val_scores)
print("AUC Score with initial RF: ", AUC_val)


# Grid search
param_grid = {
    'min_samples_leaf': list(range(1, 501))
}
grid_search_RF = GridSearchCV(clf_DT, param_grid, scoring = 'roc_auc', cv = 5, verbose = 3)
# Fit the model on training data
grid_search_RF.fit(X_train, y_train) #TAKES A LONG TIME TO RUN
# Make predictions on validation set
y_val_labels = grid_search_RF.predict(X_val)
y_val_scores = grid_search_RF.predict_proba(X_val)[:,1]
# Calculate the accuracy on validation data
accuracy_val = accuracy_score(y_val, y_val_labels)
# AUC value on validation data
AUC_val = roc_auc_score(y_val, y_val_scores)
print("AUC Score with grid search DT: ", AUC_val)




