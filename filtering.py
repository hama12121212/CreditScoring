import pandas as pd
import numpy as np



application_train = pd.read_csv('C:/Users/Barao/Downloads/New folder/application_train.csv')

categorical_list = []
numerical_list = []
for i in application_train.columns.tolist():
    if application_train[i].dtype=='object':
        categorical_list.append(i)
    else:
        numerical_list.append(i)
print('categorical features:', str(len(categorical_list)))
print('numerical features:', str(len(numerical_list)))
