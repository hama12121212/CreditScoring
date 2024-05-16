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



application_train = pd.get_dummies(application_train, drop_first=True)


X = application_train.drop(['SK_ID_CURR', 'TARGET'], axis=1)
y = application_train.TARGET
feature_name = X.columns.tolist()


def pearson_cor(X, y):
    cor_list = []
    # calculate the correlation with y for each feature
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
    # replace NaN with 0
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    # feature name
    cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-100:]].columns.tolist()
    # feature selection? 0 for not select, 1 for select
    yes_no = [True if i in cor_feature else False for i in feature_name]
    return yes_no, cor_feature
yes_no, cor_feature = pearson_cor(X, y)
print(str(len(cor_feature)), 'selected features')
print(cor_feature)