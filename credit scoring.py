import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt




application_train = pd.read_csv('C:/Users/Barao/Downloads/New folder/application_train.csv')
application_test= pd.read_csv('C:/Users/Barao/Downloads/New folder/application_test.csv')
installments_payments = pd.read_csv('C:/Users/Barao/Downloads/New folder/installments_payments.csv')
bureau = pd.read_csv('C:/Users/Barao/Downloads/New folder/bureau.csv')
credit_card_balance = pd.read_csv('C:/Users/Barao/Downloads/New folder/credit_card_balance.csv')
bureau_balance = pd.read_csv('C:/Users/Barao/Downloads/New folder/bureau_balance.csv')
POS_CASH_balance = pd.read_csv('C:/Users/Barao/Downloads/New folder/POS_CASH_balance.csv')
previous_application = pd.read_csv('C:/Users/Barao/Downloads/New folder/previous_application.csv')



def drop_rows_with_excessive_missing(data, threshold):
    # Compute the ratio of missing values in each row
    missing_ratio = data.isnull().sum(axis=1) / data.shape[1]
    
    # Filter rows that exceed the threshold
    rows_to_drop = missing_ratio[missing_ratio > threshold].index
    
    # Drop rows from the DataFrame
    data_dropped = data.drop(rows_to_drop)
    
    return data_dropped


threshold = 0.3

application_train_dropped = drop_rows_with_excessive_missing(application_train, threshold)

installments_payments_dropped = drop_rows_with_excessive_missing(installments_payments, threshold)

bureau_dropped = drop_rows_with_excessive_missing(bureau, threshold)

credit_card_balance_dropped = drop_rows_with_excessive_missing(credit_card_balance, threshold)

bureau_balance_dropped = drop_rows_with_excessive_missing(bureau_balance, threshold)

POS_CASH_balance_dropped = drop_rows_with_excessive_missing(POS_CASH_balance, threshold)

previous_application_dropped = drop_rows_with_excessive_missing(previous_application, threshold)




import csv

output_file_path = "C:/Users/Barao/Downloads/application_train_dropped.csv"

# Convert the DataFrame to a list of lists
data_to_write = application_train_dropped.values.tolist()

# Write the data to the CSV file
with open(output_file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data_to_write)



output_file_path = "C:/Users/Barao/Downloads/installments_payments_dropped.csv"

# Convert the DataFrame to a list of lists
data_to_write = installments_payments_dropped.values.tolist()

# Write the data to the CSV file
with open(output_file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data_to_write)


output_file_path = "C:/Users/Barao/Downloads/bureau_dropped.csv"

# Convert the DataFrame to a list of lists
data_to_write = bureau_dropped.values.tolist()

# Write the data to the CSV file
with open(output_file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data_to_write)




output_file_path = "C:/Users/Barao/Downloads/credit_card_balance_dropped.csv"

# Convert the DataFrame to a list of lists
data_to_write = credit_card_balance_dropped.values.tolist()

# Write the data to the CSV file
with open(output_file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data_to_write)


output_file_path = "C:/Users/Barao/Downloads/bureau_balance_dropped.csv"

# Convert the DataFrame to a list of lists
data_to_write = bureau_balance_dropped.values.tolist()

# Write the data to the CSV file
with open(output_file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data_to_write)


output_file_path = "C:/Users/Barao/Downloads/previous_application_dropped.csv"

# Convert the DataFrame to a list of lists
data_to_write = previous_application_dropped.values.tolist()

# Write the data to the CSV file
with open(output_file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data_to_write)



output_file_path = "C:/Users/Barao/Downloads/POS_CASH_balance_dropped.csv"

# Convert the DataFrame to a list of lists
data_to_write = POS_CASH_balance_dropped.values.tolist()

# Write the data to the CSV file
with open(output_file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data_to_write)












def plot_stats_pie(feature):
    temp = application_train[feature].value_counts()
    df1 = pd.DataFrame({feature: temp.index, 'Number of contracts': temp.values})

    # Calculate the percentage of target=1 per category value
    cat_perc = application_train[[feature, 'TARGET']].groupby([feature], as_index=False).mean()
    cat_perc.sort_values(by='TARGET', ascending=False, inplace=True)

    colors = ['brown', 'green']
    inverse_colors = colors[::-1]  # Reverse the order of colors

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    # Plotting the pie chart for Number of contracts
    plt.pie(df1['Number of contracts'], labels=df1[feature], autopct='%1.1f%%', startangle=90, colors=colors)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title('Number of Contracts by ' + feature)

    plt.subplot(1, 2, 2)
    # Plotting the pie chart for TARGET with inverse colors
    plt.pie(cat_perc['TARGET'], labels=cat_perc[feature], autopct='%1.1f%%', startangle=90, colors=inverse_colors)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title('Percent of target with value 1 by ' + feature)

    plt.tight_layout()
    plt.show()



plot_stats_pie('CODE_GENDER')

plot_stats_pie('FLAG_OWN_CAR')








categorical_list = []
numerical_list = []
for i in application_train.columns.tolist():
    if application_train[i].dtype=='object':
        categorical_list.append(i)
    else:
        numerical_list.append(i)
print('categorical features:', str(len(categorical_list)))
print('numerical features:', str(len(numerical_list)))




#Data filtering


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



application_bureau_train = application_train.merge(bureau, left_on='SK_ID_CURR', right_on='SK_ID_CURR', how='inner')

print("The resulting dataframe `application_bureau_train` has ",application_bureau_train.shape[0]," rows and ", 
      application_bureau_train.shape[1]," columns.")


def plot_b_stats(feature,label_rotation=False,horizontal_layout=True):
    temp = application_bureau_train[feature].value_counts()
    df1 = pd.DataFrame({feature: temp.index,'Number of contracts': temp.values})

    # Calculate the percentage of target=1 per category value
    cat_perc = application_bureau_train[[feature, 'TARGET']].groupby([feature],as_index=False).mean()
    cat_perc.sort_values(by='TARGET', ascending=False, inplace=True)
    
    if(horizontal_layout):
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))
    else:
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12,14))
    sns.set_color_codes("pastel")
    s = sns.barplot(ax=ax1, x = feature, y="Number of contracts",data=df1)
    if(label_rotation):
        s.set_xticklabels(s.get_xticklabels(),rotation=90)
    
    s = sns.barplot(ax=ax2, x = feature, y='TARGET', order=cat_perc[feature], data=cat_perc)
    if(label_rotation):
        s.set_xticklabels(s.get_xticklabels(),rotation=90)
    plt.ylabel('Percent of target with value 1 [%]', fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.show();


plot_b_stats('CREDIT_ACTIVE')
plot_b_stats('CREDIT_CURRENCY')
plot_b_stats('CREDIT_TYPE', True, True)

