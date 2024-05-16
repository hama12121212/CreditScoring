import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


application_train = pd.read_csv('C:/Users/Barao/Downloads/New folder/application_train.csv')



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