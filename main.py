#Importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pandas.plotting import parallel_coordinates

#Metrics
from sklearn import metrics
from sklearn.metrics import make_scorer, accuracy_score,precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score ,precision_score,recall_score,f1_score

#Model Select
from sklearn.cluster import KMeans
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import  LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder

def change_missing_numbers_to_nan(csv_name: str, looking_patterns: list, numeric_columns: list ):
    df = pd.read_csv(csv_name, na_values=looking_patterns)

    for column_name in numeric_columns:
        cnt = 0
        for value in df[column_name]:
            try:
                int(value)
                if int(value) <= 0:
                    df.loc[cnt, column_name] = np.nan
            except ValueError:
                df.loc[cnt, column_name] = np.nan
            cnt += 1
    df = df.drop('Unnamed: 0', 1)
    return df

def get_numeric_columns(df):
    numeric_columns = []
    non_numeric_columns = []

    for column_name in df.columns:
        for value in df[column_name]:
            if not pd.isnull(value):
                try:
                    int(value)
                    if column_name not in numeric_columns:
                        numeric_columns.append(column_name)
                except ValueError:
                    if column_name not in non_numeric_columns:
                        non_numeric_columns.append(column_name)

    return numeric_columns, non_numeric_columns

def set_correct_column_type(df):
    wrong_types_columns = []

    for column_name in df.columns:
        for value in df[column_name]:
            if not pd.isnull(value):
                try:
                    if float(value):
                        if column_name not in wrong_types_columns:
                            wrong_types_columns.append(column_name)
                except:
                    pass
    for column in wrong_types_columns:
        df[column] = pd.to_numeric(df[column], errors= 'coerce')
    return df

def set_missing_nan_values_to_median(df, numeric_columns):
    for column in numeric_columns:
        median = df[column].median()
        df[column].fillna(median, inplace=True)


def create_histogram(df, cloumn_name, property_one, property_two, property_three):
    sns.FacetGrid(df, hue=cloumn_name, height=3).map(sns.distplot, property_one).add_legend()
    sns.FacetGrid(df, hue=cloumn_name, height=3).map(sns.distplot, property_two).add_legend()
    sns.FacetGrid(df, hue=cloumn_name, height=3).map(sns.distplot, property_three).add_legend()
    plt.show()

def visualizing_data_distribution(df, cloumn_name):
    sns.set_style("whitegrid")
    sns.pairplot(df, hue=cloumn_name, height=3);
    plt.show()

def kmeans_algorithm(df):
    wcss = []
    x = df.iloc[:, [ 3,4,5]].values
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(x)
        wcss.append(kmeans.inertia_)

    plt.plot(range(1, 11), wcss)
    plt.title('The elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

    kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
    y_kmeans = kmeans.fit_predict(x)

    # Visualising the clusters
    plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s=100, c='purple', label='Adelie')
    plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s=100, c='orange', label='Chinstrap')
    plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s=100, c='green', label='Gentoo')

    # Plotting the centroids of the clusters
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='red', label='Centroids')

    plt.legend()

def classification_with_knn(df):
    print(pd.crosstab(index=df["species"], columns="count"))
    # Feature selection
    X = df.iloc[:, 3:4].values
    y = df.iloc[:, 3].values


    le = LabelEncoder()
    y = le.fit_transform(y)

    train, test = train_test_split(df, test_size=0.3,
                                   stratify=df['species'], random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    n_bins = 10
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].hist(train['bill_length_mm'], bins=n_bins);
    axs[0, 0].set_title('Bill length mm');
    axs[0, 1].hist(train['bill_depth_mm'], bins=n_bins);
    axs[0, 1].set_title('Bill depth mm');
    axs[1, 0].hist(train['flipper_length_mm'], bins=n_bins);
    axs[1, 0].set_title('Flipper length mm');
    fig.tight_layout(pad=1.0)
    plt.show()

    # K-NN (K-Nearest Neighbor)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    Y_pred = knn.predict(X_test)

    cm = metrics.confusion_matrix(y_test, Y_pred)
    accuracy = metrics.accuracy_score(y_test, Y_pred)
    precision = metrics.precision_score(y_test, Y_pred, average='micro')
    recall = metrics.recall_score(y_test, Y_pred, average='micro')
    f1 = metrics.f1_score(y_test, Y_pred, average='micro')
    print('Confusion matrix for KNN\n', cm)
    print('accuracy_KNN : %.3f' % accuracy)
    print('precision_KNN : %.3f' % precision)
    print('recall_KNN: %.3f' % recall)
    print('f1-score_KNN : %.3f' % f1)

def clustering_with_gauss(df):
    print(pd.crosstab(index=df["species"], columns="count"))
    # Feature selection
    X = df.iloc[:, 3:4].values
    y = df.iloc[:, 3].values

    # Label encoding:
    # categorical labels are transformed into numbers
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Split the data set before classification
    # Train set: 70% of data to train the model
    # Test set: 15% of data to test the model
    # Validation set: 15% of data used to evaluate
    # the performance of each classifier and fine-tune
    # the model parameters
    # Now we have small data set, therefore test set = validation set
    train, test = train_test_split(df, test_size=0.3,
                                   stratify=df['species'], random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Gaussian Naive Bayes

    # Classification algorithm for binary and
    # multi-class classification problems.

    gaussian = GaussianNB()
    gaussian.fit(X_train, y_train)
    Y_pred = gaussian.predict(X_test)
    accuracy_nb = round(metrics.accuracy_score(y_test, Y_pred) * 100, 2)
    acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)

    cm = metrics.confusion_matrix(y_test, Y_pred)
    accuracy = metrics.accuracy_score(y_test, Y_pred)
    precision = metrics.precision_score(y_test, Y_pred, average='micro')
    recall = metrics.recall_score(y_test, Y_pred, average='micro')
    f1 = metrics.f1_score(y_test, Y_pred, average='micro')
    print('Confusion matrix for Naive Bayes\n', cm)
    print('accuracy_Naive Bayes: %.3f' % accuracy)
    print('precision_Naive Bayes: %.3f' % precision)
    print('recall_Naive Bayes: %.3f' % recall)
    print('f1-score_Naive Bayes : %.3f' % f1)

def main():
    df = pd.read_csv("penguins.csv")
    cleaned_csv_name = "penguins_WithoutDuplicates.csv"

    #cleaning
    df.drop_duplicates(inplace=True)
    df.to_csv(cleaned_csv_name)
    looking_patterns = ["n/a", "na", "-", "--"]
    df = set_correct_column_type(df)
    print(df)
    print(df.isnull().sum())
    numeric_columns, non_numeric_columns = get_numeric_columns(df)
    df = change_missing_numbers_to_nan(cleaned_csv_name,
                                       looking_patterns,
                                       numeric_columns)

    set_missing_nan_values_to_median(df, numeric_columns)
    print(df)
    df.dropna(inplace=True)
    print(df)
    #classification
    create_histogram(df,"species","bill_length_mm", "bill_depth_mm", "flipper_length_mm")
    kmeans_algorithm(df)
    visualizing_data_distribution(df,"species")

    #clustering
    df.info()
    classification_with_knn(df)
    clustering_with_gauss(df)


if __name__ == '__main__':
    main()