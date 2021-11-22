import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

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

def drop_non_numeric_rows_with_nan_value(df):
    df.dropna(inplace=True)

def create_histogram(df, cloumn_name, property_one, property_two, property_three):
    sns.FacetGrid(df, hue=cloumn_name, height=3).map(sns.distplot, property_one).add_legend()
    sns.FacetGrid(df, hue=cloumn_name, height=3).map(sns.distplot, property_two).add_legend()
    sns.FacetGrid(df, hue=cloumn_name, height=3).map(sns.distplot, property_three).add_legend()
    plt.show()

def visualizing_data_distribution(df, cloumn_name):
    sns.set_style("whitegrid")
    sns.pairplot(df, hue=cloumn_name, height=3);
    plt.show()

def main():
    df = pd.read_csv("penguins.csv")
    print(df)
    cleaned_csv_name = "penguins_WithoutDuplicates.csv"
    df.drop_duplicates(inplace=True)
    df.to_csv(cleaned_csv_name)
    looking_patterns = ["n/a", "na", "-", "--"]
    df = set_correct_column_type(df)

    print(df.isnull().sum())
    numeric_columns, non_numeric_columns = get_numeric_columns(df)
    df = change_missing_numbers_to_nan(cleaned_csv_name,
                                       looking_patterns,
                                       numeric_columns)

    set_missing_nan_values_to_median(df, numeric_columns)
    drop_non_numeric_rows_with_nan_value(df)

    df.info()
    print(df)

    create_histogram(df,"species","bill_length_mm", "bill_depth_mm", "flipper_length_mm")
    visualizing_data_distribution(df,"species")

    wcss = []
    x = df.iloc[:, [ 3,4,5]].values
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(x)
        wcss.append(kmeans.inertia_)

    plt.plot(range(1, 11), wcss)
    plt.title('The elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')  # within cluster sum of squares
    plt.show()
if __name__ == '__main__':
    main()