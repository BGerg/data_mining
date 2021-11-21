import pandas as pd
import numpy as np

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


if __name__ == '__main__':
    main()