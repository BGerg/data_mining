import toml
import pandas as pd
import numpy as np

df = pd.read_csv("planets.csv")
print(df)

# Order by given field
df.sort_values("year", inplace = True)
print(df)

# Select duplicate rows based on one column
duplicateRowsDF = df[df.duplicated(['year'])]
print("Duplicate rows:", duplicateRowsDF, sep='\n')
# Remove duplicates from dataframe - keeping first occurrence
df.drop_duplicates(subset =['year'], keep = "first",
                   inplace = True)
print(df)

# Save dataframe to file
df.to_csv("planets_WithoutDuplicates.csv")

# Detect missing and invalid fields

# Making a list of missing value types
# Phyton will consider these values as NaN
missing_values = ["n/a", "na", "-", "--"]
df = pd.read_csv("planets_WithoutDuplicates.csv",
                 na_values = missing_values)

# Check if price is number - otherwise treat it as missing value
cnt=0
for value in df['mass']:
    try:
        int(value)
        if int(value)<=0 :  # price cannot be negative
            df.loc[cnt, 'mass']=np.nan
    except ValueError:
        df.loc[cnt, 'mass']=np.nan
    cnt+=1

# Total missing values for each feature
print(df.isnull().sum())

# Replace missing and incorrect PRICE values using median
median = df['mass'].median()
df['mass'].fillna(median, inplace=True)

# Select rows where value is missing
df.dropna(inplace=True)

print(df)