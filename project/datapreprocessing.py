import pandas as pd
import numpy as np

def main():
    df = pd.read_csv("wayne_county_poverty_2017_race_only.csv")

    sub_df = df[df.columns.drop(list(df.filter(regex='.*Percent below poverty level.*')))]
    sub_df = sub_df[sub_df.columns.drop(list(df.filter(regex='Label.*')))]

    sub_df.astype("str")  # ensure its all strings first
    sub_df = sub_df.replace(',','', regex=True)  # remove all commas

    data_array = sub_df.to_numpy()

    data_array = data_array.astype(int)
    print(data_array)


if __name__ == "__main__":
   main()
