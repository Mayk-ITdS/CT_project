import numpy as np
import pandas as pd


def merge_dataframes_with_transposition_check(df1: pd.DataFrame, df2: pd.DataFrame):

    np.array_equal(df1.values, df2.T.values)
    return
if __name__ == "__main__":
    books_df = pd.read_csv("./books.csv")
    books_transposed_df = pd.read_csv("./other_books.csv", index_col=0)
    merge_dataframes_with_transposition_check(books_df, books_transposed_df)
