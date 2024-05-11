import os

import pandas as pd


def get_transactions_in_date_range(start_date, end_date):
    # Load the transactions CSV file from the transactions directory
    transactions_file = os.path.join(
        os.path.dirname(__file__), "..", "transactions", "transactions.csv"
    )
    transactions_df = pd.read_csv(transactions_file)

    # Filter transactions by date range
    transactions_df = transactions_df[
        (transactions_df["date"] >= start_date) & (transactions_df["date"] <= end_date)
    ]

    return transactions_df
