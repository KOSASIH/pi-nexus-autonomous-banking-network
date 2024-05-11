import datetime
import os

import pandas as pd

from . import transactions


def generate_monthly_report(start_date, end_date):
    # Get all transactions within the specified date range
    transactions_df = transactions.get_transactions_in_date_range(start_date, end_date)

    # Group transactions by user and calculate total amount spent
    user_spending_df = transactions_df.groupby("user_id")["amount"].sum()

    # Create a new DataFrame with the user ID, total amount spent, and the date range
    report_df = pd.DataFrame(
        {
            "user_id": user_spending_df.index,
            "total_amount_spent": user_spending_df.values,
            "date_range": [f"{start_date} - {end_date}"] * len(user_spending_df),
        }
    )

    # Save the report to a CSV file in the reports directory
    report_file = f"monthly_report_{start_date}_{end_date}.csv"
    report_path = os.path.join(os.path.dirname(__file__), "reports", report_file)
    report_df.to_csv(report_path, index=False)

    return report_df
