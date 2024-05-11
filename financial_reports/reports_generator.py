import datetime
from . import reports

def generate_monthly_reports():
    # Get the current date
    current_date = datetime.date.today()

    # Generate reports for the previous 12 months
    for i in range(1, 13):
        start_date = current_date - datetime.timedelta(days=31*i)
        end_date = start_date + datetime.timedelta(days=30)
        reports.generate_monthly_report(start_date, end_date)
