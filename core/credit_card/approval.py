import random

def approve_application(application):
    # Define the minimum income requirement for credit card approval
    MIN_INCOME = 50000

    # Define the minimum credit score requirement for credit card approval
    MIN_CREDIT_SCORE = 700

    # Define the employment status requirement for credit card approval
    ACCEPTED_EMPLOYMENT_STATUSES = ["full-time", "part-time"]

    # Define the home ownership requirement for credit card approval
    ACCEPTED_HOME_OWNERSHIPS = ["owned", "rented"]

    # Check if the applicant meets the minimum income requirement
    if application.income < MIN_INCOME:
        return False

    # Check if the applicant meets the minimum credit score requirement
    if application.credit_score < MIN_CREDIT_SCORE:
        return False

    # Check if the applicant's employment status is accepted
    if application.employment_status not in ACCEPTED_EMPLOYMENT_STATUSES:
        return False

    # Check if the applicant's home ownership status is accepted
    if application.home_ownership not in ACCEPTED_HOME_OWNERSHIPS:
        return False

    # If the applicant meets all the requirements, randomly approve or deny the application
    # This simulates a real-world scenario where not all applicants who meet the requirements are approved
    return random.choice([True, False])
